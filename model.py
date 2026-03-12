"""
RAMCF model: multimodal drug-side effect frequency prediction.
Drug encoder (4 modalities: fp, img, ppi, smiles) + SE encoder (hlgt, soc, semantic)
+ cross-modal transformer + ordinal prediction head.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMLP(nn.Module):
    """Residual MLP block: x + MLP(x), with LayerNorm."""

    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


class ModalityProjector(nn.Module):
    """Project raw modality features to embed_dim."""

    def __init__(self, in_dim: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        return self.proj(x)


class AdaptiveModalFusion(nn.Module):
    """Fuse modalities via Transformer + quality-gated weighting."""

    def __init__(self, embed_dim: int, num_modalities: int, num_heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.quality_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1),
        )
        self.modality_scale = nn.Parameter(torch.ones(num_modalities))

    def forward(self, x):
        x = self.transformer(x)
        gate_logits = self.quality_gate(x).squeeze(-1)  # (batch, n_mod)
        scale = self.modality_scale.unsqueeze(0)  # (1, n_mod)
        weights = torch.softmax(gate_logits * scale, dim=1).unsqueeze(-1)  # (batch, n_mod, 1)
        return (x * weights).sum(dim=1)


class DrugEncoder(nn.Module):
    """Encode 4 drug modalities (fp, img, ppi, smiles) + optional global embed."""

    def __init__(
        self,
        num_drugs: int = 638,
        modal_dim: int = 256,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_fusion_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.projectors = nn.ModuleList([
            ModalityProjector(modal_dim, embed_dim, dropout) for _ in range(4)
        ])
        self.modality_embed = nn.Embedding(4, embed_dim)
        self.drug_global_embed = nn.Embedding(num_drugs, embed_dim)
        nn.init.normal_(self.drug_global_embed.weight, std=0.02)
        self.fusion = AdaptiveModalFusion(
            embed_dim, num_modalities=4, num_heads=num_heads,
            num_layers=num_fusion_layers, dropout=dropout,
        )
        self.post_norm = nn.LayerNorm(embed_dim)

    def forward(self, drug_feats, drug_idx=None):
        batch_size = drug_feats.size(0)
        tokens = []
        for i, proj in enumerate(self.projectors):
            tokens.append(proj(drug_feats[:, i, :]))
        x = torch.stack(tokens, dim=1)
        mod_ids = torch.arange(4, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.modality_embed(mod_ids)
        fused = self.fusion(x)

        if drug_idx is not None:
            global_emb = self.drug_global_embed(drug_idx)
            fused = fused + global_emb

        return self.post_norm(fused)


class SideEffectEncoder(nn.Module):
    """Encode 3 SE modalities (hlgt, soc, semantic) + optional global embed."""

    def __init__(
        self,
        num_se: int = 994,
        hlgt_dim: int = 105,
        soc_dim: int = 26,
        semantic_dim: int = 300,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_fusion_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj_hlgt = ModalityProjector(hlgt_dim, embed_dim, dropout)
        self.proj_soc = ModalityProjector(soc_dim, embed_dim, dropout)
        self.proj_semantic = ModalityProjector(semantic_dim, embed_dim, dropout)
        self.modality_embed = nn.Embedding(3, embed_dim)
        self.se_global_embed = nn.Embedding(num_se, embed_dim)
        nn.init.normal_(self.se_global_embed.weight, std=0.02)
        self.fusion = AdaptiveModalFusion(
            embed_dim, num_modalities=3, num_heads=num_heads,
            num_layers=num_fusion_layers, dropout=dropout,
        )
        self.post_norm = nn.LayerNorm(embed_dim)

    def forward(self, se_hlgt, se_soc, se_semantic, se_idx=None):
        batch_size = se_hlgt.size(0)
        t1 = self.proj_hlgt(se_hlgt)
        t2 = self.proj_soc(se_soc)
        t3 = self.proj_semantic(se_semantic)
        x = torch.stack([t1, t2, t3], dim=1)
        mod_ids = torch.arange(3, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.modality_embed(mod_ids)
        fused = self.fusion(x)

        if se_idx is not None:
            global_emb = self.se_global_embed(se_idx)
            fused = fused + global_emb

        return self.post_norm(fused)


class CrossModalTransformerLayer(nn.Module):
    """Cross-attention between drug and SE representations."""

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn_drug = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_se = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_d1 = nn.LayerNorm(embed_dim)
        self.norm_s1 = nn.LayerNorm(embed_dim)
        self.ffn_d = ResidualMLP(embed_dim, dropout=dropout)
        self.ffn_s = ResidualMLP(embed_dim, dropout=dropout)

    def forward(self, drug_emb, se_emb):
        d = drug_emb.unsqueeze(1)
        s = se_emb.unsqueeze(1)

        d_cross, _ = self.cross_attn_drug(d, s, s)
        d = self.norm_d1(d + d_cross)
        d = d.squeeze(1)
        d = self.ffn_d(d)

        s_cross, _ = self.cross_attn_se(s, d.unsqueeze(1), d.unsqueeze(1))
        s = self.norm_s1(s + s_cross)
        s = s.squeeze(1)
        s = self.ffn_s(s)

        return d, s


class CrossModalTransformer(nn.Module):

    def __init__(self, embed_dim=256, num_heads=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossModalTransformerLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, drug_emb, se_emb):
        for layer in self.layers:
            drug_emb, se_emb = layer(drug_emb, se_emb)
        return drug_emb, se_emb


class RankAwareContrastiveLoss(nn.Module):
    """Contrastive loss with soft positive weights and hard negative mining."""

    def __init__(self, temperature: float = 0.07, margin: float = 0.15, hard_neg_ratio: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.hard_neg_ratio = hard_neg_ratio

    def _single_branch_loss(self, z, labels):
        batch_size = z.size(0)
        if batch_size < 4:
            return torch.tensor(0.0, device=z.device, requires_grad=True)

        z = F.normalize(z, dim=-1)
        sim = torch.matmul(z, z.T) / self.temperature

        label_diff = (labels.unsqueeze(0) - labels.unsqueeze(1)).abs()
        pos_weight = torch.exp(-label_diff / self.margin)

        self_mask = ~torch.eye(batch_size, dtype=torch.bool, device=z.device)
        pos_weight = pos_weight * self_mask.float()

        neg_mask = (label_diff > 1.0) & self_mask
        k = max(int(neg_mask.float().sum(dim=1).mean().item() * self.hard_neg_ratio), 1)

        sim_max, _ = sim.detach().max(dim=1, keepdim=True)
        logits = sim - sim_max
        exp_logits = torch.exp(logits) * self_mask.float()

        if neg_mask.any():
            neg_sim = sim.clone()
            neg_sim[~neg_mask] = float('-inf')
            _, top_neg_idx = neg_sim.topk(min(k, batch_size - 1), dim=1)
            hard_neg_mask = torch.zeros_like(self_mask)
            hard_neg_mask.scatter_(1, top_neg_idx, True)
            denom_mask = hard_neg_mask | (pos_weight > 0.1)
        else:
            denom_mask = self_mask

        exp_denom = (torch.exp(logits) * denom_mask.float()).sum(dim=1, keepdim=True) + 1e-8
        log_prob = logits - torch.log(exp_denom)
        weighted_log_prob = (pos_weight * log_prob).sum(dim=1)
        denom = pos_weight.sum(dim=1).clamp(min=1e-8)
        return -(weighted_log_prob / denom).mean()

    def forward(self, drug_emb, se_emb, labels):
        loss_drug = self._single_branch_loss(drug_emb, labels)
        loss_se = self._single_branch_loss(se_emb, labels)
        loss_cross = self._single_branch_loss(drug_emb * se_emb, labels)
        return (loss_drug + loss_se + loss_cross) / 3


class OrdinalPredictionHead(nn.Module):
    """Regression head + ordinal auxiliary; fused prediction at inference."""

    def __init__(self, embed_dim: int = 256, num_levels: int = 5, dropout: float = 0.1):
        super().__init__()
        in_dim = embed_dim * 4
        hidden = embed_dim * 2

        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.reg_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.ordinal_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_levels - 1),
        )
        self.num_levels = num_levels

    def _interaction_features(self, drug_emb, se_emb):
        return torch.cat([
            drug_emb * se_emb,
            drug_emb - se_emb,
            drug_emb + se_emb,
            (drug_emb - se_emb).pow(2),
        ], dim=-1)

    def forward(self, drug_emb, se_emb):
        combined = self._interaction_features(drug_emb, se_emb)
        h = self.shared(combined)

        reg_out = self.reg_head(h).squeeze(-1)  # [0, 1]
        ordinal_logits = self.ordinal_head(h)  # (batch, K-1)

        return reg_out, ordinal_logits

    @torch.no_grad()
    def fused_predict(self, drug_emb, se_emb, alpha: float = 0.2):
        reg_out, ordinal_logits = self.forward(drug_emb, se_emb)

        cum_probs = torch.sigmoid(ordinal_logits)
        ordinal_pred = cum_probs.sum(dim=-1) / (self.num_levels - 1)

        return (1 - alpha) * reg_out + alpha * ordinal_pred


class DrugSEModel(nn.Module):
    """Full model: drug/SE encoders, cross-transformer, contrastive + ordinal loss."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_cross_layers: int = 3,
        num_fusion_layers: int = 2,
        dropout: float = 0.15,
        temperature: float = 0.07,
        contrastive_weight: float = 0.05,
        contrastive_margin: float = 0.15,
        hard_neg_ratio: float = 0.5,
        ordinal_weight: float = 0.2,
        drug_modal_dim: int = 256,
        hlgt_dim: int = 105,
        soc_dim: int = 26,
        semantic_dim: int = 300,
        num_drugs: int = 638,
        num_se: int = 994,
        num_levels: int = 5,
        huber_delta: float = 0.1,
    ):
        super().__init__()

        self.contrastive_weight = contrastive_weight
        self.ordinal_weight = ordinal_weight

        self.drug_encoder = DrugEncoder(
            num_drugs=num_drugs, modal_dim=drug_modal_dim,
            embed_dim=embed_dim, num_heads=num_heads,
            num_fusion_layers=num_fusion_layers, dropout=dropout,
        )
        self.se_encoder = SideEffectEncoder(
            num_se=num_se, hlgt_dim=hlgt_dim, soc_dim=soc_dim,
            semantic_dim=semantic_dim, embed_dim=embed_dim,
            num_heads=num_heads, num_fusion_layers=num_fusion_layers,
            dropout=dropout,
        )
        self.cross_transformer = CrossModalTransformer(
            embed_dim=embed_dim, num_heads=num_heads,
            num_layers=num_cross_layers, dropout=dropout,
        )
        self.pred_head = OrdinalPredictionHead(
            embed_dim=embed_dim, num_levels=num_levels, dropout=dropout,
        )
        self.contrastive_loss_fn = RankAwareContrastiveLoss(
            temperature=temperature, margin=contrastive_margin,
            hard_neg_ratio=hard_neg_ratio,
        )
        self.huber_delta = huber_delta
        self.num_levels = num_levels

    def _ordinal_targets(self, raw_freq):
        levels = (raw_freq * (self.num_levels - 1)).long().clamp(0, self.num_levels - 2)
        targets = torch.zeros(raw_freq.size(0), self.num_levels - 1, device=raw_freq.device)
        for k in range(self.num_levels - 1):
            targets[:, k] = (levels >= k + 1).float()
        return targets

    def encode(self, batch):
        drug_emb = self.drug_encoder(batch["drug_feats"], batch.get("drug_idx"))
        se_emb = self.se_encoder(
            batch["se_hlgt"], batch["se_soc"], batch["se_semantic"],
            batch.get("se_idx"),
        )
        drug_emb, se_emb = self.cross_transformer(drug_emb, se_emb)
        return drug_emb, se_emb

    def _weighted_huber_loss(self, pred, labels):
        diff = (pred - labels).abs()
        delta = self.huber_delta
        huber = torch.where(diff < delta, 0.5 * diff ** 2, delta * (diff - 0.5 * delta))

        center = 0.5
        dist_from_center = (labels - center).abs()
        edge_boost = 1.0 + 0.5 * dist_from_center
        return (huber * edge_boost).mean()

    def forward(self, batch):
        drug_emb, se_emb = self.encode(batch)
        reg_pred, ordinal_logits = self.pred_head(drug_emb, se_emb)
        labels = batch["label"]

        huber_loss = self._weighted_huber_loss(reg_pred, labels)

        ordinal_targets = self._ordinal_targets(labels)
        ordinal_loss = F.binary_cross_entropy_with_logits(ordinal_logits, ordinal_targets)

        cl_loss = self.contrastive_loss_fn(drug_emb, se_emb, labels)

        total_loss = (huber_loss
                      + self.contrastive_weight * cl_loss
                      + self.ordinal_weight * ordinal_loss)

        return {
            "loss": total_loss,
            "reg_loss": huber_loss,
            "cl_loss": cl_loss,
            "ord_loss": ordinal_loss,
            "pred": reg_pred,
            "drug_emb": drug_emb,
            "se_emb": se_emb,
        }

    @torch.no_grad()
    def predict(self, batch):
        drug_emb, se_emb = self.encode(batch)
        return self.pred_head.fused_predict(drug_emb, se_emb)
