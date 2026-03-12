import os, io, json, argparse, numpy as np, pandas as pd
from tqdm import tqdm
from PIL import Image
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdDepictor

# -------------------------
# Utils
# -------------------------
def canonicalize_smiles(s: str) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None: return None
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None

def tok(s: str) -> List[str]:
    i, out = 0, []
    while i < len(s):
        if i+1 < len(s) and s[i:i+2] in ("Cl","Br"):
            out.append(s[i:i+2]); i += 2
        else:
            out.append(s[i]); i += 1
    return out

def build_vocab(smiles_list, min_freq=1):
    from collections import Counter
    c = Counter()
    for s in smiles_list:
        if s: c.update(tok(s))
    stoi = {"<pad>":0,"<unk>":1,"<bos>":2,"<eos>":3}
    for t, cnt in c.items():
        if cnt >= min_freq and t not in stoi:
            stoi[t] = len(stoi)
    return stoi

def encode_smiles_ids(s: str, stoi, max_len=150):
    ids = [stoi["<bos>"]] + [stoi.get(t, stoi["<unk>"]) for t in tok(s or "")] + [stoi["<eos>"]]
    ids = ids[:max_len] + [stoi["<pad>"]]*(max_len-len(ids))
    return np.asarray(ids, np.int32)

def morgan_fp_bits(smi, nBits=2048, radius=2):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return np.zeros(nBits, dtype=np.uint8)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
        arr = np.zeros((nBits,), dtype=np.int8)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr.astype(np.uint8)
    except Exception:
        return np.zeros(nBits, dtype=np.uint8)

def parse_fp_cell(x, nBits=2048):
    """Parse fingerprint cell to 0/1 array:
       - Comma-separated '0,1,0,...'
       - JSON list '[0,1,0,...]'
       - String '0100101...' (len==nBits)
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (list, np.ndarray)):
        arr = np.asarray(x, dtype=np.uint8)
        return arr if arr.size == nBits else None
    s = str(x).strip()
    if s.startswith('[') and s.endswith(']'):
        try:
            arr = np.asarray(json.loads(s), dtype=np.uint8)
            return arr if arr.size == nBits else None
        except Exception:
            pass
    if ',' in s:
        try:
            arr = np.asarray([int(t) for t in s.replace(' ','').split(',')], dtype=np.uint8)
            return arr if arr.size == nBits else None
        except Exception:
            pass
    if set(s) <= set('01') and len(s) == nBits:
        return np.asarray([1 if ch=='1' else 0 for ch in s], dtype=np.uint8)
    return None

def svg_string_to_pil(svg_txt: str, size=(224,224)) -> Optional[Image.Image]:
    try:
        import cairosvg
        png_bytes = cairosvg.svg2png(bytestring=svg_txt.encode('utf-8'), output_width=size[0], output_height=size[1])
        return Image.open(io.BytesIO(png_bytes)).convert('RGB')
    except Exception:
        return None

def smiles_to_pil(smi: str, size=(224,224)) -> Optional[Image.Image]:
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return None
        rdDepictor.Compute2DCoords(mol)
        return Draw.MolToImage(mol, size=size).convert('RGB')
    except Exception:
        return None

# -------------------------
# Dataset
# -------------------------
class DrugCSVTwoWayDataset(Dataset):
    """
    Load from full CSV:
    - drugbank_id / name
    - smiles
    - fingerprint (optional; compute if missing)
    - image: image_path (prefer) / svg string / SMILES render (fallback)
    + target vector from target_npy (e.g. target_propagated.npy from PPI propagation)
    """
    def __init__(self, csv_path, outdir, img_size=224, max_len=150, fp_bits=2048,
                 drug_index_csv: Optional[str]=None,
                 target_npy: Optional[str]=None):
        self.df = pd.read_csv(csv_path)

        # Fallback for common column names
        col_id   = next((c for c in ['drugbank_id','DrugBankID','drug_id','id'] if c in self.df.columns), None)
        col_name = next((c for c in ['name','Name','drug_name'] if c in self.df.columns), None)
        col_smiles = next((c for c in ['smiles','SMILES'] if c in self.df.columns), None)
        col_fp  = next((c for c in ['fingerprint','fingerprints','ecfp4_bits','ecfp4','fp_bits'] if c in self.df.columns), None)
        col_img = next((c for c in ['image_path','img_path','png_path'] if c in self.df.columns), None)
        col_svg = next((c for c in ['svg','svg_string','mol_svg'] if c in self.df.columns), None)
        assert col_id and col_smiles, "CSV must have drugbank_id and smiles"

        # name may be missing
        use_cols = [col_id] + ([col_name] if col_name else []) + [col_smiles] \
                   + ([col_fp] if col_fp else []) + ([col_img] if col_img else []) + ([col_svg] if col_svg else [])
        self.df = self.df[use_cols].copy()

        # Normalize column names
        rename = {col_id: "drugbank_id", col_smiles: "smiles"}
        if col_name: rename[col_name] = "name"
        if col_fp: rename[col_fp] = "fingerprint"
        if col_img: rename[col_img] = "image_path"
        if col_svg: rename[col_svg] = "svg"
        self.df.rename(columns=rename, inplace=True)

        if "name" not in self.df.columns:
            self.df["name"] = ""

        # Canonicalize SMILES and dedupe (one row per drugbank_id)
        self.df['drugbank_id'] = self.df['drugbank_id'].astype(str)
        self.df['smiles'] = self.df['smiles'].astype(str)
        self.df['smiles_canonical'] = [canonicalize_smiles(s) for s in tqdm(self.df['smiles'], desc="Canonicalizing")]
        self.df = self.df.dropna(subset=['smiles_canonical'])
        self.df = self.df.sort_values('drugbank_id').drop_duplicates(subset=['drugbank_id'], keep='first').reset_index(drop=True)

        # Align order by drug_index.csv (must match target_propagated.npy rows)
        if drug_index_csv is not None and str(drug_index_csv).strip() != "":
            order_df = pd.read_csv(drug_index_csv)
            assert "drugbank_id" in order_df.columns, "drug_index_csv must have drugbank_id"
            order = order_df["drugbank_id"].astype(str).tolist()

            self.df = self.df.set_index("drugbank_id")
            miss = [d for d in order if d not in self.df.index]
            if len(miss) > 0:
                raise RuntimeError(f"drug_index_csv has {len(miss)} drugbank_ids not in CSV, e.g.: {miss[:10]}")
            self.df = self.df.loc[order].reset_index()

        # Vocab and sequence ids
        self.stoi = build_vocab(self.df['smiles_canonical'].tolist())
        self.smiles_ids = np.stack([encode_smiles_ids(s, self.stoi, max_len) for s in tqdm(self.df['smiles_canonical'], desc="Encoding SMILES")])

        # Fingerprint: use CSV if present, else compute
        rows = []
        fp_col = self.df["fingerprint"] if "fingerprint" in self.df.columns else [None]*len(self.df)
        for s, fp_cell in tqdm(zip(self.df['smiles_canonical'], fp_col), total=len(self.df), desc="Preparing fingerprints"):
            arr = parse_fp_cell(fp_cell, nBits=fp_bits) if fp_cell is not None else None
            if arr is None:
                arr = morgan_fp_bits(s, nBits=fp_bits, radius=2)
            rows.append(arr)
        self.fp_bits = np.stack(rows).astype(np.float32)

        # Image loading strategy
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        self.outdir = outdir
        self.img_cache_dir = os.path.join(outdir, "drug_images")
        os.makedirs(self.img_cache_dir, exist_ok=True)
        self.rendered_paths = []
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Preparing images"):
            path = None
            # 1) Existing PNG path
            if 'image_path' in self.df.columns and pd.notna(row.get('image_path', None)):
                p = str(row['image_path'])
                if os.path.isfile(p):
                    path = p
            # 2) SVG string from CSV
            if path is None and 'svg' in self.df.columns and pd.notna(row.get('svg', None)):
                img = svg_string_to_pil(str(row['svg']))
                if img is not None:
                    p = os.path.join(self.img_cache_dir, f"{row['drugbank_id']}.png")
                    img.save(p); path = p
            # 3) Fallback: render from SMILES
            if path is None:
                img = smiles_to_pil(row['smiles_canonical'])
                p = os.path.join(self.img_cache_dir, f"{row['drugbank_id']}.png")
                img.save(p); path = p
            self.rendered_paths.append(path)

        # Load target vectors (PPI propagation result)
        self.tgt = None
        if target_npy is not None and str(target_npy).strip() != "":
            self.tgt = np.load(target_npy).astype(np.float32)
            if self.tgt.shape[0] != len(self.df):
                raise RuntimeError(f"target_npy rows {self.tgt.shape[0]} != drugs {len(self.df)}. Ensure target_propagated.npy uses same drug_index.csv order.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        img = Image.open(self.rendered_paths[i]).convert('RGB')
        ids = torch.from_numpy(self.smiles_ids[i]).long()
        fp  = torch.from_numpy(self.fp_bits[i])
        imt = self.tf(img)

        if self.tgt is None:
            # Compat: no target_npy
            return ids, fp, imt

        tgt = torch.from_numpy(self.tgt[i])  # [target_dim]
        return ids, fp, imt, tgt

# -------------------------
# Encoders
# -------------------------
class SmilesEncoder(nn.Module):
    def __init__(self, vocab_size=64, d_model=256, hidden=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.lstm = nn.LSTM(d_model, hidden//2, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden, 256)
    def forward(self, ids):
        x, _ = self.lstm(self.emb(ids))
        feat = x.mean(1)
        return F.normalize(self.proj(feat), dim=-1)

class FpEncoder(nn.Module):
    def __init__(self, in_dim=2048, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, out_dim)
        )
    def forward(self, fp):
        return F.normalize(self.net(fp), dim=-1)

class ImgEncoder(nn.Module):
    def __init__(self, out_dim=256, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet18(weights=weights)
        base.fc = nn.Identity()
        self.backbone = base
        self.fc = nn.Linear(512, out_dim)
    def forward(self, img):
        x = self.backbone(img)
        return F.normalize(self.fc(x), dim=-1)

class TargetEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(512, out_dim)
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

class DrugEncoder(nn.Module):
    def __init__(self, vocab_size, out_dim=512, fp_dim=2048, target_dim: int=0):
        super().__init__()
        self.smi = SmilesEncoder(vocab_size)
        self.fp  = FpEncoder(fp_dim)
        self.img = ImgEncoder()

        self.use_target = (target_dim is not None) and (int(target_dim) > 0)
        if self.use_target:
            self.tgt = TargetEncoder(int(target_dim), out_dim=256)
            fuse_in = 256 * 4
        else:
            self.tgt = None
            fuse_in = 256 * 3

        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, out_dim)
        )

    def forward(self, ids, fp, img, tgt_vec=None):
        s = self.smi(ids)
        f = self.fp(fp)
        v = self.img(img)

        if self.use_target:
            if tgt_vec is None:
                raise RuntimeError("Model has target branch but forward received no tgt_vec.")
            t = self.tgt(tgt_vec)
            z = torch.cat([s, f, v, t], dim=-1)
        else:
            z = torch.cat([s, f, v], dim=-1)

        return F.normalize(self.fuse(z), dim=-1)

# -------------------------
# Main
# -------------------------
def main(args):
    ds = DrugCSVTwoWayDataset(
        args.csv, args.outdir,
        img_size=args.img, max_len=args.max_len, fp_bits=args.nbits,
        drug_index_csv=args.drug_index_csv,
        target_npy=args.target_npy
    )
    dl = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=0)

    model = DrugEncoder(
        vocab_size=len(ds.stoi),
        out_dim=args.out_dim,
        fp_dim=args.nbits,
        target_dim=args.target_dim
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    embs = []
    parts_smi, parts_fp, parts_img, parts_ppi = [], [], [], []

    with torch.no_grad():
        if args.target_npy and args.target_dim > 0:
            for ids, fp, img, tgt in tqdm(dl, desc="Encoding drugs (with PPI target)"):
                ids, fp, img, tgt = ids.to(device), fp.to(device), img.to(device), tgt.to(device)

                # 4 branches, 256d each
                s = model.smi(ids)  # (B,256)
                f = model.fp(fp)  # (B,256)
                v = model.img(img)  # (B,256)
                t = model.tgt(tgt)  # (B,256)

                if args.save_parts:
                    parts_smi.append(s.cpu().numpy())
                    parts_fp.append(f.cpu().numpy())
                    parts_img.append(v.cpu().numpy())
                    parts_ppi.append(t.cpu().numpy())

                # Fuse to 512d
                z = torch.cat([s, f, v, t], dim=-1)  # (B,1024)
                z = model.fuse(z)  # (B,out_dim)
                z = F.normalize(z, dim=-1)
                embs.append(z.cpu().numpy())
        else:
            for ids, fp, img in tqdm(dl, desc="Encoding drugs"):
                ids, fp, img = ids.to(device), fp.to(device), img.to(device)

                s = model.smi(ids)
                f = model.fp(fp)
                v = model.img(img)

                if args.save_parts:
                    parts_smi.append(s.cpu().numpy())
                    parts_fp.append(f.cpu().numpy())
                    parts_img.append(v.cpu().numpy())

                    # No PPI branch: zero placeholder (still output 4-part files for downstream)
                    parts_ppi.append(np.zeros((s.shape[0], 256), dtype=np.float32))

                z = torch.cat([s, f, v], dim=-1)  # (B,768)
                z = model.fuse(z)
                z = F.normalize(z, dim=-1)
                embs.append(z.cpu().numpy())

    embs = np.concatenate(embs, axis=0)

    os.makedirs(args.outdir, exist_ok=True)
    np.save(os.path.join(args.outdir, f"drug_embeddings_{args.out_dim}d.npy"), embs)

    # Save branch features
    if args.save_parts:
        np.save(os.path.join(args.outdir, "drug_smiles_256d.npy"), np.concatenate(parts_smi, axis=0))
        np.save(os.path.join(args.outdir, "drug_fp_256d.npy"), np.concatenate(parts_fp, axis=0))
        np.save(os.path.join(args.outdir, "drug_img_256d.npy"), np.concatenate(parts_img, axis=0))
        np.save(os.path.join(args.outdir, "drug_ppi_256d.npy"), np.concatenate(parts_ppi, axis=0))

    # Save index (aligned)
    idx = ds.df[['drugbank_id','name','smiles_canonical']].copy()
    idx['image_path'] = ds.rendered_paths
    idx.to_csv(os.path.join(args.outdir, "drug_index.csv"), index=False)
    with open(os.path.join(args.outdir, "smiles_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(ds.stoi, f, ensure_ascii=False, indent=2)

    print({
        "rows": len(ds),
        "vocab_size": len(ds.stoi),
        "embeddings_shape": embs.shape,
        "outdir": os.path.abspath(args.outdir),
        "use_target": bool(args.target_npy and args.target_dim > 0),
        "target_dim": int(args.target_dim)
    })

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="drugbank_all_with_fingerprints_and_svg_strict.csv")
    ap.add_argument("--outdir", default="drug_full_embeddings")
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--max_len", type=int, default=150)
    ap.add_argument("--nbits", type=int, default=2048)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--out_dim", type=int, default=512)

    ap.add_argument("--drug_index_csv", default="", help="drug_index.csv for alignment (e.g. dsnet_targets/drug_index.csv)")
    ap.add_argument("--target_npy", default="", help="target_propagated.npy from PPI propagation")
    ap.add_argument("--target_dim", type=int, default=0, help="target_propagated.npy column count (e.g. 16201)")
    ap.add_argument("--save_parts", action="store_true", help="Save 4 branch features: smiles/fp/img/ppi (256d each)")

    args = ap.parse_args()
    main(args)
