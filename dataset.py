"""
Drug-side effect frequency dataset.
Loads preprocessed embeddings, splits train/val/test, optional oversampling.
"""
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DrugSideEffectDataset(Dataset):
    """Drug-SE pairs with frequency labels; features z-score normalized."""

    FREQ_MIN = 1.0
    FREQ_MAX = 5.0

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        seed: int = 42,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        oversample: bool = True,
    ):
        self.data_dir = data_dir
        self.split = split

        # Load ID mappings and embeddings
        with open(f"{data_dir}/drug_to_idx.json") as f:
            self.drug2idx = json.load(f)
        with open(f"{data_dir}/sideeffect_to_idx.json") as f:
            self.se2idx = json.load(f)

        self.num_drugs = len(self.drug2idx)
        self.num_se = len(self.se2idx)

        self.drug_fp = self._load_and_normalize(f"{data_dir}/drug_fp_256d.npy")
        self.drug_img = self._load_and_normalize(f"{data_dir}/drug_img_256d.npy")
        self.drug_ppi = self._load_and_normalize(f"{data_dir}/drug_ppi_256d.npy")
        self.drug_smiles = self._load_and_normalize(f"{data_dir}/drug_smiles_256d.npy")

        self.se_hlgt = self._load_and_normalize(f"{data_dir}/meddra_hlgt_multi_hot.npy")
        self.se_soc = self._load_and_normalize(f"{data_dir}/meddra_soc_multi_hot.npy")
        self.se_semantic = self._load_and_normalize(f"{data_dir}/semantic_glove_300d.npy")

        # Build (drug_idx, se_idx, freq) samples from long table
        df = pd.read_csv(f"{data_dir}/drug_sideeffect_soc_freq.filtered.long_with_pseudo_hlgt.csv")
        dedup = df.groupby(["drugbank_id", "SideEffectTerm"])["FrequencyRatingValue"].max().reset_index()

        samples = []
        for _, row in dedup.iterrows():
            d_idx = self.drug2idx[row["drugbank_id"]]
            s_idx = self.se2idx[row["SideEffectTerm"]]
            freq = float(row["FrequencyRatingValue"])
            samples.append((d_idx, s_idx, freq))

        samples.sort()
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(samples))
        n_val = int(len(indices) * val_ratio)
        n_test = int(len(indices) * test_ratio)

        # Split indices by split type
        if split == "train":
            sel = indices[n_val + n_test:]
        elif split == "val":
            sel = indices[:n_val]
        elif split == "test":
            sel = indices[n_val:n_val + n_test]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.samples = [samples[i] for i in sel]

        if split == "train" and oversample:
            self.samples = self._oversample(self.samples, rng)

    @staticmethod
    def _oversample(samples, rng):
        """Oversample minority frequency levels to median count."""
        from collections import defaultdict
        by_freq = defaultdict(list)
        for s in samples:
            by_freq[int(s[2])].append(s)

        counts = {k: len(v) for k, v in by_freq.items()}
        target = int(np.median(list(counts.values())))

        result = []
        for freq, group in by_freq.items():
            result.extend(group)
            if len(group) < target:
                n_extra = target - len(group)
                extra_idx = rng.choice(len(group), size=n_extra, replace=True)
                for idx in extra_idx:
                    result.append(group[idx])
        rng.shuffle(result)
        return result

    @staticmethod
    def _load_and_normalize(path: str) -> torch.Tensor:
        """Load numpy array and z-score normalize."""
        arr = np.load(path).astype(np.float32)
        mean = arr.mean(axis=0, keepdims=True)
        std = arr.std(axis=0, keepdims=True) + 1e-8
        arr = (arr - mean) / std
        return torch.from_numpy(arr)

    @staticmethod
    def normalize_freq(freq: float) -> float:
        return (freq - DrugSideEffectDataset.FREQ_MIN) / (
            DrugSideEffectDataset.FREQ_MAX - DrugSideEffectDataset.FREQ_MIN
        )

    @staticmethod
    def denormalize_freq(norm_freq):
        return norm_freq * (
            DrugSideEffectDataset.FREQ_MAX - DrugSideEffectDataset.FREQ_MIN
        ) + DrugSideEffectDataset.FREQ_MIN

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        d_idx, s_idx, freq = self.samples[idx]
        drug_feats = torch.stack([
            self.drug_fp[d_idx],
            self.drug_img[d_idx],
            self.drug_ppi[d_idx],
            self.drug_smiles[d_idx],
        ])

        norm_freq = self.normalize_freq(freq)

        return {
            "drug_idx": d_idx,
            "se_idx": s_idx,
            "drug_feats": drug_feats,
            "se_hlgt": self.se_hlgt[s_idx],
            "se_soc": self.se_soc[s_idx],
            "se_semantic": self.se_semantic[s_idx],
            "label": torch.tensor(norm_freq, dtype=torch.float32),
            "raw_freq": torch.tensor(freq, dtype=torch.float32),
        }
