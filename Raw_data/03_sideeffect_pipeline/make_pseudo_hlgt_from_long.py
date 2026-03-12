import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict

def norm(s):
    return "" if pd.isna(s) else str(s).strip()

def main(long_csv, se_index_csv, semantic_npy, out_long_csv, k_per_soc=5, seed=0):
    # long table: must have SideEffectTerm, SOC
    df = pd.read_csv(long_csv)
    assert "SideEffectTerm" in df.columns and "SOC" in df.columns

    # se_index.csv: must have SideEffectTerm 
    se_idx = pd.read_csv(se_index_csv)
    assert "SideEffectTerm" in se_idx.columns

    X = np.load(semantic_npy).astype(np.float32)
    assert X.shape[0] == len(se_idx), "semantic_npy rows must match se_index"

    # SideEffectTerm -> row index
    term2i = {norm(t): i for i, t in enumerate(se_idx["SideEffectTerm"].tolist())}

    # Collect side-effects per SOC
    soc2terms = defaultdict(set)
    for t, soc in zip(df["SideEffectTerm"], df["SOC"]):
        soc2terms[norm(soc)].add(norm(t))

    term2pseudo = {}
    for soc, terms in soc2terms.items():
        terms = [t for t in terms if t in term2i]
        if len(terms) == 0:
            continue
        if len(terms) < k_per_soc:
            # Too few: one group per term
            for j, t in enumerate(terms):
                term2pseudo[t] = f"{soc}__H{j:02d}"
            continue

        idxs = [term2i[t] for t in terms]
        Xs = X[idxs]

        k = min(k_per_soc, len(terms))
        km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
        lab = km.fit_predict(Xs)

        for t, l in zip(terms, lab):
            term2pseudo[t] = f"{soc}__H{int(l):02d}"

    # Merge back into long table
    df["SideEffectTerm_n"] = df["SideEffectTerm"].map(norm)
    df["pseudo_hlgt"] = df["SideEffectTerm_n"].map(term2pseudo)

    df.drop(columns=["SideEffectTerm_n"], inplace=True)
    df.to_csv(out_long_csv, index=False)
    print("saved:", out_long_csv)
    print("pseudo_hlgt_nonnull:", df["pseudo_hlgt"].notna().sum(), "/", len(df))
    print("unique pseudo_hlgt:", df["pseudo_hlgt"].nunique())

if __name__ == "__main__":
    main(
        long_csv="drug_sideeffect_soc_freq.filtered.long.csv",
        se_index_csv="se_embed_from_long/side_effect_index.csv",          # Update to your path
        semantic_npy="se_embed_from_long/semantic_glove_300d.npy",         # Update to your path
        out_long_csv="drug_sideeffect_soc_freq.filtered.long_with_pseudo_hlgt.csv",
        k_per_soc=5,
        seed=0
    )
