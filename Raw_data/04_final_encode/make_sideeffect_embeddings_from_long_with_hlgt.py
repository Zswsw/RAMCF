import os, json, argparse
import numpy as np
import pandas as pd
from collections import defaultdict

def norm(s: str) -> str:
    if pd.isna(s): return ""
    s = str(s).lower().strip()
    for ch in ['\t','\n','\r','.','-','_','/','\\',',',';','(',')','[',']','{','}']:
        s = s.replace(ch, ' ')
    return ' '.join(s.split())

def load_glove_txt(path):
    w2v, dim = {}, None
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            parts = line.rstrip().split(' ')
            if len(parts) < 10:
                continue
            w = parts[0]
            vec = np.asarray(parts[1:], dtype=np.float32)
            if dim is None:
                dim = vec.size
            w2v[w] = vec
    if dim is None:
        raise ValueError("No GloVe vectors parsed; check file path and format.")
    return w2v, dim

def sent_embed(text, w2v, dim):
    toks = norm(text).split()
    vecs = [w2v[t] for t in toks if t in w2v]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)

def mlp_encode(X, out_dim=512, hidden_dim=1024, seed=0, l2norm=True):
    """
    X: (N, D) -> (N, out_dim)
    Two-layer MLP in pure numpy: ReLU + linear.
    Offline random projection/fusion (no training).
    """
    rng = np.random.RandomState(seed)
    W1 = rng.normal(0, 0.02, size=(X.shape[1], hidden_dim)).astype(np.float32)
    b1 = np.zeros(hidden_dim, dtype=np.float32)
    W2 = rng.normal(0, 0.02, size=(hidden_dim, out_dim)).astype(np.float32)
    b2 = np.zeros(out_dim, dtype=np.float32)

    H = X @ W1 + b1
    H = np.maximum(H, 0.0)          # ReLU
    Y = H @ W2 + b2

    if l2norm:
        Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
    return Y


def main(long_csv, glove_txt, outdir, hlgt_col="HLGT", freq_as_text=False, include_hlgt_in_text=True):
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(long_csv)

    need_cols = {'SideEffectTerm','SOC',hlgt_col}
    if not need_cols.issubset(df.columns):
        raise ValueError(f"Long table missing required columns: {need_cols}, current: {list(df.columns)[:30]}")

    df['SideEffectTerm'] = df['SideEffectTerm'].astype(str).str.strip()
    df['SOC'] = df['SOC'].astype(str).str.strip()
    df[hlgt_col] = df[hlgt_col].astype(str).str.strip()

    se_list = sorted(df['SideEffectTerm'].dropna().unique().tolist())
    soc_labels = sorted([s for s in df['SOC'].dropna().unique().tolist() if s.strip()])
    hlgt_labels = sorted([h for h in df[hlgt_col].dropna().unique().tolist() if h.strip() and h.lower() != "nan"])

    soc2idx = {s:i for i,s in enumerate(soc_labels)}
    hlgt2idx = {h:i for i,h in enumerate(hlgt_labels)}

    print(f"[Info] side_effects={len(se_list)}, SOC={len(soc_labels)}, HLGT={len(hlgt_labels)}")

    # Side-effect -> SOC set / HLGT set
    se2socs = (df.groupby('SideEffectTerm')['SOC']
                 .apply(lambda x: sorted(set([s for s in x.dropna().astype(str) if s.strip()])))
                 .to_dict())

    se2hlgts = (df.groupby('SideEffectTerm')[hlgt_col]
                  .apply(lambda x: sorted(set([h for h in x.dropna().astype(str) if h.strip() and h.lower() != "nan"])))
                  .to_dict())

    # Optional: append frequency to text
    if freq_as_text and 'FrequencyRatingValue' in df.columns:
        tmp = (df.dropna(subset=['SOC'])
                 .groupby(['SideEffectTerm','SOC'])['FrequencyRatingValue']
                 .max().reset_index())
        pair2freq = {(r['SideEffectTerm'], r['SOC']): r['FrequencyRatingValue'] for _, r in tmp.iterrows()}
    else:
        pair2freq = {}

    w2v, dim = load_glove_txt(glove_txt)
    N = len(se_list)
    X_sem  = np.zeros((N, dim), dtype=np.float32)
    X_soc  = np.zeros((N, len(soc_labels)), dtype=np.float32)
    X_hlgt = np.zeros((N, len(hlgt_labels)), dtype=np.float32)

    index_rows = []
    for i, se in enumerate(se_list):
        socs = se2socs.get(se, [])
        hlgts = se2hlgts.get(se, [])

        # Text: se + SOC (+ HLGT) (+ freq)
        bits = [se]
        if socs:
            if freq_as_text and pair2freq:
                bits += [f"{s} (freq {int(pair2freq.get((se,s), 0))})" for s in socs]
            else:
                bits += socs
        if include_hlgt_in_text and hlgts:
            bits += hlgts

        desc = " ; ".join(bits)
        X_sem[i] = sent_embed(desc, w2v, dim)

        for s in socs:
            X_soc[i, soc2idx[s]] = 1.0
        for h in hlgts:
            X_hlgt[i, hlgt2idx[h]] = 1.0

        index_rows.append({
            'idx': i,
            'SideEffectTerm': se,
            'SOC_joined': '; '.join(socs),
            'HLGT_joined': '; '.join(hlgts)
        })

    X_all = np.concatenate([X_soc, X_hlgt, X_sem], axis=1)

    # MLP project to 512d to align with drug embeddings
    X_se_512 = mlp_encode(
        X_all,
        out_dim=512,
        hidden_dim=1024,
        seed=0,
        l2norm=True
    )

    np.save(os.path.join(outdir, 'side_effect_embeddings_512d.npy'), X_se_512)
    np.save(os.path.join(outdir, 'semantic_glove_300d.npy'), X_sem)
    np.save(os.path.join(outdir, 'meddra_soc_multi_hot.npy'), X_soc)
    np.save(os.path.join(outdir, 'meddra_hlgt_multi_hot.npy'), X_hlgt)
    np.save(os.path.join(outdir, 'side_effect_embeddings_SOC_HLGT_glove.npy'), X_all)

    pd.DataFrame(index_rows).to_csv(os.path.join(outdir, 'side_effect_index.csv'), index=False)
    with open(os.path.join(outdir, 'soc_labels.json'), 'w', encoding='utf8') as f:
        json.dump(soc_labels, f, ensure_ascii=False, indent=2)
    with open(os.path.join(outdir, 'hlgt_labels.json'), 'w', encoding='utf8') as f:
        json.dump(hlgt_labels, f, ensure_ascii=False, indent=2)

    print({
        'num_side_effects': N,
        'num_soc': len(soc_labels),
        'num_hlgt': len(hlgt_labels),
        'semantic_dim': dim,
        'raw_feature_dim': int(X_all.shape[1]),  # 431
        'side_effect_512_shape': list(X_se_512.shape),  # [994, 512]
        'outdir': outdir
    })


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--long_csv', required=True)
    ap.add_argument('--glove', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--hlgt_col', default="HLGT", help="HLGT column name in long table")
    ap.add_argument('--freq_as_text', action='store_true')
    ap.add_argument('--include_hlgt_in_text', action='store_true', help="Append HLGT to GloVe text")
    args = ap.parse_args()
    main(args.long_csv, args.glove, args.outdir, args.hlgt_col, args.freq_as_text, args.include_hlgt_in_text)
