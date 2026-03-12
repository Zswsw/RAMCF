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
    """Load GloVe text file return {word: vector} and dim."""
    w2v, dim = {}, None
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            parts = line.rstrip().split(' ')
            if len(parts) < 10:  # Skip malformed lines
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

def main(long_csv, glove_txt, outdir, freq_as_text=False):
    os.makedirs(outdir, exist_ok=True)

    # 1) Load long table: drugbank_id, GenericName, SideEffectTerm, SOC, FrequencyRatingValue
    df = pd.read_csv(long_csv)
    if not set(['SideEffectTerm','SOC']).issubset(df.columns):
        raise ValueError("Long table missing required columns: SideEffectTerm, SOC")

    # 2) Build side-effect set and SOC labels from long table
    df['SideEffectTerm'] = df['SideEffectTerm'].astype(str).str.strip()
    df['SOC'] = df['SOC'].astype(str).str.strip()
    se_list = sorted(df['SideEffectTerm'].dropna().unique().tolist())
    soc_labels = sorted(df['SOC'].dropna().unique().tolist())
    soc2idx = {s:i for i,s in enumerate(soc_labels)}
    print(f"[Info] side_effects={len(se_list)}, SOC_labels={len(soc_labels)}")

    # 3) Collect SOC per side-effect
    se2socs = (df.groupby('SideEffectTerm')['SOC']
                 .apply(lambda x: sorted(set([s for s in x.dropna().astype(str) if s.strip()])))
                 .to_dict())

    if freq_as_text and 'FrequencyRatingValue' in df.columns:
        # Build max-frequency text per (SE, SOC) 
        tmp = (df.dropna(subset=['SOC'])
                 .groupby(['SideEffectTerm','SOC'])['FrequencyRatingValue']
                 .max().reset_index())
        pair2freq = {(r['SideEffectTerm'], r['SOC']): r['FrequencyRatingValue'] for _, r in tmp.iterrows()}
    else:
        pair2freq = {}

    # 4) Build output matrices: X_sem (300d), X_soc (multi-hot)
    w2v, dim = load_glove_txt(glove_txt)
    N = len(se_list)
    X_sem = np.zeros((N, dim), dtype=np.float32)
    X_soc = np.zeros((N, len(soc_labels)), dtype=np.float32)

    index_rows = []
    for i, se in enumerate(se_list):
        socs = se2socs.get(se, [])
        # Build GloVe text: side-effect + SOC list (+ optional freq)
        if freq_as_text and pair2freq:
            bits = [f"{se}"] + [f"{s} (freq {int(pair2freq.get((se,s), 0))})" for s in socs]
        else:
            bits = [se] + socs
        desc = " ; ".join(bits)
        X_sem[i] = sent_embed(desc, w2v, dim)

        # multi-hot
        for s in socs:
            j = soc2idx[s]
            X_soc[i, j] = 1.0

        index_rows.append({
            'idx': i,
            'SideEffectTerm': se,
            'SOC_joined': '; '.join(socs)
        })

    X_all = np.concatenate([X_soc, X_sem], axis=1)

    # 5) Save outputs
    np.save(os.path.join(outdir, 'semantic_glove_300d.npy'), X_sem)
    np.save(os.path.join(outdir, 'meddra_soc_multi_hot.npy'), X_soc)
    np.save(os.path.join(outdir, 'side_effect_embeddings_SOC_glove.npy'), X_all)
    pd.DataFrame(index_rows).to_csv(os.path.join(outdir, 'side_effect_index.csv'), index=False)
    with open(os.path.join(outdir, 'soc_labels.json'), 'w', encoding='utf8') as f:
        json.dump(soc_labels, f, ensure_ascii=False, indent=2)

    print({
        'num_side_effects': N,
        'num_soc': len(soc_labels),
        'semantic_dim': dim,
        'final_embedding_shape': list(X_all.shape),
        'outdir': outdir
    })

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--long_csv', required=True, help='drug_sideeffect_soc_freq.long.csv')
    ap.add_argument('--glove', required=True, help='Path to glove.6B.300d.txt')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--freq_as_text', action='store_true',
                    help='Append frequency to GloVe text ')
    args = ap.parse_args()
    main(args.long_csv, args.glove, args.outdir, args.freq_as_text)
