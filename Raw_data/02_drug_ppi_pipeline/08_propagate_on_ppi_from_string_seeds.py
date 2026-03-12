import gzip, json, numpy as np, pandas as pd
from scipy import sparse

def load_ppi(links_gz, score_th=700):
    nodes = {}
    u, v, w = [], [], []
    def idx(node):
        if node not in nodes:
            nodes[node] = len(nodes)
        return nodes[node]

    with gzip.open(links_gz, "rt", encoding="utf-8", errors="ignore") as f:
        header = f.readline().strip().split()
        cs_i = header.index("combined_score") if "combined_score" in header else -1

        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            p1, p2 = parts[0], parts[1]
            cs = int(parts[cs_i]) if cs_i >= 0 else int(parts[-1])
            if cs < score_th:
                continue
            i, j = idx(p1), idx(p2)
            ww = cs / 1000.0
            u += [i, j]
            v += [j, i]
            w += [ww, ww]

    n = len(nodes)
    A = sparse.csr_matrix((w, (u, v)), shape=(n, n), dtype=np.float32)
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    W = sparse.diags(1.0 / deg.astype(np.float32)) @ A

    node_list = [None] * n
    for k, i in nodes.items():
        node_list[i] = k
    return W, node_list

def rwr(W, p0, alpha=0.85, max_iter=80, eps=1e-6):
    p = p0.copy()
    for _ in range(max_iter):
        p_new = alpha * (W @ p) + (1 - alpha) * p0
        if np.abs(p_new - p).sum() < eps:
            return p_new
        p = p_new
    return p

def main():
    drug_ids = pd.read_csv("drug_index.csv")["drugbank_id"].astype(str).tolist()
    drug_to_string = json.load(open("drug_to_string_seeds_yours.json","r",encoding="utf-8"))

    links_gz = "9606.protein.links.detailed.v12.0.txt.gz"
    score_th = 700
    alpha = 0.85
    topk = 200

    W, node_list = load_ppi(links_gz, score_th=score_th)
    node2idx = {nid:i for i, nid in enumerate(node_list)}
    n = len(node_list)

    X = np.zeros((len(drug_ids), n), dtype=np.float32)
    miss = 0

    for i, dbid in enumerate(drug_ids):
        sids = drug_to_string.get(dbid, [])
        seeds = [node2idx[sid] for sid in sids if sid in node2idx]
        if not seeds:
            miss += 1
            continue

        p0 = np.zeros(n, dtype=np.float32)
        p0[seeds] = 1.0 / len(seeds)
        p = rwr(W, p0, alpha=alpha)

        idx = np.argpartition(-p, topk)[:topk]
        pp = np.zeros_like(p)
        pp[idx] = p[idx]
        X[i] = pp

        if (i+1) % 100 == 0:
            print(f"[{i+1}/{len(drug_ids)}] done")

    np.save("target_propagated.npy", X)
    json.dump(node_list, open("ppi_node_list.json","w",encoding="utf-8"), ensure_ascii=False)
    print("PPI nodes:", n)
    print("X:", X.shape)
    print("drugs_without_any_seeds:", miss, "/", len(drug_ids))

if __name__ == "__main__":
    main()
