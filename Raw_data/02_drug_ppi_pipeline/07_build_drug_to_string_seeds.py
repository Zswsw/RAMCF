import json, pandas as pd

drug_ids = pd.read_csv("drug_index.csv")["drugbank_id"].astype(str).tolist()

drug_to_uni = json.load(open("drug_to_uniprot_yours.json","r",encoding="utf-8"))
uni2str = json.load(open("uniprot_to_string.json","r",encoding="utf-8"))

drug_to_genes = json.load(open("drug_to_genes_yours.json","r",encoding="utf-8"))
gene2str = json.load(open("gene_to_string.json","r",encoding="utf-8"))

out = {}
for d in drug_ids:
    seeds = set()
    for u in drug_to_uni.get(d, []):
        s = uni2str.get(u)
        if s: seeds.add(s)
    for g in drug_to_genes.get(d, []):
        s = gene2str.get(g)
        if s: seeds.add(s)
    out[d] = sorted(list(seeds))

json.dump(out, open("drug_to_string_seeds_yours.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)

nz = sum(1 for d in drug_ids if len(out.get(d, []))>0)
print("drugs_total:", len(drug_ids))
print("drugs_with_any_seeds:", nz)
