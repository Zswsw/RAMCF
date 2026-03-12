import json, pandas as pd

drug_ids = pd.read_csv("drug_index.csv")["drugbank_id"].astype(str).tolist()
drug_to_uni = json.load(open("drug_to_uniprot_yours.json","r",encoding="utf-8"))
uni2str = json.load(open("uniprot_to_string.json","r",encoding="utf-8"))

missing = []
for d in drug_ids:
    unis = drug_to_uni.get(d, [])
    if not unis:  # No targets in DrugBank
        missing.append((d, "no_targets_in_drugbank"))
        continue
    # Has targets but none mapped to STRING
    if not any(u in uni2str for u in unis):
        missing.append((d, "targets_not_mapped_to_string"))

print("missing_count:", len(missing))
print("first_20:", missing[:20])

json.dump(missing, open("missing_mapped_drugs.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
