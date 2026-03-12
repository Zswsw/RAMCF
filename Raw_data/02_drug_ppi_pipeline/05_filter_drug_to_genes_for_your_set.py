import json

your_ids = set(json.load(open("your_drug_ids.json","r",encoding="utf-8")))
d = json.load(open("drug_to_genes.json","r",encoding="utf-8"))
d2 = {k:v for k,v in d.items() if k in your_ids}

json.dump(d2, open("drug_to_genes_yours.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)

print("your_drugs:", len(your_ids))
print("your_drugs_with_genes:", len(d2))
