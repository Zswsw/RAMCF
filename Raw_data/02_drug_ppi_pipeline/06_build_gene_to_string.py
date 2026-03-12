import gzip, json

def main(genes_json="drug_to_genes_yours.json",
         alias_gz="9606.protein.aliases.vXX.txt.gz",
         out_json="gene_to_string.json"):
    drug_to_genes = json.load(open(genes_json, "r", encoding="utf-8"))
    need = set(g for lst in drug_to_genes.values() for g in lst)

    mp = {}
    with gzip.open(alias_gz, "rt", encoding="utf-8", errors="ignore") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            string_id, alias = parts[0], parts[1]
            a = alias.strip()
            if a in need:
                # One gene may map to multiple proteins; keep first
                mp.setdefault(a, string_id)

    json.dump(mp, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("need_genes:", len(need))
    print("mapped_genes:", len(mp))
    for k in list(mp.keys())[:5]:
        print(k, "->", mp[k])

if __name__ == "__main__":
    main(
        genes_json="drug_to_genes_yours.json",
        alias_gz="9606.protein.aliases.v12.0.txt.gz",  
        out_json="gene_to_string.json"
    )
