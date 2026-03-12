import gzip, json

def main(clean_targets_json="drug_to_uniprot_clean.json",
         alias_gz="9606.protein.aliases.vXX.txt.gz",
         out_json="uniprot_to_string.json"):

    drug_to_uni = json.load(open(clean_targets_json, "r", encoding="utf-8"))
    need = set(u for lst in drug_to_uni.values() for u in lst)

    mp = {}
    with gzip.open(alias_gz, "rt", encoding="utf-8", errors="ignore") as f:
        f.readline()  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            string_id, alias, source = parts[0], parts[1], parts[2]
            a = alias.strip()
            if a.startswith("UniProtKB:"):
                a = a.split(":", 1)[1]
            if a in need and ("uniprot" in source.lower()):
                mp[a] = string_id

    json.dump(mp, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("need_uniprot:", len(need))
    print("mapped_uniprot:", len(mp))
    # Print examples
    for k in list(mp.keys())[:5]:
        print(k, "->", mp[k])

if __name__ == "__main__":
    main(
        clean_targets_json="drug_to_uniprot_clean.json",
        alias_gz="9606.protein.aliases.v12.0.txt.gz",
        out_json="uniprot_to_string.json"
    )
