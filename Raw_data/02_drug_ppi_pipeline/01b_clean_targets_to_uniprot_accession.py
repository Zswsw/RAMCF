import json
import re

def is_accession(x):
    return bool(re.match(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$", x))

def main(in_json="drug_to_uniprot.json",
         out_json="drug_to_uniprot_clean.json"):
    raw = json.load(open(in_json, "r", encoding="utf-8"))

    cleaned = {}
    drop_count = 0

    for drug, targets in raw.items():
        keep = []
        for t in targets:
            if is_accession(t):
                keep.append(t)
            else:
                drop_count += 1
        if keep:
            cleaned[drug] = sorted(set(keep))

    json.dump(cleaned, open(out_json, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print("original drugs:", len(raw))
    print("cleaned drugs :", len(cleaned))
    print("dropped non-accession targets:", drop_count)
    print("example:", list(cleaned.items())[:3])

if __name__ == "__main__":
    main()
