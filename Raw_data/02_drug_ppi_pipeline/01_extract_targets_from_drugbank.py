# Extract drugbank_id -> UniProt target list from DrugBank XML

import os, json, zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict

NS = "{http://www.drugbank.ca}"

def iter_drugs(xml_path):
    ctx = ET.iterparse(xml_path, events=("end",))
    for _, elem in ctx:
        if elem.tag == f"{NS}drug":
            yield elem
            elem.clear()

def extract_uniprot_from_target(target_elem):
    unis = set()
    pol = target_elem.find(f"{NS}polypeptide")
    if pol is None:
        return unis
    ext_ids = pol.find(f"{NS}external-identifiers")
    if ext_ids is None:
        return unis
    for ext in ext_ids.findall(f"{NS}external-identifier"):
        res = (ext.findtext(f"{NS}resource") or "").lower()
        ident = (ext.findtext(f"{NS}identifier") or "").strip()
        if "uniprot" in res and ident:
            # Handle UniProtKB:P12345 format
            if ident.startswith("UniProtKB:"):
                ident = ident.split(":", 1)[1]
            unis.add(ident)
    return unis

def main(xml_zip, out_json="drug_to_uniprot.json"):
    os.makedirs("tmp", exist_ok=True)
    with zipfile.ZipFile(xml_zip, "r") as z:
        xmls = [n for n in z.namelist() if n.endswith(".xml")]
        if len(xmls) != 1:
            raise RuntimeError(f"zip contains multiple xml files: {xmls[:5]}")
        xml_name = xmls[0]
        z.extract(xml_name, "tmp")
        xml_path = os.path.join("tmp", xml_name)
        if not os.path.isfile(xml_path):
            xml_path = os.path.join("tmp", os.path.basename(xml_name))

    drug_to_uniprot = defaultdict(set)

    for drug in iter_drugs(xml_path):
        dbid = drug.findtext(f"{NS}drugbank-id[@primary='true']")
        if not dbid:
            continue
        targets = drug.find(f"{NS}targets")
        if targets is None:
            continue
        for t in targets.findall(f"{NS}target"):
            unis = extract_uniprot_from_target(t)
            for u in unis:
                drug_to_uniprot[dbid].add(u)

    drug_to_uniprot = {k: sorted(list(v)) for k, v in drug_to_uniprot.items()}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(drug_to_uniprot, f, ensure_ascii=False)

    print("drugs_with_targets:", len(drug_to_uniprot))
    # Print example
    for k in list(drug_to_uniprot.keys())[:3]:
        print(k, drug_to_uniprot[k][:5])

if __name__ == "__main__":
    main("drugbank_all_full_database.xml.zip")
