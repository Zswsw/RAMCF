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

def extract_gene_symbols_from_target(target_elem):
    genes = set()
    pol = target_elem.find(f"{NS}polypeptide")
    if pol is None:
        return genes

    g = pol.findtext(f"{NS}gene-name")
    if g:
        genes.add(g.strip())

    # Some also in external-identifiers (HGNC, Gene Name, etc.)
    ext_ids = pol.find(f"{NS}external-identifiers")
    if ext_ids is not None:
        for ext in ext_ids.findall(f"{NS}external-identifier"):
            res = (ext.findtext(f"{NS}resource") or "").lower()
            ident = (ext.findtext(f"{NS}identifier") or "").strip()
            if not ident:
                continue
            if "hgnc" in res or "gene" in res:
                # Use identifier 
                genes.add(ident)

    return {x for x in genes if x and len(x) <= 20}

def main(xml_zip="drugbank_all_full_database.xml.zip", out_json="drug_to_genes.json"):
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

    drug_to_genes = defaultdict(set)
    for drug in iter_drugs(xml_path):
        dbid = drug.findtext(f"{NS}drugbank-id[@primary='true']")
        if not dbid:
            continue
        targets = drug.find(f"{NS}targets")
        if targets is None:
            continue
        for t in targets.findall(f"{NS}target"):
            for g in extract_gene_symbols_from_target(t):
                drug_to_genes[dbid].add(g)

    drug_to_genes = {k: sorted(list(v)) for k, v in drug_to_genes.items()}
    json.dump(drug_to_genes, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("drugs_with_gene_symbols:", len(drug_to_genes))
    print("example:", list(drug_to_genes.items())[:3])

if __name__ == "__main__":
    main()
