# RAMCF: Drug–Side Effect Frequency Prediction

## About

RAMCF (Rank-Aware Multimodal Contrastive Fusion) predicts the frequency of drug–side effect associations. It leverages multimodal drug features (fingerprint, image, PPI, SMILES) and side-effect features (MedDRA HLGT, SOC, semantic embeddings) with rank-aware contrastive learning and ordinal regression.

## Notice

**This project requires external data files that are not included in the repository.**

Please download the following resources before proceeding:

- **Pre-trained embeddings**: [GloVe 6B 300d](https://nlp.stanford.edu/projects/glove/) (`glove.6B.300d.txt`)
- **DrugBank**: [DrugBank full database](https://go.drugbank.com/releases/latest) (`drugbank_all_full_database.xml.zip`)
- **STRING PPI**: [STRING v12.0](https://string-db.org/cgi/download) – `9606.protein.aliases.v12.0.txt.gz`, `9606.protein.links.detailed.v12.0.txt.gz`

See `Raw_data/01_raw_data/DATA_SOURCES.txt` for details.

## 🚀 Features

- **Multimodal drug encoding**: SMILES, fingerprint, image, and PPI target propagation
- **Side-effect representation**: MedDRA SOC/HLGT multi-hot + GloVe semantic embeddings
- **Rank-aware contrastive learning**: Ordinal-aware loss for frequency prediction
- **Cross-modal fusion**: Transformer-based fusion for drug–side-effect pairs

## 🗂 Project Structure

```
RAMCF-main/
├── model.py              # Model definition
├── dataset.py            # Dataset and data loading
├── train.py              # Training script
├── config.yaml           # Hyperparameters
├── requirements.txt
├── checkpoints/          # Saved checkpoints
├── data/                 # Pre-processed data (ready for training)
└── Raw_data/
    ├── 01_raw_data/      # Raw sources and DATA_SOURCES.txt
    ├── 02_drug_ppi_pipeline/   # PPI target propagation
    ├── 03_sideeffect_pipeline/ # Side-effect embeddings
    └── 04_final_encode/       # Drug and side-effect encoding
```

## 📦 Installation

```bash
pip install -r requirements.txt
```

**Requirements**: Python 3.8+, PyTorch 1.12+

## 📂 Data

### Pre-processed Data (`data/`)

| File | Description |
|------|-------------|
| `drug_to_idx.json`, `sideeffect_to_idx.json` | ID mappings |
| `drug_fp_256d.npy`, `drug_img_256d.npy`, `drug_ppi_256d.npy`, `drug_smiles_256d.npy` | Drug embeddings (256d) |
| `meddra_hlgt_multi_hot.npy`, `meddra_soc_multi_hot.npy` | Side-effect MedDRA |
| `semantic_glove_300d.npy` | Side-effect GloVe embeddings |
| `drug_sideeffect_soc_freq.filtered.long_with_pseudo_hlgt.csv` | Drug–SE frequency table |

### Build from Raw Data (`Raw_data/`)

**01_raw_data/** – Raw sources (see `DATA_SOURCES.txt` for download links).

**02_drug_ppi_pipeline/** – Drug PPI target propagation:

1. `01_extract_targets_from_drugbank.py` → drug_to_uniprot.json  
2. `01b_clean_targets_to_uniprot_accession.py` → drug_to_uniprot_clean.json  
3. `02_build_uniprot_to_string_filtered.py` → uniprot_to_string.json  
4. `02b_filter_drug_to_uniprot_for_your_set.py` → drug_to_uniprot_yours.json  
5. `03_propagate_on_ppi.py` → target_propagated.npy, ppi_node_list.json  

Prepare: `drug_index.csv`, `your_drug_ids.json`.

**03_sideeffect_pipeline/** – Side-effect embeddings:

1. `make_sideeffect_embeddings_from_long.py` | long.csv, glove → se_embed_from_long/  
2. `make_pseudo_hlgt_from_long.py` | long.csv, se_embed → drug_sideeffect_soc_freq.filtered.long_with_pseudo_hlgt.csv  

**04_final_encode/** – Final encoding:

Drug encoding:
```bash
python encode_drugs_with_ppi.py \
  --csv ../01_raw_data/drugbank_all_with_fingerprints_and_svg_strict.csv \
  --outdir drug_full_embeddings_with_ppi \
  --drug_index_csv ../02_drug_ppi_pipeline/drug_index.csv \
  --target_npy ../02_drug_ppi_pipeline/target_propagated.npy \
  --target_dim 16201 --save_parts
```

Side-effect encoding:
```bash
python make_sideeffect_embeddings_from_long_with_hlgt.py \
  --long_csv ../01_raw_data/drug_sideeffect_soc_freq.filtered.long_with_pseudo_hlgt.csv \
  --glove /path/to/glove.6B.300d.txt \
  --outdir se_embed_with_pseudo_hlgt \
  --hlgt_col pseudo_hlgt --include_hlgt_in_text
```

**Note**: If the PPI pipeline is incomplete, use `--target_npy "" --target_dim 0` for drug encoding.

## 🏃 Training

```bash
python train.py --config config.yaml
```

**Key parameters** (in `config.yaml`):

| Parameter | Description |
|-----------|-------------|
| `batch_size` | Batch size (default: 512) |
| `epochs` | Training epochs (default: 250) |
| `learning_rate` | Learning rate (default: 4e-4) |
| `warmup_ratio` | Warmup ratio (default: 0.05) |
| `patience` | Early stopping patience (default: 40) |
