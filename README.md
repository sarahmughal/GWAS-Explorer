# 🧬 GWAS Explorer
### Interactive Visualization of GWAS Summary Statistics
*EPI 217: Molecular and Genetic Epidemiology Final Project  ·  Spring 2026*

An interactive Streamlit app for exploring GWAS summary statistics — no coding required. Built using real summary statistics from FinnGen Release 12 (Type 2 Diabetes, ~500,000 Finnish participants).

---

## Features
- **Manhattan plot** — interactive, hover for SNP details, color-coded by significance
- **QQ plot** — with 95% confidence band and genomic inflation factor (λ)
- **Top hits table** — filterable by chromosome, sortable, downloadable as CSV
- **Locus zoom** — regional plot around any lead SNP, colored by proxy LD (r²)
- **Upload your own data** — accepts any TSV/CSV/gz with CHR, POS, PVAL columns
- **QC filters** — MAF and INFO score sliders in sidebar, with live pre/post-QC counts

---

## Quickstart

```bash
# 1. Clone / download this folder
# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app will open at https://gwas-explorer.streamlit.app/

---

## Dataset: FinnGen R12 — Type 2 Diabetes (T2D)

This app is demonstrated using real GWAS summary statistics from **FinnGen Release 12**, the most recent data freeze from the Finnish biobank initiative.

### About FinnGen
[FinnGen](https://www.finngen.fi/en) is a large-scale genomics study that has analyzed over **500,000 Finnish biobank samples**, correlating genetic variation with health data across hundreds of disease endpoints. Finland's isolated population and comprehensive health registry linkage make it one of the most powerful GWAS resources in the world. Summary statistics are freely available via Google Cloud Storage after registration.

### Why T2D?
Type 2 Diabetes is one of the most well-studied complex diseases in human genetics. It has a well-characterized polygenic architecture with dozens of replicated associated loci, including the canonical **TCF7L2** signal on chromosome 10 — making it an ideal phenotype for validating GWAS visualization tools. If the app is working correctly, TCF7L2 should show up as the top hit. It does.

### Data preparation
The raw FinnGen R12 T2D file (`finngen_R12_T2D.gz`) is 820 MB compressed and contains ~20 million variants — too large for fast interactive use. Two preprocessing steps were applied:

1. **Convert to Parquet** — Parquet is a binary columnar storage format that pandas reads ~10x faster than a gzip-compressed TSV.
2. **Pre-filter to P < 0.001** — Variants with P > 0.001 are null noise that don't affect signal peaks or the QQ plot tail. Filtering reduced the dataset from ~20M to 274,421 variants (98% reduction) with no meaningful loss of information.

To reproduce this from the raw file:
```python
import pandas as pd
df = pd.read_csv('finngen_R12_T2D.gz', sep='\t', compression='gzip')
df[df['pval'] < 0.001].to_parquet('finngen_R12_T2D_small.parquet')
```

---

## Key Results

| Metric | Value |
|--------|-------|
| Total variants loaded | 274,421 |
| Post-QC variants (MAF ≥ 0.01, INFO ≥ 0.80) | 247,811 (90.3%) |
| Genome-wide significant hits (P < 5×10⁻⁸) | 22,032 |
| Genomic inflation factor (λ) | 32.07 |
| Lead SNP | rs7903146 (*TCF7L2*, Chr10) |
| Lead SNP p-value | 1.05×10⁻²⁸² |

**Note on λ = 32.07:** This high value is expected and appropriate for a biobank-scale GWAS with ~500,000 participants on a highly polygenic trait like T2D. The inflation reflects genuine widespread genetic associations, not population stratification or other bias.

---

## Using your own GWAS data

Upload any GWAS summary statistics file (TSV, CSV, or .gz compressed). The app auto-detects common column name variants from PLINK 2, REGENIE, SAIGE, and FinnGen outputs.

**Required columns:**

| Column | Aliases accepted |
|--------|-----------------|
| CHR    | CHROMOSOME, CHROM, #CHROM |
| POS    | POSITION, BP |
| PVAL   | P, P_VALUE, PVALUE, P-VALUE |

**Optional but recommended:** BETA, SE, MAF, INFO, RSID

---

## Downloading FinnGen data

1. Register at [finngen.fi](https://www.finngen.fi/en) (free)
2. You will receive a link to the Google Cloud Storage bucket
3. Browse to `finngen-public-data-r12/summary_stats/release/`
4. Download your endpoint of interest (e.g. `finngen_R12_T2D.gz`)
5. Pre-process with the Parquet script above
6. Update `FINNGEN_PATH` in `app.py` to point to your `.parquet` file

---

## Project structure

```
gwas_explorer/
├── app.py                         # Main Streamlit application
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── finngen_R12_T2D_small.parquet  # Pre-processed data (not included in repo)
```

---
