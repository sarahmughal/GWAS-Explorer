# 🧬 GWAS Explorer - Interactive Visualization of GWAS Summary Statistics

Interactive Streamlit app for exploring GWAS summary statistics.
Built for Epi Final Project — demo data simulates FinnGen T2D results.

---

## Features
- **Manhattan plot** — interactive, hover for SNP details, color-coded by significance
- **QQ plot** — with 95% confidence band and genomic inflation factor (λ)
- **Top hits table** — filterable, sortable, downloadable as CSV
- **Locus zoom** — regional plot around any lead SNP, colored by proxy LD (r²)
- **Upload your own data** — accepts any TSV/CSV with CHR, POS, PVAL columns
- **QC filters** — MAF and INFO score sliders in sidebar

---

## Quickstart

```bash
# 1. Clone / download this folder
# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app will open at http://localhost:8501

---

## Using your own GWAS data

Upload any GWAS summary statistics file (TSV, CSV, or .gz compressed).
The app auto-detects common column name variants. Required columns:

| Column | Aliases accepted |
|--------|-----------------|
| CHR    | CHROMOSOME, CHROM |
| POS    | POSITION, BP |
| PVAL   | P, P_VALUE, PVALUE |

Optional but recommended: BETA, SE, MAF, INFO, RSID

---

## Data sources
- **Demo data**: Simulated to match FinnGen T2D (endpoint: T2D) signal architecture
- **Real data**: Download from https://r10.finngen.fi/ (free registration)
  - Endpoint: `T2D` → download `finngen_R10_T2D.gz`

---

## Project structure
```
gwas_explorer/
├── app.py            # Main Streamlit application
├── requirements.txt  # Python dependencies
└── README.md         # This file
```
