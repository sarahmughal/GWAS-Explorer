import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import io

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GWAS Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
  }
  h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: -0.03em;
  }
  .stApp { background-color: #0d1117; color: #e6edf3; }
  .metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px 20px;
    text-align: center;
  }
  .metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }
  .metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #58a6ff;
  }
  .section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    border-bottom: 1px solid #30363d;
    padding-bottom: 6px;
    margin-bottom: 16px;
  }
</style>
""",
    unsafe_allow_html=True,
)


# ── Simulate FinnGen-style T2D summary stats ──────────────────────────────────
@st.cache_data
def generate_demo_data(n_variants: int = 150_000, seed: int = 42) -> pd.DataFrame:
    """
    Simulate GWAS summary statistics resembling FinnGen T2D results.
    Injects realistic signal peaks at known T2D loci.
    """
    rng = np.random.default_rng(seed)

    # Known T2D loci (chr, approx pos in Mb, effect size boost)
    known_loci = [
        (3, 172_000_000, 12),  # KCNQ1
        (6, 20_800_000, 15),  # CDKAL1
        (7, 28_200_000, 10),  # JAZF1
        (9, 22_100_000, 11),  # CDKN2A/B
        (10, 114_700_000, 10),  # TCF7L2 (strongest T2D signal)
        (10, 114_750_000, 22),  # TCF7L2 peak
        (11, 92_300_000, 9),  # KCNJ11
        (12, 111_300_000, 8),  # HNF1A
        (16, 53_800_000, 9),  # FTO
        (20, 44_000_000, 8),  # HNF4A
    ]

    chrom_sizes = {
        1: 249_000_000,
        2: 242_000_000,
        3: 198_000_000,
        4: 190_000_000,
        5: 181_000_000,
        6: 171_000_000,
        7: 159_000_000,
        8: 145_000_000,
        9: 138_000_000,
        10: 133_000_000,
        11: 135_000_000,
        12: 133_000_000,
        13: 115_000_000,
        14: 107_000_000,
        15: 102_000_000,
        16: 90_000_000,
        17: 83_000_000,
        18: 80_000_000,
        19: 59_000_000,
        20: 64_000_000,
        21: 47_000_000,
        22: 51_000_000,
    }

    rows = []
    var_idx = 1
    for chrom, size in chrom_sizes.items():
        n_chr = max(1, int(n_variants * size / sum(chrom_sizes.values())))
        positions = np.sort(rng.integers(1_000_000, size - 1_000_000, n_chr))

        # Base chi-squared (lambda ~ 1.05 to mimic mild inflation)
        chisq = rng.chisquare(df=1, size=n_chr) * 1.05

        # Inject locus signals
        for lc, lp, boost in known_loci:
            if lc == chrom:
                dist = np.abs(positions - lp)
                mask = dist < 500_000
                decay = np.exp(-dist[mask] / 200_000)
                chisq[mask] += boost * decay * rng.uniform(0.7, 1.3, mask.sum())

        pvals = stats.chi2.sf(chisq, df=1)
        beta = rng.normal(0, 0.1, n_chr)
        se = np.abs(rng.normal(0.05, 0.01, n_chr))
        maf = rng.uniform(0.01, 0.49, n_chr)
        info = rng.uniform(0.6, 1.0, n_chr)

        for i in range(n_chr):
            rows.append(
                {
                    "CHR": chrom,
                    "POS": int(positions[i]),
                    "RSID": f"rs{var_idx}",
                    "REF": rng.choice(["A", "C", "G", "T"]),
                    "ALT": rng.choice(["A", "C", "G", "T"]),
                    "BETA": round(float(beta[i]), 4),
                    "SE": round(float(se[i]), 4),
                    "PVAL": float(pvals[i]),
                    "MAF": round(float(maf[i]), 4),
                    "INFO": round(float(info[i]), 4),
                }
            )
            var_idx += 1

    df = pd.DataFrame(rows)
    df["-log10P"] = -np.log10(df["PVAL"].clip(lower=1e-300))
    df["OR"] = np.exp(df["BETA"]).round(4)
    return df


# ── QC filtering ──────────────────────────────────────────────────────────────
def apply_qc(df: pd.DataFrame, maf_thresh: float, info_thresh: float) -> pd.DataFrame:
    return df[(df["MAF"] >= maf_thresh) & (df["INFO"] >= info_thresh)].copy()


# ── Genomic inflation ─────────────────────────────────────────────────────────
def genomic_lambda(pvals: np.ndarray) -> float:
    chisq = stats.chi2.ppf(1 - pvals, df=1)
    return round(float(np.median(chisq) / stats.chi2.ppf(0.5, df=1)), 4)


# ── Manhattan plot ────────────────────────────────────────────────────────────
def manhattan_plot(
    df: pd.DataFrame, pval_thresh: float, highlight_rsid: str | None = None
):
    chrom_offsets: dict[int, int] = {}
    offset = 0
    for chrom in range(1, 23):
        chrom_offsets[chrom] = offset
        sub = df[df["CHR"] == chrom]
        if not sub.empty:
            offset += sub["POS"].max() + 10_000_000

    df = df.copy()
    df["ABS_POS"] = df.apply(
        lambda r: r["POS"] + chrom_offsets.get(r["CHR"], 0), axis=1
    )

    palette = ["#58a6ff", "#3fb950"]
    colors = df["CHR"].apply(lambda c: palette[(c - 1) % 2])

    sig_mask = df["PVAL"] < pval_thresh
    sugst_mask = (df["PVAL"] < 1e-5) & (~sig_mask)

    fig = go.Figure()

    # Non-significant
    sub = df[~sig_mask & ~sugst_mask]
    fig.add_trace(
        go.Scattergl(
            x=sub["ABS_POS"],
            y=sub["-log10P"],
            mode="markers",
            marker=dict(color=colors[sub.index], size=2.5, opacity=0.6),
            text=sub["RSID"],
            customdata=sub[["CHR", "POS", "PVAL", "BETA", "MAF"]],
            hovertemplate="<b>%{text}</b><br>Chr%{customdata[0]}:%{customdata[1]:,}<br>P=%{customdata[2]:.2e}<br>β=%{customdata[3]}<br>MAF=%{customdata[4]}<extra></extra>",
            name="",
            showlegend=False,
        )
    )

    # Suggestive
    sub = df[sugst_mask]
    fig.add_trace(
        go.Scattergl(
            x=sub["ABS_POS"],
            y=sub["-log10P"],
            mode="markers",
            marker=dict(color="#d29922", size=4, opacity=0.85),
            text=sub["RSID"],
            customdata=sub[["CHR", "POS", "PVAL", "BETA", "MAF"]],
            hovertemplate="<b>%{text}</b><br>Chr%{customdata[0]}:%{customdata[1]:,}<br>P=%{customdata[2]:.2e}<br>β=%{customdata[3]}<br>MAF=%{customdata[4]}<extra></extra>",
            name="Suggestive (P<1×10⁻⁵)",
            showlegend=True,
        )
    )

    # Significant
    sub = df[sig_mask]
    fig.add_trace(
        go.Scattergl(
            x=sub["ABS_POS"],
            y=sub["-log10P"],
            mode="markers",
            marker=dict(color="#f85149", size=6, opacity=0.9),
            text=sub["RSID"],
            customdata=sub[["CHR", "POS", "PVAL", "BETA", "MAF"]],
            hovertemplate="<b>%{text}</b><br>Chr%{customdata[0]}:%{customdata[1]:,}<br>P=%{customdata[2]:.2e}<br>β=%{customdata[3]}<br>MAF=%{customdata[4]}<extra></extra>",
            name=f"Significant (P<{pval_thresh:.0e})",
            showlegend=True,
        )
    )

    # Highlight selected SNP
    if highlight_rsid:
        h = df[df["RSID"] == highlight_rsid]
        if not h.empty:
            fig.add_trace(
                go.Scattergl(
                    x=h["ABS_POS"],
                    y=h["-log10P"],
                    mode="markers",
                    marker=dict(color="#ffa657", size=14, symbol="star"),
                    name=highlight_rsid,
                    showlegend=True,
                )
            )

    # Threshold lines
    fig.add_hline(
        y=-np.log10(pval_thresh),
        line_dash="dash",
        line_color="#f85149",
        annotation_text=f"P={pval_thresh:.0e}",
        annotation_font_color="#f85149",
    )
    fig.add_hline(
        y=-np.log10(1e-5),
        line_dash="dot",
        line_color="#d29922",
        annotation_text="P=1×10⁻⁵",
        annotation_font_color="#d29922",
    )

    # X-axis chromosome labels
    tick_vals, tick_text = [], []
    for chrom in range(1, 23):
        sub = df[df["CHR"] == chrom]
        if not sub.empty:
            tick_vals.append(sub["ABS_POS"].mean())
            tick_text.append(str(chrom))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        height=420,
        margin=dict(l=60, r=20, t=20, b=40),
        xaxis=dict(
            tickvals=tick_vals,
            ticktext=tick_text,
            title="Chromosome",
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(title="-log₁₀(P)", showgrid=True, gridcolor="#21262d"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        font=dict(family="IBM Plex Mono"),
        hoverlabel=dict(bgcolor="#161b22", font_family="IBM Plex Mono"),
    )
    return fig


# ── QQ plot ───────────────────────────────────────────────────────────────────
def qq_plot(pvals: np.ndarray, lam: float):
    pvals_sorted = np.sort(pvals[pvals > 0])
    n = len(pvals_sorted)
    expected = -np.log10(np.linspace(1 / n, 1, n))
    observed = -np.log10(pvals_sorted)

    # Confidence band (95%)
    ci_upper = -np.log10(
        stats.beta.ppf(0.05, np.arange(1, n + 1), n - np.arange(1, n + 1) + 1)
    )
    ci_lower = -np.log10(
        stats.beta.ppf(0.95, np.arange(1, n + 1), n - np.arange(1, n + 1) + 1)
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=expected,
            y=ci_upper,
            fill=None,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=expected,
            y=ci_lower,
            fill="tonexty",
            mode="lines",
            line=dict(width=0),
            fillcolor="rgba(88,166,255,0.15)",
            name="95% CI",
            showlegend=True,
        )
    )

    # Downsample non-extreme points for speed
    extreme_mask = observed > 1.0
    sampled_mask = ~extreme_mask
    sample_idx = np.random.choice(
        np.where(sampled_mask)[0], size=min(5000, sampled_mask.sum()), replace=False
    )
    keep = np.zeros(n, dtype=bool)
    keep[extreme_mask] = True
    keep[sample_idx] = True

    fig.add_trace(
        go.Scattergl(
            x=expected[keep],
            y=observed[keep],
            mode="markers",
            marker=dict(color="#58a6ff", size=3, opacity=0.7),
            name="Variants",
        )
    )

    # Diagonal
    max_val = max(expected.max(), observed.max()) * 1.05
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(color="#f85149", dash="dash", width=1.5),
            name="Expected",
            showlegend=True,
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        height=380,
        margin=dict(l=60, r=20, t=40, b=50),
        xaxis=dict(title="Expected -log₁₀(P)", showgrid=True, gridcolor="#21262d"),
        yaxis=dict(title="Observed -log₁₀(P)", showgrid=True, gridcolor="#21262d"),
        title=dict(
            text=f"QQ Plot  |  λ = {lam}",
            font=dict(family="IBM Plex Mono", size=14),
            x=0.5,
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        font=dict(family="IBM Plex Mono"),
    )
    return fig


# ── Locus zoom plot ───────────────────────────────────────────────────────────
def locus_zoom(df: pd.DataFrame, chrom: int, lead_pos: int, window_kb: int = 500):
    window = window_kb * 1000
    region = df[
        (df["CHR"] == chrom)
        & (df["POS"] >= lead_pos - window)
        & (df["POS"] <= lead_pos + window)
    ].copy()

    if region.empty:
        return None

    # Simulate LD (r²) with lead SNP — proxy by distance decay
    lead_row = region.loc[region["PVAL"].idxmin()]
    region["r2"] = np.exp(-np.abs(region["POS"] - lead_row["POS"]) / 150_000)
    region["r2"] = region["r2"].clip(0, 1)

    colorscale = [
        [0.0, "#1f2d3d"],
        [0.2, "#1a6fa8"],
        [0.4, "#2fa8b5"],
        [0.6, "#3fb950"],
        [0.8, "#d29922"],
        [1.0, "#f85149"],
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=region["POS"],
            y=region["-log10P"],
            mode="markers",
            marker=dict(
                color=region["r2"],
                colorscale=colorscale,
                cmin=0,
                cmax=1,
                size=7,
                colorbar=dict(
                    title="r² (proxy)",
                    thickness=12,
                    tickfont=dict(family="IBM Plex Mono", size=10),
                    titlefont=dict(family="IBM Plex Mono", size=11),
                ),
            ),
            text=region["RSID"],
            customdata=region[["PVAL", "BETA", "MAF", "r2"]],
            hovertemplate="<b>%{text}</b><br>Pos: %{x:,}<br>P=%{customdata[0]:.2e}<br>β=%{customdata[1]}<br>MAF=%{customdata[2]}<br>r²=%{customdata[3]:.2f}<extra></extra>",
        )
    )

    # Star on lead SNP
    fig.add_trace(
        go.Scatter(
            x=[lead_row["POS"]],
            y=[lead_row["-log10P"]],
            mode="markers+text",
            marker=dict(color="#ffa657", size=14, symbol="star"),
            text=[lead_row["RSID"]],
            textposition="top center",
            textfont=dict(family="IBM Plex Mono", size=10, color="#ffa657"),
            name="Lead SNP",
        )
    )

    fig.add_hline(
        y=-np.log10(5e-8),
        line_dash="dash",
        line_color="#f85149",
        annotation_text="P=5×10⁻⁸",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        height=380,
        margin=dict(l=60, r=80, t=20, b=50),
        xaxis=dict(
            title=f"Chromosome {chrom} position", showgrid=False, tickformat=","
        ),
        yaxis=dict(title="-log₁₀(P)", showgrid=True, gridcolor="#21262d"),
        font=dict(family="IBM Plex Mono"),
        showlegend=False,
    )
    return fig, region


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════


def main():
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        """
    <h1 style='font-size:2rem; margin-bottom:0'>🧬 GWAS Explorer</h1>
    <p style='color:#8b949e; font-family: IBM Plex Mono, monospace; font-size:0.85rem; margin-top:4px'>
      Interactive exploration of GWAS summary statistics · FinnGen T2D demo
    </p>
    """,
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Data Input")
        data_source = st.radio(
            "Source", ["Demo data (FinnGen T2D)", "Upload your own file"], index=0
        )

        uploaded_file = None
        if data_source == "Upload your own file":
            uploaded_file = st.file_uploader(
                "Upload GWAS summary stats",
                type=["tsv", "csv", "gz", "txt"],
                help="Needs columns: CHR, POS, PVAL (and optionally BETA, SE, MAF, INFO, RSID)",
            )

        st.divider()
        st.markdown("### 🔬 QC Filters")
        maf_thresh = st.slider("Min MAF", 0.0, 0.1, 0.01, 0.005, format="%.3f")
        info_thresh = st.slider("Min INFO", 0.0, 1.0, 0.8, 0.05, format="%.2f")

        st.divider()
        st.markdown("### 📊 Display")
        pval_thresh = st.select_slider(
            "Significance threshold",
            options=[5e-8, 1e-7, 1e-6, 1e-5],
            value=5e-8,
            format_func=lambda x: f"{x:.0e}",
        )
        locus_window = st.slider("Locus zoom window (kb)", 100, 1000, 500, 50)

    # ── Load data ──────────────────────────────────────────────────────────────
    if uploaded_file is not None:
        try:
            sep = "\t" if uploaded_file.name.endswith((".tsv", ".gz", ".txt")) else ","
            raw = pd.read_csv(uploaded_file, sep=sep, compression="infer")
            # Normalise column names
            raw.columns = [c.upper().strip() for c in raw.columns]
            col_map = {}
            for c in raw.columns:
                if c in ("P", "P_VALUE", "PVALUE", "P-VALUE"):
                    col_map[c] = "PVAL"
                if c in ("CHROMOSOME", "CHROM"):
                    col_map[c] = "CHR"
                if c in ("POSITION", "BP"):
                    col_map[c] = "POS"
                if c in ("EFFECT", "B"):
                    col_map[c] = "BETA"
                if c in ("RS_ID", "SNP", "ID", "MARKER"):
                    col_map[c] = "RSID"
            raw.rename(columns=col_map, inplace=True)
            for need in ["CHR", "POS", "PVAL"]:
                if need not in raw.columns:
                    st.error(f"Column '{need}' not found. Please rename your columns.")
                    st.stop()
            if "RSID" not in raw.columns:
                raw["RSID"] = "rs" + raw.index.astype(str)
            if "MAF" not in raw.columns:
                raw["MAF"] = 0.1
            if "INFO" not in raw.columns:
                raw["INFO"] = 1.0
            if "BETA" not in raw.columns:
                raw["BETA"] = 0.0
            raw["CHR"] = (
                raw["CHR"].astype(str).str.replace("chr", "", case=False).astype(int)
            )
            raw["PVAL"] = pd.to_numeric(raw["PVAL"], errors="coerce")
            raw = raw.dropna(subset=["PVAL"])
            raw["-log10P"] = -np.log10(raw["PVAL"].clip(lower=1e-300))
            df_raw = raw
            st.success(f"Loaded {len(df_raw):,} variants from uploaded file.")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()
    else:
        with st.spinner("Generating FinnGen T2D demo data..."):
            df_raw = generate_demo_data()

    # ── Apply QC ───────────────────────────────────────────────────────────────
    df = apply_qc(df_raw, maf_thresh, info_thresh)
    n_pre, n_post = len(df_raw), len(df)
    n_sig = (df["PVAL"] < pval_thresh).sum()
    lam = genomic_lambda(df["PVAL"].values)

    # ── Metrics row ────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, value in [
        (c1, "Total variants", f"{n_pre:,}"),
        (c2, "Post-QC variants", f"{n_post:,}"),
        (c3, f"Hits (P<{pval_thresh:.0e})", f"{n_sig:,}"),
        (c4, "Genomic λ", f"{lam}"),
        (c5, "QC pass rate", f"{100 * n_post / n_pre:.1f}%"),
    ]:
        col.markdown(
            f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
        </div>""",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Manhattan & QQ ─────────────────────────────────────────────────────────
    st.markdown(
        '<p class="section-header">Manhattan Plot & QQ Plot</p>', unsafe_allow_html=True
    )
    col_m, col_q = st.columns([2.5, 1])

    with col_m:
        fig_m = manhattan_plot(df, pval_thresh)
        clicked = st.plotly_chart(fig_m, use_container_width=True, key="manhattan")

    with col_q:
        fig_q = qq_plot(df["PVAL"].values, lam)
        st.plotly_chart(fig_q, use_container_width=True, key="qq")

    # ── Top hits table ─────────────────────────────────────────────────────────
    st.divider()
    st.markdown('<p class="section-header">Top Hits</p>', unsafe_allow_html=True)

    top_hits = (
        df[df["PVAL"] < pval_thresh]
        .sort_values("PVAL")
        .head(200)[["RSID", "CHR", "POS", "PVAL", "BETA", "SE", "MAF", "INFO"]]
        .copy()
    )
    top_hits["PVAL"] = top_hits["PVAL"].apply(lambda x: f"{x:.2e}")

    col_t1, col_t2, col_t3 = st.columns([1, 1, 1])
    p_filter = col_t1.selectbox("Filter by chromosome", ["All"] + list(range(1, 23)))
    sort_col = col_t2.selectbox("Sort by", ["PVAL", "CHR", "MAF", "BETA"])
    n_show = col_t3.slider("Rows to show", 5, 200, 20, 5)

    disp = top_hits.copy()
    if p_filter != "All":
        disp = disp[disp["CHR"] == int(p_filter)]
    if sort_col != "PVAL":
        disp = disp.sort_values(
            sort_col, key=lambda x: pd.to_numeric(x, errors="coerce")
        )

    st.dataframe(disp.head(n_show), use_container_width=True, hide_index=True)

    # Download button
    csv_buf = io.StringIO()
    top_hits.to_csv(csv_buf, index=False)
    st.download_button(
        "⬇ Download top hits CSV",
        csv_buf.getvalue(),
        file_name="gwas_top_hits.csv",
        mime="text/csv",
    )

    # ── Locus Zoom ─────────────────────────────────────────────────────────────
    st.divider()
    st.markdown('<p class="section-header">Locus Zoom</p>', unsafe_allow_html=True)

    if top_hits.empty:
        st.info("No significant hits to zoom into with current filters.")
    else:
        sig_df = df[df["PVAL"] < pval_thresh].sort_values("PVAL")
        locus_options = [
            f"{row.RSID}  (Chr{row.CHR}:{row.POS:,}  P={float(row.PVAL):.2e})"
            for _, row in sig_df.head(30).iterrows()
        ]
        selected_label = st.selectbox("Select lead variant to zoom", locus_options)

        if selected_label:
            selected_rsid = selected_label.split()[0]
            lead = sig_df[sig_df["RSID"] == selected_rsid].iloc[0]
            result = locus_zoom(df, int(lead["CHR"]), int(lead["POS"]), locus_window)

            if result:
                fig_lz, region_df = result
                st.plotly_chart(fig_lz, use_container_width=True, key="locus_zoom")

                with st.expander("📋 Region variants table"):
                    show_cols = ["RSID", "CHR", "POS", "PVAL", "BETA", "MAF", "r2"]
                    region_show = (
                        region_df[show_cols].sort_values("PVAL").head(50).copy()
                    )
                    region_show["PVAL"] = region_show["PVAL"].apply(
                        lambda x: f"{x:.2e}"
                    )
                    region_show["r2"] = region_show["r2"].round(3)
                    st.dataframe(region_show, use_container_width=True, hide_index=True)

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        """
    <p style='color:#484f58; font-family: IBM Plex Mono, monospace; font-size:0.72rem; text-align:center'>
      GWAS Explorer · Epi Final Project · Demo data simulated to match FinnGen T2D summary statistics
    </p>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
