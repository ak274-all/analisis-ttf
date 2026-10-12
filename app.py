"""Streamlit app: Time-to-Failure (TTF) distribution fitting and reliability analysis.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import reliability as rel

SAMPLE_FILE = Path(__file__).with_name("sample_ttf.csv")
MIN_POINTS = 3
COLORS = {
    "Exponential": "#EF553B",
    "Weibull": "#636EFA",
    "Normal": "#00CC96",
    "Lognormal": "#AB63FA",
    "Gamma": "#FFA15A",
}


# --------------------------------------------------------------------------
# Data loading & cleaning
# --------------------------------------------------------------------------
def is_excel(filename: str) -> bool:
    return filename.lower().endswith((".xlsx", ".xls"))


@st.cache_data(show_spinner=False)
def list_sheets(data: bytes) -> list[str]:
    return pd.ExcelFile(io.BytesIO(data)).sheet_names


@st.cache_data(show_spinner=False)
def read_table(data: bytes, filename: str, sheet: str | None = None) -> pd.DataFrame:
    """Read an Excel sheet or a CSV file (delimiter is auto-detected)."""
    if is_excel(filename):
        return pd.read_excel(io.BytesIO(data), sheet_name=sheet)
    return pd.read_csv(io.BytesIO(data), sep=None, engine="python", encoding="utf-8-sig")


def clean_ttf(series: pd.Series) -> tuple[np.ndarray, dict[str, int]]:
    """Convert a column to numeric TTF values and count what was discarded."""
    n_missing = int(series.isna().sum())
    if not pd.api.types.is_numeric_dtype(series):
        # Accept decimal commas such as "123,5" (common in Indonesian locale files).
        series = series.astype("string").str.strip().str.replace(",", ".", regex=False)
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.where(np.isfinite(numeric))
    n_invalid = int(numeric.isna().sum()) - n_missing
    values = numeric.dropna().to_numpy(dtype=float)
    return values, {"missing": n_missing, "invalid": max(n_invalid, 0)}


def default_column(df: pd.DataFrame) -> int:
    """Prefer a column that looks like TTF, otherwise the first numeric column."""
    for i, col in enumerate(df.columns):
        name = str(col).lower()
        if "ttf" in name or "failure" in name or "gagal" in name:
            return i
    numeric = df.select_dtypes("number").columns
    return int(df.columns.get_loc(numeric[0])) if len(numeric) else 0


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def summary_table(results: list[rel.FitResult], best: rel.FitResult) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "Distribusi": ("⭐ " if r is best else "") + r.name,
                "Parameter": r.params_text() if r.ok else f"— ({r.error})",
                "Log-Likelihood": r.loglik,
                "AIC": r.aic,
                "BIC": r.bic,
                "KS (D)": r.ks_stat,
                "KS p-value": r.ks_pvalue,
                "MTTF": r.dist.mean() if r.ok else np.nan,
            }
        )
    return pd.DataFrame(rows)


def highlight_best(df: pd.DataFrame):
    def style_row(row):
        css = "background-color: rgba(0, 204, 150, 0.25); font-weight: 600"
        return [css if row["Distribusi"].startswith("⭐") else ""] * len(row)

    return df.style.apply(style_row, axis=1).format(
        {
            "Log-Likelihood": "{:.3f}",
            "AIC": "{:.3f}",
            "BIC": "{:.3f}",
            "KS (D)": "{:.4f}",
            "KS p-value": "{:.4f}",
            "MTTF": "{:.4g}",
        },
        na_rep="—",
    )


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def time_grid(t: np.ndarray, n: int = 500) -> np.ndarray:
    hi = t.max() + 0.2 * (t.max() - min(t.min(), 0))
    lo = t.min() - 0.2 * (t.max() - t.min()) if t.min() <= 0 else hi / n
    return np.linspace(lo, hi, n)


def _add_curves(fig, grid, shown, best, key):
    for r in shown:
        is_best = r is best
        y = rel.evaluate(r.dist, grid)[key]
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=np.where(np.isfinite(y), y, np.nan),
                mode="lines",
                name=r.name + (" (terbaik)" if is_best else ""),
                line=dict(
                    color=COLORS[r.name],
                    width=4 if is_best else 1.8,
                    dash="solid" if is_best else "dash",
                ),
            )
        )


def _layout(fig, title, ytitle, yrange=None):
    fig.update_layout(
        title=title,
        xaxis_title="Waktu t",
        yaxis_title=ytitle,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=50, r=10, l=10),
        height=470,
    )
    if yrange:
        fig.update_yaxes(range=yrange)
    return fig


def pdf_figure(t, shown, best):
    density, edges = np.histogram(t, bins="auto", density=True)
    fig = go.Figure(
        go.Bar(
            x=(edges[:-1] + edges[1:]) / 2,
            y=density,
            width=np.diff(edges),
            name="Histogram data",
            marker=dict(color="rgba(120,120,120,0.35)", line=dict(color="gray", width=1)),
        )
    )
    _add_curves(fig, time_grid(t), shown, best, "f")
    # Cap the y-axis: PDFs with shape < 1 go to infinity near t = 0.
    return _layout(fig, "Fungsi Kepadatan Peluang f(t) vs Histogram", "f(t)", [0, density.max() * 1.6])


def cdf_figure(t, shown, best, reliability: bool):
    x, F_emp = rel.empirical_cdf(t)
    fig = go.Figure(
        go.Scatter(
            x=x,
            y=1 - F_emp if reliability else F_emp,
            mode="markers",
            name="Data empiris (median rank)",
            marker=dict(color="black", size=6, symbol="circle-open"),
        )
    )
    _add_curves(fig, time_grid(t), shown, best, "R" if reliability else "F")
    if reliability:
        return _layout(fig, "Kurva Keandalan R(t)", "R(t)", [0, 1.02])
    return _layout(fig, "Fungsi Distribusi Kumulatif / Ketidakandalan F(t)", "F(t)", [0, 1.02])


# --------------------------------------------------------------------------
# Page sections
# --------------------------------------------------------------------------
def sidebar_inputs():
    """Return (file bytes, filename) from the uploader or the bundled sample."""
    st.sidebar.header("1. Data")
    uploaded = st.sidebar.file_uploader("Unggah file Excel atau CSV", type=["xlsx", "xls", "csv"])
    use_sample = st.sidebar.checkbox(
        "Gunakan data contoh", value=False, disabled=uploaded is not None or not SAMPLE_FILE.exists()
    )
    if uploaded is not None:
        return uploaded.getvalue(), uploaded.name
    if use_sample:
        return SAMPLE_FILE.read_bytes(), SAMPLE_FILE.name
    return None, None


def load_ttf(data: bytes, filename: str) -> np.ndarray | None:
    """Read the file, let the user pick the TTF column, and return clean values."""
    try:
        sheet = None
        if is_excel(filename):
            sheets = list_sheets(data)
            sheet = st.sidebar.selectbox("Sheet", sheets) if len(sheets) > 1 else sheets[0]
        df = read_table(data, filename, sheet)
    except Exception as exc:
        st.error(f"File tidak dapat dibaca. Pastikan formatnya Excel/CSV yang valid.\n\nDetail: {exc}")
        return None

    if df.empty or len(df.columns) == 0:
        st.error("File tidak berisi data.")
        return None

    column = st.sidebar.selectbox("Kolom Time to Failure (TTF)", df.columns, index=default_column(df))
    values, dropped = clean_ttf(df[column])

    with st.expander(f"Pratinjau data — {filename} ({len(df)} baris)"):
        st.dataframe(df.head(100))

    if dropped["missing"]:
        st.warning(f"{dropped['missing']} baris kosong pada kolom '{column}' diabaikan.")
    if dropped["invalid"]:
        st.warning(f"{dropped['invalid']} nilai non-numerik pada kolom '{column}' diabaikan.")

    n_nonpos = int(np.sum(values <= 0))
    if n_nonpos:
        drop = st.sidebar.checkbox(f"Buang {n_nonpos} nilai ≤ 0", value=True)
        if drop:
            values = values[values > 0]
            st.warning(f"{n_nonpos} nilai ≤ 0 dibuang (TTF harus positif).")
        else:
            st.warning(
                f"Data memuat {n_nonpos} nilai ≤ 0. Distribusi Exponential, Weibull, Lognormal, "
                "dan Gamma membutuhkan t > 0 sehingga hanya distribusi Normal yang dapat dicocokkan."
            )

    if values.size < MIN_POINTS:
        st.error(f"Data valid hanya {values.size} titik; minimal {MIN_POINTS} titik diperlukan.")
        return None
    if np.ptp(values) == 0:
        st.error("Semua nilai TTF identik sehingga distribusi tidak dapat dicocokkan.")
        return None
    return values


def show_statistics(t: np.ndarray):
    st.subheader("Statistik Deskriptif")
    cols = st.columns(6)
    stats_ = [
        ("Jumlah data", f"{t.size}"),
        ("Rata-rata", f"{t.mean():.4g}"),
        ("Simp. baku", f"{t.std(ddof=1):.4g}"),
        ("Median", f"{np.median(t):.4g}"),
        ("Minimum", f"{t.min():.4g}"),
        ("Maksimum", f"{t.max():.4g}"),
    ]
    for col, (label, value) in zip(cols, stats_):
        col.metric(label, value)
    if t.size < 20:
        st.info("Ukuran sampel kecil (< 20): hasil pemilihan distribusi perlu ditafsirkan dengan hati-hati.")


def show_fit(results, ranked, best, criterion):
    st.subheader("Hasil Fitting Distribusi")
    st.success(f"Distribusi terbaik berdasarkan **{criterion}**: **{best.name}** — {best.params_text()}")

    cols = st.columns(len(best.params) + 1)
    for col, (label, value) in zip(cols, best.params.items()):
        col.metric(label, f"{value:.5g}")
    cols[-1].metric("MTTF", f"{best.dist.mean():.5g}")

    order = ranked + [r for r in results if not r.ok]
    st.dataframe(highlight_best(summary_table(order, best)), hide_index=True)
    st.caption(
        "Parameter diestimasi dengan Maximum Likelihood (MLE); semua data dianggap kegagalan lengkap "
        "(tanpa sensor). AIC/BIC/KS (D) lebih kecil = lebih baik. p-value KS bersifat pendekatan karena "
        "parameter diestimasi dari data yang sama."
    )
    st.download_button(
        "Unduh tabel hasil (CSV)",
        summary_table(order, best).to_csv(index=False).encode("utf-8-sig"),
        file_name="hasil_fitting_ttf.csv",
        mime="text/csv",
    )


def show_calculator(t, ranked):
    st.subheader("Kalkulator R(t) dan F(t)")
    left, right = st.columns([1, 2])
    names = [r.name for r in ranked]
    chosen = left.selectbox("Distribusi", names, index=0, help="Urutan pertama adalah distribusi terbaik.")
    time = left.number_input("Waktu t", value=float(np.median(t)), format="%.4f")
    result = next(r for r in ranked if r.name == chosen)
    m = rel.evaluate(result.dist, time)

    a, b = right.columns(2)
    a.metric("Keandalan R(t)", f"{float(m['R']):.6f}")
    b.metric("Ketidakandalan F(t)", f"{float(m['F']):.6f}")
    c, d = right.columns(2)
    c.metric("Kepadatan f(t)", f"{float(m['f']):.6g}")
    d.metric("Laju kegagalan h(t)", f"{float(m['h']):.6g}")
    if time < 0:
        left.warning("t negatif: untuk data waktu kegagalan, t seharusnya ≥ 0.")

    with st.expander("Bandingkan semua distribusi pada t ini"):
        rows = []
        for r in ranked:
            v = rel.evaluate(r.dist, time)
            rows.append(
                {"Distribusi": r.name, "R(t)": float(v["R"]), "F(t)": float(v["F"]),
                 "f(t)": float(v["f"]), "h(t)": float(v["h"])}
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True)


def show_plots(t, ranked, best):
    st.subheader("Visualisasi")
    names = [r.name for r in ranked]
    picked = st.multiselect("Distribusi yang ditampilkan", names, default=names)
    shown = [r for r in ranked if r.name in picked]
    tab_pdf, tab_cdf, tab_rel = st.tabs(["PDF f(t)", "CDF F(t)", "Keandalan R(t)"])
    with tab_pdf:
        st.plotly_chart(pdf_figure(t, shown, best))
    with tab_cdf:
        st.plotly_chart(cdf_figure(t, shown, best, reliability=False))
    with tab_rel:
        st.plotly_chart(cdf_figure(t, shown, best, reliability=True))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Analisis TTF", page_icon="📈", layout="wide")
    st.title("📈 Analisis Time to Failure (TTF)")
    st.caption("Fitting distribusi, estimasi parameter, serta perhitungan keandalan R(t) dan ketidakandalan F(t).")

    data, filename = sidebar_inputs()
    if data is None:
        st.info(
            "Unggah file Excel (.xlsx/.xls) atau CSV berisi data waktu kegagalan melalui panel kiri, "
            "atau centang **Gunakan data contoh**."
        )
        return

    t = load_ttf(data, filename)
    if t is None:
        return

    st.sidebar.header("2. Pengaturan Fitting")
    criterion = st.sidebar.radio("Kriteria distribusi terbaik", list(rel.CRITERIA), index=0)
    free_loc = st.sidebar.checkbox(
        "Estimasi parameter lokasi γ",
        value=False,
        help="Jika tidak dicentang, γ = 0 (Weibull 2-parameter, dst.). Estimasi γ dapat kurang "
        "stabil pada sampel kecil.",
    )

    results = rel.fit_all(t, free_loc)
    ranked = rel.rank_results(results, criterion)
    if not ranked:
        st.error("Tidak ada distribusi yang berhasil dicocokkan dengan data ini.")
        for r in results:
            st.write(f"- **{r.name}**: {r.error}")
        return
    best = ranked[0]

    show_statistics(t)
    st.divider()
    show_fit(results, ranked, best, criterion)
    st.divider()
    show_calculator(t, ranked)
    st.divider()
    show_plots(t, ranked, best)


if __name__ == "__main__":
    main()
