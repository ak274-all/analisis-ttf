"""Supplementary page: concepts behind the five lifetime distributions.

For every distribution the page shows the formulas of f(t), F(t), R(t), h(t)
and MTTF, the meaning of its parameters, and example curves that illustrate
how the parameters change the shape.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

import reliability as rel

PALETTE = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]

# Each entry: formulas are LaTeX strings; ``curves`` are (label, frozen dist)
# pairs drawn in the example chart; f_max / h_max cap the y-axis where the
# curve is unbounded (e.g. Weibull with β < 1 near t = 0).
CONCEPTS = [
    {
        "tab": "Eksponensial",
        "title": "Distribusi Eksponensial",
        "desc": (
            "Memodelkan kegagalan yang terjadi secara **acak dengan laju konstan**: peluang gagal pada "
            "interval berikutnya tidak bergantung pada umur komponen (*memoryless*). Cocok untuk fase "
            "*useful life* pada kurva bak mandi (*bathtub curve*), misalnya komponen elektronik."
        ),
        "support": r"t \ge 0",
        "f": r"f(t) = \lambda\, e^{-\lambda t}",
        "F": r"F(t) = 1 - e^{-\lambda t}",
        "R": r"R(t) = e^{-\lambda t}",
        "h": r"h(t) = \lambda \quad \text{(konstan)}",
        "mttf": r"\text{MTTF} = \dfrac{1}{\lambda} = \theta",
        "params": [
            (r"$\lambda$", "Laju kegagalan", r"$\lambda > 0$", "Jumlah kegagalan per satuan waktu; makin besar, makin cepat gagal."),
            (r"$\theta = 1/\lambda$", "Skala (rata-rata umur)", r"$\theta > 0$", "Sama dengan MTTF. Pada $t = \\theta$, $R(t) = e^{-1} \\approx 36{,}8\\%$."),
        ],
        "notes": [
            "Kasus khusus Weibull dengan $\\beta = 1$ dan Gamma dengan $\\alpha = 1$.",
            "Hanya memiliki satu parameter, sehingga bentuk kurvanya tidak fleksibel.",
        ],
        "curves": [
            ("λ = 0,5", stats.expon(scale=1 / 0.5)),
            ("λ = 1", stats.expon(scale=1.0)),
            ("λ = 2", stats.expon(scale=1 / 2.0)),
        ],
        "t_max": 5.0,
        "f_max": 2.1,
        "h_max": 2.5,
    },
    {
        "tab": "Weibull",
        "title": "Distribusi Weibull",
        "desc": (
            "Distribusi paling populer dalam rekayasa keandalan karena **sangat fleksibel**: dengan mengubah "
            "parameter bentuk $\\beta$, ia dapat memodelkan laju kegagalan yang menurun, konstan, maupun "
            "meningkat. Banyak dipakai untuk komponen mekanis (bantalan, pompa, kelelahan material)."
        ),
        "support": r"t \ge 0",
        "f": r"f(t) = \dfrac{\beta}{\eta}\left(\dfrac{t}{\eta}\right)^{\beta-1} \exp\!\left[-\left(\dfrac{t}{\eta}\right)^{\beta}\right]",
        "F": r"F(t) = 1 - \exp\!\left[-\left(\dfrac{t}{\eta}\right)^{\beta}\right]",
        "R": r"R(t) = \exp\!\left[-\left(\dfrac{t}{\eta}\right)^{\beta}\right]",
        "h": r"h(t) = \dfrac{\beta}{\eta}\left(\dfrac{t}{\eta}\right)^{\beta-1}",
        "mttf": r"\text{MTTF} = \eta\,\Gamma\!\left(1 + \dfrac{1}{\beta}\right)",
        "params": [
            (r"$\beta$", "Bentuk (*shape*)", r"$\beta > 0$", "Menentukan pola laju kegagalan (lihat catatan di bawah)."),
            (r"$\eta$", "Skala (*characteristic life*)", r"$\eta > 0$", "Waktu saat 63,2% unit telah gagal: $F(\\eta) = 1 - e^{-1}$, berapa pun nilai $\\beta$."),
            (r"$\gamma$", "Lokasi (opsional)", r"$\gamma \le t$", "Waktu bebas kegagalan. Jika dipakai, ganti $t$ dengan $(t - \\gamma)$."),
        ],
        "notes": [
            "$\\beta < 1$: laju kegagalan **menurun** (*infant mortality* / kegagalan dini).",
            "$\\beta = 1$: laju kegagalan **konstan**; identik dengan distribusi Eksponensial.",
            "$\\beta > 1$: laju kegagalan **meningkat** (*wear-out* / keausan). $\\beta \\approx 3{,}5$ menyerupai distribusi Normal.",
        ],
        "curves": [
            ("β = 0,5; η = 1", stats.weibull_min(0.5)),
            ("β = 1; η = 1", stats.weibull_min(1.0)),
            ("β = 1,5; η = 1", stats.weibull_min(1.5)),
            ("β = 3,5; η = 1", stats.weibull_min(3.5)),
        ],
        "t_max": 2.5,
        "f_max": 2.0,
        "h_max": 5.0,
    },
    {
        "tab": "Normal",
        "title": "Distribusi Normal",
        "desc": (
            "Kurva lonceng yang **simetris** terhadap rata-rata. Cocok untuk kegagalan akibat **keausan** "
            "yang terkumpul di sekitar umur tertentu dengan variasi kecil (misalnya keausan pahat atau ban)."
        ),
        "support": r"-\infty < t < \infty",
        "f": r"f(t) = \dfrac{1}{\sigma\sqrt{2\pi}} \exp\!\left[-\dfrac{1}{2}\left(\dfrac{t-\mu}{\sigma}\right)^{2}\right]",
        "F": r"F(t) = \Phi\!\left(\dfrac{t-\mu}{\sigma}\right)",
        "R": r"R(t) = 1 - \Phi\!\left(\dfrac{t-\mu}{\sigma}\right)",
        "h": r"h(t) = \dfrac{f(t)}{R(t)} \quad \text{(selalu meningkat)}",
        "mttf": r"\text{MTTF} = \mu",
        "params": [
            (r"$\mu$", "Rata-rata (lokasi)", r"$-\infty < \mu < \infty$", "Pusat kurva; sekaligus median dan modus. $R(\\mu) = 50\\%$."),
            (r"$\sigma$", "Simpangan baku (skala)", r"$\sigma > 0$", "Sebaran umur; makin kecil, kegagalan makin terkumpul di sekitar $\\mu$."),
        ],
        "notes": [
            "$\\Phi(z)$ adalah CDF normal baku; tidak memiliki bentuk tertutup sehingga dihitung secara numerik atau dari tabel $z$.",
            "Secara teori mengizinkan $t < 0$. Model ini layak dipakai bila $\\mu$ jauh lebih besar dari $\\sigma$ (kira-kira $\\mu > 3\\sigma$) sehingga peluang waktu negatif dapat diabaikan.",
        ],
        "curves": [
            ("μ = 5; σ = 0,5", stats.norm(5, 0.5)),
            ("μ = 5; σ = 1", stats.norm(5, 1.0)),
            ("μ = 5; σ = 2", stats.norm(5, 2.0)),
            ("μ = 3; σ = 1", stats.norm(3, 1.0)),
        ],
        "t_max": 10.0,
        "f_max": None,
        "h_max": 6.0,
    },
    {
        "tab": "Lognormal",
        "title": "Distribusi Lognormal",
        "desc": (
            "Waktu kegagalan $T$ berdistribusi Lognormal bila $\\ln T$ berdistribusi Normal. Kurvanya "
            "**menceng ke kanan** (ekor panjang). Cocok untuk kegagalan akibat degradasi yang bersifat "
            "multiplikatif (retak lelah, korosi) dan sangat umum untuk **waktu perbaikan**."
        ),
        "support": r"t > 0",
        "f": r"f(t) = \dfrac{1}{t\,\sigma\sqrt{2\pi}} \exp\!\left[-\dfrac{1}{2}\left(\dfrac{\ln t-\mu}{\sigma}\right)^{2}\right]",
        "F": r"F(t) = \Phi\!\left(\dfrac{\ln t-\mu}{\sigma}\right)",
        "R": r"R(t) = 1 - \Phi\!\left(\dfrac{\ln t-\mu}{\sigma}\right)",
        "h": r"h(t) = \dfrac{f(t)}{R(t)} \quad \text{(naik lalu turun)}",
        "mttf": r"\text{MTTF} = \exp\!\left(\mu + \dfrac{\sigma^{2}}{2}\right)",
        "params": [
            (r"$\mu$", "Rata-rata dari $\\ln t$", r"$-\infty < \mu < \infty$", "Berperan sebagai skala: median umur $= e^{\\mu}$."),
            (r"$\sigma$", "Simpangan baku dari $\\ln t$", r"$\sigma > 0$", "Berperan sebagai bentuk: makin besar, kurva makin menceng ke kanan."),
            (r"$\gamma$", "Lokasi (opsional)", r"$\gamma < t$", "Jika dipakai, ganti $t$ dengan $(t - \\gamma)$."),
        ],
        "notes": [
            "$\\mu$ dan $\\sigma$ **bukan** rata-rata dan simpangan baku dari $t$, melainkan dari $\\ln t$.",
            "Laju kegagalan naik sampai puncak lalu menurun, sehingga kurang sesuai untuk keausan murni jangka panjang.",
        ],
        "curves": [
            ("μ = 0; σ = 0,25", stats.lognorm(0.25)),
            ("μ = 0; σ = 0,5", stats.lognorm(0.5)),
            ("μ = 0; σ = 1", stats.lognorm(1.0)),
            ("μ = 0,5; σ = 0,5", stats.lognorm(0.5, scale=np.exp(0.5))),
        ],
        "t_max": 4.0,
        "f_max": None,
        "h_max": 5.0,
    },
    {
        "tab": "Gamma",
        "title": "Distribusi Gamma",
        "desc": (
            "Fleksibel seperti Weibull. Secara fisik menggambarkan waktu sampai terjadinya kejadian "
            "ke-$\\alpha$ pada proses dengan laju konstan, sehingga cocok untuk sistem dengan **cadangan "
            "(*standby*)** atau kegagalan akibat **akumulasi kejutan/kerusakan**."
        ),
        "support": r"t \ge 0",
        "f": r"f(t) = \dfrac{t^{\alpha-1}\, e^{-t/\theta}}{\Gamma(\alpha)\,\theta^{\alpha}}",
        "F": r"F(t) = \dfrac{\gamma\!\left(\alpha,\ t/\theta\right)}{\Gamma(\alpha)}",
        "R": r"R(t) = 1 - \dfrac{\gamma\!\left(\alpha,\ t/\theta\right)}{\Gamma(\alpha)}",
        "h": r"h(t) = \dfrac{f(t)}{R(t)}",
        "mttf": r"\text{MTTF} = \alpha\,\theta",
        "params": [
            (r"$\alpha$", "Bentuk (*shape*)", r"$\alpha > 0$", "Menentukan pola laju kegagalan (lihat catatan di bawah)."),
            (r"$\theta$", "Skala (*scale*)", r"$\theta > 0$", "Meregangkan sumbu waktu. Sebagian buku memakai laju $\\lambda = 1/\\theta$."),
            (r"$\gamma$", "Lokasi (opsional)", r"$\gamma \le t$", "Jika dipakai, ganti $t$ dengan $(t - \\gamma)$."),
        ],
        "notes": [
            "$\\Gamma(\\alpha)$ adalah fungsi gamma dan $\\gamma(\\alpha, x) = \\int_0^x u^{\\alpha-1} e^{-u}\\,du$ adalah fungsi gamma tak lengkap bawah (berbeda dari parameter lokasi $\\gamma$); keduanya dihitung secara numerik.",
            "$\\alpha < 1$: laju kegagalan menurun; $\\alpha = 1$: konstan (Eksponensial); $\\alpha > 1$: meningkat dan mendekati $1/\\theta$.",
        ],
        "curves": [
            ("α = 1; θ = 1", stats.gamma(1.0)),
            ("α = 2; θ = 1", stats.gamma(2.0)),
            ("α = 5; θ = 1", stats.gamma(5.0)),
            ("α = 2; θ = 2", stats.gamma(2.0, scale=2.0)),
        ],
        "t_max": 12.0,
        "f_max": None,
        "h_max": 1.3,
    },
]

SUMMARY = """
| Distribusi | Parameter | Pola laju kegagalan $h(t)$ | Penggunaan tipikal |
|---|---|---|---|
| Eksponensial | $\\lambda$ | Konstan | Kegagalan acak, komponen elektronik, fase *useful life* |
| Weibull | $\\beta,\\ \\eta$ | Menurun, konstan, atau meningkat | Komponen mekanis, analisis umur secara umum |
| Normal | $\\mu,\\ \\sigma$ | Selalu meningkat | Keausan dengan variasi kecil |
| Lognormal | $\\mu,\\ \\sigma$ | Naik lalu turun | Kelelahan, korosi, waktu perbaikan |
| Gamma | $\\alpha,\\ \\theta$ | Menurun, konstan, atau meningkat menuju $1/\\theta$ | Sistem *standby*, akumulasi kerusakan |
"""


def concept_figure(concept: dict) -> go.Figure:
    """2x2 chart of f(t), F(t), R(t) and h(t) for the example parameter sets."""
    panels = [("f", "f(t) — kepadatan peluang"), ("F", "F(t) — ketidakandalan"),
              ("R", "R(t) — keandalan"), ("h", "h(t) — laju kegagalan")]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[p[1] for p in panels],
                        vertical_spacing=0.14, horizontal_spacing=0.09)
    t = np.linspace(concept["t_max"] / 500, concept["t_max"], 500)
    for i, (label, dist) in enumerate(concept["curves"]):
        values = rel.evaluate(dist, t)
        for j, (key, _) in enumerate(panels):
            y = values[key]
            fig.add_trace(
                go.Scatter(
                    x=t, y=np.where(np.isfinite(y), y, np.nan), mode="lines", name=label,
                    legendgroup=label, showlegend=(j == 0),
                    line=dict(color=PALETTE[i % len(PALETTE)], width=2.5),
                ),
                row=j // 2 + 1, col=j % 2 + 1,
            )
    if concept["f_max"]:
        fig.update_yaxes(range=[0, concept["f_max"]], row=1, col=1)
    fig.update_yaxes(range=[0, 1.03], row=1, col=2)
    fig.update_yaxes(range=[0, 1.03], row=2, col=1)
    if concept["h_max"]:
        fig.update_yaxes(range=[0, concept["h_max"]], row=2, col=2)
    fig.update_xaxes(title_text="Waktu t", row=2)
    fig.update_layout(height=640, margin=dict(t=40, r=10, l=10, b=10),
                      legend=dict(orientation="h", y=-0.12))
    return fig


def show_concept(concept: dict):
    st.subheader(concept["title"])
    st.markdown(concept["desc"])

    st.markdown(f"**Rumus** (untuk ${concept['support']}$)")
    left, right = st.columns(2)
    with left:
        st.markdown("Fungsi kepadatan peluang $f(t)$")
        st.latex(concept["f"])
        st.markdown("Fungsi keandalan $R(t)$")
        st.latex(concept["R"])
        st.markdown("Rata-rata waktu kegagalan")
        st.latex(concept["mttf"])
    with right:
        st.markdown("Fungsi distribusi kumulatif / ketidakandalan $F(t)$")
        st.latex(concept["F"])
        st.markdown("Laju kegagalan $h(t)$")
        st.latex(concept["h"])

    st.markdown("**Parameter distribusi**")
    rows = ["| Simbol | Nama | Syarat | Makna |", "|---|---|---|---|"]
    rows += [f"| {sym} | {name} | {cond} | {meaning} |" for sym, name, cond, meaning in concept["params"]]
    st.markdown("\n".join(rows))

    st.markdown("**Catatan**")
    st.markdown("\n".join(f"- {note}" for note in concept["notes"]))

    st.markdown("**Grafik distribusi** — pengaruh nilai parameter terhadap bentuk kurva")
    st.plotly_chart(concept_figure(concept))


def show_theory():
    """Render the whole supplementary page."""
    st.header("Konsep Distribusi Waktu Kegagalan")
    st.markdown(
        "Misalkan $T$ adalah waktu sampai suatu peralatan gagal (*Time to Failure*). Perilaku $T$ "
        "dijelaskan oleh empat fungsi yang saling terkait:"
    )
    left, right = st.columns(2)
    with left:
        st.markdown("**Kepadatan peluang** — sebaran kegagalan terhadap waktu")
        st.latex(r"f(t) = \dfrac{dF(t)}{dt} = -\dfrac{dR(t)}{dt}")
        st.markdown("**Keandalan** — peluang masih berfungsi sampai waktu $t$")
        st.latex(r"R(t) = P(T > t) = 1 - F(t)")
    with right:
        st.markdown("**Ketidakandalan (CDF)** — peluang sudah gagal sebelum waktu $t$")
        st.latex(r"F(t) = P(T \le t) = \int_{0}^{t} f(u)\,du")
        st.markdown("**Laju kegagalan** — risiko gagal sesaat bagi unit yang masih bertahan")
        st.latex(r"h(t) = \dfrac{f(t)}{R(t)}")
    st.markdown("Umur rata-rata (*Mean Time to Failure*) diperoleh dari luas di bawah kurva keandalan:")
    st.latex(r"\text{MTTF} = \int_{0}^{\infty} t\,f(t)\,dt = \int_{0}^{\infty} R(t)\,dt")
    st.caption(
        "Rumus di bawah ditulis untuk parameter lokasi γ = 0, sesuai pengaturan bawaan aplikasi. "
        "Simbol parameter sama dengan yang tampil pada tabel hasil fitting."
    )

    for tab, concept in zip(st.tabs([c["tab"] for c in CONCEPTS]), CONCEPTS):
        with tab:
            show_concept(concept)

    st.divider()
    st.subheader("Ringkasan Perbandingan")
    st.markdown(SUMMARY)
