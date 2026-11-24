from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from src.analyze_bls import run_bls_analysis

st.set_page_config(page_title="Transit Hunter — BLS", layout="wide")
st.title("Transit Hunter — Deteção de Trânsitos (BLS)")
st.caption("Exploração interativa de curvas de luz com *detrending* e Box Least Squares.")

data_dir = Path("data/processed")
parquet_files = sorted(data_dir.glob("*.parquet")) if data_dir.exists() else []

if not parquet_files:
    st.warning("Nenhum ficheiro `.parquet` encontrado em `data/processed/`.")
    st.stop()

if "bls_results" not in st.session_state:
    st.session_state["bls_results"] = None

from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from src.analyze_bls import run_bls_analysis

st.set_page_config(page_title="Transit Hunter — BLS", layout="wide")
st.title("Transit Hunter — Deteção de Trânsitos (BLS)")
st.caption("Exploração interativa de curvas de luz com *detrending* e Box Least Squares.")

data_dir = Path("data/processed")
parquet_files = sorted(data_dir.glob("*.parquet")) if data_dir.exists() else []

if not parquet_files:
    st.warning("Nenhum ficheiro `.parquet` encontrado em `data/processed/`.")
    st.stop()

if "bls_results" not in st.session_state:
    st.session_state["bls_results"] = None

# 1. Métricas e visualizações principais no topo/meio
if st.session_state["bls_results"] is not None:
    results = st.session_state["bls_results"]

    # Indicadores principais
    st.subheader("Resultados Principais")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Melhor período", f"{results['period_days']:.6f} d")
    m2.metric("Época central (t0)", f"{results['t0_jd']:.6f} JD")
    m3.metric("Profundidade aprox.", f"{results['depth']:.2e}")
    m4.metric("Razão de Raios (Rp/Rs)", f"{results.get('rp_rs_ratio', 0.0):.4f}")

    # Diagnósticos adicionais
    d1, d2, d3, d4 = st.columns(4)
    max_p = results.get("max_power")
    snr = results.get("snr")
    half_pow = results.get("harmonic_half_power")
    double_pow = results.get("harmonic_double_power")

    d1.metric("Potência Máxima", f"{max_p:.2f}" if max_p is not None else "N/A")
    d2.metric("SNR do Trânsito", f"{snr:.1f}" if snr is not None and not np.isnan(snr) else "N/A")
    d3.metric(
        "Potência em 0.5×P",
        f"{half_pow:.2f}" if half_pow is not None else "Fora da grelha",
        help=f"0.5 × P = {results.get('harmonic_half_period', 0):.4f} d",
    )
    d4.metric(
        "Potência em 2.0×P",
        f"{double_pow:.2f}" if double_pow is not None else "Fora da grelha",
        help=f"2.0 × P = {results.get('harmonic_double_period', 0):.4f} d",
    )

    # Gráficos Interativos Plotly
    if "bls_periods" in results and "bls_power" in results:
        st.subheader("Visualizações Interativas")
        p_col1, p_col2 = st.columns(2)

        with p_col1:
            # Gráfico do Periodograma BLS
            fig_bls = go.Figure()
            fig_bls.add_trace(
                go.Scatter(
                    x=results["bls_periods"],
                    y=results["bls_power"],
                    mode="lines",
                    name="BLS Power",
                    line=dict(color="#1f77b4", width=1.5),
                    hovertemplate="Período: %{x:.4f} d<br>Potência: %{y:.2f}<extra></extra>",
                )
            )
            # Destacar o melhor período
            fig_bls.add_vline(
                x=results["period_days"],
                line=dict(color="#d62728", width=2, dash="dash"),
                annotation_text=f"P = {results['period_days']:.4f} d",
                annotation_position="top right",
            )
            # Destacar harmônicos se estiverem dentro do intervalo
            if half_pow is not None:
                fig_bls.add_vline(
                    x=results["harmonic_half_period"],
                    line=dict(color="#ff7f0e", width=1.2, dash="dot"),
                    annotation_text="0.5×P",
                    annotation_position="top left",
                )
            if double_pow is not None:
                fig_bls.add_vline(
                    x=results["harmonic_double_period"],
                    line=dict(color="#2ca02c", width=1.2, dash="dot"),
                    annotation_text="2.0×P",
                    annotation_position="top right",
                )

            fig_bls.update_layout(
                title=f"Periodograma BLS (Pico em {results['period_days']:.4f} dias)",
                xaxis_title="Período (dias)",
                yaxis_title="Potência BLS",
                template="plotly_white",
                height=420,
                margin=dict(l=40, r=30, t=50, b=40),
            )
            st.plotly_chart(fig_bls, use_container_width=True)

        with p_col2:
            # Gráfico Curva Dobrada em Fase (com zoom inicial no trânsito)
            fig_fold = go.Figure()

            # Subamostragem leve se houver dezenas de milhares de pontos para alta performance no navegador
            ph = results["folded_phase"]
            fl = results["folded_flux"]
            if len(ph) > 15000:
                step = int(np.ceil(len(ph) / 15000))
                ph_plot = ph[::step]
                fl_plot = fl[::step]
            else:
                ph_plot = ph
                fl_plot = fl

            fig_fold.add_trace(
                go.Scattergl(
                    x=ph_plot,
                    y=fl_plot,
                    mode="markers",
                    name="Pontos individuais",
                    marker=dict(color="#4ea8de", size=3, opacity=0.35),
                    hovertemplate="Fase: %{x:.4f} d<br>Fluxo: %{y:.5f}<extra></extra>",
                )
            )

            # Sobreposição dos pontos binned (médias)
            if "binned_phase" in results and "binned_flux" in results:
                fig_fold.add_trace(
                    go.Scatter(
                        x=results["binned_phase"],
                        y=results["binned_flux"],
                        mode="markers+lines",
                        name="Média agrupada (binned)",
                        marker=dict(color="#d62728", size=6),
                        line=dict(color="#d62728", width=1.5),
                        hovertemplate="Fase: %{x:.4f} d<br>Fluxo Médio: %{y:.5f}<extra></extra>",
                    )
                )

            fig_fold.add_vline(x=0.0, line=dict(color="#333333", width=1, dash="dot"))
            fig_fold.add_hline(y=1.0, line=dict(color="#888888", width=1, dash="dash"))

            # Intervalo focado no trânsito (por exemplo +/- 0.5 dia ou proporção da duração)
            transit_zoom = max(0.4, results.get("harmonic_half_period", 1.0) * 0.25)
            fig_fold.update_layout(
                title=f"Curva Dobrada em Fase (P = {results['period_days']:.4f} d)",
                xaxis_title="Fase (dias relativos a t0)",
                yaxis_title="Fluxo Normalizado",
                xaxis=dict(range=[-transit_zoom, transit_zoom]),
                template="plotly_white",
                height=420,
                margin=dict(l=40, r=30, t=50, b=40),
            )
            st.plotly_chart(fig_fold, use_container_width=True)

    # Imagens PNG salvas em outputs/ preservadas
    st.subheader("Gráficos Estáticos Gerados (PNG)")
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.image(
            results["flat_plot"],
            caption="Curva achatada (Detrended)",
            use_container_width=True,
        )
    with img_col2:
        st.image(
            results["folded_plot"],
            caption="Curva dobrada em fase",
            use_container_width=True,
        )
else:
    st.info("Ajuste os parâmetros abaixo e clique em **Executar análise BLS** para gerar os gráficos.")

st.divider()

# 2. Controles no bottom da página
st.subheader("Parâmetros da Análise")

row1_c1, row1_c2, row1_c3 = st.columns(3)
with row1_c1:
    selected_file = st.selectbox(
        "Curva de luz (.parquet)",
        options=parquet_files,
        format_func=lambda p: p.name,
    )
with row1_c2:
    window_length = st.slider(
        "Window length (flatten)",
        min_value=101,
        max_value=1001,
        value=401,
        step=2,
    )
with row1_c3:
    duration = st.slider(
        "Duração do trânsito (dias)",
        min_value=0.02,
        max_value=0.50,
        value=0.10,
        step=0.02,
    )

row2_c1, row2_c2, row2_c3, row2_c4 = st.columns([1, 1, 1.5, 1])
with row2_c1:
    period_min = st.number_input(
        "Período mínimo (dias)",
        min_value=0.1,
        max_value=50.0,
        value=0.5,
        step=0.1,
    )
with row2_c2:
    period_max = st.number_input(
        "Período máximo (dias)",
        min_value=0.5,
        max_value=100.0,
        value=15.0,
        step=0.5,
    )
with row2_c3:
    num_periods = st.slider(
        "Número de períodos na grelha",
        min_value=500,
        max_value=5000,
        value=2000,
        step=250,
    )
with row2_c4:
    st.write("")
    st.write("")
    run_clicked = st.button("Executar análise BLS", type="primary", use_container_width=True)

if period_min >= period_max:
    st.error("O período mínimo deve ser menor do que o período máximo.")
    st.stop()

if run_clicked:
    with st.spinner("A processar curva de luz e a correr o BLS..."):
        st.session_state["bls_results"] = run_bls_analysis(
            parquet_path=str(selected_file),
            output_dir="outputs",
            window_length=int(window_length),
            period_min=float(period_min),
            period_max=float(period_max),
            num_periods=int(num_periods),
            duration=float(duration),
        )
    st.rerun()