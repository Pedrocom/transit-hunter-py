from pathlib import Path
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

# 1. Métricas e 2 telas principais no topo/meio
if st.session_state["bls_results"] is not None:
    results = st.session_state["bls_results"]

    m1, m2, m3 = st.columns(3)
    m1.metric("Melhor período", f"{results['period_days']:.6f} d")
    m2.metric("Época central (t0)", f"{results['t0_jd']:.6f} JD")
    m3.metric("Profundidade aprox.", f"{results['depth']:.2e}")

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