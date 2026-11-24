from pathlib import Path
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run_bls_analysis(
    parquet_path: str,
    output_dir: str = "outputs",
    window_length: int = 401,
    period_min: float = 0.5,
    period_max: float = 15.0,
    num_periods: int = 2000,
    duration: float = 0.1,
) -> dict:
    """Aplica detrending, executa busca BLS e gera gráficos refinados."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(parquet_path)

    lc = lk.LightCurve(
        time=df["time"].values,
        flux=df["flux"].values,
        flux_err=df["flux_err"].values,
    )

    # Remove tendências estelares lentas mantendo quedas curtas
    lc_flat = lc.flatten(window_length=window_length).remove_outliers(sigma=5)

    # Busca de trânsito via Box Least Squares (BLS)
    periods = np.linspace(period_min, period_max, num_periods)
    bls = lc_flat.to_periodogram(method="bls", period=periods, duration=duration)

    best_period = bls.period_at_max_power
    best_t0 = bls.transit_time_at_max_power
    best_depth = bls.depth_at_max_power

    period_val = float(best_period.value)
    t0_val = float(best_t0.value)
    depth_val = float(best_depth.value)
    max_power = float(bls.max_power.value if hasattr(bls.max_power, "value") else bls.max_power)

    # Métricas diagnósticas extras
    best_idx = int(np.nanargmax(bls.power.value))
    snr_val = float(bls.snr.value[best_idx]) if hasattr(bls, "snr") and bls.snr is not None else float("nan")

    # Razão de raios estimada (Rp / Rs)
    rp_rs_ratio = float(np.sqrt(max(depth_val, 0.0)))

    # Checagem de harmônicos (0.5 * P e 2.0 * P)
    periods_arr = np.asarray(bls.period.value, dtype=float)
    power_arr = np.asarray(bls.power.value, dtype=float)

    half_p = 0.5 * period_val
    double_p = 2.0 * period_val

    half_p_power = (
        float(np.interp(half_p, periods_arr, power_arr))
        if (half_p >= periods_arr[0] and half_p <= periods_arr[-1])
        else None
    )
    double_p_power = (
        float(np.interp(double_p, periods_arr, power_arr))
        if (double_p >= periods_arr[0] and double_p <= periods_arr[-1])
        else None
    )

    print(f"Melhor período detetado: {period_val:.6f} dias")
    print(f"Época central (t0): {t0_val:.6f} JD")
    print(f"Profundidade aproximada: {depth_val:.6e}")
    print(f"Potência máxima: {max_power:.4f}")
    print(f"SNR aproximado: {snr_val:.2f}")
    print(f"Rp/Rs estimado: {rp_rs_ratio:.4f}")

    # 1. Gráfico da curva achatada (detrended)
    fig, ax = plt.subplots(figsize=(10, 4))
    lc_flat.scatter(ax=ax, s=1.8, alpha=0.45, color="#4ea8de", label="Fluxo achatado")
    ax.axhline(1.0, color="#adb5bd", linestyle="--", linewidth=1, label="Linha base (1.0)")
    ax.set_title("Curva de Luz Achatada (Detrended) — Kepler-10", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time [JD]")
    ax.set_ylabel("Normalized Flux")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    flat_file = out_path / "kepler-10_flat.png"
    fig.savefig(flat_file, dpi=180)
    plt.close(fig)

    # 2. Gráfico dobrado em fase com curva agrupada (binned)
    lc_folded = lc_flat.fold(period=best_period, epoch_time=best_t0)
    lc_binned = lc_folded.bin(time_bin_size=0.015)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    lc_folded.scatter(ax=ax, s=1.8, alpha=0.35, color="#74c0fc", label="Pontos individuais")
    lc_binned.scatter(ax=ax, s=18, alpha=0.95, color="#e63946", label="Média agrupada (binned)")
    ax.axvline(0.0, color="#2b2d42", linestyle=":", linewidth=1.2, label="Centro da fase (0)")
    ax.axhline(1.0, color="#adb5bd", linestyle="--", linewidth=1)

    metrics_text = (
        f"Período: {period_val:.6f} d | "
        f"t0: {t0_val:.4f} JD | "
        f"Profundidade: {depth_val:.2e}"
    )
    ax.set_title(
        f"Curva Dobrada em Fase — Kepler-10\n{metrics_text}",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Phase (dias em relação a t0)")
    ax.set_ylabel("Normalized Flux")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    folded_file = out_path / "kepler-10_folded.png"
    fig.savefig(folded_file, dpi=180)
    plt.close(fig)

    # Retorna chaves originais preservadas + novos arrays e métricas diagnósticas
    return {
        # Chaves originais
        "period_days": period_val,
        "t0_jd": t0_val,
        "depth": depth_val,
        "flat_plot": str(flat_file),
        "folded_plot": str(folded_file),
        # Métricas diagnósticas extras
        "max_power": max_power,
        "snr": snr_val,
        "rp_rs_ratio": rp_rs_ratio,
        "harmonic_half_period": half_p,
        "harmonic_half_power": half_p_power,
        "harmonic_double_period": double_p,
        "harmonic_double_power": double_p_power,
        # Arrays para gráficos interativos
        "bls_periods": periods_arr,
        "bls_power": power_arr,
        "folded_phase": np.asarray(lc_folded.time.value, dtype=float),
        "folded_flux": np.asarray(lc_folded.flux.value, dtype=float),
        "binned_phase": np.asarray(lc_binned.time.value, dtype=float),
        "binned_flux": np.asarray(lc_binned.flux.value, dtype=float),
    }


if __name__ == "__main__":
    run_bls_analysis("data/processed/kepler-10_kepler.parquet")