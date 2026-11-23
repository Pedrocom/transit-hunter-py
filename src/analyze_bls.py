from pathlib import Path
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run_bls_analysis(parquet_path: str, output_dir: str = "outputs") -> None:
    """Aplica detrending simples e procura sinais de trânsito via BLS."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(parquet_path)

    lc = lk.LightCurve(
        time=df["time"].values,
        flux=df["flux"].values,
        flux_err=df["flux_err"].values,
    )

    # Remove tendências lentas mantendo variações curtas (como trânsitos)
    lc_flat = lc.flatten(window_length=401).remove_outliers(sigma=5)

    # Procura períodos de trânsito entre 0.5 e 15 dias
    periods = np.linspace(0.5, 15.0, 2000)
    bls = lc_flat.to_periodogram(method="bls", period=periods, duration=0.1)

    best_period = bls.period_at_max_power
    best_t0 = bls.transit_time_at_max_power
    best_depth = bls.depth_at_max_power

    print(f"Melhor período detetado: {best_period.value:.6f} dias")
    print(f"Época central (t0): {best_t0.value:.6f}")
    print(f"Profundidade aproximada: {best_depth.value:.6e}")

    # 1. Gráfico da curva achatada
    fig, ax = plt.subplots(figsize=(10, 4))
    lc_flat.scatter(ax=ax, s=2, alpha=0.6)
    ax.set_title("Curva de Luz Achatada (Detrended) — Kepler-10")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "kepler-10_flat.png", dpi=150)
    plt.close(fig)

    # 2. Gráfico dobrado em fase no melhor período encontrado
    lc_folded = lc_flat.fold(period=best_period, epoch_time=best_t0)
    fig, ax = plt.subplots(figsize=(10, 4))
    lc_folded.scatter(ax=ax, s=2, alpha=0.6)
    ax.set_title(f"Curva Dobrada em Fase (Período = {best_period.value:.4f} d)")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "kepler-10_folded.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run_bls_analysis("data/processed/kepler-10_kepler.parquet")