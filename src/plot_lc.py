from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_lightcurve(parquet_path: str, output_img: str = "outputs/kepler-10_raw.png") -> Path:
    """Lê uma curva de luz em Parquet e guarda um gráfico de fluxo ao longo do tempo."""
    df = pd.read_parquet(parquet_path)
    Path(output_img).parent.mkdir(parents=True, exist_ok=True)

    print(f"Linhas carregadas: {len(df)}")
    print(df.head())

    plt.figure(figsize=(10, 4))
    plt.plot(df["time"], df["flux"], ".", markersize=1.5, alpha=0.7)
    plt.title("Curva de Luz Normalizada — Kepler-10")
    plt.xlabel("Tempo (dias)")
    plt.ylabel("Fluxo Normalizado")
    plt.tight_layout()
    plt.savefig(output_img, dpi=150)
    plt.close()

    return Path(output_img)


if __name__ == "__main__":
    img_path = plot_lightcurve("data/processed/kepler-10_kepler.parquet")
    print(f"Gráfico guardado em: {img_path}")