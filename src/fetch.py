from pathlib import Path
import lightkurve as lk
import pandas as pd


def fetch_lightcurve(
    target: str, mission: str = "Kepler", output_dir: str = "data/processed"
) -> Path:
    """Descarrega a primeira curva de luz disponível para o alvo e guarda em Parquet."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    search = lk.search_lightcurve(target, mission=mission)
    if len(search) == 0:
        raise ValueError(
            f"Nenhuma curva de luz encontrada para '{target}' na missão {mission}."
        )

    lc = search[0].download().remove_nans().normalize()
    df = pd.DataFrame({
        "time": lc.time.value,
        "flux": lc.flux.value,
        "flux_err": lc.flux_err.value,
    })

    safe_name = target.lower().replace(" ", "_")
    out_path = Path(output_dir) / f"{safe_name}_{mission.lower()}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


if __name__ == "__main__":
    saved_file = fetch_lightcurve("Kepler-10", mission="Kepler")
    print(f"Curva guardada em: {saved_file}")