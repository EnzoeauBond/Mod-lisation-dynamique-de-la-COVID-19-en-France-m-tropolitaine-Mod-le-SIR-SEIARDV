"""
donnees.py — Chargement des données réelles COVID-19 (France métropolitaine).

Source : Santé Publique France, jeu de données « donnees-hospitalieres-covid19 »
publié sur data.gouv.fr. Ce fichier recense, par département et par date, le
nombre de personnes hospitalisées (hosp), en réanimation (rea), décédées à
l'hôpital (dc, cumulé) et rentrées à domicile (rad, cumulé).

Le module télécharge le CSV, l'agrège au niveau national, met le résultat en
cache local dans ``data/`` puis le réutilise hors-ligne lors des exécutions
suivantes. En l'absence de réseau, le cache committé dans le dépôt prend le
relais.
"""

from __future__ import annotations

import io
import os

import pandas as pd

# URL stable du fichier « donnees-hospitalieres-covid19 » (séparateur ';').
URL_HOSPITALIERES = "https://www.data.gouv.fr/fr/datasets/r/63352e38-d353-4b54-bfd1-f1b3ee1cabd7"

DOSSIER_DATA = os.path.join(os.path.dirname(__file__), "data")
CACHE_NATIONAL = os.path.join(DOSSIER_DATA, "covid_france_national.csv")


def _telecharger_csv(url: str, timeout: int = 30) -> str:
    """Télécharge le contenu texte d'un CSV. Lève une exception en cas d'échec."""
    import requests  # import local : le module reste utilisable hors-ligne via le cache

    reponse = requests.get(url, timeout=timeout)
    reponse.raise_for_status()
    return reponse.text


def _agreger_national(csv_texte: str) -> pd.DataFrame:
    """Agrège les données départementales hospitalières au niveau national.

    On ne conserve que le sexe agrégé (colonne ``sexe == 0``) pour éviter de
    compter deux fois hommes + femmes, puis on somme sur les départements.
    """
    df = pd.read_csv(io.StringIO(csv_texte), sep=";")

    # La colonne « sexe » vaut 0 (tous), 1 (hommes), 2 (femmes) : on garde 0.
    if "sexe" in df.columns:
        df = df[df["sexe"] == 0]

    df["jour"] = pd.to_datetime(df["jour"])
    national = (
        df.groupby("jour")[["hosp", "rea", "dc", "rad"]]
        .sum()
        .reset_index()
        .sort_values("jour")
        .reset_index(drop=True)
    )
    national = national.rename(columns={
        "jour": "date",
        "hosp": "hospitalises",
        "rea": "reanimation",
        "dc": "deces_cumul",
        "rad": "retours_domicile_cumul",
    })
    return national


def charger_donnees_france(forcer_telechargement: bool = False) -> pd.DataFrame:
    """Renvoie les données nationales COVID France sous forme de DataFrame.

    Colonnes : date, hospitalises, reanimation, deces_cumul, retours_domicile_cumul.

    Stratégie :
      1. Si un cache existe et ``forcer_telechargement`` est False, on le lit.
      2. Sinon on télécharge depuis data.gouv.fr, on agrège et on met en cache.
      3. En cas d'échec réseau alors qu'un cache existe, on retombe sur le cache.
    """
    os.makedirs(DOSSIER_DATA, exist_ok=True)

    if os.path.exists(CACHE_NATIONAL) and not forcer_telechargement:
        df = pd.read_csv(CACHE_NATIONAL, parse_dates=["date"])
        return df

    try:
        csv_texte = _telecharger_csv(URL_HOSPITALIERES)
        national = _agreger_national(csv_texte)
        national.to_csv(CACHE_NATIONAL, index=False)
        return national
    except Exception as exc:  # réseau indisponible ou format inattendu
        if os.path.exists(CACHE_NATIONAL):
            print(f"[donnees] Téléchargement impossible ({exc}). Utilisation du cache local.")
            return pd.read_csv(CACHE_NATIONAL, parse_dates=["date"])
        raise RuntimeError(
            "Impossible de télécharger les données et aucun cache local n'est "
            f"disponible dans {CACHE_NATIONAL}. Vérifiez la connexion réseau."
        ) from exc


if __name__ == "__main__":
    print("--- Chargement des données réelles COVID-19 (France) ---")
    df = charger_donnees_france()
    print(f"{len(df)} jours chargés, du {df['date'].min().date()} au {df['date'].max().date()}.")
    print("\nAperçu :")
    print(df.head(10).to_string(index=False))
    print("\nPic d'hospitalisations :")
    pic = df.loc[df["hospitalises"].idxmax()]
    print(f"  {int(pic['hospitalises']):,} personnes le {pic['date'].date()}".replace(",", " "))
    print(f"\nCache : {CACHE_NATIONAL}")
