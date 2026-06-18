"""
calibration.py — Ajustement du modèle SIR aux données réelles (première vague).

On confronte le modèle à la courbe d'hospitalisations de Santé Publique France.
Le nombre d'hospitalisés est supposé proportionnel au nombre d'infectés actifs :
    hospitalises(t) ≈ k · I(t).

Le taux de transmission ``beta`` (et donc R0 = beta/gamma) ainsi que le facteur
d'observation ``k`` sont estimés par moindres carrés (scipy.least_squares) sur la
première vague. Le script reporte le beta estimé, le R0 implicite, l'erreur
quadratique moyenne (RMSE) et trace le modèle calibré face aux observations.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

import models
from donnees import charger_donnees_france

DOSSIER_FIGURES = os.path.join(os.path.dirname(__file__), "figures")

# Contexte épidémique fixé (cf. parametres.py).
N = 67_000_000.0
GAMMA = 0.1  # durée moyenne d'infection ≈ 10 jours

# Fenêtre de calibration : phase de CROISSANCE de la première vague (printemps
# 2020), avant que le confinement (17 mars 2020) ne fasse chuter la transmission.
# Un SIR à beta constant ne modélise correctement que cette montée exponentielle ;
# estimer R0 sur la croissance initiale est l'approche épidémiologique standard.
DEBUT_VAGUE = "2020-03-18"
FIN_VAGUE = "2020-04-14"  # pic d'hospitalisations de la première vague


def _courbe_modele(params, jours, I0):
    """Simule I(t) pour le SIR avec beta donné. Renvoie le vecteur I (longueur jours)."""
    beta = params
    y0 = [N - I0, I0, 0.0]
    t = np.arange(0, jours, dtype=float)
    res = models.odeint(models.modele_sir, y0, t, args=(beta, GAMMA))
    return res[:, 1]


def calibrer(sauvegarder: bool = True):
    """Calibre beta et k sur la première vague, puis affiche les résultats."""
    df = charger_donnees_france()
    masque = (df["date"] >= DEBUT_VAGUE) & (df["date"] <= FIN_VAGUE)
    vague = df.loc[masque].reset_index(drop=True)

    y_obs = vague["hospitalises"].to_numpy(dtype=float)
    jours = len(y_obs)

    # I0 estimé à partir des premières hospitalisations (ordre de grandeur).
    I0 = max(y_obs[0] * 10.0, 1_000.0)

    def residus(theta):
        """Résidus k·I(t) - hospitalises(t). theta = [log(beta), log(k)]."""
        beta = np.exp(theta[0])
        k = np.exp(theta[1])
        I = _courbe_modele(beta, jours, I0)
        return k * I - y_obs

    # Point de départ : beta tel que R0 = 2.5, k = ratio initial.
    theta0 = [np.log(0.25), np.log(y_obs[0] / I0)]
    resultat = least_squares(residus, theta0, method="lm")

    beta_est = float(np.exp(resultat.x[0]))
    k_est = float(np.exp(resultat.x[1]))
    R0_est = models.r0_sir(beta_est, GAMMA)

    I_modele = _courbe_modele(beta_est, jours, I0)
    y_pred = k_est * I_modele
    rmse = float(np.sqrt(np.mean((y_pred - y_obs) ** 2)))

    print("\n============ Calibration du SIR sur données réelles ============")
    print(f"Fenêtre ........................ {DEBUT_VAGUE} -> {FIN_VAGUE} ({jours} jours)")
    print(f"beta estimé .................... {beta_est:.4f}")
    print(f"R0 implicite (beta/gamma) ...... {R0_est:.2f}")
    print(f"Facteur d'observation k ........ {k_est:.4f}")
    print(f"RMSE ........................... {rmse:,.0f} hospitalisations".replace(",", " "))

    # --- Graphique modèle vs observations ---
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(vague["date"], y_obs, "o", markersize=3, color="black",
            label="Hospitalisations observées (SPF)")
    ax.plot(vague["date"], y_pred, color="red",
            label=f"SIR calibré (R0 = {R0_est:.2f})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Personnes hospitalisées")
    ax.set_title("Calibration du modèle SIR — première vague COVID-19 (France)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.autofmt_xdate()
    fig.tight_layout()

    if sauvegarder:
        os.makedirs(DOSSIER_FIGURES, exist_ok=True)
        chemin = os.path.join(DOSSIER_FIGURES, "calibration.png")
        fig.savefig(chemin, dpi=120)
        print(f"\nFigure enregistrée : {chemin}")

    plt.show()


if __name__ == "__main__":
    calibrer()
