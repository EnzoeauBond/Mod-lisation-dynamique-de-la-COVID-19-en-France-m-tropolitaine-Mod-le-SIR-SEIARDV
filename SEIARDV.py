"""
SEIARDV.py — Simulation du modèle SEIARDV pour la COVID-19 en France.

Modèle à 7 compartiments (Susceptibles, Exposés, Infectés symptomatiques,
Asymptomatiques, Rétablis, Vaccinés, Décédés) calibré pour la France
métropolitaine au début de la première vague (2020).

Le nombre de reproduction de base R0 est calculé rigoureusement par la matrice
de prochaine génération (et non par une approximation). Le script affiche
également le seuil d'immunité collective, les caractéristiques du pic et le
bilan final (rétablis / décès / vaccinés), et trace le R effectif R(t).
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt

import models
from parametres import ParametresSEIARDV, SEIARDV_FRANCE_2020

DOSSIER_FIGURES = os.path.join(os.path.dirname(__file__), "figures")


def simuler_et_afficher(params: ParametresSEIARDV, sauvegarder: bool = True):
    """Résout les EDO du modèle SEIARDV et affiche résultats + graphiques."""
    t, resultats = models.integrer(
        models.modele_seiardv, params.y0, params.jours, args=params.args_edo
    )
    S, E, I, A, R, V, D = resultats.T

    # --- Indicateurs épidémiologiques ---
    R0 = models.r0_seiardv(*params.args_edo)
    R0_formule = models.r0_seiardv_formule(
        params.beta_I, params.beta_A, params.sigma, params.p,
        params.gamma_I, params.gamma_A, params.alpha,
    )
    seuil = models.seuil_immunite_collective(R0)
    infectes = I + A  # ensemble des individus contagieux
    jour_pic, amplitude_pic = models.pic_epidemique(t, infectes)
    Rt = models.reproduction_effective(S, params.N, R0)

    print("\n========= Résultats du modèle SEIARDV =========")
    print(f"R0 (matrice de prochaine génération) ....... {R0:.2f}")
    print(f"R0 (formule analytique, contrôle) .......... {R0_formule:.2f}")
    print(f"Seuil d'immunité collective ................ {seuil * 100:.1f} %")
    print(f"Pic de contagieux (I+A) .................... {amplitude_pic:,.0f} au jour {jour_pic:.0f}".replace(",", " "))
    print("\nBilan au terme de la simulation :")
    print(f"  Rétablis (R) ............................. {R[-1]:,.0f}".replace(",", " "))
    print(f"  Décès (D) ................................ {D[-1]:,.0f}".replace(",", " "))
    print(f"  Vaccinés (V) ............................. {V[-1]:,.0f}".replace(",", " "))

    # --- Graphiques (proportions de la population) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), height_ratios=[3, 1], sharex=True)

    ax1.plot(t, S / params.N, label="Susceptibles (S)", color="blue")
    ax1.plot(t, I / params.N, label="Symptomatiques (I)", color="red")
    ax1.plot(t, A / params.N, label="Asymptomatiques (A)", color="orange")
    ax1.plot(t, R / params.N, label="Rétablis (R)", color="green")
    ax1.plot(t, D / params.N, label="Décédés (D)", color="black", linestyle="--")
    ax1.plot(t, V / params.N, label="Vaccinés (V)", color="purple", linestyle=":")
    ax1.axvline(jour_pic, color="grey", linestyle=":", alpha=0.7,
                label=f"Pic (jour {jour_pic:.0f})")
    ax1.set_ylabel("Proportion de la population")
    ax1.set_title(f"Modèle SEIARDV — COVID-19 France (R0 = {R0:.2f})")
    ax1.legend(ncol=2)
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(t, Rt, color="purple", label="R effectif R(t)")
    ax2.axhline(1.0, color="black", linestyle="--", alpha=0.7, label="Seuil R = 1")
    ax2.set_xlabel("Jours")
    ax2.set_ylabel("R(t)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()

    if sauvegarder:
        os.makedirs(DOSSIER_FIGURES, exist_ok=True)
        chemin = os.path.join(DOSSIER_FIGURES, "seiardv.png")
        fig.savefig(chemin, dpi=120)
        print(f"\nFigure enregistrée : {chemin}")

    plt.show()


def simuler_seiardv():
    """Point d'entrée : simulation avec le preset France 2020."""
    print("--- Simulation du Modèle SEIARDV pour la COVID-19 en France ---")
    simuler_et_afficher(SEIARDV_FRANCE_2020)


if __name__ == "__main__":
    simuler_seiardv()
