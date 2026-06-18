"""
comparaison.py — Comparaison des modèles SIR et SEIARDV.

Simule les deux modèles avec le même contexte (France 2020), superpose les
courbes d'infectés et produit un tableau comparatif des grandeurs clés :
nombre de reproduction de base R0, date et amplitude du pic, et part totale de
la population infectée.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt

import models
from parametres import SIR_FRANCE_2020, SEIARDV_FRANCE_2020

DOSSIER_FIGURES = os.path.join(os.path.dirname(__file__), "figures")


def comparer(sauvegarder: bool = True):
    """Compare SIR et SEIARDV et affiche graphique + tableau récapitulatif."""
    # --- Simulation SIR ---
    sir = SIR_FRANCE_2020
    t_sir, res_sir = models.integrer(
        models.modele_sir, sir.y0, sir.jours, args=(sir.beta, sir.gamma)
    )
    S_sir, I_sir, R_sir = res_sir.T
    R0_sir = models.r0_sir(sir.beta, sir.gamma)
    pic_sir = models.pic_epidemique(t_sir, I_sir)
    taille_sir = models.taille_finale_sir(R0_sir)

    # --- Simulation SEIARDV ---
    se = SEIARDV_FRANCE_2020
    t_se, res_se = models.integrer(
        models.modele_seiardv, se.y0, se.jours, args=se.args_edo
    )
    S_se, E_se, I_se, A_se, R_se, V_se, D_se = res_se.T
    infectes_se = I_se + A_se
    R0_se = models.r0_seiardv(*se.args_edo)
    pic_se = models.pic_epidemique(t_se, infectes_se)
    # Part totale infectée (proxy) : rétablis + décès en fin de simulation.
    taille_se = (R_se[-1] + D_se[-1]) / se.N

    # --- Tableau comparatif ---
    print("\n================== Comparaison SIR vs SEIARDV ==================")
    entete = f"{'Indicateur':<32}{'SIR':>14}{'SEIARDV':>16}"
    print(entete)
    print("-" * len(entete))
    print(f"{'R0 (reproduction de base)':<32}{R0_sir:>14.2f}{R0_se:>16.2f}")
    print(f"{'Jour du pic':<32}{pic_sir[0]:>14.0f}{pic_se[0]:>16.0f}")
    print(f"{'Amplitude du pic (infectés)':<32}{pic_sir[1]:>14,.0f}{pic_se[1]:>16,.0f}".replace(",", " "))
    print(f"{'Part totale infectée':<32}{taille_sir * 100:>13.1f}%{taille_se * 100:>15.1f}%")
    print(f"{'Décès cumulés (SEIARDV)':<32}{'—':>14}{D_se[-1]:>16,.0f}".replace(",", " "))

    # --- Graphique ---
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(t_sir, I_sir / sir.N, label="SIR — Infectés (I)", color="red")
    ax.plot(t_se, I_se / se.N, label="SEIARDV — Symptomatiques (I)", color="darkorange")
    ax.plot(t_se, infectes_se / se.N, label="SEIARDV — Contagieux (I+A)",
            color="orange", linestyle="--")

    ax.axvline(pic_sir[0], color="red", linestyle=":", alpha=0.5)
    ax.axvline(pic_se[0], color="orange", linestyle=":", alpha=0.5)

    ax.set_xlabel("Jours")
    ax.set_ylabel("Proportion de la population infectée")
    ax.set_title("Comparaison des modèles SIR et SEIARDV (France 2020)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()

    if sauvegarder:
        os.makedirs(DOSSIER_FIGURES, exist_ok=True)
        chemin = os.path.join(DOSSIER_FIGURES, "comparaison.png")
        fig.savefig(chemin, dpi=120)
        print(f"\nFigure enregistrée : {chemin}")

    plt.show()


if __name__ == "__main__":
    comparer()
