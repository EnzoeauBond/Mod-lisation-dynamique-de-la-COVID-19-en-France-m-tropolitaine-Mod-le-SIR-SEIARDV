"""
SIR.py — Simulation du modèle SIR pour la COVID-19.

Deux modes d'exécution sont proposés :
  1) Saisie manuelle des paramètres au clavier ;
  2) Preset « France 2020 » calibré (reproductible, sans saisie).

En plus des courbes S/I/R, le script calcule et affiche les grandeurs
épidémiologiques clés : nombre de reproduction de base R0, seuil d'immunité
collective, date et amplitude du pic, taille finale de l'épidémie, et trace le
nombre de reproduction effectif R(t).
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt

import models
from parametres import ParametresSIR, SIR_FRANCE_2020

DOSSIER_FIGURES = os.path.join(os.path.dirname(__file__), "figures")


def _saisir_parametres() -> ParametresSIR | None:
    """Saisie interactive des paramètres. Renvoie None en cas d'erreur de saisie."""
    try:
        N = float(input("Population totale (N) : "))
        I0 = float(input("Nombre initial d'infectés (I0) : "))
        R0_init = float(input("Nombre initial de retirés/immunisés (R0) : "))
        jours = int(input("Nombre de jours à simuler : "))
        beta = float(input("Taux de transmission (beta, ex: 0.25) : "))
        gamma = float(input("Taux de guérison/retrait (gamma, ex: 0.1) : "))
    except ValueError:
        print("Erreur de saisie. Veuillez entrer des nombres valides.")
        return None

    params = ParametresSIR(N=N, I0=I0, R0_init=R0_init, beta=beta, gamma=gamma, jours=jours)
    if params.S0 < 0:
        print("Erreur : I0 + R0 dépasse la population totale N.")
        return None
    return params


def simuler_et_afficher(params: ParametresSIR, sauvegarder: bool = True):
    """Résout les EDO du modèle SIR et affiche résultats + graphiques."""
    # Résolution numérique (pas de 1 jour).
    t, resultats = models.integrer(
        models.modele_sir, params.y0, params.jours, args=(params.beta, params.gamma)
    )
    S, I, R = resultats.T

    # --- Indicateurs épidémiologiques ---
    R0 = models.r0_sir(params.beta, params.gamma)
    seuil = models.seuil_immunite_collective(R0)
    jour_pic, amplitude_pic = models.pic_epidemique(t, I)
    taille_finale = models.taille_finale_sir(R0)
    Rt = models.reproduction_effective(S, params.N, R0)

    print("\n=========== Résultats du modèle SIR ===========")
    print(f"Nombre de reproduction de base R0 .......... {R0:.2f}")
    if R0 > 1:
        print("  -> R0 > 1 : l'épidémie se propage.")
        print(f"Seuil d'immunité collective ................ {seuil * 100:.1f} %")
        print(f"Pic épidémique ............................. {amplitude_pic:,.0f} infectés au jour {jour_pic:.0f}".replace(",", " "))
        print(f"Taille finale de l'épidémie ................ {taille_finale * 100:.1f} % de la population")
        print(f"  ({taille_finale * params.N:,.0f} personnes infectées au total)".replace(",", " "))
    else:
        print("  -> R0 < 1 : l'épidémie s'éteint.")

    # --- Graphiques ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), height_ratios=[3, 1], sharex=True)

    ax1.plot(t, S, label="Susceptibles (S)", color="blue")
    ax1.plot(t, I, label="Infectés (I)", color="red")
    ax1.plot(t, R, label="Retirés (R)", color="green")
    if R0 > 1:
        ax1.axvline(jour_pic, color="grey", linestyle=":", alpha=0.7,
                    label=f"Pic (jour {jour_pic:.0f})")
    ax1.set_ylabel("Nombre d'individus")
    ax1.set_title(f"Modèle SIR — COVID-19 (R0 = {R0:.2f})")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Nombre de reproduction effectif R(t).
    ax2.plot(t, Rt, color="purple", label="R effectif R(t)")
    ax2.axhline(1.0, color="black", linestyle="--", alpha=0.7, label="Seuil R = 1")
    ax2.set_xlabel("Jours")
    ax2.set_ylabel("R(t)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()

    if sauvegarder:
        os.makedirs(DOSSIER_FIGURES, exist_ok=True)
        chemin = os.path.join(DOSSIER_FIGURES, "sir.png")
        fig.savefig(chemin, dpi=120)
        print(f"\nFigure enregistrée : {chemin}")

    plt.show()


def simuler_sir():
    """Point d'entrée interactif : choix du mode puis simulation."""
    print("--- Simulation du Modèle SIR pour la COVID-19 ---")
    print("  1) Saisie manuelle des paramètres")
    print("  2) Preset France 2020 (R0 = 2.5)")
    choix = input("Votre choix [1/2] (par défaut 2) : ").strip()

    if choix == "1":
        params = _saisir_parametres()
        if params is None:
            return
    else:
        params = SIR_FRANCE_2020
        print("Preset France 2020 sélectionné.")

    simuler_et_afficher(params)


if __name__ == "__main__":
    simuler_sir()
