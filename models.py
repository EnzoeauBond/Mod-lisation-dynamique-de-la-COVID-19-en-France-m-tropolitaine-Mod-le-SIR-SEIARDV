"""
models.py — Cœur scientifique partagé des modèles épidémiologiques.

Ce module regroupe :
  * les systèmes d'équations différentielles (EDO) des modèles SIR et SEIARDV ;
  * les outils mathématiques associés (nombre de reproduction de base R0,
    nombre de reproduction effectif R(t), seuil d'immunité collective,
    caractéristiques du pic épidémique, taille finale de l'épidémie) ;
  * un utilitaire d'intégration numérique commun aux deux modèles.

Les fonctions sont volontairement « pures » (sans affichage ni saisie) afin de
pouvoir être réutilisées par les scripts SIR.py, SEIARDV.py, comparaison.py et
calibration.py, et testées indépendamment.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# 1. Systèmes d'équations différentielles
# ---------------------------------------------------------------------------

def modele_sir(y, t, beta, gamma):
    """Système d'EDO du modèle SIR classique.

    Compartiments :
        S — Susceptibles, I — Infectés, R — Retirés (guéris ou décédés).

    Paramètres
    ----------
    y : séquence [S, I, R]
        État courant des compartiments.
    t : float
        Temps (en jours). Non utilisé ici (système autonome) mais requis par odeint.
    beta : float
        Taux de transmission (par contact effectif et par jour).
    gamma : float
        Taux de retrait (1 / durée moyenne d'infection).

    Retour
    ------
    list[float] : dérivées [dS/dt, dI/dt, dR/dt].
    """
    S, I, R = y
    N = S + I + R  # population totale (constante dans le SIR simple)

    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


def modele_seiardv(y, t, beta_I, beta_A, sigma, p, gamma_I, gamma_A, alpha, delta):
    """Système d'EDO du modèle SEIARDV (7 compartiments).

    Compartiments :
        S — Susceptibles
        E — Exposés (infectés en incubation, non encore contagieux)
        I — Infectés symptomatiques (contagieux)
        A — Infectés asymptomatiques (contagieux)
        R — Rétablis / immunisés
        V — Vaccinés
        D — Décédés

    La population N(t) = S+E+I+A+R+V+D est dynamique : elle reste constante car
    les décès D restent comptabilisés dans N (population « fermée »), mais la
    force d'infection est normalisée par N(t) à chaque instant.

    Paramètres
    ----------
    beta_I, beta_A : float
        Taux de transmission des symptomatiques / asymptomatiques.
    sigma : float
        Taux de sortie d'incubation (1 / durée moyenne d'incubation).
    p : float
        Proportion d'exposés devenant symptomatiques (0 <= p <= 1).
    gamma_I, gamma_A : float
        Taux de guérison des symptomatiques / asymptomatiques.
    alpha : float
        Taux de mortalité des symptomatiques (I -> D).
    delta : float
        Taux de vaccination (S -> V).

    Retour
    ------
    list[float] : dérivées des 7 compartiments.
    """
    S, E, I, A, R, V, D = y
    N_t = S + E + I + A + R + V + D  # population totale courante

    # Force d'infection (Lambda) : seuls I et A sont contagieux.
    Lambda = (beta_I * I + beta_A * A) / N_t

    dSdt = -Lambda * S - delta * S
    dEdt = Lambda * S - sigma * E
    dIdt = p * sigma * E - (gamma_I + alpha) * I
    dAdt = (1 - p) * sigma * E - gamma_A * A
    dRdt = gamma_I * I + gamma_A * A
    dVdt = delta * S
    dDdt = alpha * I
    return [dSdt, dEdt, dIdt, dAdt, dRdt, dVdt, dDdt]


# ---------------------------------------------------------------------------
# 2. Intégration numérique commune
# ---------------------------------------------------------------------------

def integrer(modele, y0, jours, args):
    """Intègre un système d'EDO sur ``jours`` jours avec un pas de 1 jour.

    Utilise ``np.arange(0, jours + 1)`` pour obtenir exactement un point par jour
    (corrige le pas erroné de ``np.linspace(0, jours, jours)``).

    Retour
    ------
    (t, resultats) : tuple
        ``t`` est le vecteur temps (jours+1 points), ``resultats`` la matrice
        (jours+1, n_compartiments) renvoyée par ``odeint``.
    """
    t = np.arange(0, jours + 1, dtype=float)
    resultats = odeint(modele, y0, t, args=args)
    return t, resultats


# ---------------------------------------------------------------------------
# 3. Nombres de reproduction
# ---------------------------------------------------------------------------

def r0_sir(beta, gamma):
    """Nombre de reproduction de base du modèle SIR : R0 = beta / gamma."""
    return beta / gamma


def r0_seiardv(beta_I, beta_A, sigma, p, gamma_I, gamma_A, alpha, delta):
    """Nombre de reproduction de base du SEIARDV par la matrice de prochaine génération.

    Méthode de van den Driessche & Watmough (2002). Sur les compartiments
    infectés (E, I, A), on linéarise au voisinage de l'équilibre sans maladie
    (S0 ≈ N) et on écrit le système comme ``x' = (F - V) x`` où :

      * F décrit les *nouvelles* infections,
      * V décrit les *transferts* entre compartiments (et sorties).

    R0 = rayon spectral de la matrice de prochaine génération K = F · V⁻¹.

    Le calcul analytique donne :

        R0 = (sigma / (sigma + delta_E_loss)) · [ p · beta_I / (gamma_I + alpha)
                                                 + (1 - p) · beta_A / gamma_A ]

    Comme un susceptible peut être vacciné (taux delta) avant d'être infecté à
    l'équilibre sans maladie S0 = N, on a ici delta_E_loss = 0 dans la dynamique
    de E (delta n'agit que sur S) : la fraction d'exposés qui survit à
    l'incubation est sigma/sigma = 1. On expose néanmoins le facteur de
    réduction lié à la vaccination via ``facteur_vaccination`` ci-dessous pour
    le R0 *effectif* en présence de vaccination.

    Cette implémentation calcule directement le rayon spectral de K pour rester
    exacte quelle que soit la paramétrisation.
    """
    # Matrice F (nouvelles infections) — lignes/colonnes ordre [E, I, A].
    # Les nouvelles infections n'arrivent que dans E, générées par I et A.
    F = np.array([
        [0.0, beta_I, beta_A],
        [0.0, 0.0,    0.0],
        [0.0, 0.0,    0.0],
    ])

    # Matrice V (transferts) — sorties et flux internes.
    #   E sort au taux sigma.
    #   I est alimenté par p*sigma*E et sort au taux (gamma_I + alpha).
    #   A est alimenté par (1-p)*sigma*E et sort au taux gamma_A.
    V = np.array([
        [sigma,           0.0,             0.0],
        [-p * sigma,      gamma_I + alpha, 0.0],
        [-(1 - p) * sigma, 0.0,            gamma_A],
    ])

    K = F @ np.linalg.inv(V)  # matrice de prochaine génération
    valeurs_propres = np.linalg.eigvals(K)
    R0 = max(abs(valeurs_propres))
    return float(R0)


def r0_seiardv_formule(beta_I, beta_A, sigma, p, gamma_I, gamma_A, alpha):
    """Formule analytique fermée du R0 du SEIARDV (vérification de r0_seiardv).

    R0 = p · beta_I / (gamma_I + alpha) + (1 - p) · beta_A / gamma_A
    """
    return p * beta_I / (gamma_I + alpha) + (1 - p) * beta_A / gamma_A


def reproduction_effective(S, N, R0):
    """Nombre de reproduction effectif R(t) = R0 · S(t) / N (vectorisé)."""
    return R0 * np.asarray(S) / N


# ---------------------------------------------------------------------------
# 4. Seuils et caractéristiques épidémiques
# ---------------------------------------------------------------------------

def seuil_immunite_collective(R0):
    """Seuil d'immunité collective : fraction immunisée nécessaire = 1 - 1/R0.

    Renvoie 0.0 si R0 <= 1 (pas d'épidémie, aucun seuil requis).
    """
    if R0 <= 1:
        return 0.0
    return 1.0 - 1.0 / R0


def pic_epidemique(t, I):
    """Date et amplitude du pic d'infectés.

    Retour
    ------
    (jour_pic, amplitude_pic) : tuple(float, float)
    """
    I = np.asarray(I)
    idx = int(np.argmax(I))
    return float(t[idx]), float(I[idx])


def taille_finale_sir(R0):
    """Taille finale de l'épidémie (modèle SIR), fraction totale infectée R∞.

    Résout l'équation transcendante de la « final size relation » :

        R∞ = 1 - exp(-R0 · R∞)

    Renvoie 0.0 si R0 <= 1 (la seule solution est R∞ = 0).
    """
    if R0 <= 1:
        return 0.0
    # Recherche de la racine non nulle dans (0, 1).
    f = lambda r: 1.0 - np.exp(-R0 * r) - r
    # f(0)=0 ; on cherche la racine sur un intervalle évitant 0.
    return float(brentq(f, 1e-9, 1.0 - 1e-12))
