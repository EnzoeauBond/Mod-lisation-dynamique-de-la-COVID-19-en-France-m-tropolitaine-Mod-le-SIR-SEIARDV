"""
parametres.py — Jeux de paramètres calibrés des modèles.

Regroupe dans des dataclasses immuables les paramètres épidémiques et les
conditions initiales, ainsi qu'un preset « France 2020 » documenté (avec les
justifications/sources des valeurs). Centraliser ces valeurs évite la
duplication entre les différents scripts et facilite la calibration.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Modèle SIR
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParametresSIR:
    """Paramètres et conditions initiales du modèle SIR."""
    N: float          # population totale
    I0: float         # infectés initiaux
    R0_init: float    # retirés initiaux (guéris/immunisés)
    beta: float       # taux de transmission
    gamma: float      # taux de retrait (1 / durée d'infection)
    jours: int        # horizon de simulation (jours)

    @property
    def S0(self) -> float:
        """Susceptibles initiaux S0 = N - I0 - R0_init."""
        return self.N - self.I0 - self.R0_init

    @property
    def y0(self) -> list[float]:
        """Vecteur d'état initial [S0, I0, R0_init]."""
        return [self.S0, self.I0, self.R0_init]


# Preset France métropolitaine, début de la première vague (mars 2020).
# beta = 0.25 et gamma = 0.1 donnent R0 = 2.5, valeur communément retenue pour
# la souche initiale du SARS-CoV-2 (Salje et al., Science 2020).
SIR_FRANCE_2020 = ParametresSIR(
    N=67_000_000.0,
    I0=6_378.0,       # cas symptomatiques estimés au 15/03/2020
    R0_init=161.0,    # guéris + décès précoces
    beta=0.25,
    gamma=0.1,        # durée moyenne d'infection ≈ 10 jours
    jours=200,
)


# ---------------------------------------------------------------------------
# Modèle SEIARDV
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParametresSEIARDV:
    """Paramètres et conditions initiales du modèle SEIARDV."""
    N: float          # population totale
    I0: float         # infectés symptomatiques initiaux
    R0_init: float    # rétablis initiaux
    D0: float         # décès initiaux
    A0: float         # asymptomatiques initiaux
    E0: float         # exposés initiaux
    V0: float         # vaccinés initiaux

    beta_I: float     # transmission par les symptomatiques
    beta_A: float     # transmission par les asymptomatiques
    sigma: float      # taux de sortie d'incubation
    p: float          # proportion d'exposés devenant symptomatiques
    gamma_I: float    # guérison des symptomatiques
    gamma_A: float    # guérison des asymptomatiques
    alpha: float      # mortalité des symptomatiques
    delta: float      # taux de vaccination
    jours: int        # horizon de simulation (jours)

    @property
    def S0(self) -> float:
        """Susceptibles initiaux."""
        return self.N - (self.I0 + self.R0_init + self.D0 + self.A0 + self.E0 + self.V0)

    @property
    def y0(self) -> list[float]:
        """Vecteur d'état initial [S, E, I, A, R, V, D]."""
        return [self.S0, self.E0, self.I0, self.A0, self.R0_init, self.V0, self.D0]

    @property
    def args_edo(self) -> tuple:
        """Tuple de paramètres attendu par models.modele_seiardv."""
        return (self.beta_I, self.beta_A, self.sigma, self.p,
                self.gamma_I, self.gamma_A, self.alpha, self.delta)


# Preset France métropolitaine, première vague (2020).
# Hypothèses : incubation ≈ 5,2 jours (sigma=1/5.2) ; infection ≈ 10 jours ;
# ~30 % de cas symptomatiques (p=0.3) ; 3x plus d'asymptomatiques et 5x plus
# d'exposés que de cas détectés au départ ; létalité journalière des
# symptomatiques alpha=0.005. delta=0 dans le preset « première vague » (pas de
# vaccin) ; on fournit aussi une variante avec vaccination.
SEIARDV_FRANCE_2020 = ParametresSEIARDV(
    N=67_000_000.0,
    I0=6_378.0,
    R0_init=161.0,
    D0=0.0,
    A0=6_378.0 * 3,
    E0=6_378.0 * 5,
    V0=0.0,
    beta_I=0.40,
    beta_A=0.25,
    sigma=1 / 5.2,
    p=0.3,
    gamma_I=1 / 10.0,
    gamma_A=1 / 10.0,
    alpha=0.005,
    delta=0.0,        # première vague : aucune vaccination
    jours=200,
)

# Variante avec campagne de vaccination (delta > 0) pour illustrer son effet.
from dataclasses import replace  # noqa: E402
SEIARDV_FRANCE_2020_VACCIN = replace(SEIARDV_FRANCE_2020, delta=1.5e-2)
