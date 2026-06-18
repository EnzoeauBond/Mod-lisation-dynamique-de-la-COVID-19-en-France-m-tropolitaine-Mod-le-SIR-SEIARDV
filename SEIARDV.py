import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# --- 1. Définition du Modèle SEIARDV ---

def modele_seiardv(y, t, N, beta_I, beta_A, sigma, p, gamma_I, gamma_A, alpha, delta):
    """
    Fonction définissant le système de 7 équations différentielles (EDO) pour le modèle SEIARDV.

    Paramètres:
    y : Liste des valeurs actuelles [S, E, I, A, R, V, D]
    t : Le temps (jours)
    N : Population totale (qui varie dans ce modèle en raison des décès D)
    beta_I : Taux de transmission par les Symptomatiques (I)
    beta_A : Taux de transmission par les Asymptomatiques (A)
    sigma : Taux de sortie de E (Incubation)
    p : Proportion des exposés E qui deviennent symptomatiques (I)
    gamma_I : Taux de guérison pour les Symptomatiques (I -> R)
    gamma_A : Taux de guérison pour les Asymptomatiques (A -> R)
    alpha : Taux de mortalité (I -> D)
    delta : Taux de vaccination (S -> V)
    """
    S, E, I, A, R, V, D = y

    # La population totale N(t) est dynamique (elle diminue avec D)
    N_t = S + E + I + A + R + V + D

    # Force d'Infection (Lambda)
    Lambda = (beta_I * I + beta_A * A) / N_t

    # --- Équations Différentielles (EDO) ---

    # 1. Susceptible (S)
    dSdt = -Lambda * S - delta * S

    # 2. Exposé (E - Incubation)
    dEdt = Lambda * S - sigma * E

    # 3. Infecté Symptomatique (I)
    dIdt = p * sigma * E - (gamma_I + alpha) * I

    # 4. Infecté Asymptomatique (A)
    dAdt = (1 - p) * sigma * E - gamma_A * A

    # 5. Retiré/Rétabli (R)
    dRdt = gamma_I * I + gamma_A * A

    # 6. Vacciné (V)
    dVdt = delta * S

    # 7. Décédé (D)
    dDdt = alpha * I

    return [dSdt, dEdt, dIdt, dAdt, dRdt, dVdt, dDdt]

# --- 2. Simulation avec Paramètres Calibrés (Exemple pour la France) ---

def simuler_seiardv():
    print("--- Simulation du Modèle SEIARDV pour la COVID-19 en France ---")

    # --- Paramètres de Population et Conditions Initiales (Début 2020) ---
    N_initial = 67000000.0  # Population totale (France métropolitaine)
    I0 = 6378.0             # Cas initiaux symptomatiques (estimé au 15/03/2020)
    R0 = 161.0              # Retirés initiaux (décès + guérisons précoces)
    D0 = 0.0                # Décès initiaux (inclus dans R0, mais séparé pour le modèle)
    A0 = I0 * 3             # Estimation : 3 fois plus d'asymptomatiques que de cas détectés I0
    E0 = I0 * 5             # Estimation : 5 fois plus d'exposés que de cas symptomatiques I0
    V0 = 0.0                # Aucun vacciné au début de la crise
    S0 = N_initial - (I0 + R0 + D0 + A0 + E0 + V0) # Susceptibles initiaux

    Jours = 200 # Nombre de jours à simuler
    t = np.linspace(0, Jours, Jours)
    y0 = [S0, E0, I0, A0, R0, V0, D0]

    # --- Paramètres Épidémiques (Calibrés pour R0 ≈ 3.0) ---

    # Taux d'Incubation (sigma) : Incubation moyenne ≈ 5.2 jours
    sigma = 1 / 5.2

    # Taux de Transmission (Beta) : Calibré pour R0 élevé (pré-confinement)
    beta_I = 0.40  # Les symptomatiques transmettent plus
    beta_A = 0.25  # Les asymptomatiques transmettent moins

    # Taux de Guérison (Gamma) : Durée moyenne de l'infection ≈ 10 jours
    gamma_I = 1 / 10.0
    gamma_A = 1 / 10.0

    # Taux de Mortalité (Alpha) : Létalité estimée à 1-2% des symptomatiques
    alpha = 0.005 # Simule 0.5% de mortalité par jour dans I

    # Proportion Symptomatique (p) : Environ 30% des cas deviennent symptomatiques
    p = 0.3

    # Taux de Vaccination (Delta) : Mis à zéro pour simuler la première vague sans vaccin
    # Pour simuler une campagne future, changez cette valeur (ex: 0.0001 = 6700 personnes/jour)
    delta = 1.5e-2

    # Résolution des EDO
    # Note : Le paramètre N est inclus mais est recalculé dynamiquement dans la fonction
    resultats = odeint(modele_seiardv, y0, t, args=(N_initial, beta_I, beta_A, sigma, p, gamma_I, gamma_A, alpha, delta))

    # Extraction des résultats
    S, E, I, A, R, V, D = resultats.T

    # --- 3. Affichage des Résultats ---

    # Calcul du R0 initial (pour l'affichage)
    # R0 = (p * beta_I * (1/gamma_I) + (1-p) * beta_A * (1/gamma_A)) / (1/sigma)
    # (Formule très complexe, nous utilisons une approximation simple ici)
    R_approx = (beta_I / gamma_I) * p + (beta_A / gamma_A) * (1-p)
    print(f"Le Taux de Reproduction Approximatif Initial (R0) est : {R_approx:.2f}")

    plt.figure(figsize=(12, 8))

    # Affichage des populations principales
    plt.plot(t, S / N_initial, label='Susceptibles (S)', color='blue')
    plt.plot(t, I / N_initial, label='Infectés Symptomatiques (I)', color='red')
    plt.plot(t, A / N_initial, label='Asymptomatiques (A)', color='orange')
    plt.plot(t, R / N_initial, label='Rétablis (R)', color='green')
    plt.plot(t, D / N_initial, label='Décédés (D)', color='black', linestyle='--')
    plt.plot(t, V / N_initial, label='Vaccinés (V)', color='purple', linestyle=':')

    plt.xlabel("Jours")
    plt.ylabel("Proportion de la Population")
    plt.title(f"Simulation du Modèle SEIARDV (R0 initial ≈ {R_approx:.2f})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# Exécuter la simulation
if __name__ == "__main__":
    simuler_seiardv()