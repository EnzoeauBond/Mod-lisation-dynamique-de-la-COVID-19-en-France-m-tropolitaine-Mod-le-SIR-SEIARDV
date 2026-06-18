import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# --- 1. Définition du Modèle SIR ---

def modele_sir(y, t, beta, gamma):
    """
    Fonction définissant le système d'équations différentielles ordinaires (EDO) du modèle SIR.

    Paramètres:
    y : Liste des valeurs actuelles [S, I, R]
    t : Le temps (jours)
    beta : Taux de transmission
    gamma : Taux de guérison/retrait
    """
    S, I, R = y
    N = S + I + R  # Population totale (constante dans le modèle SIR simple)

    # Équations Différentielles :
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I

    return [dSdt, dIdt, dRdt]

# --- 2. Saisie des Paramètres et Simulation ---

def simuler_sir():
    """
    Fonction principale pour saisir les paramètres, résoudre les EDO et afficher le graphique.
    """
    print("--- Simulation du Modèle SIR pour la COVID-19 ---")

    # Saisie des paramètres par l'utilisateur
    try:
        # Paramètres de population et conditions initiales
        N = float(input("Population totale (N) : "))
        I0 = float(input("Nombre initial d'infectés (I0) : "))
        R0 = float(input("Nombre initial de retirés/immunisés (R0) : "))
        Jours = int(input("Nombre de jours à simuler : "))

        # Calculer S0
        S0 = N - I0 - R0
        if S0 < 0:
            print("Erreur: La somme I0 + R0 est supérieure à la population totale N.")
            return

        # Paramètres épidémiques
        beta = float(input("Taux de transmission (beta, ex: 0.25) : "))
        gamma = float(input("Taux de guérison/retrait (gamma, ex: 0.1) : "))

    except ValueError:
        print("Erreur de saisie. Veuillez entrer des nombres valides.")
        return

    # Calcul du R0 théorique
    R_zero = beta / gamma
    print(f"\nLe Taux de Reproduction de Base (R0) est : {R_zero:.2f}")

    # Vérification du R0
    if R_zero > 1:
        print("-> L'épidémie va se propager (R0 > 1).")
    else:
        print("-> L'épidémie va s'éteindre (R0 < 1).")

    # Vecteur temps
    t = np.linspace(0, Jours, Jours)
    # Conditions initiales [S0, I0, R0]
    y0 = [S0, I0, R0]

    # Résolution des EDO en utilisant odeint (Runge-Kutta implicite)
    resultats = odeint(modele_sir, y0, t, args=(beta, gamma))

    # Extraction des résultats
    S = resultats[:, 0]
    I = resultats[:, 1]
    R = resultats[:, 2]

    # --- 3. Affichage des Résultats ---

    plt.figure(figsize=(10, 6))
    plt.plot(t, S, label='Susceptibles (S)', color='blue')
    plt.plot(t, I, label='Infectés (I)', color='red')
    plt.plot(t, R, label='Retirés (R)', color='green')

    plt.xlabel("Jours")
    plt.ylabel("Nombre d'individus")
    plt.title(f"Simulation du Modèle SIR (R0 = {R_zero:.2f})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# Exécuter la simulation
if __name__ == "__main__":
    simuler_sir()