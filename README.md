#  Poker AI — Exploitative Decision & EV Engine (Flop & Turn)

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![LightGBM](https://img.shields.io/badge/ML-LightGBM-brightgreen)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi)
![Status](https://img.shields.io/badge/Status-Active-success)

**Poker AI** est un moteur d'aide à la décision exploitatif conçu pour le Texas Hold'em No-Limit (Cash Game). Contrairement aux solveurs GTO (*Game Theory Optimal*) classiques qui supposent un adversaire théoriquement parfait, ce système combine **l'apprentissage automatique sur données réelles**, **l'inférence bayésienne de ranges** et **la simulation Monte Carlo** pour maximiser l'espérance de gain ($EV$) face aux tendances réelles d'un pool de joueurs.

---

##  Problématique & Approche Technique

Dans un environnement à **information imparfaite et multi-agents adversariaux** comme le poker :
* Les adversaires réels dévient fortement de l'équilibre de Nash (ex. : *over-fold* sur certaines textures de board, passivité hors de position, profils *Calling Station* vs *Nit*).
* Un solveur en arbre complet (*expectimax* sur toutes les streets) souffre d'explosion combinatoire et nécessite plusieurs minutes de calcul par main.

**Architecture hybride retenue :**
1. **Prédiction comportementale par Machine Learning (`LightGBM`) :** Modèles multiclasses entraînés sur un historique réel de **86 625 mains** (999 fichiers d'historiques Winamax) pour prédire les probabilités de réaction de l'adversaire ($P(\text{Fold})$, $P(\text{Call})$, $P(\text{Raise})$) selon son profil HUD, la texture du board et le *sizing* utilisé.
2. **Reconstruction combinatoire & Filtrage de Range :** Génération des 1 326 combinaisons de départ (heuristique de Chen), suppression des bloqueurs et filtrage dynamique du top $(1 - P(\text{Fold}))$ de la range adverse selon la force faite et les tirages sur le board.
3. **Simulation Monte Carlo (`treys`) & Calcul d'EV :** Évaluation rapide par tirages aléatoires sur les streets restantes, pondérée par des coefficients de réalisation d'équité ($EQR$) dépendant de la position (`IP` vs `OOP`) et du prélèvement (*rake*).
4. **Audit automatisé des fuites (*Leak Finder*) :** Analyse rétrospective des décisions historiques du joueur pour identifier et quantifier les pertes d'EV en *big blinds* (`bb`).

---

##  Résultats Empiriques & Insights Extraits

L'entraînement et l'évaluation sur le dataset propriétaire (pots relancés simples en Heads-Up) ont permis d'extraire des métriques comportementales précises :

* **Flop (C-Bet SRP — ~8 600 spots) :** Taux de fold moyen du pool de **45,4 %**.
* **Turn (2e Barrel SRP — 2 149 spots, Log-Loss validation : `0.4555`) :**
  * Taux de fold global en baisse à **39,0 %** (range adverse filtrée après le flop), avec **52,8 %** de *call* et **8,2 %** de *raise*.
  * **Impact de la dynamique de Turn :**
    * *Overcard au flop* : **43,9 %** de fold (meilleure texture de bluff).
    * *3e carte de la couleur (Flush)* : **39,8 %** de fold / **11,8 %** de raise.
    * *Doublette au board* : **38,2 %** de fold / **6,4 %** de raise.
    * *4 cartes connectées (Quinte)* : **29,6 %** de fold / **12,7 %** de raise.

---

## Architecture du Projet

```text
Poker_AI/
├── data/
│   ├── hero_cbet_spots.parquet        # Spots extraits du joueur (Flop SRP + détection Check-Raise)
│   └── dataset_turn_barrel.parquet    # Dataset d'entraînement 2e barrel (Turn)
├── models/
│   ├── cbet_response_lgbm.joblib      # Modèle LightGBM multiclasse (Flop)
│   └── turn_barrel_lgbm.joblib        # Modèle LightGBM multiclasse (Turn)
├── src/
│   ├── features.py                    # Feature engineering sur les textures de board
│   ├── extract_hero_spots.py          # Parseur regex des historiques bruts & lissage bayésien HUD
│   ├── ev_engine.py                   # Moteur combinatoire, Monte Carlo & calcul d'EV (Flop)
│   ├── turn_pipeline.py               # Extraction & entraînement du modèle LightGBM (Turn)
│   ├── turn_ev_engine.py              # Moteur combinatoire & calcul d'EV (Turn)
│   ├── leak_finder.py                 # Audit statistique des pertes d'EV sur l'historique
│   └── app.py                         # API REST FastAPI (/api/evaluate & /api/evaluate_turn)
├── static/
│   └── index.html                     # Interface web interactive (Flop & Turn, presets HUD)
└── requirements.txt
