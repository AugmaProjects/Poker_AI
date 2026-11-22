from pathlib import Path
import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import log_loss

DATASET_PATH = Path("data/processed/dataset_flop.parquet")
MODEL_PATH = Path("models/cbet_response_lgbm.joblib")

FEATURES = [
    "is_ip",
    "spr",
    "bet_size_pct",
    "aggressor_vpip_smooth",
    "aggressor_pfr_smooth",
    "villain_vpip_smooth",
    "villain_pfr_smooth",
    "villain_fold_cbet_smooth",
    "villain_raise_cbet_smooth",
    "flop_high",
    "flop_mid",
    "flop_low",
    "flop_is_paired",
    "flop_is_monotone",
    "flop_is_two_tone",
    "flop_is_connected",
    "flop_has_ace",
]


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable : {DATASET_PATH}")

    df = pd.read_parquet(DATASET_PATH)
    df = df.sort_values(["timestamp", "hand_id"]).reset_index(drop=True)

    X = df[FEATURES]
    y = df["target_action"]

    # Découpage chronologique strict : 80% passé / 20% futur
    split_idx = int(len(df) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Train : {len(X_train)} spots | Test (futur) : {len(X_test)} spots")

    # 1. Baseline naïve
    dummy = DummyClassifier(strategy="prior")
    dummy.fit(X_train, y_train)
    baseline_loss = log_loss(y_test, dummy.predict_proba(X_test), labels=[0, 1, 2])

    # 2. LightGBM calibré pour ~10k+ observations
    model = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=180,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=12,
        min_child_samples=60,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    test_probs = model.predict_proba(X_test)
    model_loss = log_loss(y_test, test_probs, labels=[0, 1, 2])

    print("\n--- ÉVALUATION SUR LE JEU DE TEST (MULTI-CLASS LOG-LOSS) ---")
    print(f"Baseline naïve : {baseline_loss:.4f}")
    print(f"LightGBM       : {model_loss:.4f} (plus bas = meilleur)")

    importances = (
        pd.Series(model.feature_importances_, index=FEATURES)
        .sort_values(ascending=False)
        .head(8)
    )
    print("\n--- TOP 8 VARIABLES LES PLUS UTILISÉES PAR L'IA ---")
    for feat, score in importances.items():
        print(f"  {feat:<26} : {score} splits")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)
    print(f"\nModèle sauvegardé -> {MODEL_PATH}")

    # 3. Simulation sur un flop K♠ 7♦ 2♣ avec ton profil (VPIP 17.1%, PFR 14.9%)
    base_spot = {
        "is_ip": 1, "spr": 14.0,
        "aggressor_vpip_smooth": 0.171, "aggressor_pfr_smooth": 0.149,
        "flop_high": 13, "flop_mid": 7, "flop_low": 2,
        "flop_is_paired": 0, "flop_is_monotone": 0, "flop_is_two_tone": 0,
        "flop_is_connected": 0, "flop_has_ace": 0,
    }

    scenarios = pd.DataFrame([
        {
            **base_spot,
            "label": "Vs Pool moyen | Bet 33% pot",
            "bet_size_pct": 0.33,
            "villain_vpip_smooth": 0.274, "villain_pfr_smooth": 0.166,
            "villain_fold_cbet_smooth": 0.454, "villain_raise_cbet_smooth": 0.080,
        },
        {
            **base_spot,
            "label": "Vs Pool moyen | Bet 75% pot",
            "bet_size_pct": 0.75,
            "villain_vpip_smooth": 0.274, "villain_pfr_smooth": 0.166,
            "villain_fold_cbet_smooth": 0.454, "villain_raise_cbet_smooth": 0.080,
        },
        {
            **base_spot,
            "label": "Vs Calling Station (VPIP 48%, Fold 28%) | Bet 33%",
            "bet_size_pct": 0.33,
            "villain_vpip_smooth": 0.48, "villain_pfr_smooth": 0.12,
            "villain_fold_cbet_smooth": 0.28, "villain_raise_cbet_smooth": 0.05,
        },
        {
            **base_spot,
            "label": "Vs Nit Folder (VPIP 16%, Fold 62%) | Bet 33%",
            "bet_size_pct": 0.33,
            "villain_vpip_smooth": 0.16, "villain_pfr_smooth": 0.13,
            "villain_fold_cbet_smooth": 0.62, "villain_raise_cbet_smooth": 0.07,
        },
    ])

    sim_probs = model.predict_proba(scenarios[FEATURES])
    print("\n--- TEST DE SENSIBILITÉ (FLOP K♠ 7♦ 2♣ EN POSITION) ---")
    for i, row in scenarios.iterrows():
        p_fold, p_call, p_raise = sim_probs[i]
        print(
            f"{row['label']:<52} -> "
            f"Fold: {p_fold*100:4.1f}% | Call: {p_call*100:4.1f}% | Raise: {p_raise*100:4.1f}%"
        )


if __name__ == "__main__":
    main()