import sys
from pathlib import Path
import joblib
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from features import evaluate_board_features, parse_card

MODEL_PATH = Path("models/cbet_response_lgbm.joblib")

BOARD_TEXTURES = [
    ("1. Sec Haut (As)",       ["As", "8d", "3c"]),
    ("2. Sec Haut (Roi)",      ["Ks", "7d", "2c"]),
    ("3. Sec Bas",             ["7s", "4d", "2c"]),
    ("4. Pairé Haut",          ["Js", "Jd", "4c"]),
    ("5. Two-Tone (Draw)",     ["Ks", "9s", "4d"]),
    ("6. Connecté (Très Wet)", ["9s", "8h", "6s"]),
    ("7. Monocolore",          ["Qs", "8s", "3s"]),
]

SIZINGS = [0.33, 0.50, 0.75, 1.00]

ALPHA_BLUFF = {
    0.33: 24.8,
    0.50: 33.3,
    0.75: 42.9,
    1.00: 50.0,
}

PROFILES = {
    "Pool Moyen":      {"vpip": 0.274, "pfr": 0.166, "fold": 0.454, "raise": 0.080},
    "Calling Station": {"vpip": 0.480, "pfr": 0.120, "fold": 0.280, "raise": 0.050},
    "Nit / Serré":     {"vpip": 0.160, "pfr": 0.130, "fold": 0.620, "raise": 0.070},
}


def predict_spot(model, features, board_feats, is_ip, bet_size, profile):
    row = pd.DataFrame([{
        "is_ip": is_ip,
        "spr": 14.0,
        "bet_size_pct": bet_size,
        "aggressor_vpip_smooth": 0.171,
        "aggressor_pfr_smooth": 0.149,
        "villain_vpip_smooth": profile["vpip"],
        "villain_pfr_smooth": profile["pfr"],
        "villain_fold_cbet_smooth": profile["fold"],
        "villain_raise_cbet_smooth": profile["raise"],
        **board_feats,
    }])[features]
    return model.predict_proba(row)[0]


def run_simulation():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}. Lance train.py d'abord.")

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    features = bundle["features"]
    pool = PROFILES["Pool Moyen"]

    print("=========================================================================================================")
    print("        COMPARAISON C-BET IP (TOI EN POSITION) VS C-BET OOP (TOI HORS DE POSITION) - POOL MOYEN        ")
    print("=========================================================================================================")

    for label, cards in BOARD_TEXTURES:
        board_feats = evaluate_board_features([parse_card(c) for c in cards])
        print(f"\n--- {label} [{' '.join(cards)}] ---")
        for s in SIZINGS:
            f_ip, c_ip, r_ip = predict_spot(model, features, board_feats, 1, s, pool)
            f_oop, c_oop, r_oop = predict_spot(model, features, board_feats, 0, s, pool)
            diff_fold = (f_oop - f_ip) * 100

            print(
                f"  Bet {int(s*100):3d}% (seuil {ALPHA_BLUFF[s]:4.1f}%) | "
                f"IP -> Fold:{f_ip*100:4.1f}% Call:{c_ip*100:4.1f}% Raise:{r_ip*100:4.1f}%  ||  "
                f"OOP -> Fold:{f_oop*100:4.1f}% Call:{c_oop*100:4.1f}% Raise:{r_oop*100:4.1f}%  "
                f"(Écart Fold: {diff_fold:+5.1f}%)"
            )

    print("\n=========================================================================================================")
    print("                 COMPARAISON DES PROFILS EN IP VS OOP (BET 50% POT SUR SEC VS WET)                       ")
    print("=========================================================================================================")

    for label, cards in [("Board Sec [Ks 7d 2c]", ["Ks", "7d", "2c"]), ("Board Wet [9s 8h 6s]", ["9s", "8h", "6s"])]:
        board_feats = evaluate_board_features([parse_card(c) for c in cards])
        print(f"\n--- {label} (Bet 50% pot) ---")
        for prof_name, prof in PROFILES.items():
            f_ip, c_ip, r_ip = predict_spot(model, features, board_feats, 1, 0.50, prof)
            f_oop, c_oop, r_oop = predict_spot(model, features, board_feats, 0, 0.50, prof)
            print(
                f"  {prof_name:<16} | "
                f"IP -> Fold:{f_ip*100:4.1f}% Call:{c_ip*100:4.1f}% Raise:{r_ip*100:4.1f}%  ||  "
                f"OOP -> Fold:{f_oop*100:4.1f}% Call:{c_oop*100:4.1f}% Raise:{r_oop*100:4.1f}%"
            )


if __name__ == "__main__":
    run_simulation()