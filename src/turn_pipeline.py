import re
from pathlib import Path
from collections import defaultdict
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss

# --- CONFIGURATION ---
# Remets ici ton chemin exact qui a trouvé tes 999 fichiers !
HISTORY_DIR = Path(r"C:\Users\nknry\AppData\Roaming\winamax\documents\accounts\SHINIGAMI_GM\history")
DATA_TURN_PATH = Path("data/dataset_turn_barrel.parquet")
MODEL_TURN_PATH = Path("models/turn_barrel_lgbm.joblib")

RANK_MAP = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}


def parse_amount_bb(text: str, bb_size: float) -> float:
    m = re.search(r"([\d\.]+)", text)
    if not m:
        return 0.0
    val = float(m.group(1))
    return val if "bb" in text.lower() else val / bb_size


def extract_turn_features(flop_cards: list[str], turn_card: str) -> dict:
    f_ranks = [RANK_MAP[c[0]] for c in flop_cards]
    f_suits = [c[1] for c in flop_cards]
    t_rank = RANK_MAP[turn_card[0]]
    t_suit = turn_card[1]

    all_ranks = sorted(f_ranks + [t_rank])
    all_suits = f_suits + [t_suit]

    is_overcard = int(t_rank > max(f_ranks))
    is_paired_turn = int(t_rank in f_ranks)
    is_flush_completing = int(all_suits.count(t_suit) >= 3 and f_suits.count(t_suit) == 2)
    is_monotone_4 = int(all_suits.count(t_suit) == 4)

    # Connectivité quinte sur 4 cartes
    unique_r = sorted(set(all_ranks))
    max_span = max(unique_r) - min(unique_r)
    is_4_straight_board = int(len(unique_r) == 4 and max_span <= 4)

    return {
        "turn_rank": t_rank,
        "flop_max_rank": max(f_ranks),
        "turn_is_overcard": is_overcard,
        "turn_pairs_board": is_paired_turn,
        "turn_completes_flush": is_flush_completing,
        "turn_4_flush": is_monotone_4,
        "turn_4_straight": is_4_straight_board,
    }


def build_and_train_turn_model():
    if not HISTORY_DIR.exists():
        print(f"ERREUR : Le dossier {HISTORY_DIR} n'existe pas.")
        return

    txt_files = [f for f in HISTORY_DIR.rglob("*.txt") if "summary" not in f.name.lower()]
    print(f"Lecture de {len(txt_files)} fichiers pour extraction des spots de 2e barrel (Turn)...")

    stats = defaultdict(lambda: {"hands": 0, "vpip": 0, "pfr": 0, "turn_opp": 0, "fold_turn": 0, "raise_turn": 0})
    raw_spots = []

    hand_split_re = re.compile(r"(?=Winamax Poker - )")
    blind_re = re.compile(r"\(([\d\.]+)€?/([\d\.]+)€?\)")

    for file_path in txt_files:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for hand in hand_split_re.split(content):
            if "CashGame" not in hand or "*** FLOP ***" not in hand or "*** TURN ***" not in hand:
                continue

            m_bb = blind_re.search(hand.splitlines()[0])
            if not m_bb:
                continue
            bb_size = float(m_bb.group(2))
            if bb_size <= 0:
                continue

            preflop_txt, rest_post = hand.split("*** FLOP ***", 1)
            flop_txt, rest_turn = rest_post.split("*** TURN ***", 1)
            turn_txt = rest_turn.split("*** RIVER ***")[0].split("*** SUMMARY ***")[0]

            seats = re.findall(r"Seat \d+: ([^(]+) \([\d\.]+€?\)", preflop_txt)
            players = [s.strip() for s in seats]
            if len(players) < 2 or "*** PRE-FLOP ***" not in preflop_txt:
                continue

            pf_actions = preflop_txt.split("*** PRE-FLOP ***")[1]
            raisers, vpip_p, pfr_p, folded_pf = [], set(), set(), set()
            pot_bb = 1.5

            for line in pf_actions.splitlines():
                line = line.strip()
                for p in players:
                    if line.startswith(p + " "):
                        act = line[len(p) + 1 :]
                        if act.startswith("folds"):
                            folded_pf.add(p)
                        elif act.startswith("calls"):
                            vpip_p.add(p)
                            pot_bb += parse_amount_bb(act, bb_size)
                        elif act.startswith("raises"):
                            vpip_p.add(p)
                            pfr_p.add(p)
                            raisers.append(p)
                            m_to = re.search(r"to ([\d\.]+)", act)
                            if m_to:
                                pot_bb += float(m_to.group(1)) / bb_size

            for p in players:
                stats[p]["hands"] += 1
                if p in vpip_p:
                    stats[p]["vpip"] += 1
                if p in pfr_p:
                    stats[p]["pfr"] += 1

            active = [p for p in players if p not in folded_pf]
            if len(raisers) != 1 or len(active) != 2:
                continue

            pfr_player = raisers[0]
            caller_player = [p for p in active if p != pfr_player][0]

            # Cartes flop & turn
            m_flop = re.search(r"\[([2-9TJQKA][shdc]) ([2-9TJQKA][shdc]) ([2-9TJQKA][shdc])\]", flop_txt.splitlines()[0])
            m_turn = re.search(r"\[([2-9TJQKA][shdc])\]\s*$", turn_txt.splitlines()[0])
            if not m_flop or not m_turn:
                continue
            flop_cards = [m_flop.group(1), m_flop.group(2), m_flop.group(3)]
            turn_card = m_turn.group(1)

            # Vérifier que le flop a été C-bet par PFR et payé par Caller (sans raise)
            flop_lines = [l.strip() for l in flop_txt.splitlines()[1:] if l.strip()]
            first_actor = None
            cbet_bb = 0.0
            cbet_called = False
            for line in flop_lines:
                actor = next((p for p in active if line.startswith(p + " ")), None)
                if not actor:
                    continue
                if first_actor is None:
                    first_actor = actor
                act = line[len(actor) + 1 :]
                if actor == pfr_player and cbet_bb == 0:
                    if act.startswith("bets"):
                        cbet_bb = parse_amount_bb(act, bb_size)
                    else:
                        break
                elif actor == caller_player and cbet_bb > 0:
                    if act.startswith("calls"):
                        cbet_called = True
                    break

            if not cbet_called:
                continue
            pot_bb += 2.0 * cbet_bb

            # Vérifier le 2e barrel à la Turn par PFR et la réponse de Caller
            turn_lines = [l.strip() for l in turn_txt.splitlines()[1:] if l.strip()]
            barrel_bb = 0.0
            turn_resp = None
            for line in turn_lines:
                actor = next((p for p in active if line.startswith(p + " ")), None)
                if not actor:
                    continue
                act = line[len(actor) + 1 :]
                if actor == pfr_player and barrel_bb == 0:
                    if act.startswith("bets"):
                        barrel_bb = parse_amount_bb(act, bb_size)
                    else:
                        break
                elif actor == caller_player and barrel_bb > 0:
                    if act.startswith("folds"):
                        turn_resp = "fold"
                    elif act.startswith("calls"):
                        turn_resp = "call"
                    elif act.startswith("raises"):
                        turn_resp = "raise"
                    break

            if barrel_bb <= 0 or not turn_resp:
                continue

            stats[caller_player]["turn_opp"] += 1
            if turn_resp == "fold":
                stats[caller_player]["fold_turn"] += 1
            elif turn_resp == "raise":
                stats[caller_player]["raise_turn"] += 1

            is_ip = 1 if first_actor == caller_player else 0
            raw_spots.append({
                "aggressor": pfr_player,
                "villain": caller_player,
                "is_ip": is_ip,
                "pot_bb": pot_bb,
                "bet_size_pct": min(barrel_bb / max(pot_bb, 0.1), 2.0),
                "target": {"fold": 0, "call": 1, "raise": 2}[turn_resp],
                **extract_turn_features(flop_cards, turn_card),
            })

    # Construction du DataFrame avec stats lissées
    rows = []
    for s in raw_spots:
        v = stats[s["villain"]]
        vn, vt = v["hands"], v["turn_opp"]
        rows.append({
            "is_ip": s["is_ip"],
            "pot_bb": s["pot_bb"],
            "bet_size_pct": s["bet_size_pct"],
            "villain_vpip_smooth": (v["vpip"] + 20 * 0.274) / (vn + 20),
            "villain_pfr_smooth": (v["pfr"] + 20 * 0.166) / (vn + 20),
            "villain_fold_turn_smooth": (v["fold_turn"] + 8 * 0.50) / (vt + 8),
            "villain_raise_turn_smooth": (v["raise_turn"] + 8 * 0.08) / (vt + 8),
            "turn_rank": s["turn_rank"],
            "flop_max_rank": s["flop_max_rank"],
            "turn_is_overcard": s["turn_is_overcard"],
            "turn_pairs_board": s["turn_pairs_board"],
            "turn_completes_flush": s["turn_completes_flush"],
            "turn_4_flush": s["turn_4_flush"],
            "turn_4_straight": s["turn_4_straight"],
            "target": s["target"],
        })

    df = pd.DataFrame(rows)
    DATA_TURN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_TURN_PATH, index=False)

    print(f"\n=== DATASET TURN EXTRAIT : {len(df)} spots de 2e barrel ===")
    dist = df["target"].value_counts(normalize=True).sort_index() * 100
    print(f"Répartition globale : Fold = {dist.get(0, 0):.1f}% | Call = {dist.get(1, 0):.1f}% | Raise = {dist.get(2, 0):.1f}%")

    print("\n=== TAUX DE FOLD DU FIELD WINAMAX SELON LA CARTE DE TURN ===")
    for col, label in [
        ("turn_is_overcard", "Overcard au flop (ex: As/Roi qui tombe)"),
        ("turn_completes_flush", "3e carte de la couleur (Flush rentre)"),
        ("turn_pairs_board", "Doublette au board"),
        ("turn_4_straight", "4 cartes connectées (Quinte possible)"),
    ]:
        sub = df[df[col] == 1]
        if len(sub) > 0:
            fold_rate = (sub["target"] == 0).mean() * 100
            raise_rate = (sub["target"] == 2).mean() * 100
            print(f"  - {label:42s} ({len(sub):4d} spots) -> Fold: {fold_rate:5.1f}% | Raise: {raise_rate:4.1f}%")

    # Entraînement LightGBM Turn
    feature_cols = [c for c in df.columns if c != "target"]
    split = int(len(df) * 0.8)
    X_train, y_train = df[feature_cols].iloc[:split], df["target"].iloc[:split]
    X_val, y_val = df[feature_cols].iloc[split:], df["target"].iloc[split:]

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=200,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=25,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    val_preds = model.predict_proba(X_val)
    loss = log_loss(y_val, val_preds)
    MODEL_TURN_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": feature_cols}, MODEL_TURN_PATH)
    print(f"\nModèle Turn sauvegardé dans {MODEL_TURN_PATH} (Log-Loss validation : {loss:.4f}).")


if __name__ == "__main__":
    build_and_train_turn_model()