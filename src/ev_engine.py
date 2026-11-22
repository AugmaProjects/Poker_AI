from itertools import combinations
from pathlib import Path
import random
import joblib
import numpy as np
import pandas as pd
from treys import Card, Evaluator

from features import evaluate_board_features, parse_card

MODEL_PATH = Path("models/cbet_response_lgbm.joblib")
RANKS = "23456789TJQKA"
SUITS = "shdc"
ALL_CARDS_STR = [r + s for r in RANKS for s in SUITS]
RANK_VAL = {r: i + 2 for i, r in enumerate(RANKS)}


def preflop_hand_score(c1: str, c2: str) -> float:
    """Score heuristique (formule de Chen) pour ordonner les 1326 combos preflop."""
    r1, s1 = RANK_VAL[c1[0]], c1[1]
    r2, s2 = RANK_VAL[c2[0]], c2[1]
    high, low = max(r1, r2), min(r1, r2)
    suited = s1 == s2
    gap = high - low

    base_map = {14: 10, 13: 8, 12: 7, 11: 6}
    score = base_map.get(high, high / 2.0)

    if high == low:
        score = max(5.0, score * 2.0)
    else:
        if suited:
            score += 2.0
        if gap == 1:
            score += 1.0
        elif gap == 2:
            score -= 1.0
        elif gap == 3:
            score -= 2.0
        elif gap >= 4:
            score -= 4.0
    return score


def build_villain_preflop_range(
    villain_vpip: float,
    villain_pfr: float,
    dead_cards: set[str],
) -> list[tuple[str, str]]:
    """Construit la range de call preflop de Vilain (exclut sa range de 3-bet et les bloqueurs)."""
    combos = [
        (c1, c2)
        for c1, c2 in combinations(ALL_CARDS_STR, 2)
        if c1 not in dead_cards and c2 not in dead_cards
    ]
    combos.sort(key=lambda c: preflop_hand_score(c[0], c[1]), reverse=True)

    n_total = len(combos)
    three_bet_pct = np.clip(villain_pfr * 0.35, 0.03, 0.14)
    call_pct = max(villain_vpip - three_bet_pct, 0.08)

    start_idx = int(n_total * three_bet_pct)
    end_idx = min(int(n_total * (three_bet_pct + call_pct)), n_total)
    return combos[start_idx:end_idx]


def combo_flop_strength(
    evaluator: Evaluator,
    flop_int: list[int],
    combo: tuple[str, str],
) -> int:
    """Évalue la force faite + tirage couleur d'un combo sur le flop (plus bas = meilleur)."""
    hand_int = [Card.new(combo[0]), Card.new(combo[1])]
    made_score = evaluator.evaluate(flop_int, hand_int)

    suits = [combo[0][1], combo[1][1]] + [Card.int_to_str(c)[1] for c in flop_int]
    has_fd = any(suits.count(s) == 4 for s in SUITS)
    if has_fd and made_score > 3000:
        made_score -= 2200
    return made_score


def monte_carlo_equity(
    evaluator: Evaluator,
    hero_int: list[int],
    flop_int: list[int],
    continuing_range: list[tuple[str, str]],
    n_sims: int = 1500,
) -> float:
    """Calcule l'équité de Hero contre la range de continuation de Vilain sur Turn + River."""
    if not continuing_range:
        return 1.0

    used_base = set(hero_int + flop_int)
    full_deck = [Card.new(c) for c in ALL_CARDS_STR]
    remaining_deck = [c for c in full_deck if c not in used_base]

    wins = 0.0
    for _ in range(n_sims):
        v_c1_str, v_c2_str = random.choice(continuing_range)
        v_int = [Card.new(v_c1_str), Card.new(v_c2_str)]

        while True:
            t, r = random.sample(remaining_deck, 2)
            if t not in v_int and r not in v_int:
                break

        board_5 = flop_int + [t, r]
        h_score = evaluator.evaluate(board_5, hero_int)
        v_score = evaluator.evaluate(board_5, v_int)

        if h_score < v_score:
            wins += 1.0
        elif h_score == v_score:
            wins += 0.5

    return wins / n_sims


class ExploitativeFlopEngine:
    def __init__(self, model_path: Path = MODEL_PATH):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.evaluator = Evaluator()

    def recommend(
        self,
        hero_hand: list[str],
        flop: list[str],
        pot_bb: float,
        eff_stack_bb: float,
        is_ip: int,
        hero_stats: dict,
        villain_stats: dict,
        sizings: list[float] = [0.33, 0.50, 0.75],
        rake_pct: float = 0.0575,
    ) -> pd.DataFrame:
        dead = set(hero_hand + flop)
        hero_int = [Card.new(c) for c in hero_hand]
        flop_int = [Card.new(c) for c in flop]

        # 1. Range preflop de Vilain triée par force sur ce flop spécifique
        v_range = build_villain_preflop_range(
            villain_stats["vpip"], villain_stats["pfr"], dead
        )
        v_range.sort(key=lambda c: combo_flop_strength(self.evaluator, flop_int, c))

        # 2. Équité de base et réalisation d'équité réaliste (les bluffs abandonnent souvent Turn)
        eq_full = monte_carlo_equity(self.evaluator, hero_int, flop_int, v_range, n_sims=1500)
        eqr_check = 0.90 if is_ip else 0.75
        ev_check = pot_bb * (1.0 - rake_pct) * min(eq_full * eqr_check, 1.0)

        results = [
            {
                "Action": "Check",
                "Mise (bb)": 0.0,
                "P(Fold)": 0.0,
                "P(Call)": 100.0,
                "P(Raise)": 0.0,
                "Équité vs Call": round(eq_full * 100, 1),
                "EV (bb)": round(ev_check, 2),
            }
        ]

        # 3. Interrogation de LightGBM pour chaque sizing candidat
        board_feats = evaluate_board_features([parse_card(c) for c in flop])
        spr = min(eff_stack_bb / pot_bb, 30.0)

        rows_ml = []
        for s in sizings:
            rows_ml.append(
                {
                    "is_ip": is_ip,
                    "spr": spr,
                    "bet_size_pct": s,
                    "aggressor_vpip_smooth": hero_stats["vpip"],
                    "aggressor_pfr_smooth": hero_stats["pfr"],
                    "villain_vpip_smooth": villain_stats["vpip"],
                    "villain_pfr_smooth": villain_stats["pfr"],
                    "villain_fold_cbet_smooth": villain_stats["fold_cbet"],
                    "villain_raise_cbet_smooth": villain_stats["raise_cbet"],
                    **board_feats,
                }
            )

        probs = self.model.predict_proba(pd.DataFrame(rows_ml)[self.features])

        # 4. Calcul d'EV par sizing en filtrant la range de continuation de Vilain
        for idx, s in enumerate(sizings):
            p_fold, p_call, p_raise = probs[idx]
            bet_bb = min(s * pot_bb, eff_stack_bb)

            # Vilain ne continue qu'avec le top (1 - p_fold) de sa range sur ce board
            keep_count = max(int(len(v_range) * (1.0 - p_fold)), 5)
            continuing_range = v_range[:keep_count]

            eq_vs_call = monte_carlo_equity(
                self.evaluator, hero_int, flop_int, continuing_range, n_sims=1500
            )

            # Une main faite réalise ~100% de son équité IP ; un bluff pur ~62% (abandon Turn)
            if eq_vs_call >= 0.50:
                eqr_bet = 1.02 if is_ip else 0.90
            elif eq_vs_call >= 0.30:
                eqr_bet = 0.85 if is_ip else 0.72
            else:
                eqr_bet = 0.62 if is_ip else 0.50

            realized_eq = min(eq_vs_call * eqr_bet, 1.0)

            # EV si Vilain Call
            pot_if_called = (pot_bb + 2.0 * bet_bb) * (1.0 - rake_pct)
            ev_when_called = pot_if_called * realized_eq - bet_bb

            # EV si Vilain Raise (~3x bet)
            raise_to_bb = min(3.0 * bet_bb, eff_stack_bb)
            if eq_vs_call >= 0.62:
                pot_if_raised = (pot_bb + 2.0 * raise_to_bb) * (1.0 - rake_pct)
                ev_when_raised = pot_if_raised * (eq_vs_call * 0.78) - raise_to_bb
            else:
                ev_when_raised = -bet_bb

            total_ev = (
                p_fold * pot_bb
                + p_call * ev_when_called
                + p_raise * ev_when_raised
            )

            results.append(
                {
                    "Action": f"Bet {int(s*100)}%",
                    "Mise (bb)": round(bet_bb, 2),
                    "P(Fold)": round(p_fold * 100, 1),
                    "P(Call)": round(p_call * 100, 1),
                    "P(Raise)": round(p_raise * 100, 1),
                    "Équité vs Call": round(eq_vs_call * 100, 1),
                    "EV (bb)": round(total_ev, 2),
                }
            )

        return pd.DataFrame(results)


if __name__ == "__main__":
    random.seed(42)
    engine = ExploitativeFlopEngine()

    hero = {"vpip": 0.171, "pfr": 0.149}
    pool_villain = {"vpip": 0.274, "pfr": 0.166, "fold_cbet": 0.454, "raise_cbet": 0.080}
    station_villain = {"vpip": 0.480, "pfr": 0.120, "fold_cbet": 0.280, "raise_cbet": 0.050}

    spots = [
        ("Bluff pur (As 4d) sur Ks 7d 2c IP vs Pool Moyen", ["As", "4d"], ["Ks", "7d", "2c"], pool_villain),
        ("Bluff pur (Ah 4d) sur 9s 8h 6s IP vs Calling Station", ["Ah", "4d"], ["9s", "8h", "6s"], station_villain),
        ("Top Paire Top Kicker (Ad Kc) sur Ks 7d 2c IP vs Calling Station", ["Ad", "Kc"], ["Ks", "7d", "2c"], station_villain),
    ]

    for title, hand, board, vil in spots:
        print(f"\n=== {title} (Pot: 6.5 bb) ===")
        df_res = engine.recommend(
            hero_hand=hand,
            flop=board,
            pot_bb=6.5,
            eff_stack_bb=93.5,
            is_ip=1,
            hero_stats=hero,
            villain_stats=vil,
        )
        print(df_res.to_string(index=False))