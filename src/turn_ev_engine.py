from pathlib import Path
import random
import joblib
import pandas as pd
from treys import Card, Evaluator

from ev_engine import (
    ALL_CARDS_STR,
    build_villain_preflop_range,
    combo_flop_strength,
)
from turn_pipeline import extract_turn_features

MODEL_TURN_PATH = Path("models/turn_barrel_lgbm.joblib")


def combo_turn_strength(
    evaluator: Evaluator,
    board_4_int: list[int],
    combo: tuple[str, str],
) -> int:
    """Évalue la force faite + tirage d'un combo sur le board de 4 cartes (plus bas = meilleur)."""
    hand_int = [Card.new(combo[0]), Card.new(combo[1])]
    made_score = evaluator.evaluate(board_4_int, hand_int)

    suits = [combo[0][1], combo[1][1]] + [Card.int_to_str(c)[1] for c in board_4_int]
    has_fd = any(suits.count(s) == 4 for s in "shdc")
    if has_fd and made_score > 3000:
        made_score -= 1800
    return made_score


def monte_carlo_turn_equity(
    evaluator: Evaluator,
    hero_int: list[int],
    board_4_int: list[int],
    continuing_range: list[tuple[str, str]],
    n_sims: int = 1500,
) -> float:
    """Calcule l'équité de Hero contre la range de continuation de Vilain sur la River."""
    if not continuing_range:
        return 1.0

    used_base = set(hero_int + board_4_int)
    full_deck = [Card.new(c) for c in ALL_CARDS_STR]
    remaining_deck = [c for c in full_deck if c not in used_base]

    wins = 0.0
    for _ in range(n_sims):
        v_c1_str, v_c2_str = random.choice(continuing_range)
        v_int = [Card.new(v_c1_str), Card.new(v_c2_str)]

        while True:
            river_card = random.choice(remaining_deck)
            if river_card not in v_int:
                break

        board_5 = board_4_int + [river_card]
        h_score = evaluator.evaluate(board_5, hero_int)
        v_score = evaluator.evaluate(board_5, v_int)

        if h_score < v_score:
            wins += 1.0
        elif h_score == v_score:
            wins += 0.5

    return wins / n_sims


class ExploitativeTurnEngine:
    def __init__(self, model_path: Path = MODEL_TURN_PATH):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.evaluator = Evaluator()

    def recommend(
        self,
        hero_hand: list[str],
        flop: list[str],
        turn_card: str,
        pot_bb: float,
        eff_stack_bb: float,
        is_ip: int,
        villain_stats: dict,
        sizings: list[float] = [0.33, 0.50, 0.75],
        rake_pct: float = 0.0575,
    ) -> pd.DataFrame:
        dead = set(hero_hand + flop + [turn_card])
        hero_int = [Card.new(c) for c in hero_hand]
        flop_int = [Card.new(c) for c in flop]
        board_4_int = flop_int + [Card.new(turn_card)]

        # 1. Range preflop de Vilain filtrée par son call au flop (~55% de continuation moyenne)
        v_range = build_villain_preflop_range(
            villain_stats["vpip"], villain_stats["pfr"], dead
        )
        v_range.sort(key=lambda c: combo_flop_strength(self.evaluator, flop_int, c))
        flop_call_keep = max(int(len(v_range) * 0.55), 10)
        v_turn_range = v_range[:flop_call_keep]

        # Re-tri de la range sur la nouvelle carte de Turn
        v_turn_range.sort(key=lambda c: combo_turn_strength(self.evaluator, board_4_int, c))

        # 2. Équité de base en cas de Check à la Turn
        eq_full = monte_carlo_turn_equity(
            self.evaluator, hero_int, board_4_int, v_turn_range, n_sims=1500
        )
        eqr_check = 0.94 if is_ip else 0.80
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

        # 3. Interrogation de LightGBM Turn pour chaque sizing de 2e barrel
        turn_feats = extract_turn_features(flop, turn_card)
        rows_ml = []
        for s in sizings:
            rows_ml.append(
                {
                    "is_ip": is_ip,
                    "pot_bb": pot_bb,
                    "bet_size_pct": s,
                    "villain_vpip_smooth": villain_stats["vpip"],
                    "villain_pfr_smooth": villain_stats["pfr"],
                    "villain_fold_turn_smooth": villain_stats.get("fold_turn", 0.39),
                    "villain_raise_turn_smooth": villain_stats.get("raise_turn", 0.082),
                    **turn_feats,
                }
            )

        probs = self.model.predict_proba(pd.DataFrame(rows_ml)[self.features])

        # 4. Calcul d'EV par sizing de 2e barrel
        for idx, s in enumerate(sizings):
            p_fold, p_call, p_raise = probs[idx]
            bet_bb = min(s * pot_bb, eff_stack_bb)

            keep_count = max(int(len(v_turn_range) * (1.0 - p_fold)), 4)
            continuing_range = v_turn_range[:keep_count]

            eq_vs_call = monte_carlo_turn_equity(
                self.evaluator, hero_int, board_4_int, continuing_range, n_sims=1500
            )

            # À la Turn il ne reste qu'une seule carte (la River) : réalisation d'équité plus directe
            if eq_vs_call >= 0.50:
                eqr_bet = 1.02 if is_ip else 0.94
            elif eq_vs_call >= 0.25:
                eqr_bet = 0.92 if is_ip else 0.82
            else:
                eqr_bet = 0.72 if is_ip else 0.60

            realized_eq = min(eq_vs_call * eqr_bet, 1.0)

            pot_if_called = (pot_bb + 2.0 * bet_bb) * (1.0 - rake_pct)
            ev_when_called = pot_if_called * realized_eq - bet_bb

            raise_to_bb = min(2.6 * bet_bb, eff_stack_bb)
            if eq_vs_call >= 0.65:
                pot_if_raised = (pot_bb + 2.0 * raise_to_bb) * (1.0 - rake_pct)
                ev_when_raised = pot_if_raised * (eq_vs_call * 0.82) - raise_to_bb
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