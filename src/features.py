import re
import sys
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from parser import RAW_HISTORY_DIR

HANDS_PARQUET = Path("data/processed/hands_all.parquet")
OUTPUT_DATASET = Path("data/processed/dataset_flop.parquet")

RANK_MAP = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
    "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
}

RE_HEADER = re.compile(
    r"Winamax Poker - CashGame - HandId: #(?P<hand_id>[\d-]+) - "
    r"Holdem no limit \((?P<sb>\d+(?:\.\d+)?)€/(?P<bb>\d+(?:\.\d+)?)€\) - "
    r"(?P<dt>\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})"
)
RE_SEAT = re.compile(r"^Seat \d+: (?P<player>.+?) \((?P<stack>\d+(?:\.\d+)?)€\)$")
RE_FLOP = re.compile(
    r"^\*\*\* FLOP \*\*\* \[(?P<c1>[2-9TJQKA][shdc]) (?P<c2>[2-9TJQKA][shdc]) (?P<c3>[2-9TJQKA][shdc])\]"
)
RE_POST = re.compile(r"^(?P<player>.+?) posts (?:small blind|big blind|dead blind) (?P<amt>\d+(?:\.\d+)?)€")
RE_FOLD = re.compile(r"^(?P<player>.+?) folds$")
RE_CHECK = re.compile(r"^(?P<player>.+?) checks$")
RE_CALL = re.compile(r"^(?P<player>.+?) calls (?P<amt>\d+(?:\.\d+)?)€")
RE_BET = re.compile(r"^(?P<player>.+?) bets (?P<amt>\d+(?:\.\d+)?)€")
RE_RAISE = re.compile(r"^(?P<player>.+?) raises (?P<inc>\d+(?:\.\d+)?)€ to (?P<to>\d+(?:\.\d+)?)€")
RE_UNCALLED = re.compile(r"^Uncalled bet \((?P<amt>\d+(?:\.\d+)?)€\) returned to (?P<player>.+)$")


def parse_card(card_str: str) -> tuple[int, str]:
    return RANK_MAP[card_str[0]], card_str[1]


def evaluate_board_features(board_cards: list[tuple[int, str]]) -> dict:
    """Calcule la texture structurelle du flop."""
    b_ranks = sorted([c[0] for c in board_cards], reverse=True)
    b_suits = [c[1] for c in board_cards]

    b_rank_counts = Counter(b_ranks)
    b_suit_counts = Counter(b_suits)

    flop_high = b_ranks[0]
    flop_mid = b_ranks[1]
    flop_low = b_ranks[2]
    flop_is_paired = int(len(b_rank_counts) < 3)
    flop_is_monotone = int(max(b_suit_counts.values()) == 3)
    flop_is_two_tone = int(max(b_suit_counts.values()) == 2)
    flop_spread = flop_high - flop_low
    flop_is_connected = int(flop_spread <= 4 and not flop_is_paired)
    flop_has_ace = int(flop_high == 14)

    return {
        "flop_high": flop_high,
        "flop_mid": flop_mid,
        "flop_low": flop_low,
        "flop_is_paired": flop_is_paired,
        "flop_is_monotone": flop_is_monotone,
        "flop_is_two_tone": flop_is_two_tone,
        "flop_is_connected": flop_is_connected,
        "flop_has_ace": flop_has_ace,
    }


def build_historical_stats_lookup(parquet_path: Path) -> dict[tuple[str, str], tuple]:
    """Calcule les stats cumulées avant chaque main avec lissage bayésien."""
    print("Calcul des profils historiques sans fuite temporelle...")
    df = pd.read_parquet(parquet_path)
    df = df.sort_values(["timestamp", "hand_id"]).reset_index(drop=True)

    pool_vpip = df["vpip"].mean()
    pool_pfr = df["pfr"].mean()
    pool_fold_cbet = df["folded_to_cbet_flop"].sum() / max(df["faced_cbet_flop"].sum(), 1)
    pool_raise_cbet = df["raised_cbet_flop"].sum() / max(df["faced_cbet_flop"].sum(), 1)

    print(
        f"Priors du pool -> VPIP: {pool_vpip:.1%} | PFR: {pool_pfr:.1%} | "
        f"Fold C-bet: {pool_fold_cbet:.1%} | Raise C-bet: {pool_raise_cbet:.1%}"
    )

    g = df.groupby("player_name", sort=False)
    df["hist_hands"] = g.cumcount()

    for col in ["vpip", "pfr", "faced_cbet_flop", "folded_to_cbet_flop", "raised_cbet_flop"]:
        df[f"hist_{col}"] = g[col].cumsum() - df[col].astype(int)

    # Neutralisation des sièges anonymes Incognito
    is_incognito = df["player_name"].str.startswith("Incognito", na=False)
    df.loc[is_incognito, "hist_hands"] = 0
    for col in ["vpip", "pfr", "faced_cbet_flop", "folded_to_cbet_flop", "raised_cbet_flop"]:
        df.loc[is_incognito, f"hist_{col}"] = 0

    k_pre = 30.0
    k_cbet = 15.0

    vpip_smooth = (df["hist_vpip"] + k_pre * pool_vpip) / (df["hist_hands"] + k_pre)
    pfr_smooth = (df["hist_pfr"] + k_pre * pool_pfr) / (df["hist_hands"] + k_pre)
    fold_cbet_smooth = (df["hist_folded_to_cbet_flop"] + k_cbet * pool_fold_cbet) / (
        df["hist_faced_cbet_flop"] + k_cbet
    )
    raise_cbet_smooth = (df["hist_raised_cbet_flop"] + k_cbet * pool_raise_cbet) / (
        df["hist_faced_cbet_flop"] + k_cbet
    )

    keys = zip(df["hand_id"], df["player_name"])
    vals = zip(
        np.round(vpip_smooth.to_numpy(), 4),
        np.round(pfr_smooth.to_numpy(), 4),
        np.round(fold_cbet_smooth.to_numpy(), 4),
        np.round(raise_cbet_smooth.to_numpy(), 4),
    )
    return dict(zip(keys, vals))


def extract_table_cbet_spot(hand_text: str, stats_lookup: dict) -> dict | None:
    """Extrait tout C-bet au flop en Heads-Up (SRP) joué par n'importe quel joueur de la table."""
    if "*** FLOP ***" not in hand_text:
        return None

    lines = [line.strip() for line in hand_text.splitlines() if line.strip()]
    m_header = RE_HEADER.match(lines[0])
    if not m_header:
        return None

    hand_id = m_header.group("hand_id")
    bb = float(m_header.group("bb"))
    timestamp = m_header.group("dt")

    stacks = {}
    for line in lines[2:]:
        if line.startswith("***"):
            break
        m_seat = RE_SEAT.match(line)
        if m_seat:
            stacks[m_seat.group("player")] = float(m_seat.group("stack"))

    if len(stacks) < 2:
        return None

    active_players = set(stacks.keys())
    street_invested = {p: 0.0 for p in stacks}
    total_pot = 0.0
    street = "BLINDS"

    board_cards = None
    last_pfr = None
    preflop_raises = 0

    pot_at_flop = 0.0
    eff_stack_flop = 0.0
    aggressor_name = None
    defender_name = None

    flop_actions = []

    for line in lines[2:]:
        if line.startswith("*** "):
            street_invested = {p: 0.0 for p in stacks}
            if "ANTE/BLINDS" in line:
                street = "BLINDS"
            elif "PRE-FLOP" in line:
                street = "PREFLOP"
            elif "FLOP" in line:
                if len(active_players) != 2 or preflop_raises != 1 or last_pfr not in active_players:
                    return None
                m_flop = RE_FLOP.match(line)
                if not m_flop:
                    return None

                board_cards = [
                    parse_card(m_flop.group("c1")),
                    parse_card(m_flop.group("c2")),
                    parse_card(m_flop.group("c3")),
                ]
                aggressor_name = last_pfr
                defender_name = [p for p in active_players if p != aggressor_name][0]
                pot_at_flop = total_pot
                eff_stack_flop = min(stacks[aggressor_name], stacks[defender_name])
                street = "FLOP"
            elif "TURN" in line or "RIVER" in line or "SHOW DOWN" in line or "SUMMARY" in line:
                break
            continue

        if street == "BLINDS":
            m_post = RE_POST.match(line)
            if m_post:
                p = m_post.group("player")
                amt = float(m_post.group("amt"))
                if p in stacks:
                    street_invested[p] += amt
                    stacks[p] -= amt
                    total_pot += amt
            continue

        if street == "PREFLOP":
            m_unc = RE_UNCALLED.match(line)
            if m_unc:
                p = m_unc.group("player")
                amt = float(m_unc.group("amt"))
                if p in stacks:
                    street_invested[p] -= amt
                    stacks[p] += amt
                    total_pot -= amt
                continue

            if RE_FOLD.match(line):
                active_players.discard(RE_FOLD.match(line).group("player"))
            elif RE_CALL.match(line):
                m_call = RE_CALL.match(line)
                p = m_call.group("player")
                amt = float(m_call.group("amt"))
                if p in stacks:
                    street_invested[p] += amt
                    stacks[p] -= amt
                    total_pot += amt
            elif RE_RAISE.match(line):
                m_raise = RE_RAISE.match(line)
                p = m_raise.group("player")
                to_amt = float(m_raise.group("to"))
                if p in stacks:
                    inc = to_amt - street_invested[p]
                    street_invested[p] = to_amt
                    stacks[p] -= inc
                    total_pot += inc
                last_pfr = p
                preflop_raises += 1
            continue

        if street == "FLOP":
            if RE_FOLD.match(line):
                flop_actions.append((RE_FOLD.match(line).group("player"), "FOLD", 0.0))
            elif RE_CHECK.match(line):
                flop_actions.append((RE_CHECK.match(line).group("player"), "CHECK", 0.0))
            elif RE_CALL.match(line):
                m_call = RE_CALL.match(line)
                flop_actions.append((m_call.group("player"), "CALL", float(m_call.group("amt"))))
            elif RE_BET.match(line):
                m_bet = RE_BET.match(line)
                flop_actions.append((m_bet.group("player"), "BET", float(m_bet.group("amt"))))
            elif RE_RAISE.match(line):
                m_raise = RE_RAISE.match(line)
                flop_actions.append((m_raise.group("player"), "RAISE", float(m_raise.group("to"))))

    if not flop_actions or pot_at_flop <= 0 or eff_stack_flop <= 0 or not defender_name:
        return None

    # Séquence de C-bet : OOP (Bet direct) ou IP (Check de Vilain puis Bet de l'agresseur)
    if flop_actions[0][0] == aggressor_name and flop_actions[0][1] == "BET":
        is_ip = 0
        bet_amt = flop_actions[0][2]
        next_idx = 1
    elif (
        len(flop_actions) >= 2
        and flop_actions[0][0] == defender_name
        and flop_actions[0][1] == "CHECK"
        and flop_actions[1][0] == aggressor_name
        and flop_actions[1][1] == "BET"
    ):
        is_ip = 1
        bet_amt = flop_actions[1][2]
        next_idx = 2
    else:
        return None

    bet_size_pct = bet_amt / pot_at_flop
    if bet_size_pct <= 0 or bet_size_pct > 2.5:
        return None

    if len(flop_actions) <= next_idx or flop_actions[next_idx][0] != defender_name:
        return None

    v_act = flop_actions[next_idx][1]
    target_map = {"FOLD": 0, "CALL": 1, "RAISE": 2}
    if v_act not in target_map:
        return None

    agg_stats = stats_lookup.get((hand_id, aggressor_name))
    def_stats = stats_lookup.get((hand_id, defender_name))
    if agg_stats is None or def_stats is None:
        return None

    agg_vpip, agg_pfr, _, _ = agg_stats
    def_vpip, def_pfr, def_fold_cbet, def_raise_cbet = def_stats

    row = {
        "hand_id": hand_id,
        "timestamp": timestamp,
        "aggressor_name": aggressor_name,
        "villain_name": defender_name,
        "is_ip": is_ip,
        "spr": round(min(eff_stack_flop / pot_at_flop, 30.0), 2),
        "bet_size_pct": round(bet_size_pct, 3),
        "aggressor_vpip_smooth": float(agg_vpip),
        "aggressor_pfr_smooth": float(agg_pfr),
        "villain_vpip_smooth": float(def_vpip),
        "villain_pfr_smooth": float(def_pfr),
        "villain_fold_cbet_smooth": float(def_fold_cbet),
        "villain_raise_cbet_smooth": float(def_raise_cbet),
        "target_action": target_map[v_act],
    }
    row.update(evaluate_board_features(board_cards))
    return row


def main():
    if not HANDS_PARQUET.exists():
        raise FileNotFoundError(f"Lance d'abord parser.py pour générer {HANDS_PARQUET}")

    stats_lookup = build_historical_stats_lookup(HANDS_PARQUET)

    txt_files = list(RAW_HISTORY_DIR.rglob("*.txt"))
    spots = []
    seen_hands = set()

    print("Extraction de tous les spots de C-bet HU de la table...")
    for fp in txt_files:
        name_lower = fp.name.lower()
        if "summary" in name_lower or "_play_" in name_lower:
            continue

        content = fp.read_text(encoding="utf-8-sig", errors="replace")
        if "Winamax Poker - CashGame" not in content:
            continue

        for raw_h in content.split("Winamax Poker - CashGame - ")[1:]:
            full_hand = "Winamax Poker - CashGame - " + raw_h
            spot = extract_table_cbet_spot(full_hand, stats_lookup)
            if spot and spot["hand_id"] not in seen_hands:
                seen_hands.add(spot["hand_id"])
                spots.append(spot)

    df_out = pd.DataFrame(spots)
    df_out = df_out.sort_values(["timestamp", "hand_id"]).reset_index(drop=True)

    OUTPUT_DATASET.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(OUTPUT_DATASET, index=False)

    print(f"\nDataset ML généré : {len(df_out)} spots de C-bet HU -> {OUTPUT_DATASET}")
    dist = df_out["target_action"].value_counts(normalize=True).sort_index() * 100
    labels = {0: "Fold ", 1: "Call ", 2: "Raise"}
    print("\n--- DISTRIBUTION DES RÉACTIONS DE VILAIN ---")
    for k, pct in dist.items():
        print(f"  {labels[k]} ({k}) : {pct:.1f} %")
    print(f"\nSizing moyen des C-bets du pool : {df_out['bet_size_pct'].mean()*100:.1f} % du pot")


if __name__ == "__main__":
    main()