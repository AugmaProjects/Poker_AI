import re
from pathlib import Path
from datetime import datetime
import pandas as pd

# =====================================================================
# 1. CONFIGURATION : Mets ici le chemin vers tes fichiers .txt Winamax
# =====================================================================
RAW_HISTORY_DIR = Path(r"C:\Users\nknry\AppData\Roaming\winamax\documents\accounts\SHINIGAMI_GM\history")
OUTPUT_PARQUET = Path("data/processed/hands_all.parquet")

# Regex précompilées pour la performance
RE_HEADER = re.compile(
    r"Winamax Poker - CashGame - HandId: #(?P<hand_id>[\d-]+) - "
    r"Holdem no limit \((?P<sb>\d+(?:\.\d+)?)€/(?P<bb>\d+(?:\.\d+)?)€\) - "
    r"(?P<dt>\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})"
)
RE_TABLE = re.compile(r"Table: '.+?' (?P<max_seats>\d+)-max .* Seat #(?P<btn>\d+) is the button")
RE_SEAT = re.compile(r"^Seat (?P<seat>\d+): (?P<player>.+?) \((?P<stack>\d+(?:\.\d+)?)€\)$")
RE_POST = re.compile(r"^(?P<player>.+?) posts (?:small blind|big blind|dead blind) (?P<amt>\d+(?:\.\d+)?)€")
RE_FOLD = re.compile(r"^(?P<player>.+?) folds$")
RE_CHECK = re.compile(r"^(?P<player>.+?) checks$")
RE_CALL = re.compile(r"^(?P<player>.+?) calls (?P<amt>\d+(?:\.\d+)?)€")
RE_BET = re.compile(r"^(?P<player>.+?) bets (?P<amt>\d+(?:\.\d+)?)€")
RE_RAISE = re.compile(r"^(?P<player>.+?) raises (?P<inc>\d+(?:\.\d+)?)€ to (?P<to>\d+(?:\.\d+)?)€")
RE_UNCALLED = re.compile(r"^Uncalled bet \((?P<amt>\d+(?:\.\d+)?)€\) returned to (?P<player>.+)$")
RE_COLLECTED = re.compile(r"^(?P<player>.+?) collected (?P<amt>\d+(?:\.\d+)?)€ from")


def assign_positions(seats_dict: dict[int, str], btn_seat: int) -> dict[str, str]:
    """Attribue les positions (BTN, SB, BB, UTG/HJ, CO) dans l'ordre horaire."""
    ordered_seats = sorted(seats_dict.keys())
    if btn_seat not in ordered_seats:
        return {p: "UNK" for p in seats_dict.values()}

    idx = ordered_seats.index(btn_seat)
    rotated = ordered_seats[idx:] + ordered_seats[:idx]
    n = len(rotated)

    pos_maps = {
        2: ["BTN", "BB"],
        3: ["BTN", "SB", "BB"],
        4: ["BTN", "SB", "BB", "CO"],
        5: ["BTN", "SB", "BB", "HJ", "CO"],
        6: ["BTN", "SB", "BB", "UTG", "MP", "CO"],
    }
    labels = pos_maps.get(n, [f"POS_{i}" for i in range(n)])
    return {seats_dict[s]: labels[i] for i, s in enumerate(rotated)}


def parse_single_hand(hand_text: str) -> list[dict]:
    """Parse une main individuelle de Cash Game Winamax."""
    lines = [line.strip() for line in hand_text.splitlines() if line.strip()]
    if not lines:
        return []

    m_header = RE_HEADER.match(lines[0])
    if not m_header:
        return []  # Ignore Tournois, Expresso, Sit&Go

    hand_id = m_header.group("hand_id")
    bb = float(m_header.group("bb"))
    timestamp = datetime.strptime(m_header.group("dt"), "%Y/%m/%d %H:%M:%S")

    if len(lines) < 3:
        return []

    m_table = RE_TABLE.match(lines[1])
    if not m_table:
        return []
    max_seats = int(m_table.group("max_seats"))
    btn_seat = int(m_table.group("btn"))

    # Extraction des joueurs assis
    seats = {}
    stacks = {}
    for line in lines[2:]:
        if line.startswith("***"):
            break
        m_seat = RE_SEAT.match(line)
        if m_seat:
            s_num = int(m_seat.group("seat"))
            p_name = m_seat.group("player")
            seats[s_num] = p_name
            stacks[p_name] = float(m_seat.group("stack"))

    if len(seats) < 2:
        return []

    positions = assign_positions(seats, btn_seat)

    # Initialisation de l'état par joueur
    players_data = {
        p: {
            "hand_id": hand_id,
            "timestamp": timestamp,
            "bb_euro": bb,
            "max_seats": max_seats,
            "num_players": len(seats),
            "player_name": p,
            "position": positions.get(p, "UNK"),
            "stack_bb": round(stacks[p] / bb, 2),
            "vpip": False,
            "pfr": False,
            "is_pfr_aggressor": False,
            "saw_flop": False,
            "faced_cbet_flop": False,
            "folded_to_cbet_flop": False,
            "called_cbet_flop": False,
            "raised_cbet_flop": False,
            "net_bb": 0.0,
        }
        for p in seats.values()
    }

    street = "BLINDS"
    street_invested = {p: 0.0 for p in seats.values()}
    total_invested = {p: 0.0 for p in seats.values()}
    total_won = {p: 0.0 for p in seats.values()}
    active_players = set(seats.values())

    last_pfr = None
    flop_donk_or_bet = False
    cbet_active = False
    cbet_raised = False

    for line in lines[2:]:
        # Changement de street
        if line.startswith("*** "):
            # Clôture de l'investissement de la street précédente
            for p, amt in street_invested.items():
                total_invested[p] += amt
            street_invested = {p: 0.0 for p in seats.values()}

            if "ANTE/BLINDS" in line:
                street = "BLINDS"
            elif "PRE-FLOP" in line:
                street = "PREFLOP"
            elif "FLOP" in line:
                street = "FLOP"
                for p in active_players:
                    players_data[p]["saw_flop"] = True
                if last_pfr and last_pfr in active_players:
                    players_data[last_pfr]["is_pfr_aggressor"] = True
            elif "TURN" in line:
                street = "TURN"
            elif "RIVER" in line:
                street = "RIVER"
            elif "SHOW DOWN" in line or "SUMMARY" in line:
                street = "END"
            continue

        if street == "END":
            m_col = RE_COLLECTED.match(line)
            if m_col and m_col.group("player") in total_won:
                total_won[m_col.group("player")] += float(m_col.group("amt"))
            continue

        # Blindes
        if street == "BLINDS":
            m_post = RE_POST.match(line)
            if m_post:
                p = m_post.group("player")
                if p in street_invested:
                    street_invested[p] += float(m_post.group("amt"))
            continue

        # Retour de mise non payée
        m_unc = RE_UNCALLED.match(line)
        if m_unc:
            p = m_unc.group("player")
            if p in street_invested:
                street_invested[p] -= float(m_unc.group("amt"))
            continue

        # Actions de jeu (Preflop, Flop, Turn, River)
        m_fold = RE_FOLD.match(line)
        if m_fold:
            p = m_fold.group("player")
            active_players.discard(p)
            if street == "FLOP" and cbet_active and not cbet_raised and p != last_pfr:
                players_data[p]["faced_cbet_flop"] = True
                players_data[p]["folded_to_cbet_flop"] = True
            continue

        m_check = RE_CHECK.match(line)
        if m_check:
            continue

        m_call = RE_CALL.match(line)
        if m_call:
            p = m_call.group("player")
            amt = float(m_call.group("amt"))
            if p in street_invested:
                street_invested[p] += amt
            if street == "PREFLOP":
                players_data[p]["vpip"] = True
            elif street == "FLOP" and cbet_active and not cbet_raised and p != last_pfr:
                players_data[p]["faced_cbet_flop"] = True
                players_data[p]["called_cbet_flop"] = True
            continue

        m_bet = RE_BET.match(line)
        if m_bet:
            p = m_bet.group("player")
            amt = float(m_bet.group("amt"))
            if p in street_invested:
                street_invested[p] += amt
            if street == "FLOP":
                if p == last_pfr and not flop_donk_or_bet:
                    cbet_active = True
                flop_donk_or_bet = True
            continue

        m_raise = RE_RAISE.match(line)
        if m_raise:
            p = m_raise.group("player")
            to_amt = float(m_raise.group("to"))
            if p in street_invested:
                # Sur Winamax, "raises X to Y" indique le montant total misé sur la street
                street_invested[p] = to_amt
            if street == "PREFLOP":
                players_data[p]["vpip"] = True
                players_data[p]["pfr"] = True
                last_pfr = p
            elif street == "FLOP" and cbet_active and not cbet_raised and p != last_pfr:
                players_data[p]["faced_cbet_flop"] = True
                players_data[p]["raised_cbet_flop"] = True
                cbet_raised = True
            continue

    # Calcul final du gain net en Big Blinds
    for p, p_dict in players_data.items():
        net_euro = total_won[p] - total_invested[p]
        p_dict["net_bb"] = round(net_euro / bb, 2)

    return list(players_data.values())


def main():
    if not RAW_HISTORY_DIR.exists():
        raise FileNotFoundError(f"Dossier introuvable : {RAW_HISTORY_DIR}")

    txt_files = list(RAW_HISTORY_DIR.rglob("*.txt"))
    print(f"Fichiers .txt trouvés : {len(txt_files)}")

    all_rows = []
    cash_files_count = 0

    for fp in txt_files:
        name_lower = fp.name.lower()
        if "summary" in name_lower or "_play_" in name_lower:
            continue

        content = fp.read_text(encoding="utf-8-sig", errors="replace")
        if "Winamax Poker - CashGame" not in content:
            continue

        cash_files_count += 1
        # Découpage main par main
        raw_hands = content.split("Winamax Poker - CashGame - ")
        for raw_h in raw_hands[1:]:
            full_hand = "Winamax Poker - CashGame - " + raw_h
            all_rows.extend(parse_single_hand(full_hand))

    if not all_rows:
        print("Aucune main de Cash Game trouvée. Vérifie le chemin RAW_HISTORY_DIR.")
        return

    df = pd.DataFrame(all_rows)

    # Dédoublonnage (au cas où tu as des backups de fichiers) et tri chronologique strict
    df = df.drop_duplicates(subset=["hand_id", "player_name"])
    df = df.sort_values(["timestamp", "hand_id"]).reset_index(drop=True)

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PARQUET, index=False)

    print(f"\nFichiers Cash Game lus : {cash_files_count}")
    print(f"Mains uniques extraites : {df['hand_id'].nunique()}")
    print(f"Lignes joueur-main enregistrées : {len(df)} -> {OUTPUT_PARQUET}")

    # Vérification rapide : Top 10 des adversaires les plus rencontrés
    summary = (
        df.groupby("player_name")
        .agg(
            mains=("hand_id", "count"),
            vpip=("vpip", lambda s: round(s.mean() * 100, 1)),
            pfr=("pfr", lambda s: round(s.mean() * 100, 1)),
            faced_cbet=("faced_cbet_flop", "sum"),
            fold_cbet=(
                "folded_to_cbet_flop",
                lambda s: round(
                    (s.sum() / max(df.loc[s.index, "faced_cbet_flop"].sum(), 1)) * 100, 1
                ),
            ),
        )
        .sort_values("mains", ascending=False)
        .head(10)
    )
    print("\n--- TOP 10 JOUEURS DANS TA BASE ---")
    print(summary.to_string())


if __name__ == "__main__":
    main()
