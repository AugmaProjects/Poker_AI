import re
from pathlib import Path
from collections import defaultdict
import pandas as pd

# --- CONFIGURATION ---
HERO_NAME = "SHINIGAMI_GM"
# Remets ici ton chemin exact qui a trouvé tes 999 fichiers !
HISTORY_DIR = Path(r"C:\Users\nknry\AppData\Roaming\winamax\documents\accounts\SHINIGAMI_GM\history")
OUTPUT_PATH = Path("data/hero_cbet_spots.parquet")


def parse_amount_bb(text: str, bb_size: float) -> float:
    m = re.search(r"([\d\.]+)", text)
    if not m:
        return 0.0
    val = float(m.group(1))
    return val if "bb" in text.lower() else val / bb_size


def bucket_action(bet_bb: float, pot_bb: float) -> str:
    if bet_bb <= 0.01:
        return "Check"
    ratio = bet_bb / max(pot_bb, 0.1)
    if ratio < 0.42:
        return "Bet 33%"
    elif ratio < 0.62:
        return "Bet 50%"
    elif ratio < 0.87:
        return "Bet 75%"
    else:
        return "Bet 100%"


def extract_all_spots():
    if not HISTORY_DIR.exists():
        print(f"ERREUR : Le dossier {HISTORY_DIR} n'existe pas.")
        return

    txt_files = [f for f in HISTORY_DIR.rglob("*.txt") if "summary" not in f.name.lower()]
    print(f"{len(txt_files)} fichiers d'historiques trouvés. Lecture en cours...")

    stats = defaultdict(lambda: {"hands": 0, "vpip": 0, "pfr": 0, "cbet_opp": 0, "fold_cbet": 0, "raise_cbet": 0})
    raw_hero_spots = []

    hand_split_re = re.compile(r"(?=Winamax Poker - )")
    blind_re = re.compile(r"\(([\d\.]+)€?/([\d\.]+)€?\)")

    for file_path in txt_files:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        hands = hand_split_re.split(content)
        for hand in hands:
            if "CashGame" not in hand or "*** FLOP ***" not in hand:
                continue

            m_bb = blind_re.search(hand.splitlines()[0])
            if not m_bb:
                continue
            bb_size = float(m_bb.group(2))
            if bb_size <= 0:
                continue

            parts = hand.split("*** FLOP ***")
            preflop_txt = parts[0]
            postflop_txt = parts[1].split("*** TURN ***")[0].split("*** SUMMARY ***")[0]

            seats = re.findall(r"Seat \d+: ([^(]+) \([\d\.]+€?\)", preflop_txt)
            players = [s.strip() for s in seats]
            if len(players) < 2 or "*** PRE-FLOP ***" not in preflop_txt:
                continue

            m_cards = re.search(rf"Dealt to {re.escape(HERO_NAME)} \[([2-9TJQKA][shdc]) ([2-9TJQKA][shdc])\]", preflop_txt)
            hero_cards = [m_cards.group(1), m_cards.group(2)] if m_cards else None

            pf_actions = preflop_txt.split("*** PRE-FLOP ***")[1]

            raisers = []
            vpip_players = set()
            pfr_players = set()
            folded_pf = set()
            pf_pot_bb = 1.5

            for line in pf_actions.splitlines():
                line = line.strip()
                for p in players:
                    if line.startswith(p + " "):
                        rest = line[len(p) + 1 :]
                        if rest.startswith("folds"):
                            folded_pf.add(p)
                        elif rest.startswith("calls"):
                            vpip_players.add(p)
                            pf_pot_bb += parse_amount_bb(rest, bb_size)
                        elif rest.startswith("raises"):
                            vpip_players.add(p)
                            pfr_players.add(p)
                            raisers.append(p)
                            m_to = re.search(r"to ([\d\.]+)", rest)
                            if m_to:
                                pf_pot_bb += float(m_to.group(1)) / bb_size

            for p in players:
                stats[p]["hands"] += 1
                if p in vpip_players:
                    stats[p]["vpip"] += 1
                if p in pfr_players:
                    stats[p]["pfr"] += 1

            active_flop_players = [p for p in players if p not in folded_pf]
            if len(raisers) != 1 or len(active_flop_players) != 2:
                continue

            pfr_player = raisers[0]
            caller_player = [p for p in active_flop_players if p != pfr_player][0]

            flop_header = postflop_txt.splitlines()[0]
            m_flop = re.search(r"\[([2-9TJQKA][shdc]) ([2-9TJQKA][shdc]) ([2-9TJQKA][shdc])\]", flop_header)
            if not m_flop:
                continue
            flop_cards = [m_flop.group(1), m_flop.group(2), m_flop.group(3)]

            # Analyse complète des actions du flop (incluant Check-Raise)
            flop_lines = [l.strip() for l in postflop_txt.splitlines()[1:] if l.strip()]
            first_actor = None
            pfr_bet_bb = 0.0
            pfr_acted = False
            pfr_checked_first = False
            pfr_check_raised = False
            caller_bet_after_check = False
            caller_resp = None

            for line in flop_lines:
                actor = next((p for p in active_flop_players if line.startswith(p + " ")), None)
                if not actor:
                    continue
                if first_actor is None:
                    first_actor = actor

                rest = line[len(actor) + 1 :]
                if actor == pfr_player:
                    if not pfr_acted:
                        if rest.startswith("checks"):
                            pfr_acted = True
                            pfr_checked_first = True
                        elif rest.startswith("bets"):
                            pfr_acted = True
                            pfr_bet_bb = parse_amount_bb(rest, bb_size)
                    elif pfr_checked_first and caller_bet_after_check:
                        if rest.startswith("raises"):
                            pfr_check_raised = True
                        break
                elif actor == caller_player:
                    if pfr_acted and pfr_bet_bb > 0 and caller_resp is None:
                        if rest.startswith("folds"):
                            caller_resp = "fold"
                        elif rest.startswith("calls"):
                            caller_resp = "call"
                        elif rest.startswith("raises"):
                            caller_resp = "raise"
                        break
                    elif pfr_checked_first and rest.startswith("bets"):
                        caller_bet_after_check = True

            if pfr_acted and pfr_bet_bb > 0 and caller_resp:
                stats[caller_player]["cbet_opp"] += 1
                if caller_resp == "fold":
                    stats[caller_player]["fold_cbet"] += 1
                elif caller_resp == "raise":
                    stats[caller_player]["raise_cbet"] += 1

            if pfr_player == HERO_NAME and hero_cards and pfr_acted:
                is_ip = 1 if first_actor == caller_player else 0
                action_label = "Check-Raise" if pfr_check_raised else bucket_action(pfr_bet_bb, pf_pot_bb)
                raw_hero_spots.append({
                    "hero_hand": hero_cards,
                    "flop": flop_cards,
                    "pot_bb": round(pf_pot_bb, 1),
                    "eff_stack_bb": 100.0,
                    "is_ip": is_ip,
                    "villain_name": caller_player,
                    "hero_action": action_label,
                })

    rows = []
    h_st = stats[HERO_NAME]
    h_n = max(h_st["hands"], 1)
    hero_vpip = h_st["vpip"] / h_n
    hero_pfr = h_st["pfr"] / h_n

    for spot in raw_hero_spots:
        v_st = stats[spot["villain_name"]]
        vn = v_st["hands"]
        vc = v_st["cbet_opp"]

        v_vpip = (v_st["vpip"] + 20 * 0.274) / (vn + 20)
        v_pfr = (v_st["pfr"] + 20 * 0.166) / (vn + 20)
        v_fold = (v_st["fold_cbet"] + 10 * 0.454) / (vc + 10)
        v_raise = (v_st["raise_cbet"] + 10 * 0.080) / (vc + 10)

        rows.append({
            "hero_hand": spot["hero_hand"],
            "flop": spot["flop"],
            "pot_bb": spot["pot_bb"],
            "eff_stack_bb": spot["eff_stack_bb"],
            "is_ip": spot["is_ip"],
            "hero_vpip": hero_vpip,
            "hero_pfr": hero_pfr,
            "villain_vpip": v_vpip,
            "villain_pfr": v_pfr,
            "villain_fold_cbet": v_fold,
            "villain_raise_cbet": v_raise,
            "hero_action": spot["hero_action"],
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_parquet(OUTPUT_PATH, index=False)
    xr_count = (out_df["hero_action"] == "Check-Raise").sum()
    print(f"Extraction terminée : {len(out_df)} spots sauvegardés (dont {xr_count} Check-Raises isolés).")


if __name__ == "__main__":
    extract_all_spots()