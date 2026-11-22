import random
import sys
import time
from pathlib import Path
import pandas as pd
from ev_engine import ExploitativeFlopEngine


def print_progress_bar(current: int, total: int, start_time: float, bar_len: int = 32):
    elapsed = max(time.time() - start_time, 0.001)
    pct = current / total
    filled = int(bar_len * pct)
    bar = "█" * filled + "░" * (bar_len - filled)
    speed = current / elapsed
    eta = (total - current) / max(speed, 0.1)
    sys.stdout.write(
        f"\r  Analyse IA : [{bar}] {pct*100:5.1f}% | {current}/{total} spots | {speed:4.1f} mains/s | Reste ~{eta:4.1f}s "
    )
    sys.stdout.flush()


def run_leak_audit(
    dataset_path: str = "data/hero_cbet_spots.parquet",
    max_hands: int = 500,
):
    path = Path(dataset_path)
    if not path.exists():
        print(f"Fichier introuvable : {path}. Lance d'abord : python src/extract_hero_spots.py")
        return

    df = pd.read_parquet(path)
    if len(df) == 0:
        print("Aucun spot trouvé dans le fichier parquet.")
        return

    if len(df) > max_hands:
        df = df.sample(n=max_hands, random_state=42)

    total = len(df)
    print(f"\nLancement de l'audit exploitatif sur {total} spots...\n")
    t0 = time.time()

    engine = ExploitativeFlopEngine()
    random.seed(42)

    audit_rows = []
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        hero_hand = list(row["hero_hand"])
        flop = list(row["flop"])

        recs = engine.recommend(
            hero_hand=hero_hand,
            flop=flop,
            pot_bb=float(row["pot_bb"]),
            eff_stack_bb=float(row["eff_stack_bb"]),
            is_ip=int(row["is_ip"]),
            hero_stats={"vpip": float(row["hero_vpip"]), "pfr": float(row["hero_pfr"])},
            villain_stats={
                "vpip": float(row["villain_vpip"]),
                "pfr": float(row["villain_pfr"]),
                "fold_cbet": float(row["villain_fold_cbet"]),
                "raise_cbet": float(row["villain_raise_cbet"]),
            },
        )

        best_row = recs.loc[recs["EV (bb)"].idxmax()]
        played_action = row["hero_action"]
        played_match = recs[recs["Action"] == played_action]
        if not played_match.empty:
            played_ev = float(played_match.iloc[0]["EV (bb)"])
            best_ev = float(best_row["EV (bb)"])
            ev_loss = best_ev - played_ev

            audit_rows.append({
                "Main": "".join(hero_hand),
                "Flop": "".join(flop),
                "Pos": "IP" if int(row["is_ip"]) == 1 else "OOP",
                "Vilain_Fold%": round(float(row["villain_fold_cbet"]) * 100, 1),
                "Joue": played_action,
                "Optimal": best_row["Action"],
                "EV_Joue": round(played_ev, 2),
                "EV_Opt": round(best_ev, 2),
                "Perte_bb": round(ev_loss, 2),
            })

        if idx % 5 == 0 or idx == total:
            print_progress_bar(idx, total, t0)

    print("\n")
    results = pd.DataFrame(audit_rows)

    print("=======================================================")
    print("       BILAN GLOBAL DE TES LEAKS AU FLOP (SRP)         ")
    print("=======================================================")
    print(f"Spots analysés : {len(results)}")
    print(f"Perte d'EV moyenne par spot : -{results['Perte_bb'].mean():.2f} bb")
    print(f"Perte totale cumulée sur l'échantillon : -{results['Perte_bb'].sum():.1f} bb\n")

    print("=== TOP 12 DE TES PLUS GROSSES ERREURS D'EV ===")
    top_leaks = results.sort_values(by="Perte_bb", ascending=False).head(12)
    print(top_leaks.to_string(index=False))

    print("\n=== RÉPARTITION DES PERTES PAR TYPE D'ERREUR ===")
    results["Type_Erreur"] = results["Joue"] + "  -->  " + results["Optimal"]
    summary = (
        results[results["Perte_bb"] > 0.15]
        .groupby("Type_Erreur")
        .agg(
            Nb_Spots=("Perte_bb", "count"),
            Perte_Totale_bb=("Perte_bb", "sum"),
            Perte_Moy_bb=("Perte_bb", "mean"),
        )
        .sort_values(by="Perte_Totale_bb", ascending=False)
        .head(8)
    )
    print(summary.to_string())

    print("\n=== PERFORMANCE PAR POSITION (IP vs OOP) ===")
    pos_summary = (
        results.groupby("Pos")
        .agg(
            Nb_Spots=("Perte_bb", "count"),
            Perte_Moy_bb=("Perte_bb", "mean"),
            Perte_Totale_bb=("Perte_bb", "sum"),
        )
    )
    print(pos_summary.to_string())


if __name__ == "__main__":
    run_leak_audit()