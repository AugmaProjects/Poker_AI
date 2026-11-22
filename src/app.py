import random
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parent))
from ev_engine import ExploitativeFlopEngine
from turn_ev_engine import ExploitativeTurnEngine

app = FastAPI(title="Poker AI - Exploitative Flop & Turn Engine")
flop_engine = ExploitativeFlopEngine()
turn_engine = ExploitativeTurnEngine()

STATIC_HTML = Path("static/index.html")


class SpotRequest(BaseModel):
    hero_hand: list[str]
    flop: list[str]
    pot_bb: float = 6.5
    eff_stack_bb: float = 93.5
    is_ip: int = 1
    hero_vpip: float = 17.1
    hero_pfr: float = 14.9
    villain_vpip: float = 27.4
    villain_pfr: float = 16.6
    villain_fold_cbet: float = 45.4
    villain_raise_cbet: float = 8.0


class TurnSpotRequest(BaseModel):
    hero_hand: list[str]
    flop: list[str]
    turn_card: str
    pot_bb: float = 11.0
    eff_stack_bb: float = 89.0
    is_ip: int = 1
    villain_vpip: float = 27.4
    villain_pfr: float = 16.6
    villain_fold_turn: float = 39.0
    villain_raise_turn: float = 8.2


@app.get("/")
def serve_ui():
    return FileResponse(STATIC_HTML)


@app.post("/api/evaluate")
def evaluate_spot(req: SpotRequest):
    all_cards = req.hero_hand + req.flop
    if len(req.hero_hand) != 2 or len(req.flop) != 3:
        raise HTTPException(status_code=400, detail="Sélectionne exactement 2 cartes en main et 3 cartes au flop.")
    if len(set(all_cards)) != 5:
        raise HTTPException(status_code=400, detail="Carte en double détectée entre ta main et le flop.")

    random.seed(42)
    df = flop_engine.recommend(
        hero_hand=req.hero_hand,
        flop=req.flop,
        pot_bb=req.pot_bb,
        eff_stack_bb=req.eff_stack_bb,
        is_ip=req.is_ip,
        hero_stats={"vpip": req.hero_vpip / 100.0, "pfr": req.hero_pfr / 100.0},
        villain_stats={
            "vpip": req.villain_vpip / 100.0,
            "pfr": req.villain_pfr / 100.0,
            "fold_cbet": req.villain_fold_cbet / 100.0,
            "raise_cbet": req.villain_raise_cbet / 100.0,
        },
    )

    records = df.to_dict(orient="records")
    best_action = max(records, key=lambda r: r["EV (bb)"])
    return {"results": records, "best_action": best_action["Action"], "best_ev": best_action["EV (bb)"]}


@app.post("/api/evaluate_turn")
def evaluate_turn_spot(req: TurnSpotRequest):
    all_cards = req.hero_hand + req.flop + [req.turn_card]
    if len(req.hero_hand) != 2 or len(req.flop) != 3 or not req.turn_card:
        raise HTTPException(status_code=400, detail="Sélectionne 2 cartes en main, 3 cartes au flop et 1 carte à la Turn.")
    if len(set(all_cards)) != 6:
        raise HTTPException(status_code=400, detail="Carte en double détectée sur le board ou dans ta main.")

    random.seed(42)
    df = turn_engine.recommend(
        hero_hand=req.hero_hand,
        flop=req.flop,
        turn_card=req.turn_card,
        pot_bb=req.pot_bb,
        eff_stack_bb=req.eff_stack_bb,
        is_ip=req.is_ip,
        villain_stats={
            "vpip": req.villain_vpip / 100.0,
            "pfr": req.villain_pfr / 100.0,
            "fold_turn": req.villain_fold_turn / 100.0,
            "raise_turn": req.villain_raise_turn / 100.0,
        },
    )

    records = df.to_dict(orient="records")
    best_action = max(records, key=lambda r: r["EV (bb)"])
    return {"results": records, "best_action": best_action["Action"], "best_ev": best_action["EV (bb)"]}