from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, List
import time


router = APIRouter(tags=["Stations"])


class StationKPI(BaseModel):
    good: int
    bad: int
    active: int
    streak: int
    lastText: str
    delay: bool = False


_STATIONS: Dict[str, StationKPI] = {
    "sorting": StationKPI(good=0, bad=0, active=1, streak=0, lastText="RUNNING", delay=False),
    "packaging": StationKPI(good=0, bad=0, active=1, streak=0, lastText="PALLETIZING", delay=False),
    "cnc": StationKPI(good=0, bad=0, active=1, streak=0, lastText="MACHINING", delay=False),
}


@router.get("/stations")
def list_stations():
    # Provide a lightweight snapshot, increasing counters to simulate activity
    now = int(time.time())
    tick = now % 3 == 0
    if tick:
        s = _STATIONS["sorting"]; s.good += 1; s.active = 1; s.lastText = "PASS"
        p = _STATIONS["packaging"]; p.good += 1; p.active = 1
        c = _STATIONS["cnc"]; c.good += 1; c.active = 1
    items = []
    for sid, k in _STATIONS.items():
        items.append({"id": sid, "name": sid.capitalize(), "kpi": k.dict()})
    return {"items": items}


@router.post("/stations/{station_id}/delay")
def set_delay(station_id: str, payload: dict):
    target = _STATIONS.get(station_id)
    if not target:
        return {"ok": False}
    target.delay = bool(payload.get("delay"))
    target.lastText = "DELAY" if target.delay else target.lastText
    return {"ok": True, "station": {"id": station_id, "kpi": target.dict()}}


