from django.shortcuts import render
from django.conf import settings
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import requests


def _api_base() -> str:
    return getattr(settings, "API_BASE", "http://127.0.0.1:8000")


def _tpl(request, name: str) -> str:
    """Choose partial template when coming from HTMX or when explicitly requested.

    Some proxies or environments may strip the HX-Request header; allow URL hints.
    """
    hx_hdr = (request.headers.get("HX-Request") or "").lower() == "true"
    hx_qs = request.GET.get("hx") in {"1", "true", "True"}
    partial_qs = request.GET.get("_partial") in {"1", "true", "True"}
    is_partial = hx_hdr or hx_qs or partial_qs
    return f"dashboard/{name}_partial.html" if is_partial else f"dashboard/{name}.html"


def index(request):
    api = _api_base()
    model_info = None
    health = None
    errors = []
    chart_points = None
    try:
        r = requests.get(f"{api}/api/v1/model_info", timeout=5)
        if r.ok:
            model_info = r.json()
        else:
            errors.append(f"model_info HTTP {r.status_code}")
    except Exception as e:
        errors.append(f"model_info error: {e}")

    try:
        r = requests.get(f"{api}/healthz", timeout=3)
        if r.ok:
            health = r.json()
        else:
            errors.append(f"healthz HTTP {r.status_code}")
    except Exception as e:
        errors.append(f"healthz error: {e}")

    # Fetch a small prediction sample for the chart
    try:
        payload = {
            "machine_id": "M001",
            "horizon_min": [1, 5, 15, 30],
            "algo": "lstm",
            "scope": "per_machine",
            "want_quantiles": False,
        }
        r = requests.post(f"{api}/api/v1/predict", json=payload, timeout=10)
        if r.ok:
            data = r.json()
            # Normalize to array of one result
            res = data.get("result")
            if isinstance(res, dict):
                points = res.get("points", [])
            elif isinstance(res, list) and res:
                points = res[0].get("points", [])
            else:
                points = []
            chart_points = [
                {"h": p.get("horizon_min"), "y": p.get("yhat")} for p in points if isinstance(p, dict)
            ]
        else:
            errors.append(f"predict sample HTTP {r.status_code}")
    except Exception as e:
        errors.append(f"predict sample error: {e}")

    context = {
        "api_base": api,
        "model_info": model_info,
        "health": health,
        "errors": errors,
        "chart_points": chart_points,
    }
    return render(request, _tpl(request, "index"), context)


def predict_view(request):
    api = _api_base()
    result = None
    error = None
    chart_points = None
    ann = None

    if request.method == "POST":
        machine_id = request.POST.get("machine_id", "M001").strip()
        horizon_raw = request.POST.get("horizon_min", "15").strip()
        scope = request.POST.get("scope", "per_machine")
        algo = request.POST.get("algo", "lstm")
        want_quantiles = request.POST.get("want_quantiles") == "on"

        # Support comma-separated horizons or a single integer
        try:
            if "," in horizon_raw:
                horizon_min = [int(x) for x in horizon_raw.split(",") if x.strip()]
            else:
                horizon_min = int(horizon_raw)
        except Exception:
            error = "Invalid horizon_min. Use an integer or comma-separated integers, e.g., 1,5,15"
            horizon_min = 15

        payload = {
            "machine_id": machine_id if "," not in machine_id else [m.strip() for m in machine_id.split(",") if m.strip()],
            "horizon_min": horizon_min,
            "algo": algo,
            "scope": scope,
            "want_quantiles": want_quantiles,
        }

        try:
            r = requests.post(f"{api}/api/v1/predict", json=payload, timeout=20)
            if r.ok:
                result = r.json()
                data = result.get("result", result)
                # Normalize to points array
                if isinstance(data, dict) and "points" in data:
                    pts = data.get("points", [])
                elif isinstance(data, list) and data and isinstance(data[0], dict) and "points" in data[0]:
                    pts = data[0].get("points", [])
                else:
                    pts = []
                chart_points = [
                    {"h": p.get("horizon_min"), "y": p.get("yhat")}
                    for p in pts if isinstance(p, dict)
                ]
                # Try ANN score for the first machine id
                mid = machine_id if isinstance(machine_id, str) else (machine_id[0] if machine_id else "M001")
                ann = _ann_score(api, mid, pts)
            else:
                error = f"Predict HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            error = f"Predict error: {e}"

    context = {
        "api_base": api,
        "result": result,
        "error": error,
        "chart_points": chart_points,
        "ann": ann or {},
    }
    return render(request, _tpl(request, "predict"), context)


@csrf_exempt
def api_proxy(request, path: str):
    """Proxy /api/* requests to FastAPI (API_BASE). Useful for same-origin browser calls.
    Example: /api/v1/model_info -> {API_BASE}/api/v1/model_info
    """
    api = _api_base().rstrip("/")
    target = f"{api}/api/{path}"

    method = request.method.upper()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in {"host", "content-length"}}
    try:
        resp = requests.request(
            method,
            target,
            params=request.GET.dict(),
            data=request.body if method in {"POST", "PUT", "PATCH"} else None,
            headers=headers,
            timeout=30,
        )
        dj = HttpResponse(resp.content, status=resp.status_code)
        ct = resp.headers.get("content-type")
        if ct:
            dj["Content-Type"] = ct
        # Pass through useful headers
        if "x-request-id" in resp.headers:
            dj["X-Request-ID"] = resp.headers["x-request-id"]
        return dj
    except Exception as e:
        return HttpResponse(f"Proxy error: {e}", status=502)


@csrf_exempt
def health_proxy(request):
    """Proxy /healthz to FastAPI /healthz"""
    api = _api_base().rstrip("/")
    target = f"{api}/healthz"
    try:
        resp = requests.get(target, timeout=10)
        dj = HttpResponse(resp.content, status=resp.status_code)
        ct = resp.headers.get("content-type")
        if ct:
            dj["Content-Type"] = ct
        return dj
    except Exception as e:
        return HttpResponse(f"Proxy error: {e}", status=502)


def _fetch_predict(api: str, machine_id: str, horizons: list[int]):
    try:
        payload = {
            "machine_id": machine_id,
            "horizon_min": horizons,
            "algo": "lstm",
            "scope": "per_machine",
            "want_quantiles": False,
        }
        r = requests.post(f"{api}/api/v1/predict", json=payload, timeout=20)
        if not r.ok:
            return None, f"predict {machine_id} HTTP {r.status_code}"
        j = r.json()
        data = j.get("result", j)
        # Try common shapes: {result:{points:[...]}} or {points:[...]} or list variant
        if isinstance(data, dict) and "points" in data:
            points = data.get("points", [])
        elif isinstance(data, list) and data and isinstance(data[0], dict) and "points" in data[0]:
            points = data[0].get("points", [])
        elif isinstance(j, dict) and "points" in j:
            points = j.get("points", [])
        else:
            points = []
        return points, None
    except Exception as e:
        return None, str(e)


def _ann_score(api: str, machine_id: str, points: list[dict] | None):
    """Call unified ANN endpoint with a tiny series from prediction points.
    Falls back to None on any error.
    """
    try:
        series = []
        for p in (points or [])[:20]:
            t = p.get("horizon_min") or p.get("h") or 0
            y = p.get("yhat") or p.get("y") or 0.0
            series.append({"t": float(t), "y": float(y)})
        payload = {"machine_id": machine_id, "series": series}
        r = requests.post(f"{api}/api/v1/ann/anomaly_score", json=payload, timeout=10)
        if not r.ok:
            return None
        return r.json()
    except Exception:
        return None

def _fetch_eval_baseline(api: str, machine_id: str):
    """Try to get an RMSE-like baseline via /api/v1/evaluate for horizon 15."""
    try:
        payload = {"machine_id": machine_id, "algo": "lstm", "lookback": 24, "horizons": [15]}
        r = requests.post(f"{api}/api/v1/evaluate", json=payload, timeout=30)
        if not r.ok:
            return None
        mets = r.json().get("metrics", {})
        h15 = mets.get("15") or next(iter(mets.values()), None)
        if not h15:
            return None
        return float(h15.get("rmse") or 0.0) or None
    except Exception:
        return None


def anomalies_view(request):
    api = _api_base()
    from django.conf import settings as djset

    machines = getattr(djset, "MACHINES", ["M001"]) or ["M001"]
    # Ensure we have multiple machines for a richer digital twin view
    if isinstance(machines, list) and len(machines) <= 1:
        machines = ["M001", "M002", "M003", "M004", "M005", "M006"]
    horizons = [1, 5, 15, 30]

    rows = []
    errors = []
    for m in machines:
        points, err = _fetch_predict(api, m, horizons)
        if err:
            errors.append(f"{m}: {err}")
            continue
        # Severity rules based on drift_flag occurrence across horizons
        drift_h = [p.get("horizon_min") for p in (points or []) if p.get("drift_flag")]
        if any(h >= 30 for h in drift_h):
            severity = "High"
        elif drift_h:
            severity = "Medium"
        else:
            severity = "None"

        ann = _ann_score(api, m, points)
        rows.append({
            "machine_id": m,
            "points": points or [],
            "severity": severity,
            "drifts": drift_h,
            "ann": ann or {},
        })

    context = {
        "rows": rows,
        "errors": errors,
        "api_base": api,
    }
    return render(request, _tpl(request, "anomalies"), context)


def maintenance_view(request):
    api = _api_base()
    from django.conf import settings as djset

    machines = getattr(djset, "MACHINES", ["M001"]) or ["M001"]
    recs = []
    try:
        params = {
            "machine_id": ",".join(machines),
            "algo": request.GET.get("algo", "lstm"),
        }
        r = requests.get(f"{api}/api/v1/maintenance/recommendations", params=params, timeout=45)
        if r.ok:
            payload = r.json()
            recs = payload.get("items", []) if isinstance(payload, dict) else []
        else:
            # fallback to local heuristic if backend not ready
            recs = _local_maintenance_fallback(api, machines)
    except Exception:
        recs = _local_maintenance_fallback(api, machines)

    context = {"recs": recs, "api_base": api}
    return render(request, _tpl(request, "maintenance"), context)


def compare_view(request):
    api = _api_base()
    from django.conf import settings as djset
    machines = getattr(djset, "MACHINES", ["M001"]) or ["M001"]

    # Inputs
    m = request.GET.get("machine", machines[0])
    horizons_raw = request.GET.get("horizons", "5,15,30").strip()
    algos_raw = request.GET.get("algos", "lstm,gru,xgboost,rf").strip()
    try:
        horizons = [int(x) for x in horizons_raw.split(",") if x.strip()]
    except Exception:
        horizons = [5, 15, 30]
    algos = [a.strip().lower() for a in algos_raw.split(",") if a.strip()]

    series = {}
    errors = []
    for algo in algos:
        try:
            payload = {"machine_id": m, "algo": algo, "lookback": 24, "horizons": horizons}
            r = requests.post(f"{api}/api/v1/evaluate", json=payload, timeout=60)
            if r.ok:
                series[algo] = r.json().get("metrics", {})
            else:
                errors.append(f"{algo}: HTTP {r.status_code}")
        except Exception as e:
            errors.append(f"{algo}: {e}")

    context = {
        "machines": machines,
        "selected_machine": m,
        "horizons": horizons,
        "algos": algos,
        "series": series,
        "errors": errors,
        "api_base": api,
    }
    return render(request, _tpl(request, "compare"), context)


def upload_data_view(request):
    """Simple CSV upload that stores files under artifacts/uploads/ for later ingestion."""
    import os
    from pathlib import Path
    saved_path = None
    error = None
    if request.method == "POST" and request.FILES.get("file"):
        f = request.FILES["file"]
        if not f.name.lower().endswith(".csv"):
            error = "Only CSV files are allowed."
        else:
            out_dir = Path("artifacts") / "uploads"
            out_dir.mkdir(parents=True, exist_ok=True)
            # prefix with timestamp to avoid clashes
            import time
            fname = f"{int(time.time())}_{f.name}"
            out_path = out_dir / fname
            with out_path.open("wb") as dst:
                for chunk in f.chunks():
                    dst.write(chunk)
            saved_path = str(out_path)
    context = {"saved_path": saved_path, "error": error}
    return render(request, _tpl(request, "upload"), context)


def machine_detail_view(request, machine_id: str):
    api = _api_base()
    points, err = _fetch_predict(api, machine_id, [1, 5, 15, 30])
    eval_rmse = _fetch_eval_baseline(api, machine_id)
    ann = _ann_score(api, machine_id, points)
    context = {
        "machine_id": machine_id,
        "points": points or [],
        "rmse": eval_rmse,
        "api_base": api,
        "error": err,
        "ann": ann or {},
    }
    return render(request, _tpl(request, "machine_detail"), context)


def stations3d_view(request):
    """Render the interactive 3D stations digital twin page (partial-aware)."""
    return render(request, _tpl(request, "stations3d"))


def _local_maintenance_fallback(api: str, machines: list[str]):
    """Generate simple heuristic maintenance recommendations when backend isn't available.

    Heuristic:
      - Use /healthz when possible to detect degraded status
      - Otherwise emit time-based preventive tasks for each machine
    """
    recs = []
    try:
        # Best-effort health probe
        import requests as _rq
        health = None
        try:
            r = _rq.get(f"{api}/healthz", timeout=5)
            if r.ok:
                health = r.json()
        except Exception:
            health = None

        for m in machines:
            items = [
                {
                    "machine_id": m,
                    "type": "Preventive",
                    "task": "Inspect belts and rollers",
                    "priority": "Medium",
                    "due_in_hours": 72,
                },
                {
                    "machine_id": m,
                    "type": "Preventive",
                    "task": "Lubricate moving assemblies",
                    "priority": "Low",
                    "due_in_hours": 120,
                },
            ]
            # If health indicates degraded, add corrective
            if isinstance(health, dict) and not health.get("ok", True):
                items.append({
                    "machine_id": m,
                    "type": "Corrective",
                    "task": "Investigate backend health degradation",
                    "priority": "High",
                    "due_in_hours": 4,
                })
            recs.extend(items)
    except Exception:
        # On any unexpected error, still return generic recs
        for m in machines:
            recs.append({
                "machine_id": m,
                "type": "Preventive",
                "task": "General inspection",
                "priority": "Low",
                "due_in_hours": 96,
            })
    return recs