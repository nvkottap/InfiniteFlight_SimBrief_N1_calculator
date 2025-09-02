# app.py — Infinite Flight Takeoff N1 & Trim Estimator (Streamlit)
# v1.3.1 — Boeing dial refined, per-airframe N1 caps, compact flaps + trim diagrams

import re, json, math
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st

# ----------------------------- Robust calibration loader -----------------------------

CALIB_PATH = Path(__file__).with_name("calibrations.json")

_MINIMAL_FALLBACK = {
    "engines": {
        "generic": { "a": 95.0, "b_temp": -0.06, "c_pa": 0.24, "d_derate": -1.5, "w_ref": 300.0, "e_wt": 0.018 }
    },
    "airframes": {
        "generic": {
            "engine_id": "generic", "brand": "boeing",
            "mtow_klb": 300.0, "n1_max_pct": 101.5,
            "detents": ["0","5","15","25","30"],
            "baseline_trim": 8.0, "baseline_flaps": 5, "cg_ref": 25.0,
            "if_trim_default_pct": 12.0
        }
    }
}

def load_calibrations_unified() -> Dict[str, Any]:
    try:
        with open(CALIB_PATH, "r", encoding="utf-8") as f:
            raw = f.read()
        if not raw.strip().startswith("{"):
            raise ValueError("File does not start with '{' – likely not JSON.")
        return json.loads(raw)
    except FileNotFoundError:
        st.warning(f"`{CALIB_PATH.name}` not found. Using minimal built-in calibration.")
        return _MINIMAL_FALLBACK
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON in `{CALIB_PATH.name}` at line {e.lineno}, column {e.colno}: {e.msg}")
        try:
            with open(CALIB_PATH, "r", encoding="utf-8") as f:
                bad_lines = f.readlines()
            start = max(0, e.lineno - 3); end = min(len(bad_lines), e.lineno + 2)
            preview = "".join(f"{i+1:>4}: {bad_lines[i]}" for i in range(start, end))
            st.code(preview, language="json")
        except Exception:
            pass
        st.info("Falling back to a minimal internal calibration so the app can still run.")
        return _MINIMAL_FALLBACK
    except Exception as e:
        st.error(f"Could not read `{CALIB_PATH.name}`: {e}")
        return _MINIMAL_FALLBACK

CALIB = load_calibrations_unified()
ENGINES = CALIB.get("engines", {})
AIRFRAMES = CALIB.get("airframes", {})

# ---------------------------------- Utilities ----------------------------------------

def pressure_altitude_ft(elev_ft: float, qnh_inhg: float) -> float:
    """Approx pressure altitude from field elevation and QNH (inHg)."""
    # PA ≈ Elev + (29.92 - QNH)*1000* (1 inHg ≈ 1000 ft)
    return (elev_ft or 0.0) + (29.92 - float(qnh_inhg or 29.92)) * 1000.0

def derate_level_from_mode(mode: str) -> int:
    """Map thrust mode strings to step levels."""
    m = (mode or "").upper()
    if "D-TO2" in m or "DTO2" in m: return 2
    if "D-TO1" in m or "DTO1" in m: return 1
    if "D-TO"  in m or "DTO"  in m: return 1
    # FLEX maps to assumed temp; treat as derate step 1 by default, we use deltaT too
    if "FLEX" in m: return 1
    return 0

def get_flap_detents(brand: str, series: str) -> List[str]:
    af = AIRFRAMES.get(series, {})
    det = af.get("detents")
    if det: return det
    return ["0","1","1+F","2","3","FULL"] if (brand or "").lower()=="airbus" else ["0","1","2","5","10","15","25","30","40"]

def choose_selected_label(detents: List[str], flaps_numeric: int, brand: str) -> str:
    # For Airbus, force "1+F" if flaps ~1 (as we agreed earlier)
    if (brand or "").lower()=="airbus":
        if flaps_numeric <= 0: return "0"
        if flaps_numeric == 1: return "1+F"
        if flaps_numeric in (2,3): return str(flaps_numeric)
        return "FULL"
    # Boeing: choose the nearest detent not less than requested numeric
    try:
        nums = []
        for d in detents:
            n = float(d.replace("°","")) if d not in ("FULL","1+F") else (4.0 if d=="FULL" else 1.0)
            nums.append((d, n))
        target = float(flaps_numeric)
        cands = sorted(nums, key=lambda x: (x[1] < target, abs(x[1]-target)))
        # prefer >= target if available
        ge = [d for d,n in nums if n >= target]
        return ge[0] if ge else cands[0][0]
    except:
        return str(flaps_numeric)

def detent_numeric_value(label: str, brand: str) -> float:
    lab = (label or "").upper()
    if (brand or "").lower()=="airbus":
        if lab == "FULL": return 4.0
        if lab in ("3","2"): return float(lab)
        if lab in ("1","1+F"): return 1.0
        return 0.0
    try:
        return float(lab.replace("°",""))
    except:
        return 0.0

def apply_n1_overrides(series: str, flaps_label: str, tow_klb: float, n1_est: float) -> float:
    af = AIRFRAMES.get(series, {})
    overs = af.get("n1_overrides", [])
    if not overs: return n1_est
    mtow = float(af.get("mtow_klb", 0) or 0)
    if mtow <= 0: return n1_est
    wf = max(0.0, min(1.0, (tow_klb or 0.0)/mtow))
    cands = [ov for ov in overs if ov.get("flaps_label","").upper()==(flaps_label or "").upper()]
    if not cands: return n1_est
    best = min(cands, key=lambda ov: abs(wf - float(ov.get("weight_frac", 0.6))))
    try:
        target = float(best["n1_target"])
        return 0.55*n1_est + 0.45*target
    except:
        return n1_est

# --------------------------------- Estimators ----------------------------------------

def estimate_n1(engine_id: str, oat_c: float, sel_temp_c: float, qnh_inhg: float,
                elev_ft: float, tow_klb: float, bleeds_on: bool, anti_ice_on: bool,
                thrust_mode: str, brand: str, flaps: Optional[int],
                header_text: str, series: str) -> Dict[str, Any]:
    """
    Heuristic N1% (sim-use only):
      n1 = a + b*(SEL-OAT) + c*(PA_kft) + d*(derate_steps) + e*(TOW - w_ref) + flap_term [+ bleed/ice]
      Then: adjust derate effectiveness by weight, floor, apply per-airframe cap & overrides.
    """
    cal = ENGINES.get(engine_id, ENGINES.get("generic", {}))
    a = cal.get("a", 95.0); b = cal.get("b_temp", -0.06); c = cal.get("c_pa", 0.24)
    d = cal.get("d_derate", -1.5); w_ref = cal.get("w_ref", 300.0); e = cal.get("e_wt", 0.018)

    deltaT = (sel_temp_c - oat_c) if (sel_temp_c is not None and oat_c is not None) else 0.0
    pa_ft = pressure_altitude_ft(elev_ft or 0.0, qnh_inhg or 29.92)
    pa_kft = max(pa_ft, 0.0) / 1000.0
    derate_steps = derate_level_from_mode(thrust_mode or "")
    wt_term = ((tow_klb or w_ref) - w_ref)

    detents = get_flap_detents(brand, series)
    sel_label = choose_selected_label(detents, int(flaps or 0), brand)
    sel_val = detent_numeric_value(sel_label or ("1" if (brand or "").lower()=="airbus" else "5"), brand)
    base_val = 1.0 if (brand or "").lower()=="airbus" else 5.0

    # Base model
    n1 = a + b*deltaT + c*pa_kft + d*derate_steps + e*wt_term
    if bleeds_on:   n1 += 0.2
    if anti_ice_on: n1 += 0.5
    n1 += 0.35 * max(0.0, sel_val - base_val)  # flap drag compensation

    # Weight-dependent derate effectiveness (apply AFTER base n1 is defined)
    af = AIRFRAMES.get(series, {})
    mtow = float(af.get("mtow_klb", 0) or 0)
    wf = max(0.0, min(1.0, (tow_klb or 0.0) / mtow)) if mtow > 0 else 0.6
    n1 += 0.8 * derate_steps * (wf - 0.6)

    # Guardrail floor + per-airframe cap
    weight_ratio = max(0.8, (tow_klb or w_ref) / w_ref)
    n1_floor = 88.0 + 5.5*weight_ratio + 0.35*pa_kft + 0.25*max(0.0, sel_val - base_val)
    n1 = max(n1, n1_floor)
    n1_cap = float(af.get("n1_max_pct", 103.0))
    n1 = min(n1, n1_cap)

    # Blend toward any airframe overrides
    n1 = apply_n1_overrides(series, sel_label or "", tow_klb, n1)

    conf_pm = 0.6 + 0.06*pa_kft + 0.15*derate_steps
    return {"n1": n1, "conf_pm": conf_pm, "pa_ft": pa_ft, "flaps_label": sel_label}

def estimate_trim_if(series: str, brand: str, flaps_label: str, tow_klb: float) -> float:
    """
    Infinite Flight trim % (−100..+100), NU positive.
    Uses per-airframe `if_trim_default_pct` → small weight & flap nudges → optional overrides.
    """
    af = AIRFRAMES.get(series, {})
    base_map = { "boeing": 12.0, "airbus": 8.0 }
    base = float(af.get("if_trim_default_pct", base_map.get((brand or "").lower(), 10.0)))

    mtow = float(af.get("mtow_klb", 0) or 0)
    wf = max(0.0, min(1.0, (tow_klb or 0.0) / mtow)) if mtow > 0 else 0.6
    k_w = 22.0 if (brand or "").lower()=="boeing" else 16.0
    base += k_w * (wf - 0.6)

    def detent_numeric(lab: str) -> float:
        lab = (lab or "").upper()
        if (brand or "").lower()=="airbus":
            if lab in ("FULL",): return 4.0
            if lab in ("3",): return 3.0
            if lab in ("2",): return 2.0
            if lab in ("1+F","1"): return 1.0
            return 0.0
        try: return float(lab.replace("°",""))
        except: return 0.0

    sel_val = detent_numeric(flaps_label)
    base_val = 1.0 if (brand or "").lower()=="airbus" else 5.0
    base += 0.9 * max(0.0, sel_val - base_val)

    overs = af.get("trim_overrides", [])
    if overs and mtow > 0:
        cands = [o for o in overs if o.get("flaps_label","").upper()==(flaps_label or "").upper()]
        if cands:
            best = min(cands, key=lambda o: abs(wf - float(o.get("weight_frac", 0.6))))
            try:
                target = float(best["if_trim_pct"])
                base = 0.6*base + 0.4*target
            except:
                pass

    return max(-100.0, min(100.0, round(base, 1)))

# --------------------------------- Dial renderers ------------------------------------

def draw_n1_dial_boeing(n1_percent: float, max_n1_pct: float = 102.0):
    """Boeing-style N1 dial (thin bezel, inward yellow chevron, square readout box)."""
    def pct_to_scale(p): return max(0.0, min(10.0, (p / 100.0) * 10.0))
    n1_sc = pct_to_scale(n1_percent); max_sc = pct_to_scale(max_n1_pct)

    start_deg = 0.0                          # 3 o'clock
    end_deg   = -225.0 * min(1.0, max_sc/10.0)  # to ~10:30
    to_rad    = np.deg2rad

    bg = "#0a0f14"; white = "#ffffff"
    bezel_color = white; tick_color = white; label_color = white
    red_tick = "#d84b4b"; chevron = "#ffd21f"

    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    ax.set_aspect("equal"); ax.axis("off")
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)

    R_ring = 0.98; ring_lw = 2.6
    tick_len_major = 0.11; tick_len_minor = 0.06
    font_small = 9; font_big = 15

    def scale_to_angle(v):
        frac = max(0.0, min(1.0, v / max(1e-6, max_sc)))
        return to_rad(start_deg + frac * (end_deg - start_deg))

    # Bezel
    theta = np.linspace(to_rad(start_deg), to_rad(end_deg), 720)
    ax.plot(R_ring*np.cos(theta), R_ring*np.sin(theta), color=bezel_color,
            linewidth=ring_lw, solid_capstyle='round', zorder=2)

    # Internal ticks & even-number labels
    for v in np.arange(0, 10.1, 1.0):
        if v > max_sc + 1e-6: continue
        ang = scale_to_angle(v)
        is_major = (int(v) % 2 == 0)
        tlen = tick_len_major if is_major else tick_len_minor
        x1, y1 = R_ring*np.cos(ang), R_ring*np.sin(ang)
        x2, y2 = (R_ring - tlen)*np.cos(ang), (R_ring - tlen)*np.sin(ang)
        ax.plot([x1, x2], [y1, y2], color=tick_color, linewidth=2 if is_major else 1.2, zorder=3)
        if is_major:
            rl = R_ring - tlen - 0.08
            ax.text(rl*np.cos(ang), rl*np.sin(ang), f"{int(v)}",
                    ha="center", va="center", fontsize=font_small, color=label_color, zorder=4)

    # Red max tick
    ang_max = scale_to_angle(max_sc)
    x1m, y1m = R_ring*np.cos(ang_max), R_ring*np.sin(ang_max)
    x2m, y2m = (R_ring - tick_len_major - 0.02)*np.cos(ang_max), (R_ring - tick_len_major - 0.02)*np.sin(ang_max)
    ax.plot([x1m, x2m], [y1m, y2m], color=red_tick, linewidth=3, zorder=4)

    # Yellow chevron (outline), sharp, pointing inward
    ang_n1  = scale_to_angle(min(max(n1_sc, 0.0), max_sc))
    tip_r   = R_ring - 0.008
    base_r  = R_ring + 0.045
    spread  = math.radians(3.0)
    tip     = np.array([tip_r*np.cos(ang_n1),  tip_r*np.sin(ang_n1)])
    left    = np.array([base_r*np.cos(ang_n1 + spread), base_r*np.sin(ang_n1 + spread)])
    right   = np.array([base_r*np.cos(ang_n1 - spread), base_r*np.sin(ang_n1 - spread)])
    ax.plot([left[0],  tip[0]],  [left[1],  tip[1]],  color=chevron, linewidth=2.4, solid_capstyle='round', zorder=6)
    ax.plot([right[0], tip[0]], [right[1], tip[1]], color=chevron, linewidth=2.4, solid_capstyle='round', zorder=6)

    # Square digital box at 0 mark (lifted)
    ang0 = scale_to_angle(0.0)
    anchor_r = R_ring + 0.34
    anchor_x, anchor_y = anchor_r*np.cos(ang0), anchor_r*np.sin(ang0)
    box_w, box_h = 0.60, 0.26
    ll = (anchor_x - box_w/2, anchor_y)
    rect = patches.Rectangle(ll, box_w, box_h, linewidth=2, edgecolor=white, facecolor=(0,0,0,0.0), zorder=10)
    ax.add_patch(rect)
    cx, cy = (anchor_x, anchor_y + box_h/2)
    ax.text(cx, cy + 0.01, f"{(n1_percent/10):.1f}" if False else f"{(n1_percent/100.0)*10.0:.1f}",
            ha="center", va="center", fontsize=font_big, fontweight="bold", color=white, zorder=11)

    return fig

def draw_n1_dial_airbus(n1_percent: float, max_n1_pct: float = 100.0):
    """Airbus-style N1 dial: thin arc + inner ticks, small white needle, green centered readout."""
    def pct_to_scale(p): return max(0.0, min(10.0, (p / 100.0) * 10.0))
    n1_sc = pct_to_scale(n1_percent); max_sc = pct_to_scale(max_n1_pct)

    start_deg = -225.0; end_deg = -45.0; to_rad = np.deg2rad
    bg = "#0a0f14"; white = "#ffffff"; green = "#7CFF5A"; red = "#d84b4b"

    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    ax.set_aspect("equal"); ax.axis("off")
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)

    R_outer = 1.00; ring_lw = 2.6; tick_len = 0.06
    font_mark = 10; font_val = 18

    def scale_to_angle(v):
        frac = max(0.0, min(1.0, v/10.0))
        return to_rad(start_deg + frac * (end_deg - start_deg))

    theta = np.linspace(to_rad(start_deg), to_rad(end_deg), 720)
    ax.plot(R_outer*np.cos(theta), R_outer*np.sin(theta), color=white,
            linewidth=ring_lw, solid_capstyle='round', zorder=2)

    for v in np.arange(0, 10.1, 1.0):
        ang = scale_to_angle(v)
        x1, y1 = R_outer*np.cos(ang), R_outer*np.sin(ang)
        x2, y2 = (R_outer - tick_len)*np.cos(ang), (R_outer - tick_len)*np.sin(ang)
        ax.plot([x1, x2], [y1, y2], color=white, linewidth=1.4, zorder=3)
        if abs(v-5.0) < 0.01 or abs(v-10.0) < 0.01:
            rl = R_outer - tick_len - 0.10
            ax.text(rl*np.cos(ang), rl*np.sin(ang), f"{int(v)}",
                    ha="center", va="center", fontsize=font_mark, color=white, zorder=4)

    # Red overboost segment: last ~0.8 near max
    over_lo = max(0.0, 9.2); over_hi = min(10.0, max_sc)
    if over_hi > over_lo:
        t_over = np.linspace(scale_to_angle(over_lo), scale_to_angle(over_hi), 60)
        ax.plot(R_outer*np.cos(t_over), R_outer*np.sin(t_over),
                color=red, linewidth=ring_lw+0.6, solid_capstyle='butt', zorder=5)

    # Small white needle
    ang_n1 = scale_to_angle(n1_sc)
    needle_r1 = 0.78; needle_r2 = R_outer - 0.04
    ax.plot([needle_r1*np.cos(ang_n1), needle_r2*np.cos(ang_n1)],
            [needle_r1*np.sin(ang_n1), needle_r2*np.sin(ang_n1)],
            color=white, linewidth=2.2, solid_capstyle='round', zorder=6)

    # Centered green readout
    ax.text(0, 0.08, "N1%", ha="center", va="center", fontsize=12, color=green, zorder=7)
    ax.text(0, -0.02, f"{n1_percent:.1f}", ha="center", va="center",
            fontsize=font_val, fontweight="bold", color=green, zorder=7)
    ax.text(0, -0.13, "%", ha="center", va="center", fontsize=12, color=green, zorder=7)

    return fig

# ----------------------------- Compact flaps + trim diagrams --------------------------

def draw_flap_detents_small(brand: str, detents: List[str], selected_label: str):
    fig, ax = plt.subplots(figsize=(2.2, 3.0))
    ax.set_facecolor("#0a0f14"); fig.patch.set_facecolor("#0a0f14")
    for s in ax.spines.values(): s.set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.set_ylim(-0.5, len(detents) - 0.5); ax.set_xlim(0, 1)
    ax.plot([0.5, 0.5], [0, len(detents)-1], color="#ffffff", linewidth=2.0)
    for i, lab in enumerate(detents):
        ax.plot([0.42, 0.58], [i, i], color="#ffffff", linewidth=1.6)
        ax.text(0.62, i, lab, va="center", ha="left", fontsize=10, color="#ffffff")
    if selected_label in detents:
        idx = detents.index(selected_label)
        ax.add_patch(plt.Rectangle((0.25, idx-0.35), 0.5, 0.7, fill=False,
                                   edgecolor="#ffd21f", linewidth=2.0))
    ax.text(0.5, len(detents)-0.2, "FLAPS", ha="center", va="bottom",
            fontsize=10, color="#ffffff", fontweight="bold")
    ax.text(0.5, -0.2, f"{selected_label}", ha="center", va="top", fontsize=10, color="#ffffff")
    ax.set_yticks([])
    return fig

def draw_trim_bar(trim_pct: float):
    trim_pct = max(-100.0, min(100.0, float(trim_pct or 0)))
    fig, ax = plt.subplots(figsize=(2.0, 3.0))
    ax.set_facecolor("#0a0f14"); fig.patch.set_facecolor("#0a0f14")
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xlim(0, 1); ax.set_ylim(-100, 100); ax.get_xaxis().set_visible(False)
    ax.plot([0.5, 0.5], [-100, 100], color="#ffffff", linewidth=2.2)
    for t in range(-100, 101, 20):
        ax.plot([0.42, 0.58], [t, t], color="#ffffff", linewidth=1.2)
    ax.plot([0.38, 0.62], [0, 0], color="#ffffff", linewidth=2.0)
    ax.plot([0.5], [trim_pct], marker="o", markersize=8, markerfacecolor="#ffd21f", markeredgecolor="#ffd21f")
    ax.text(0.5, 105, "TRIM", ha="center", va="bottom", fontsize=10, color="#ffffff", fontweight="bold")
    ax.text(0.5, -115, f"{trim_pct:.0f}%", ha="center", va="top", fontsize=10, color="#ffffff")
    ax.set_yticks([])
    return fig

# --------------------------------- Paste parser --------------------------------------

PAT_FIELD = re.compile(r"\b(OAT|QNH|ELEV|WEIGHT|SEL\s*TEMP|THRUST|FLAPS|APT|RWY|BLEEDS|A/ICE)\b[:\s]+([A-Z0-9+\-./]+)", re.I)
PAT_HEADER = re.compile(r"TAKEOFF PERFORMANCE\s+([A-Z0-9\-]+)\s+([A-Z0-9\-\s/]+)\s+([A-Z0-9\-\s/]+)", re.I)

def parse_block(text: str) -> Dict[str, Any]:
    out = {}
    m = PAT_HEADER.search(text or "")
    if m:
        out["reg"] = m.group(1).strip()
        # m.group(2) likely type (e.g., A350-900)
        out["type"] = m.group(2).strip()
        out["engine_text"] = m.group(3).strip()
    for k, v in PAT_FIELD.findall(text or ""):
        key = k.upper().replace(" ", "")
        out[key] = v
    # Also capture numeric V1/VR/V2 if needed later (not used for N1 calc)
    return out

def infer_series_and_brand(ac_type: str) -> (str, str):
    t = (ac_type or "").upper()
    if "737" in t and "MAX" in t: return "boeing_737_max8", "boeing"
    if "777" in t: return "boeing_777", "boeing"
    if "787" in t: return "boeing_787", "boeing"
    if "757" in t: return "boeing_757", "boeing"
    if "A380" in t: return "airbus_a380", "airbus"
    if "A350" in t: return "airbus_a350", "airbus"
    if "A321" in t: return "airbus_a321", "airbus"
    if "A320" in t: return "airbus_a320", "airbus"
    if "A330" in t: return "airbus_a330", "airbus"
    return "generic", "boeing"

# ------------------------------------ UI ---------------------------------------------

st.set_page_config(page_title="IF N1 & Trim", page_icon="✈️", layout="wide")
st.title("Infinite Flight • Takeoff N1% & Trim Estimator")

st.caption("Paste your SimBrief-style **TAKEOFF PERFORMANCE** block below, then click **Estimate N1%**.")

paste = st.text_area("Paste block", height=260, placeholder="Paste your TAKEOFF PERFORMANCE text here...")
if st.button("Estimate N1%", type="primary"):
    data = parse_block(paste)
    missing_keys = []
    for req in ("OAT","QNH","ELEV","WEIGHT","SELTEMP","THRUST","FLAPS"):
        if req not in data:
            # accept alternative "SEL TEMP" w/ space already normalized to SELTEMP above
            if req=="SELTEMP" and "SELTEMP" in (key.replace(" ","") for key in data.keys()):
                continue
            missing_keys.append(req)
    if missing_keys:
        st.error("Some inputs are missing. Ensure your paste includes OAT, QNH, ELEV, WEIGHT, SEL TEMP, THRUST, and FLAPS.")
        st.stop()

    # Resolve inputs
    ac_type = data.get("type","")
    series, brand = infer_series_and_brand(ac_type)
    engine_id = AIRFRAMES.get(series, {}).get("engine_id", "generic")

    try:
        oat = float(data.get("OAT"))
        qnh = float(data.get("QNH"))
        elev = float(data.get("ELEV"))
        tow_klb = float(data.get("WEIGHT"))
        sel_temp = float(data.get("SELTEMP") or data.get("SELTEMP", "0"))
    except Exception:
        st.error("Could not parse numeric values (OAT, QNH, ELEV, WEIGHT, SEL TEMP).")
        st.stop()

    thrust_mode = (data.get("THRUST") or "").upper()
    bleeds_on = (data.get("BLEEDS","ON").upper() == "ON")
    anti_ice_on = (data.get("A/ICE","OFF").upper() == "ON")

    try:
        flaps_val = int(data.get("FLAPS") or "5")
    except:
        flaps_val = 5

    # Estimate N1
    res = estimate_n1(
        engine_id=engine_id, oat_c=oat, sel_temp_c=sel_temp, qnh_inhg=qnh,
        elev_ft=elev, tow_klb=tow_klb, bleeds_on=bleeds_on, anti_ice_on=anti_ice_on,
        thrust_mode=thrust_mode, brand=brand, flaps=flaps_val,
        header_text=paste, series=series
    )
    n1 = res["n1"]; conf = res["conf_pm"]; pa_ft = res["pa_ft"]; selected_flaps_label = res["flaps_label"]

    # Estimate Trim
    trim_pct = estimate_trim_if(series=series, brand=brand, flaps_label=selected_flaps_label, tow_klb=tow_klb)

    # Performance card
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Aircraft", ac_type or series.replace("_"," ").title())
    colB.metric("Thrust Mode", thrust_mode or "TO/GA")
    colC.metric("Selected Temp", f"{sel_temp:.0f}°C")
    colD.metric("Pressure Alt", f"{pa_ft:.0f} ft")

    # Dial + small diagrams
    col_dial, col_right = st.columns([3, 2], gap="large")
    with col_dial:
        max_cap = AIRFRAMES.get(series, {}).get("n1_max_pct", 102.0)
        if (brand or "").lower()=="airbus":
            st.pyplot(draw_n1_dial_airbus(n1_percent=n1, max_n1_pct=max_cap), use_container_width=False)
        else:
            st.pyplot(draw_n1_dial_boeing(n1_percent=n1, max_n1_pct=max_cap), use_container_width=False)

    with col_right:
        s1, s2 = st.columns(2)
        with s1:
            detents = AIRFRAMES.get(series, {}).get("detents", get_flap_detents(brand, series))
            st.pyplot(draw_flap_detents_small(brand, detents, selected_flaps_label), use_container_width=False)
        with s2:
            st.pyplot(draw_trim_bar(trim_pct), use_container_width=False)

    # Big, easy-to-read headline (top-center style)
    st.markdown(
        f"""
        <div style="text-align:center; font-size:1.6rem; font-weight:800; margin-top:-0.5rem;">
            N1% = {n1:.1f}% &nbsp;&nbsp;&nbsp; Trim = {trim_pct:.0f}%
        </div>
        """, unsafe_allow_html=True
    )
