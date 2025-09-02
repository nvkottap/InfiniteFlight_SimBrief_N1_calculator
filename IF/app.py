# app.py
# Infinite Flight – Takeoff N1% & Trim Estimator (Sim-use only)
# Adds IF Trim estimation (NU/ND %) from brand + flaps + weight + CG% (if found)

import re
import json
import math
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# Global style (seaborn)
# --------------------------
sns.set_theme(style="whitegrid", context="talk")

# --------------------------
# Engine calibration (heuristic; sim-use only)
# --------------------------
CAL_PATH = Path(__file__).with_name("engine_calibrations.json")

def load_calibrations() -> Dict[str, Any]:
    if CAL_PATH.exists():
        with open(CAL_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "leap-1b28": { "a": 93.5, "b_temp": -0.10, "c_pa": 0.25, "d_derate": -2.0, "w_ref": 160.0, "e_wt": 0.01 },
        "ge90-94b":  { "a": 95.0, "b_temp": -0.11, "c_pa": 0.20, "d_derate": -2.0, "w_ref": 460.0, "e_wt": 0.01 },
        "trent-970-84": { "a": 96.7, "b_temp": -0.15, "c_pa": 0.15, "d_derate": 0.0, "w_ref": 800.0, "e_wt": 0.002 },
        "trent-xwb-84": { "a": 91.6, "b_temp": -0.10, "c_pa": 0.18, "d_derate": 0.0, "w_ref": 590.0, "e_wt": 0.005 },
        "cfm56-5b3": { "a": 93.0, "b_temp": -0.10, "c_pa": 0.18, "d_derate": 0.0, "w_ref": 158.0, "e_wt": 0.01 },
        "cfm56-5b4": { "a": 92.0, "b_temp": -0.12, "c_pa": 0.20, "d_derate": 0.0, "w_ref": 148.0, "e_wt": 0.01 },
        "pw2037":    { "a": 94.0, "b_temp": -0.10, "c_pa": 0.20, "d_derate": -2.0, "w_ref": 235.0, "e_wt": 0.01 },
        "generic":   { "a": 92.0, "b_temp": -0.10, "c_pa": 0.20, "d_derate": -2.0, "w_ref": 300.0, "e_wt": 0.00 }
    }

CAL = load_calibrations()

ENGINE_ALIASES = {
    "LEAP-1B28": "leap-1b28",
    "GE90-94B": "ge90-94b",
    "TRENT 970-84": "trent-970-84",
    "TRENT XWB-84": "trent-xwb-84",
    "CFM56-5B3": "cfm56-5b3",
    "CFM56-5B4": "cfm56-5b4",
    "PW2037": "pw2037"
}

ENGINE_PRETTY = {
    "leap-1b28": "LEAP-1B",
    "ge90-94b": "GE90-94B",
    "trent-970-84": "Trent 970-84",
    "trent-xwb-84": "Trent XWB-84",
    "cfm56-5b3": "CFM56-5B3/P",
    "cfm56-5b4": "CFM56-5B4/P",
    "pw2037": "PW2037",
    "generic": "Generic"
}

# --------------------------
# Brand detection (Airbus vs Boeing)
# --------------------------
AIRBUS_TOKENS  = [
    r"\bA3(18|19|20|21)\b", r"\bA(330|340|350|380)\b",
    r"\bA320-?200\b", r"\bA321-?200\b", r"\bA350-900\b", r"\bA380-800\b"
]
BOEING_TOKENS  = [
    r"\bB(737|747|757|767|777|787)\b", r"\b737\s*MAX\b", r"\bMAX\s*8\b", r"\b777-200\b", r"\b757-200\b"
]

def detect_brand(text: str) -> str:
    t = (text or "").upper()
    for pat in AIRBUS_TOKENS:
        if re.search(pat, t, flags=re.I): return "airbus"
    for pat in BOEING_TOKENS:
        if re.search(pat, t, flags=re.I): return "boeing"
    if "TRENT XWB" in t or "CFM56-5B" in t: return "airbus"
    return "boeing"

# --------------------------
# Flaps detents (for guide)
# --------------------------
AIRBUS_DETENTS = ["0", "1", "1+F", "2", "3", "FULL"]
BOEING_DETENTS_DEFAULT = ["0", "1", "2", "5", "10", "15", "25", "30", "40"]
BOEING_FAMILY_DETENTS = {
    "737": ["0", "1", "2", "5", "10", "15", "25", "30", "40"],
    "757": ["0", "1", "5", "15", "20", "25", "30"],
    "767": ["0", "5", "15", "20", "25", "30"],
    "777": ["0", "5", "15", "20", "25", "30"],
    "787": ["0", "5", "10", "15", "17", "18", "20", "25", "30"],  # tunable
}

def get_boeing_series(header_or_text: str) -> Optional[str]:
    t = (header_or_text or "").upper()
    for key in BOEING_FAMILY_DETENTS.keys():
        if re.search(rf"\b{key}\b", t): return key
    if re.search(r"\bMAX\s*8\b", t) or re.search(r"\b737\s*MAX\b", t): return "737"
    return None

def get_flap_detents(brand: str, header_or_text: str) -> list[str]:
    if brand == "airbus": return AIRBUS_DETENTS
    series = get_boeing_series(header_or_text) or ""
    return BOEING_FAMILY_DETENTS.get(series, BOEING_DETENTS_DEFAULT)

# --------------------------
# Parsing & helpers
# --------------------------
def detect_engine_in_text(txt: str) -> Tuple[str, Optional[str]]:
    for token, eng_id in ENGINE_ALIASES.items():
        if re.search(rf"\b{re.escape(token)}\b", txt, flags=re.I):
            return eng_id, token
    return "generic", None

def pressure_altitude_ft(field_elev_ft: float, qnh_inhg: float) -> float:
    return float(field_elev_ft) + (29.92 - float(qnh_inhg)) * 1000.0

def parse_cg_percent(txt: str) -> Optional[float]:
    """
    Try to extract a %MAC CG from SimBrief text. Looks for TOCG, ZFWCG, MACZFW, etc.
    Returns a float (e.g., 24.3) or None.
    """
    patterns = [
        r"\bTOCG\s+(\d{1,2}\.?\d*)\s*%", r"\bZFWCG\s+(\d{1,2}\.?\d*)\s*%",
        r"\bMACZFW\s+(\d{1,2}\.?\d*)\s*%", r"\bCG\s+(\d{1,2}\.?\d*)\s*%MAC"
    ]
    t = txt.replace(",", " ")
    for pat in patterns:
        m = re.search(pat, t, flags=re.I)
        if m: 
            try:
                val = float(m.group(1))
                if 5.0 <= val <= 45.0:  # sanity range for %MAC in airliners
                    return val
            except:
                pass
    return None

def parse_airframe_series(header_or_text: str) -> str:
    t = (header_or_text or "").upper()
    # Airbus
    if re.search(r"\bA321\b", t): return "airbus_a321"
    if re.search(r"\bA320\b", t): return "airbus_a320"
    if re.search(r"\bA350\b", t): return "airbus_a350"
    if re.search(r"\bA380\b", t): return "airbus_a380"
    # Boeing
    if re.search(r"\b737\b", t):  return "boeing_737"
    if re.search(r"\b757\b", t):  return "boeing_757"
    if re.search(r"\b777\b", t):  return "boeing_777"
    if re.search(r"\b787\b", t):  return "boeing_787"
    return "generic"

def parse_takeoff_text(txt: str) -> Dict[str, Any]:
    d: Dict[str, Any] = {}
    header_line = next((l for l in txt.splitlines() if re.search(r"\b[A-Z0-9]{3,}.*", l)), "")
    d["header"] = header_line
    d["brand"] = detect_brand(header_line or txt)
    d["series"] = parse_airframe_series(header_line or txt)

    # Airport/runway
    apt = re.search(r"APT\s+([A-Z]{4})/([A-Z]{3})", txt)
    d["apt_icao"] = apt.group(1) if apt else None
    rwy = re.search(r"RWY\s+(\d+[LRC]?)/\+?0?", txt)
    d["runway"] = rwy.group(1) if rwy else None

    # Weather
    m = re.search(r"OAT\s+(-?\d+)", txt);        d["oat_c"] = float(m.group(1)) if m else None
    m = re.search(r"QNH\s+(\d{2}\.\d{2})", txt); d["qnh_inhg"] = float(m.group(1)) if m else None
    m = re.search(r"ELEV\s+(-?\d+)", txt);       d["elev_ft"] = float(m.group(1)) if m else None

    # Inputs/outputs
    m = re.search(r"WEIGHT\s+(\d+\.?\d*)", txt); d["tow_klb"] = float(m.group(1)) if m else None
    m = re.search(r"FLAPS\s+(\d+)", txt);        d["flaps"] = int(m.group(1)) if m else None
    m = re.search(r"THRUST\s+(FLEX|D-TO2|D-TO1|D-TO)", txt); d["thrust_mode"] = m.group(1) if m else None
    m = re.search(r"SEL\s+TEMP\s+(\d+)", txt);   d["sel_temp_c"] = float(m.group(1)) if m else None
    m = re.search(r"BLEEDS\s+(ON|OFF)", txt);    d["bleeds_on"] = (m.group(1) == "ON") if m else True
    m = re.search(r"A/ICE\s+(ON|OFF)", txt);     d["anti_ice_on"] = (m.group(1) == "ON") if m else False
    m = re.search(r"RWY\s+COND\s+(DRY|WET|SLUSH|ICE)", txt); d["rwy_cond"] = m.group(1) if m else "DRY"

    # V-speeds
    v1 = re.search(r"V1\s+(\d+)", txt); d["v1"] = int(v1.group(1)) if v1 else None
    vr = re.search(r"VR\s+(\d+)", txt); d["vr"] = int(vr.group(1)) if vr else None
    v2 = re.search(r"V2\s+(\d+)", txt); d["v2"] = int(v2.group(1)) if v2 else None

    # Engine detection
    eng_id, token = detect_engine_in_text(txt)
    d["engine_id"] = eng_id
    d["engine_token"] = token
    d["engine_pretty"] = ENGINE_PRETTY.get(eng_id, "Generic")

    # Try to parse CG %MAC from full text (OFP often includes TOCG on another page)
    d["cg_percent_mac"] = parse_cg_percent(txt)

    return d

def derate_level_from_mode(mode: str) -> int:
    if not mode: return 0
    m = mode.upper()
    if m == "D-TO2": return 2
    if m == "D-TO1": return 1
    if m == "D-TO":  return 1
    return 0  # FLEX or unknown

def estimate_n1(engine_id: str, oat_c: float, sel_temp_c: float, qnh_inhg: float,
                elev_ft: float, tow_klb: float, bleeds_on: bool, anti_ice_on: bool,
                thrust_mode: str) -> Dict[str, Any]:
    cal = CAL.get(engine_id, CAL.get("generic"))
    a = cal["a"]; b = cal["b_temp"]; c = cal["c_pa"]; d = cal["d_derate"]; w_ref = cal["w_ref"]; e = cal["e_wt"]
    deltaT = (sel_temp_c - oat_c) if (sel_temp_c is not None and oat_c is not None) else 0.0
    pa_ft = pressure_altitude_ft(elev_ft or 0.0, qnh_inhg or 29.92)
    pa_kft = max(pa_ft, 0.0) / 1000.0
    derate_steps = derate_level_from_mode(thrust_mode or "")
    wt_term = ((tow_klb or w_ref) - w_ref)
    n1 = a + b*deltaT + c*pa_kft + d*derate_steps + e*wt_term
    if bleeds_on:   n1 += 0.2
    if anti_ice_on: n1 += 0.5
    conf_pm = 0.4 + 0.05*pa_kft + 0.1*derate_steps
    return {"n1": n1, "conf_pm": conf_pm, "pa_ft": pa_ft}

# --------------------------
# IF Trim estimation (heuristic; sim-use only)
# --------------------------
# Baseline trims (% IF UI) at typical takeoff flaps and nominal CG.
# Values are heuristic starting points for Infinite Flight, not real aircraft "units".
BASE_TRIM = {
    "boeing_737":  {"baseline_flaps": 5, "baseline_trim": 8.0, "cg_ref": 24.0},
    "boeing_757":  {"baseline_flaps": 5, "baseline_trim": 7.0, "cg_ref": 24.0},
    "boeing_777":  {"baseline_flaps": 5, "baseline_trim": 6.0, "cg_ref": 28.0},
    "boeing_787":  {"baseline_flaps": 5, "baseline_trim": 6.0, "cg_ref": 28.0},
    "airbus_a320": {"baseline_flaps": 1, "baseline_trim": 5.0, "cg_ref": 25.0},
    "airbus_a321": {"baseline_flaps": 1, "baseline_trim": 6.0, "cg_ref": 25.0},
    "airbus_a350": {"baseline_flaps": 1, "baseline_trim": 4.0, "cg_ref": 25.0},
    "airbus_a380": {"baseline_flaps": 1, "baseline_trim": 4.0, "cg_ref": 25.0},
    "generic":     {"baseline_flaps": 5, "baseline_trim": 6.0, "cg_ref": 25.0},
}

def nearest_label_by_value(detents: list[str], value: int) -> Optional[str]:
    numeric_pairs = []
    for lab in detents:
        try:
            numeric_pairs.append((lab, float(lab.replace("°", ""))))
        except:
            pass
    if not numeric_pairs: return None
    best = min(numeric_pairs, key=lambda p: abs(p[1] - value))
    return best[0]

def choose_selected_label(detents: list[str], numeric_flaps: Optional[int], brand: str) -> Optional[str]:
    if numeric_flaps is None: return None
    if brand == "airbus":
        if numeric_flaps >= 4: return "FULL" if "FULL" in detents else detents[-1]
        if numeric_flaps == 3 and "3" in detents: return "3"
        if numeric_flaps == 2 and "2" in detents: return "2"
        if numeric_flaps == 1: return "1+F" if "1+F" in detents else ("1" if "1" in detents else detents[1])
        return "0" if "0" in detents else detents[0]
    return nearest_label_by_value(detents, numeric_flaps)

def detent_numeric_value(label: str, brand: str) -> float:
    """Map detent label to a numeric angle-ish value for deltas."""
    t = label.upper().strip()
    if brand == "airbus":
        # Treat FULL as 4, 1+F ~ 1
        if t == "FULL": return 4.0
        if t == "1+F":  return 1.0
        try: return float(t)
        except: return 1.0
    else:
        try: return float(t.replace("°",""))
        except: return 5.0

def estimate_if_trim(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Heuristic Infinite Flight trim (%) for takeoff:
      - Start from airframe baseline (brand + typical flap)
      - Adjust for CG %MAC delta from a reference CG
      - Adjust for actual flap vs baseline flap
      - Small weight tweak vs engine calibration reference weight

    Output: dict { 'if_trim_pct': float (NU positive), 'label': 'NU'/'ND' }
    """
    series = data.get("series", "generic")
    brand  = data.get("brand", "boeing")
    flaps  = int(data.get("flaps") or 0)
    cg_pct = data.get("cg_percent_mac")  # may be None
    tow_klb = float(data.get("tow_klb") or 0.0)

    base = BASE_TRIM.get(series, BASE_TRIM["generic"])
    trim = base["baseline_trim"]
    cg_ref = base["cg_ref"]
    baseline_flaps = base["baseline_flaps"]

    # Use our flaps detent guide to interpret labels consistently
    detents = get_flap_detents(brand, data.get("header",""))
    selected_label = choose_selected_label(detents, flaps, brand) or (str(flaps) if brand=="boeing" else "1")
    selected_val = detent_numeric_value(selected_label, brand)
    baseline_val = float(baseline_flaps)

    # --- CG adjustment (forward CG => more NU trim)
    # sensitivity ~ +1.2% IF trim per 1.0 %MAC forward of reference
    if cg_pct is not None:
        cg_delta = (cg_ref - float(cg_pct))  # positive if CG is forward of ref
        trim += 1.2 * cg_delta

    # --- Flaps adjustment
    # More flaps (higher angle) usually needs LESS NU trim at rotate.
    # Heuristic: -0.25% IF trim per unit of flaps above baseline (Boeing degrees or Airbus steps)
    trim += -0.25 * (selected_val - baseline_val)

    # --- Weight tweak (vs engine-cal w_ref)
    w_ref = CAL.get(data.get("engine_id","generic"), CAL["generic"])["w_ref"]
    trim += 0.3 * ((tow_klb - w_ref) / 50.0)  # +0.3% per 50k lb above ref

    # Clamp to a sane sim range
    trim = max(-20.0, min(20.0, trim))

    label = "NU" if trim >= 0 else "ND"
    return {"if_trim_pct": trim, "label": label, "selected_flaps_label": selected_label}

# --------------------------
# Dials (readout below) & Flaps guide
# --------------------------
def draw_n1_dial_boeing(n1: float, conf_pm: float, show_title=False):
    import numpy as np
    min_n1, max_n1 = 0, 110
    start_deg, end_deg = -210, 30
    def clamp(v, lo, hi): return max(min(v, hi), lo)
    def n1_to_angle(v):
        frac = (clamp(v, min_n1, max_n1) - min_n1) / (max_n1 - min_n1)
        return math.radians(start_deg + frac * (end_deg - start_deg))
    bg, bezel, tick, label = "#0a0f14", (1,1,1,0.05), "#e8edf6", "#e8edf6"
    normal, bug, band = "#3ccc8c", "#ffd21f", (1,1,1,0.14)
    fig, ax = plt.subplots(figsize=(3.2, 3.2)); ax.set_aspect("equal"); ax.axis("off")
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    R_outer, R_inner = 1.00, 0.74
    theta = np.linspace(math.radians(start_deg), math.radians(end_deg), 240)
    ax.fill(np.r_[R_outer*np.cos(theta), R_inner*np.cos(theta[::-1])],
            np.r_[R_outer*np.sin(theta), R_inner*np.sin(theta[::-1])], color=bezel)
    # Normal band
    t_band = np.linspace(n1_to_angle(80), n1_to_angle(102), 120)
    ax.fill(np.r_[R_outer*np.cos(t_band), R_inner*np.cos(t_band[::-1])],
            np.r_[R_outer*np.sin(t_band), R_inner*np.sin(t_band[::-1])],
            color=(60/255,204/255,140/255, 0.22))
    # Confidence band
    t_pm = np.linspace(n1_to_angle(n1 - conf_pm), n1_to_angle(n1 + conf_pm), 90)
    ax.fill(np.r_[R_outer*np.cos(t_pm), (R_outer-0.06)*np.cos(t_pm[::-1])],
            np.r_[R_outer*np.sin(t_pm), (R_outer-0.06)*np.sin(t_pm[::-1])], color=band)
    # Ticks
    for val in range(0, 111, 5):
        ang = n1_to_angle(val); major = (val % 10 == 0)
        r2 = R_inner - (0.06 if major else 0.03)
        x1, y1 = R_outer*np.cos(ang), R_outer*np.sin(ang)
        x2, y2 = r2*np.cos(ang), r2*np.sin(ang)
        ax.plot([x1, x2], [y1, y2], color=tick, linewidth=2)
        if major and 0 < val < 110:
            rl = r2 - 0.10
            ax.text(rl*np.cos(ang), rl*np.sin(ang), f"{val}", ha="center", va="center", fontsize=9, color=label)
    # Bug
    ang_tgt = n1_to_angle(n1)
    ax.plot([0, (R_inner-0.18)*np.cos(ang_tgt)], [0, (R_inner-0.18)*np.sin(ang_tgt)], color=bug, linewidth=3)
    ax.add_artist(plt.Circle((0,0), 0.05, color=label))
    # Readouts BELOW
    ax.text(0, -0.92, f"N1  {n1:.1f}%", ha="center", va="center", fontsize=12, fontweight="bold", color=normal)
    ax.text(0, -1.08, f"±{conf_pm:.1f}%", ha="center", va="center", fontsize=8.5, color=label)
    if show_title: ax.text(0, 1.15, "ENGINE N1", ha="center", va="center", fontsize=10, color=label)
    return fig

def draw_n1_dial_airbus(n1: float, conf_pm: float):
    import numpy as np
    min_n1, max_n1 = 0, 110
    start_deg, end_deg = -210, 30
    def clamp(v, lo, hi): return max(min(v, hi), lo)
    def n1_to_angle(v):
        frac = (clamp(v, min_n1, max_n1) - min_n1) / (max_n1 - min_n1)
        return math.radians(start_deg + frac * (end_deg - start_deg))
    bg, bezel, tick, label = "#0b0b0b", (1,1,1,0.055), "#e8f2ff", "#e8f2ff"
    normal, bug, band = "#00ff90", "#ffd21f", (1,1,1,0.13)
    fig, ax = plt.subplots(figsize=(3.2, 3.2)); ax.set_aspect("equal"); ax.axis("off")
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    R_outer, R_inner = 1.00, 0.74
    theta = np.linspace(math.radians(start_deg), math.radians(end_deg), 240)
    ax.fill(np.r_[R_outer*np.cos(theta), R_inner*np.cos(theta[::-1])],
            np.r_[R_outer*np.sin(theta), R_inner*np.sin(theta[::-1])], color=bezel)
    # Confidence band
    t_pm = np.linspace(n1_to_angle(n1 - conf_pm), n1_to_angle(n1 + conf_pm), 90)
    ax.fill(np.r_[R_outer*np.cos(t_pm), (R_outer-0.06)*np.cos(t_pm[::-1])],
            np.r_[R_outer*np.sin(t_pm), (R_outer-0.06)*np.sin(t_pm[::-1])], color=band)
    # Normal band
    t_band = np.linspace(n1_to_angle(80), n1_to_angle(102), 120)
    ax.fill(np.r_[R_outer*np.cos(t_band), R_inner*np.cos(t_band[::-1])],
            np.r_[R_outer*np.sin(t_band), R_inner*np.sin(t_band[::-1])],
            color=(0,1,0.55,0.20))
    # Ticks
    for val in range(0, 111, 5):
        ang = n1_to_angle(val); major = (val % 10 == 0)
        r2 = 0.74 - (0.06 if major else 0.03)
        x1, y1 = 1.00*np.cos(ang), 1.00*np.sin(ang)
        x2, y2 = r2*np.cos(ang), r2*np.sin(ang)
        ax.plot([x1, x2], [y1, y2], color=tick, linewidth=2)
        if major and 0 < val < 110:
            rl = r2 - 0.10
            ax.text(rl*np.cos(ang), rl*np.sin(ang), f"{val}", ha="center", va="center", fontsize=9, color=label)
    # Bug
    ang_tgt = n1_to_angle(n1)
    ax.plot([0, (0.74-0.18)*np.cos(ang_tgt)], [0, (0.74-0.18)*np.sin(ang_tgt)], color=bug, linewidth=2.5)
    ax.add_artist(plt.Circle((0,0), 0.05, color=label))
    # Readouts BELOW
    ax.text(0, -0.92, f"N1  {n1:.1f}%", ha="center", va="center", fontsize=12, fontweight="bold", color=normal)
    ax.text(0, -1.08, f"±{conf_pm:.1f}%", ha="center", va="center", fontsize=8.5, color=label)
    return fig

def draw_flap_detent_guide(detents: list[str], selected_label: Optional[str] = None):
    bg, lane, tick, label = "#0a0f14", (1,1,1,0.05), "#e8edf6", "#e8edf6"
    highlight = "#3ccc8c"
    fig, ax = plt.subplots(figsize=(3.0, 4.6)); ax.axis("off")
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    top_margin, bottom_margin, x_lane, lane_w = 0.90, 0.10, 0.35, 0.14
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    header_val = (selected_label or "-")
    ax.text(0.5, 0.965, f"Flap setting: {header_val}", ha="center", va="center",
            fontsize=11, color=label, fontweight="bold")
    ax.add_patch(plt.Rectangle((x_lane - lane_w/2, bottom_margin),
                               lane_w, top_margin - bottom_margin,
                               facecolor=lane, edgecolor=None))
    n = len(detents)
    if n == 0: return fig
    sel_norm = (selected_label or "").strip().upper()
    for i, lab in enumerate(detents):
        frac = 1.0 - (i / (n - 1)) if n > 1 else 0.5
        y = bottom_margin + frac * (top_margin - bottom_margin)
        ax.plot([x_lane - lane_w*0.55, x_lane + lane_w*0.55], [y, y], color=tick, linewidth=2)
        ax.text(x_lane + lane_w*0.8, y, lab, ha="left", va="center", fontsize=10, color=label)
        if lab.strip().upper() == sel_norm and sel_norm:
            ax.add_patch(plt.Rectangle((x_lane - lane_w*0.6, y - 0.028), lane_w*1.2, 0.056,
                                       facecolor=(60/255,204/255,140/255,0.25), edgecolor=None))
            ax.plot([x_lane - lane_w*0.7, x_lane - lane_w*0.6], [y, y], color=highlight, linewidth=4)
    return fig

# --------------------------
# Performance Card (no mini-graph)
# --------------------------
def draw_perf_card(meta: Dict[str, Any], result: Dict[str, Any], trim: Optional[Dict[str, Any]] = None):
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.axis("off")
    title = (meta.get("header") or "TAKEOFF PERFORMANCE").strip()
    ax.text(0.02, 0.92, title, fontsize=13, fontweight="bold")

    left_y = 0.82
    def L(label, value):
        nonlocal left_y
        ax.text(0.02, left_y, f"{label}: {value}", fontsize=10); left_y -= 0.06

    L("Airport", f'{meta.get("apt_icao") or "-"} / RWY {meta.get("runway") or "-"} / {meta.get("rwy_cond","-")}')
    L("Engine",  meta.get("engine_pretty", "Generic"))
    L("Flaps",   meta.get("flaps") or "-")
    L("Thrust",  meta.get("thrust_mode") or "-")
    L("SEL TEMP (°C)", meta.get("sel_temp_c") or "-")
    L("Bleeds / A-I", f'{"ON" if meta.get("bleeds_on") else "OFF"} / {"ON" if meta.get("anti_ice_on") else "OFF"}')
    L("TOW (k lb)", meta.get("tow_klb") or "-")

    mid_y = 0.82
    def M(label, value):
        nonlocal mid_y
        ax.text(0.42, mid_y, f"{label}: {value}", fontsize=10); mid_y -= 0.06

    M("OAT (°C)", meta.get("oat_c") or "-")
    M("QNH (inHg)", meta.get("qnh_inhg") or "-")
    M("Field Elev (ft)", meta.get("elev_ft") or "-")
    M("Press Alt (ft)", f'{result.get("pa_ft",0):,.0f}')
    if meta.get("v1") and meta.get("vr") and meta.get("v2"):
        M("V1 / VR / V2", f'{meta["v1"]} / {meta["vr"]} / {meta["v2"]}')
    if meta.get("cg_percent_mac") is not None:
        M("TOCG (%MAC)", f'{meta["cg_percent_mac"]:.1f}%')

    # Right column: headline numbers
    ax.text(0.72, 0.82, "N1 Target", fontsize=10, fontweight="bold")
    ax.text(0.72, 0.76, f'{result["n1"]:.1f}% ±{result["conf_pm"]:.1f}%', fontsize=16)

    if trim is not None:
        ax.text(0.72, 0.64, "IF Trim", fontsize=10, fontweight="bold")
        sign = "+" if trim["if_trim_pct"] >= 0 else ""
        ax.text(0.72, 0.58, f'{sign}{trim["if_trim_pct"]:.0f}% {trim["label"]}', fontsize=16)

    return fig

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="IF Takeoff N1 & Trim", page_icon="🛫", layout="wide")
st.markdown("<h1 style='font-size:2.1rem; margin-bottom:0;'>🛫 Infinite Flight – Takeoff N1% & Trim Estimator</h1>", unsafe_allow_html=True)
st.caption("Paste your TAKEOFF PERFORMANCE / SimBrief text → estimate N1% and Infinite Flight trim. **For simulation only; not for real-world operations.**")

txt = st.text_area("Paste your TAKEOFF PERFORMANCE (or full SimBrief OFP) text:", height=280, placeholder="Paste the whole block here…")
go = st.button("Estimate N1%")
st.markdown("---")

if go and txt.strip():
    data = parse_takeoff_text(txt)

    # Engine/brand labels
    st.write(f"**Detected engine:** {data['engine_pretty']}")
    brand = data.get("brand", detect_brand(data.get("header","")))

    # Validate inputs needed for N1
    if None in (data["oat_c"], data["sel_temp_c"], data["qnh_inhg"], data["elev_ft"], data["tow_klb"]):
        st.warning("Some inputs are missing. Ensure your paste includes OAT, QNH, Elevation, Weight, and SEL TEMP.")
    else:
        # Estimate N1
        res = estimate_n1(
            engine_id=data["engine_id"], oat_c=data["oat_c"], sel_temp_c=data["sel_temp_c"],
            qnh_inhg=data["qnh_inhg"], elev_ft=data["elev_ft"], tow_klb=data["tow_klb"],
            bleeds_on=data["bleeds_on"], anti_ice_on=data["anti_ice_on"], thrust_mode=data["thrust_mode"] or ""
        )
        n1 = res["n1"]; conf = res["conf_pm"]

        # Trim (IF %) estimate
        trim_res = estimate_if_trim(data)

        # Top metrics (include Trim)
        m1, m2, m3, m4 = st.columns([1.2, 1, 1, 1])
        with m1: st.metric("Estimated N1%", f"{n1:.1f}%", f"±{conf:.1f}%")
        with m2: st.metric("Pressure Altitude", f'{res["pa_ft"]:,.0f} ft')
        with m3:
            if data["v1"] and data["vr"] and data["v2"]:
                st.metric("V1 / VR / V2", f'{data["v1"]} / {data["vr"]} / {data["v2"]}')
        with m4:
            sign = "+" if trim_res["if_trim_pct"] >= 0 else ""
            st.metric("IF Trim", f"{sign}{trim_res['if_trim_pct']:.0f}% {trim_res['label']}")

        # Takeoff Thrust / Flaps
        st.markdown("<h3 style='margin-top:0.25rem;'>Takeoff Thrust / Flaps</h3>", unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="large")
        
        with c1:
            st.caption("Engine N1")
            if brand == "airbus":
                st.pyplot(draw_n1_dial_airbus(n1, conf), use_container_width=False)
            else:
                st.pyplot(draw_n1_dial_boeing(n1, conf), use_container_width=False)
            st.markdown(f"<div style='text-align:center; margin-top:-10px;'>"
                f"<span style='font-weight:700;'>N1:</span> {n1:.1f}% "
                f"<span style='opacity:0.7;'>(±{conf:.1f}%)</span>"
                f"</div>", unsafe_allow_html=True)

        with c2:
            st.caption("Flaps")
            detents = get_flap_detents(brand, data.get("header",""))
            flv = int(data.get("flaps") or 0)
            selected = trim_res["selected_flaps_label"]  # already brand-aware, forces 1+F for Airbus at flaps=1
            fig_flaps = draw_flap_detent_guide(detents, selected_label=selected)
            st.pyplot(fig_flaps, use_container_width=False)

        # Performance card (now also shows IF Trim and TOCG if parsed)
        st.subheader("Takeoff Performance Card")
        fig_card = draw_perf_card(data, res, trim=trim_res)
        st.pyplot(fig_card, use_container_width=True)

        # Sensitivity section (unchanged)
        st.subheader("Sensitivity")
        c_s1, c_s2 = st.columns(2, gap="large")

        with c_s1:
            st.caption("N1 vs Selected Temperature")
            sel_range = np.arange(max(0, data["sel_temp_c"]-12), data["sel_temp_c"]+13, 1)
            n1_curve = []
            for s in sel_range:
                tmp = estimate_n1(data["engine_id"], data["oat_c"], float(s), data["qnh_inhg"], data["elev_ft"],
                                  data["tow_klb"], data["bleeds_on"], data["anti_ice_on"], data["thrust_mode"])
                n1_curve.append(tmp["n1"])
            fig2, ax2 = plt.subplots(figsize=(4.4, 2.4))
            ax2.plot(sel_range, n1_curve, linewidth=2)
            ax2.axvline(data["sel_temp_c"], linestyle="--")
            ax2.set_xlabel("Selected Temperature (°C)")
            ax2.set_ylabel("Estimated N1 (%)")
            st.pyplot(fig2, use_container_width=True)

        with c_s2:
            st.caption("N1 vs Field Elevation (Pressure Altitude Effect)")
            elev_demo = np.arange(0, 9001, 500)
            pa_curve = []
            for e_ft in elev_demo:
                tmp = estimate_n1(data["engine_id"], data["oat_c"], data["sel_temp_c"], data["qnh_inhg"], float(e_ft),
                                  data["tow_klb"], data["bleeds_on"], data["anti_ice_on"], data["thrust_mode"])
                pa_curve.append(tmp["n1"])
            fig3, ax3 = plt.subplots(figsize=(4.4, 2.4))
            ax3.plot(elev_demo, pa_curve, linewidth=2)
            ax3.axvline(float(data["elev_ft"]), linestyle="--")
            ax3.set_xlabel("Field Elevation (ft) at same QNH")
            ax3.set_ylabel("Estimated N1 (%)")
            st.pyplot(fig3, use_container_width=True)
