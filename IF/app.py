# app.py
# Infinite Flight – Takeoff N1% Estimator (Sim-use only)
# Final build with:
#  - Parsing of TAKEOFF PERFORMANCE text
#  - Engine + brand (Airbus/Boeing) auto-detection
#  - Heuristic N1% estimate (for sim only)
#  - Boeing/Airbus compact MFD N1 dials + matching flaps indicators
#  - Takeoff Performance Card
#  - Seaborn-styled, smaller sensitivity plots

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
sns.set_theme(style="whitegrid", context="talk")  # clean visuals

# --------------------------
# Engine calibration (heuristic)
# --------------------------
CAL_PATH = Path(__file__).with_name("engine_calibrations.json")

def load_calibrations() -> Dict[str, Any]:
    if CAL_PATH.exists():
        with open(CAL_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    # fallback minimal defaults
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
    """Return 'airbus' or 'boeing' (default to boeing if unclear)."""
    t = (text or "").upper()
    for pat in AIRBUS_TOKENS:
        if re.search(pat, t, flags=re.I):
            return "airbus"
    for pat in BOEING_TOKENS:
        if re.search(pat, t, flags=re.I):
            return "boeing"
    # engine hints
    if "TRENT XWB" in t or "CFM56-5B" in t:
        return "airbus"
    return "boeing"

# --------------------------
# Parsing & basic physics helpers
# --------------------------
def detect_engine_in_text(txt: str) -> Tuple[str, Optional[str]]:
    for token, eng_id in ENGINE_ALIASES.items():
        if re.search(rf"\b{re.escape(token)}\b", txt, flags=re.I):
            return eng_id, token
    return "generic", None

def pressure_altitude_ft(field_elev_ft: float, qnh_inhg: float) -> float:
    # Approx: PA = Elev + (29.92 - QNH) * 1000
    return float(field_elev_ft) + (29.92 - float(qnh_inhg)) * 1000.0

def parse_takeoff_text(txt: str) -> Dict[str, Any]:
    d: Dict[str, Any] = {}

    header_line = next((l for l in txt.splitlines() if re.search(r"\b[A-Z0-9]{3,}.*", l)), "")
    d["header"] = header_line
    d["brand"] = detect_brand(header_line or txt)

    # Airport/runway (for card)
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
    m = re.search(r"THRUST\s+(FLEX|D-TO2|D-TO1|D-TO)", txt)
    d["thrust_mode"] = m.group(1) if m else None
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
    """
    Heuristic model (sim-use only):
      N1 = a + b*(SEL - OAT) + c*(PA_kft) + d*(derate_steps) + e*(TOW - w_ref) + small taps
    """
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

    # Confidence: grows slightly with PA and derate
    conf_pm = 0.4 + 0.05*pa_kft + 0.1*derate_steps
    return {"n1": n1, "conf_pm": conf_pm, "pa_ft": pa_ft}

# --------------------------
# Dials & Flaps (Boeing / Airbus)
# --------------------------
def draw_n1_dial_boeing(n1: float, conf_pm: float, show_title=False):
    # Geometry
    min_n1, max_n1 = 0, 110
    start_deg, end_deg = -210, 30
    span_deg = end_deg - start_deg
    def clamp(v, lo, hi): return max(min(v, hi), lo)
    def n1_to_angle(v):
        v = clamp(v, min_n1, max_n1)
        frac = (v - min_n1) / (max_n1 - min_n1)
        return math.radians(start_deg + frac * span_deg)

    # Colors (Boeing glass)
    bg     = "#0a0f14"
    bezel  = (1,1,1,0.05)
    tick   = "#e8edf6"
    label  = "#e8edf6"
    normal = "#3ccc8c"
    bug    = "#ffd21f"
    band   = (1,1,1,0.14)

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.set_aspect("equal"); ax.axis("off")
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)

    R_outer, R_inner = 1.00, 0.74

    # Bezel
    theta = np.linspace(math.radians(start_deg), math.radians(end_deg), 240)
    x_o, y_o = R_outer*np.cos(theta), R_outer*np.sin(theta)
    x_i, y_i = R_inner*np.cos(theta[::-1]), R_inner*np.sin(theta[::-1])
    ax.fill(np.r_[x_o, x_i], np.r_[y_o, y_i], color=bezel)

    # Normal band (80–102)
    g0, g1 = 80, 102
    t_band = np.linspace(n1_to_angle(g0), n1_to_angle(g1), 120)
    xb_o, yb_o = R_outer*np.cos(t_band), R_outer*np.sin(t_band)
    xb_i, yb_i = R_inner*np.cos(t_band[::-1]), R_inner*np.sin(t_band[::-1])
    ax.fill(np.r_[xb_o, xb_i], np.r_[yb_o, yb_i], color=(60/255, 204/255, 140/255, 0.22))

    # Confidence band
    lo, hi = n1 - conf_pm, n1 + conf_pm
    t_pm = np.linspace(n1_to_angle(lo), n1_to_angle(hi), 90)
    xp_o, yp_o = R_outer*np.cos(t_pm), R_outer*np.sin(t_pm)
    xp_i, yp_i = (R_outer-0.06)*np.cos(t_pm[::-1]), (R_outer-0.06)*np.sin(t_pm[::-1])
    ax.fill(np.r_[xp_o, xp_i], np.r_[yp_o, yp_i], color=band)

    # Ticks and labels
    for val in range(0, 111, 5):
        ang = n1_to_angle(val)
        major = (val % 10 == 0)
        r2 = R_inner - (0.06 if major else 0.03)
        x1, y1 = R_outer*np.cos(ang), R_outer*np.sin(ang)
        x2, y2 = r2*np.cos(ang), r2*np.sin(ang)
        ax.plot([x1, x2], [y1, y2], color=tick, linewidth=2)
        if major and 0 < val < 110:
            rl = r2 - 0.10
            xl, yl = rl*np.cos(ang), rl*np.sin(ang)
            ax.text(xl, yl, f"{val}", ha="center", va="center", fontsize=9, color=label)

    # Bug
    ang_tgt = n1_to_angle(n1)
    xh, yh = (R_inner-0.18)*np.cos(ang_tgt), (R_inner-0.18)*np.sin(ang_tgt)
    ax.plot([0, xh], [0, yh], color=bug, linewidth=3)
    ax.add_artist(plt.Circle((0,0), 0.05, color=label))

    # Readouts
    ax.text(0, -0.53, f"N1  {n1:.1f}%", ha="center", va="center",
            fontsize=12, fontweight="bold", color=normal)
    ax.text(0, -0.71, f"±{conf_pm:.1f}%", ha="center", va="center",
            fontsize=8.5, color=label)
    if show_title:
        ax.text(0, 1.15, "ENGINE N1", ha="center", va="center", fontsize=10, color=label)
    return fig

def draw_flaps_boeing(flaps_value: int):
    bg     = "#0a0f14"
    bezel  = (1,1,1,0.05)
    tick   = "#e8edf6"
    label  = "#e8edf6"
    normal = "#3ccc8c"

    marks = [0,1,2,5,10,15,25,30,40]
    w, h = 7.2, 2.2
    fig, ax = plt.subplots(figsize=(w/2.2, h/2.2))
    ax.axis("off"); fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    ax.set_xlim(0, w); ax.set_ylim(0, h)
    ax.add_patch(plt.Rectangle((0.2, 0.35), w-0.4, h-0.7, facecolor=bezel, edgecolor=None))

    # Scale
    x0, x1 = 0.6, w-0.6
    y_mid = h/2
    ax.plot([x0, x1], [y_mid, y_mid], color=tick, linewidth=2)

    for m in marks:
        frac = (marks.index(m))/(len(marks)-1)
        x = x0 + frac*(x1 - x0)
        ax.plot([x, x], [y_mid-0.25, y_mid+0.25], color=tick, linewidth=2)
        ax.text(x, y_mid+0.45, f"{m}", ha="center", va="center", fontsize=9, color=label)

    nearest = min(marks, key=lambda v: abs(v - (flaps_value or 0)))
    frac_p = (marks.index(nearest))/(len(marks)-1)
    xp = x0 + frac_p*(x1 - x0)
    ax.plot([xp, xp], [y_mid-0.45, y_mid+0.45], color=normal, linewidth=4)

    # Readout
    box_w = 1.3; box_h = 0.9
    ax.add_patch(plt.Rectangle((w-0.2-box_w, h-0.2-box_h), box_w, box_h,
                               facecolor=(1,1,1,0.06), edgecolor=None))
    ax.text(w-0.2-box_w/2, h-0.2-box_h/2, f"FLAPS {nearest}",
            ha="center", va="center", fontsize=11, fontweight="bold", color=normal)
    return fig

def draw_n1_dial_airbus(n1: float, conf_pm: float):
    # Geometry
    min_n1, max_n1 = 0, 110
    start_deg, end_deg = -210, 30
    span_deg = end_deg - start_deg
    def clamp(v, lo, hi): return max(min(v, hi), lo)
    def n1_to_angle(v):
        v = clamp(v, min_n1, max_n1)
        frac = (v - min_n1) / (max_n1 - min_n1)
        return math.radians(start_deg + frac * span_deg)

    # Colors (Airbus ECAM)
    bg     = "#0b0b0b"
    bezel  = (1,1,1,0.055)
    tick   = "#e8f2ff"
    label  = "#e8f2ff"
    normal = "#00ff90"
    bug    = "#ffd21f"
    band   = (1,1,1,0.13)

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.set_aspect("equal"); ax.axis("off")
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)

    R_outer, R_inner = 1.00, 0.74

    theta = np.linspace(math.radians(start_deg), math.radians(end_deg), 240)
    x_o, y_o = R_outer*np.cos(theta), R_outer*np.sin(theta)
    x_i, y_i = R_inner*np.cos(theta[::-1]), R_inner*np.sin(theta[::-1])
    ax.fill(np.r_[x_o, x_i], np.r_[y_o, y_i], color=bezel)

    # Confidence band
    lo, hi = n1 - conf_pm, n1 + conf_pm
    t_pm = np.linspace(n1_to_angle(lo), n1_to_angle(hi), 90)
    xp_o, yp_o = R_outer*np.cos(t_pm), R_outer*np.sin(t_pm)
    xp_i, yp_i = (R_outer-0.06)*np.cos(t_pm[::-1]), (R_outer-0.06)*np.sin(t_pm[::-1])
    ax.fill(np.r_[xp_o, xp_i], np.r_[yp_o, yp_i], color=band)

    # Normal band (80–102)
    g0, g1 = 80, 102
    t_band = np.linspace(n1_to_angle(g0), n1_to_angle(g1), 120)
    xb_o, yb_o = R_outer*np.cos(t_band), R_outer*np.sin(t_band)
    xb_i, yb_i = R_inner*np.cos(t_band[::-1]), R_inner*np.sin(t_band[::-1])
    ax.fill(np.r_[xb_o, xb_i], np.r_[yb_o, yb_i], color=(0,1,0.55,0.20))

    # Ticks/labels
    for val in range(0, 111, 5):
        ang = n1_to_angle(val)
        major = (val % 10 == 0)
        r2 = R_inner - (0.06 if major else 0.03)
        x1, y1 = R_outer*np.cos(ang), R_outer*np.sin(ang)
        x2, y2 = r2*np.cos(ang), r2*np.sin(ang)
        ax.plot([x1, x2], [y1, y2], color=tick, linewidth=2)
        if major and 0 < val < 110:
            rl = r2 - 0.10
            xl, yl = rl*np.cos(ang), rl*np.sin(ang)
            ax.text(xl, yl, f"{val}", ha="center", va="center", fontsize=9, color=label)

    # Target bug
    ang_tgt = n1_to_angle(n1)
    xh, yh = (R_inner-0.18)*np.cos(ang_tgt), (R_inner-0.18)*np.sin(ang_tgt)
    ax.plot([0, xh], [0, yh], color=bug, linewidth=2.5)
    ax.add_artist(plt.Circle((0,0), 0.05, color=label))

    # Readouts
    ax.text(0, -0.53, f"N1  {n1:.1f}%", ha="center", va="center",
            fontsize=12, fontweight="bold", color=normal)
    ax.text(0, -0.71, f"±{conf_pm:.1f}%", ha="center", va="center",
            fontsize=8.5, color=label)
    return fig

def draw_flaps_airbus(flaps_value: int):
    bg     = "#0b0b0b"
    bezel  = (1,1,1,0.055)
    tick   = "#e8f2ff"
    label  = "#e8f2ff"
    normal = "#00ff90"

    steps = ["0", "1", "2", "3", "FULL"]
    def nearest_step(v):
        if v is None: return 0
        if v >= 4: return 4
        return int(round(max(0, min(3, v))))
    nearest = nearest_step(flaps_value)

    w, h = 7.2, 2.2
    fig, ax = plt.subplots(figsize=(w/2.2, h/2.2))
    ax.axis("off"); fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    ax.set_xlim(0, w); ax.set_ylim(0, h)
    ax.add_patch(plt.Rectangle((0.2, 0.35), w-0.4, h-0.7, facecolor=bezel, edgecolor=None))

    x0, x1 = 0.7, w-1.6
    y_mid = h/2
    ax.plot([x0, x1], [y_mid, y_mid], color=tick, linewidth=2)

    for idx, lab in enumerate(steps):
        frac = idx / (len(steps)-1)
        x = x0 + frac*(x1 - x0)
        ax.plot([x, x], [y_mid-0.25, y_mid+0.25], color=tick, linewidth=2)
        ax.text(x, y_mid+0.45, lab, ha="center", va="center", fontsize=9, color=label)
        if lab == "1":
            ax.text(x, y_mid+0.82, "1+F", ha="center", va="center", fontsize=7.5, color=label, alpha=0.8)

    frac_p = nearest / (len(steps)-1)
    xp = x0 + frac_p*(x1 - x0)
    ax.plot([xp, xp], [y_mid-0.45, y_mid+0.45], color=normal, linewidth=4)

    box_w = 1.5; box_h = 0.9
    ax.add_patch(plt.Rectangle((w-0.2-box_w, h-0.2-box_h), box_w, box_h,
                               facecolor=(1,1,1,0.06), edgecolor=None))
    right_label = "FULL" if nearest == 4 else steps[nearest]
    ax.text(w-0.2-box_w/2, h-0.2-box_h/2, f"FLAPS {right_label}",
            ha="center", va="center", fontsize=11, fontweight="bold", color=normal)
    return fig

# --------------------------
# Performance Card
# --------------------------
def draw_perf_card(meta: Dict[str, Any], result: Dict[str, Any]):
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
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

    ax.text(0.72, 0.82, "N1 Target", fontsize=10, fontweight="bold")
    ax.text(0.72, 0.76, f'{result["n1"]:.1f}% ±{result["conf_pm"]:.1f}%', fontsize=16)

    g_left, g_bottom, g_width, g_height = 0.68, 0.18, 0.28, 0.16
    ax_in = fig.add_axes([g_left, g_bottom, g_width, g_height])
    ax_in.set_xlim(80, 102); ax_in.set_ylim(0, 1)
    ax_in.axvspan(max(80, result["n1"]-result["conf_pm"]), min(102, result["n1"]+result["conf_pm"]), alpha=0.15)
    ax_in.axvline(result["n1"], linewidth=3)
    ax_in.set_yticks([]); ax_in.set_xlabel("N1 (%)", fontsize=8)

    ax.text(0.02, 0.05, "Simulation aid only • Heuristic estimate", fontsize=8)
    return fig

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="IF Takeoff N1 Estimator", page_icon="🛫", layout="wide")

# Larger main title
st.markdown("<h1 style='font-size:2.1rem; margin-bottom:0;'>🛫 Infinite Flight – Takeoff N1% Estimator</h1>", unsafe_allow_html=True)
st.caption("Paste your TAKEOFF PERFORMANCE text → estimate N1% with brand-matched gauges and a Takeoff Performance Card. **Sim-use only; not for real-world operations.**")

txt = st.text_area(
    "Paste your TAKEOFF PERFORMANCE text:",
    height=280,
    placeholder="Paste the whole block here…"
)

go = st.button("Estimate N1%")
st.markdown("---")

if go and txt.strip():
    data = parse_takeoff_text(txt)

    # Engine/brand labels
    st.write(f"**Detected engine:** {data['engine_pretty']}")
    brand = data.get("brand", detect_brand(data.get("header","")))

    # Validate inputs
    if None in (data["oat_c"], data["sel_temp_c"], data["qnh_inhg"], data["elev_ft"], data["tow_klb"]):
        st.warning("Some inputs are missing. Ensure your paste includes OAT, QNH, Elevation, Weight, and SEL TEMP.")
    else:
        # Estimate N1
        res = estimate_n1(
            engine_id=data["engine_id"],
            oat_c=data["oat_c"],
            sel_temp_c=data["sel_temp_c"],
            qnh_inhg=data["qnh_inhg"],
            elev_ft=data["elev_ft"],
            tow_klb=data["tow_klb"],
            bleeds_on=data["bleeds_on"],
            anti_ice_on=data["anti_ice_on"],
            thrust_mode=data["thrust_mode"] or ""
        )
        n1 = res["n1"]; conf = res["conf_pm"]

        # Top metrics
        m1, m2, m3 = st.columns([1.2, 1, 1])
        with m1: st.metric("Estimated N1%", f"{n1:.1f}%", f"±{conf:.1f}%")
        with m2: st.metric("Pressure Altitude", f'{res["pa_ft"]:,.0f} ft')
        with m3:
            if data["v1"] and data["vr"] and data["v2"]:
                st.metric("V1 / VR / V2", f'{data["v1"]} / {data["vr"]} / {data["v2"]}')

        # Takeoff Thrust + Flaps (brand-aware)
        st.markdown("<h3 style='margin-top:0.25rem;'>Takeoff Thrust / Flaps</h3>", unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.caption("Engine N1")
            if brand == "airbus":
                st.pyplot(draw_n1_dial_airbus(n1, conf), use_container_width=False)
            else:
                st.pyplot(draw_n1_dial_boeing(n1, conf), use_container_width=False)
        with c2:
            st.caption("Flaps")
            flv = int(data.get("flaps") or 0)
            if brand == "airbus":
                st.pyplot(draw_flaps_airbus(flv), use_container_width=False)
            else:
                st.pyplot(draw_flaps_boeing(flv), use_container_width=False)

        # Performance card
        st.subheader("Takeoff Performance Card")
        fig_card = draw_perf_card(data, res)
        st.pyplot(fig_card, use_container_width=True)

        # Sensitivity (smaller side-by-side)
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
