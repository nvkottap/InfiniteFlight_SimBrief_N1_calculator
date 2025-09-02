# app.py
# Infinite Flight – Takeoff N1% & Trim Estimator (Sim-use only)
# - Uses unified calibrations.json (engines + airframes + optional overrides)
# - N1 readout printed UNDER the dial (big, black text)
# - QNH parser accepts inHg OR hPa
# - Stronger N1 model with weight/PA/flaps floor + optional override blending
# - Stronger IF Trim model + optional override blending
# - Airbus FLAPS=1 forces "1+F"; Flaps detent guide shows "Flap setting: X"

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
# Unified calibration loader
# --------------------------
# ---- Robust unified calibration loader ----
CALIB_PATH = Path(__file__).with_name("calibrations.json")

_MINIMAL_FALLBACK = {
    "engines": {
        "generic": { "a": 95.0, "b_temp": -0.06, "c_pa": 0.24, "d_derate": -1.5, "w_ref": 300.0, "e_wt": 0.018 }
    },
    "airframes": {
        "generic": { "engine_id": "generic", "brand": "boeing",
                     "detents": ["0","5","15","25","30"],
                     "baseline_trim": 8.0, "baseline_flaps": 5, "cg_ref": 25.0 }
    }
}

def load_calibrations_unified() -> Dict[str, Any]:
    try:
        with open(CALIB_PATH, "r", encoding="utf-8") as f:
            raw = f.read()
        # Quick sanity check to catch HTML or blank files
        if not raw.strip().startswith("{"):
            raise ValueError("File does not start with '{' – likely not JSON (maybe a bad paste or HTML).")
        data = json.loads(raw)
        return data
    except FileNotFoundError:
        st.warning(f"`{CALIB_PATH.name}` not found. Using minimal built-in calibration.")
        return _MINIMAL_FALLBACK
    except json.JSONDecodeError as e:
        # Point you to the exact spot
        st.error(
            f"Your `{CALIB_PATH.name}` contains invalid JSON at line {e.lineno}, column {e.colno}: {e.msg}."
        )
        # Show a small preview around the error location
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

# Pretty names for engines (optional)
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

ENGINE_ALIASES = {
    "LEAP-1B28": "leap-1b28",
    "GE90-94B": "ge90-94b",
    "TRENT 970-84": "trent-970-84",
    "TRENT XWB-84": "trent-xwb-84",
    "CFM56-5B3": "cfm56-5b3",
    "CFM56-5B4": "cfm56-5b4",
    "PW2037": "pw2037"
}

# --------------------------
# Brand & series detection
# --------------------------
AIRBUS_TOKENS  = [
    r"\bA3(18|19|20|21)\b", r"\bA(330|340|350|380)\b",
    r"\bA320-?200\b", r"\bA321-?200\b", r"\bA350-900\b", r"\bA380-800\b"
]
BOEING_TOKENS  = [
    r"\bB(737|747|757|767|777|787)\b", r"\b737\s*MAX\b", r"\bMAX\s*8\b",
    r"\b777-200\b", r"\b757-200\b"
]

def detect_brand(text: str) -> str:
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

def parse_airframe_series(header_or_text: str) -> str:
    t = (header_or_text or "").upper()
    # See if a JSON airframe key matches a known type in the text
    if "A321" in t: return "airbus_a321"
    if "A320" in t: return "airbus_a320"
    if "A350" in t: return "airbus_a350"
    if "A380" in t: return "airbus_a380"
    if re.search(r"\b737\b", t) or "MAX" in t: return "boeing_737_max8" if "MAX" in t else "boeing_737"
    if "757" in t: return "boeing_757"
    if "777" in t: return "boeing_777"
    if "787" in t: return "boeing_787"
    return "generic"

# --------------------------
# Flaps detents (from JSON or sensible defaults)
# --------------------------
AIRBUS_DEFAULT_DETENTS = ["0", "1", "1+F", "2", "3", "FULL"]
BOEING_DEFAULT_DETENTS = ["0", "1", "2", "5", "10", "15", "25", "30", "40"]

def get_airframe(series: str) -> Dict[str, Any]:
    return AIRFRAMES.get(series, AIRFRAMES.get("generic", {}))

def get_flap_detents(brand: str, series: str) -> list[str]:
    af = get_airframe(series)
    det = af.get("detents")
    if det: return det
    return AIRBUS_DEFAULT_DETENTS if brand == "airbus" else BOEING_DEFAULT_DETENTS

# --------------------------
# Parsing helpers
# --------------------------
def detect_engine_in_text(txt: str) -> Tuple[str, Optional[str]]:
    for token, eng_id in ENGINE_ALIASES.items():
        if re.search(rf"\b{re.escape(token)}\b", txt, flags=re.I):
            return eng_id, token
    # fallback from airframe default
    series = parse_airframe_series(txt)
    eng_from_af = get_airframe(series).get("engine_id", "generic")
    return eng_from_af, None

def pressure_altitude_ft(field_elev_ft: float, qnh_inhg: float) -> float:
    # PA = Elev + (29.92 - QNH) * 1000
    return float(field_elev_ft) + (29.92 - float(qnh_inhg)) * 1000.0

def _parse_qnh_inhg(txt: str) -> Optional[float]:
    """
    Accept QNH in inHg (e.g. 29.92) or hPa (e.g. 1021 / 1021 hPa) and return inHg.
    """
    # inHg like "QNH 29.92"
    m = re.search(r"\bQNH[:\s]+(\d{2}\.\d{2})\b", txt, flags=re.I)
    if m:
        try: return float(m.group(1))
        except: pass
    # hPa integer like "QNH 1021", optional HPA token
    m = re.search(r"\bQNH[:\s]+(\d{3,4})(?:\s*HPA)?\b", txt, flags=re.I)
    if m:
        try:
            hpa = float(m.group(1))
            if 850.0 <= hpa <= 1100.0:
                return hpa * 0.0295299830714
        except: pass
    # hPa decimal like "QNH 1013.2"
    m = re.search(r"\bQNH[:\s]+(\d{3,4}\.\d)\s*(HPA)?\b", txt, flags=re.I)
    if m:
        try:
            hpa = float(m.group(1))
            if 850.0 <= hpa <= 1100.0:
                return hpa * 0.0295299830714
        except: pass
    return None

def parse_cg_percent(txt: str) -> Optional[float]:
    patterns = [
        r"\bTOCG\s+(\d{1,2}\.?\d*)\s*%",
        r"\bZFWCG\s+(\d{1,2}\.?\d*)\s*%",
        r"\bMACZFW\s+(\d{1,2}\.?\d*)\s*%",
        r"\bCG\s+(\d{1,2}\.?\d*)\s*%MAC"
    ]
    t = txt.replace(",", " ")
    for pat in patterns:
        m = re.search(pat, t, flags=re.I)
        if m:
            try:
                val = float(m.group(1))
                if 5.0 <= val <= 45.0:
                    return val
            except:
                pass
    return None

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

    # Weather (OAT, QNH (any unit), ELEV)
    m = re.search(r"OAT\s+(-?\d+)", txt);        d["oat_c"] = float(m.group(1)) if m else None
    d["qnh_inhg"] = _parse_qnh_inhg(txt)
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

    # Engine detection (from text or default from airframe)
    eng_id, token = detect_engine_in_text(txt)
    d["engine_id"] = eng_id
    d["engine_token"] = token
    d["engine_pretty"] = ENGINE_PRETTY.get(eng_id, "Generic")

    # CG %MAC (if present in full OFP)
    d["cg_percent_mac"] = parse_cg_percent(txt)

    return d

# --------------------------
# Flaps label mapping
# --------------------------
def detent_numeric_value(label: str, brand: str) -> float:
    t = (label or "").upper().strip()
    if brand == "airbus":
        if t == "FULL": return 4.0
        if t == "1+F":  return 1.0
        try: return float(t)
        except: return 1.0
    else:
        try: return float(t.replace("°",""))
        except: return 5.0

def nearest_label_by_value(detents: list[str], value: int) -> Optional[str]:
    numeric_pairs = []
    for lab in detents:
        try: numeric_pairs.append((lab, float(lab.replace("°",""))))
        except: pass
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

# --------------------------
# Model helpers & overrides
# --------------------------
def derate_level_from_mode(mode: str) -> int:
    if not mode: return 0
    m = mode.upper()
    if m == "D-TO2": return 2
    if m == "D-TO1": return 1
    if m == "D-TO":  return 1
    return 0  # FLEX or unknown

def apply_n1_overrides(series: str, flaps_label: str, tow_klb: float, n1_est: float) -> float:
    af = AIRFRAMES.get(series, {})
    overrides = af.get("n1_overrides", [])
    if not overrides:
        return n1_est
    mtow = float(af.get("mtow_klb", 0) or 0)
    if mtow <= 0:
        return n1_est
    wf = max(0.0, min(1.0, (tow_klb or 0.0) / mtow))
    # choose same flaps label & nearest weight_frac
    cands = [ov for ov in overrides if ov.get("flaps_label","").upper() == (flaps_label or "").upper()]
    if not cands:
        return n1_est
    best = min(cands, key=lambda ov: abs(wf - float(ov.get("weight_frac", 0.6))))
    try:
        target = float(best["n1_target"])
        # blend toward target
        return 0.55 * n1_est + 0.45 * target
    except:
        return n1_est

def maybe_apply_trim_override(series: str, flaps_label: str, tow_klb: float, trim_est: float) -> float:
    af = get_airframe(series)
    ovs = af.get("trim_overrides", [])
    if not ovs: return trim_est
    candidates = [ov for ov in ovs if ov.get("flaps_label","").upper() == (flaps_label or "").upper()]
    if not candidates: return trim_est
    ov = candidates[min(len(candidates)//2, len(candidates)-1)]
    try:
        tgt = float(ov["if_trim_pct"])
        return 0.5 * trim_est + 0.5 * tgt
    except:
        return trim_est

# --------------------------
# N1 estimator (stronger + floor + flaps)
# --------------------------
def estimate_n1(engine_id: str, oat_c: float, sel_temp_c: float, qnh_inhg: float,
                elev_ft: float, tow_klb: float, bleeds_on: bool, anti_ice_on: bool,
                thrust_mode: str, brand: str, flaps: Optional[int],
                header_text: str, series: str) -> Dict[str, Any]:
    """
    Heuristic model (sim-use only):
      N1_raw = a + b*(SEL-OAT) + c*(PA_kft) + d*(derate_steps) + e*(TOW - w_ref) + flap_drag
      Then apply: derate effectiveness vs weight, floor, and airframe overrides.
    """
    cal = ENGINES.get(engine_id, ENGINES.get("generic", {}))
    a = cal.get("a", 95.0); b = cal.get("b_temp", -0.06); c = cal.get("c_pa", 0.24)
    d = cal.get("d_derate", -1.5); w_ref = cal.get("w_ref", 300.0); e = cal.get("e_wt", 0.018)

    # Inputs & environment
    deltaT = (sel_temp_c - oat_c) if (sel_temp_c is not None and oat_c is not None) else 0.0
    pa_ft = pressure_altitude_ft(elev_ft or 0.0, qnh_inhg or 29.92)
    pa_kft = max(pa_ft, 0.0) / 1000.0
    derate_steps = derate_level_from_mode(thrust_mode or "")
    wt_term = ((tow_klb or w_ref) - w_ref)

    # Flaps (select label/value)
    detents = get_flap_detents(brand, series)
    sel_label = choose_selected_label(detents, int(flaps or 0), brand)
    sel_val = detent_numeric_value(sel_label or ("1" if brand=="airbus" else "5"), brand)
    base_val = 1.0 if brand == "airbus" else 5.0

    # Base model (define n1 BEFORE any further adjustments)
    n1 = a + b*deltaT + c*pa_kft + d*derate_steps + e*wt_term
    if bleeds_on:   n1 += 0.2
    if anti_ice_on: n1 += 0.5
    # Flap drag term (more flaps → more thrust)
    n1 += 0.35 * max(0.0, sel_val - base_val)

    # NEW: derate effectiveness depends on weight (apply AFTER n1 exists)
    af = AIRFRAMES.get(series, {})
    mtow = float(af.get("mtow_klb", 0) or 0)
    wf = max(0.0, min(1.0, (tow_klb or 0.0) / mtow)) if mtow > 0 else 0.6
    # Heavier (wf>0.6) regains some N1 despite derate; lighter benefits more from derate
    n1 += 0.8 * derate_steps * (wf - 0.6)

    # Guardrail floor & cap
    weight_ratio = max(0.8, (tow_klb or w_ref) / w_ref)
    n1_floor = 88.0 + 5.5*weight_ratio + 0.35*pa_kft + 0.25*max(0.0, sel_val - base_val)
    n1 = max(n1, n1_floor)
    n1 = min(n1, 103.0)

    # Blend toward any airframe overrides
    n1 = apply_n1_overrides(series, sel_label or "", tow_klb, n1)

    # Confidence band (heuristic)
    conf_pm = 0.6 + 0.06*pa_kft + 0.15*derate_steps
    return {"n1": n1, "conf_pm": conf_pm, "pa_ft": pa_ft}


# --------------------------
# IF Trim estimator (stronger + optional overrides)
# --------------------------
def estimate_if_trim(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Heuristic Infinite Flight trim (%):
      - Start from airframe baseline (brand + baseline flaps)
      - +1.8% per 1%MAC forward of reference CG
      - -0.6% per flaps unit above baseline
      - +0.8% per 50k lb above engine cal weight
      - Optional JSON override blend
    """
    series = data.get("series", "generic")
    brand  = data.get("brand", "boeing")
    flaps  = int(data.get("flaps") or 0)
    cg_pct = data.get("cg_percent_mac")
    tow_klb = float(data.get("tow_klb") or 0.0)

    af = get_airframe(series)
    baseline_trim = float(af.get("baseline_trim", 8.0))
    baseline_flaps = float(af.get("baseline_flaps", 5.0))
    cg_ref = float(af.get("cg_ref", 25.0))

    detents = get_flap_detents(brand, series)
    selected_label = choose_selected_label(detents, flaps, brand) or (str(flaps) if brand=="boeing" else "1")
    selected_val = detent_numeric_value(selected_label, brand)

    trim = baseline_trim

    # CG effect
    if cg_pct is not None:
        cg_delta = (cg_ref - float(cg_pct))  # forward positive
        trim += 1.8 * cg_delta

    # Flaps effect (more flaps => less NU)
    trim += -0.6 * (selected_val - baseline_flaps)

    # Weight tweak vs engine reference
    w_ref = ENGINES.get(data.get("engine_id","generic"), ENGINES["generic"]).get("w_ref", 300.0)
    trim += 0.8 * ((tow_klb - w_ref) / 50.0)

    # Optional overrides from JSON
    trim = maybe_apply_trim_override(series, selected_label, tow_klb, trim)

    # Clamp
    trim = max(-40.0, min(40.0, trim))
    label = "NU" if trim >= 0 else "ND"
    return {"if_trim_pct": trim, "label": label, "selected_flaps_label": selected_label}

# --------------------------
# Dials (numeric readout is printed below via Streamlit)
# --------------------------
def draw_n1_dial_boeing(n1: float, conf_pm: float):
    import numpy as np
    min_n1, max_n1 = 0, 110
    start_deg, end_deg = -210, 30

    def clamp(v, lo, hi): return max(min(v, hi), lo)
    def n1_to_angle(v):
        frac = (clamp(v, min_n1, max_n1) - min_n1) / (max_n1 - min_n1)
        return math.radians(start_deg + frac * (end_deg - start_deg))

    # Colors
    bg = "#0a0f14"
    bezel = (1, 1, 1, 0.06)
    tick = "#e8edf6"
    label = "#e8edf6"
    band = (1, 1, 1, 0.14)
    txt_color = "#ffffff"

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.set_aspect("equal"); ax.axis("off")
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)

    R_outer, R_inner = 1.00, 0.74
    theta = np.linspace(math.radians(start_deg), math.radians(end_deg), 240)
    # Bezel ring
    ax.fill(np.r_[R_outer*np.cos(theta), R_inner*np.cos(theta[::-1])],
            np.r_[R_outer*np.sin(theta), R_inner*np.sin(theta[::-1])],
            color=bezel)

    # Confidence arc
    t_pm = np.linspace(n1_to_angle(n1 - conf_pm), n1_to_angle(n1 + conf_pm), 80)
    ax.fill(np.r_[R_outer*np.cos(t_pm), (R_outer-0.06)*np.cos(t_pm[::-1])],
            np.r_[R_outer*np.sin(t_pm), (R_outer-0.06)*np.sin(t_pm[::-1])],
            color=band)

    # Normal band 80–102
    t_norm = np.linspace(n1_to_angle(80), n1_to_angle(102), 120)
    ax.fill(np.r_[R_outer*np.cos(t_norm), R_inner*np.cos(t_norm[::-1])],
            np.r_[R_outer*np.sin(t_norm), R_inner*np.sin(t_norm[::-1])],
            color=(60/255, 204/255, 140/255, 0.20))

    # Ticks & numerals
    for val in range(0, 111, 5):
        ang = n1_to_angle(val); major = (val % 10 == 0)
        r2 = R_inner - (0.06 if major else 0.03)
        ax.plot([R_outer*np.cos(ang), r2*np.cos(ang)],
                [R_outer*np.sin(ang), r2*np.sin(ang)], color=tick, linewidth=2)
        if major and 0 < val < 110:
            rl = r2 - 0.10
            ax.text(rl*np.cos(ang), rl*np.sin(ang), f"{val}",
                    ha="center", va="center", fontsize=9, color=label)

    # --- Centered three-line white text ---
    ax.text(0, 0.08, "N1%", ha="center", va="center",
            fontsize=11, fontweight="bold", color=txt_color, zorder=6)
    ax.text(0, -0.02, f"{n1:.1f}%", ha="center", va="center",
            fontsize=13, fontweight="bold", color=txt_color, zorder=6)
    ax.text(0, -0.12, f"±{conf_pm:.1f}%", ha="center", va="center",
            fontsize=10, color=txt_color, alpha=0.9, zorder=6)

    return fig

def draw_n1_dial_airbus(n1: float, conf_pm: float):
    import numpy as np
    min_n1, max_n1 = 0, 110
    start_deg, end_deg = -210, 30

    def clamp(v, lo, hi): return max(min(v, hi), lo)
    def n1_to_angle(v):
        frac = (clamp(v, min_n1, max_n1) - min_n1) / (max_n1 - min_n1)
        return math.radians(start_deg + frac * (end_deg - start_deg))

    # Airbus-style palette
    bg = "#0b0b0b"
    bezel = (1, 1, 1, 0.055)
    tick = "#e8f2ff"
    label = "#e8f2ff"
    band = (1, 1, 1, 0.13)
    txt_color = "#e8f2ff"  # WHITE(ish)

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.set_aspect("equal"); ax.axis("off")
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)

    R_outer, R_inner = 1.00, 0.74
    theta = np.linspace(math.radians(start_deg), math.radians(end_deg), 240)
    # Bezel ring
    ax.fill(np.r_[R_outer*np.cos(theta), R_inner*np.cos(theta[::-1])],
            np.r_[R_outer*np.sin(theta), R_inner*np.sin(theta[::-1])],
            color=bezel)

    # Confidence arc
    t_pm = np.linspace(n1_to_angle(n1 - conf_pm), n1_to_angle(n1 + conf_pm), 80)
    ax.fill(np.r_[R_outer*np.cos(t_pm), (R_outer-0.06)*np.cos(t_pm[::-1])],
            np.r_[R_outer*np.sin(t_pm), (R_outer-0.06)*np.sin(t_pm[::-1])],
            color=band)

    # Normal band 80–102
    t_norm = np.linspace(n1_to_angle(80), n1_to_angle(102), 120)
    ax.fill(np.r_[R_outer*np.cos(t_norm), R_inner*np.cos(t_norm[::-1])],
            np.r_[R_outer*np.sin(t_norm), R_inner*np.sin(t_norm[::-1])],
            color=(60/255, 204/255, 140/255, 0.20))

    # Ticks & numerals
    for val in range(0, 111, 5):
        ang = n1_to_angle(val); major = (val % 10 == 0)
        r2 = R_inner - (0.06 if major else 0.03)
        ax.plot([R_outer*np.cos(ang), r2*np.cos(ang)],
                [R_outer*np.sin(ang), r2*np.sin(ang)], color=tick, linewidth=2)
        if major and 0 < val < 110:
            rl = r2 - 0.10
            ax.text(rl*np.cos(ang), rl*np.sin(ang), f"{val}",
                    ha="center", va="center", fontsize=9, color=label)

    # --- Centered three-line white text ---
    ax.text(0, 0.08, "N1%", ha="center", va="center",
            fontsize=11, fontweight="bold", color=txt_color, zorder=6)
    ax.text(0, -0.02, f"{n1:.1f}%", ha="center", va="center",
            fontsize=13, fontweight="bold", color=txt_color, zorder=6)
    ax.text(0, -0.12, f"±{conf_pm:.1f}%", ha="center", va="center",
            fontsize=10, color=txt_color, alpha=0.9, zorder=6)

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
# Performance Card
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
    if meta.get("qnh_inhg") is not None:
        M("QNH (inHg)", f'{meta["qnh_inhg"]:.2f}')
    else:
        M("QNH (inHg)", "-")
    M("Field Elev (ft)", meta.get("elev_ft") or "-")
    M("Press Alt (ft)", f'{result.get("pa_ft",0):,.0f}')
    if meta.get("v1") and meta.get("vr") and meta.get("v2"):
        M("V1 / VR / V2", f'{meta["v1"]} / {meta["vr"]} / {meta["v2"]}')
    if meta.get("cg_percent_mac") is not None:
        M("TOCG (%MAC)", f'{meta["cg_percent_mac"]:.1f}%')

    # Right column: headline numbers
    ax.text(0.72, 0.82, "N1 Target", fontsize=10, fontweight="bold")
    ax.text(0.72, 0.76, f'N1% = {result["n1"]:.1f}% (±{result["conf_pm"]:.1f}%)', fontsize=16)
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
    series = data.get("series", "generic")
    brand = data.get("brand", detect_brand(data.get("header","")))
    data["series"] = series
    data["brand"] = brand

    # Engine label
    st.write(f"**Detected engine:** {data.get('engine_pretty', 'Generic')}")

    # Validate required inputs
    missing = []
    if data["oat_c"] is None: missing.append("OAT")
    if data["sel_temp_c"] is None: missing.append("SEL TEMP")
    if data["qnh_inhg"] is None: missing.append("QNH")
    if data["elev_ft"] is None: missing.append("Elevation")
    if data["tow_klb"] is None: missing.append("Weight")
    if missing:
        st.error("Missing: " + ", ".join(missing) + ". Ensure your paste includes these fields.")
    else:
        # N1 estimate
        res = estimate_n1(
            engine_id=data["engine_id"],
            oat_c=data["oat_c"],
            sel_temp_c=data["sel_temp_c"],
            qnh_inhg=data["qnh_inhg"],
            elev_ft=data["elev_ft"],
            tow_klb=data["tow_klb"],
            bleeds_on=data["bleeds_on"],
            anti_ice_on=data["anti_ice_on"],
            thrust_mode=data["thrust_mode"] or "",
            brand=brand,
            flaps=data["flaps"],
            header_text=data.get("header",""),
            series=series
        )
        n1 = res["n1"]; conf = res["conf_pm"]

        # Trim estimate
        trim_res = estimate_if_trim(data)

        # Top metrics
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
        

        with c2:
            st.caption("Flaps")
            detents = get_flap_detents(brand, series)
            flv = int(data.get("flaps") or 0)
            selected = trim_res["selected_flaps_label"]  # includes Airbus 1+F behavior
            fig_flaps = draw_flap_detent_guide(detents, selected_label=selected)
            st.pyplot(fig_flaps, use_container_width=False)

        # Performance card
        st.subheader("Takeoff Performance Card")
        fig_card = draw_perf_card(data, res, trim=trim_res)
        st.pyplot(fig_card, use_container_width=True)
