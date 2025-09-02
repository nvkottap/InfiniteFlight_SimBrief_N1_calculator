# app.py
import re, json, math
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

st.set_page_config(page_title="SimBrief to IF TO Tool", page_icon="🛫", layout="wide")

# --- UI polish CSS (readability + spacing) ---
st.markdown("""
<style>
/* Info bar */
.info-bar {
  display:flex; flex-wrap:wrap; gap:16px; align-items:center;
  padding:14px 16px; margin: 4px 0 10px 0;
  border:1px solid #2a3542; border-radius:12px; 
  background:#111824;
}
.info-kv { display:flex; gap:6px; align-items:baseline; }
.info-kv .k { color:#a9c4db; font-size:18px; letter-spacing:0.2px; }
.info-kv .v { color:#ffffff; font-weight:800; font-size:22px; }

/* Reduce default Streamlit figure top/bottom spacing */
.block-container { padding-top: 1.0rem; padding-bottom: 2.0rem; }
.element-container:has(.stPlotlyChart), 
.element-container:has(canvas),
.element-container:has(img),
.element-container:has(svg) { margin-bottom: 0.25rem !important; }

/* Columns spacing harmonize */
div[data-testid="stHorizontalBlock"] { gap: 16px; }

/* Make Matplotlib figures appear tighter */
.figure-tight img { margin: 0 !important; display:block; }
</style>
""", unsafe_allow_html=True)


# ----------------------------- Utilities -----------------------------
def pressure_altitude_ft(field_elev_ft: float, qnh_inhg: float) -> float:
    """Approx pressure altitude using US std formula."""
    if not qnh_inhg:
        qnh_inhg = 29.92
    return (field_elev_ft or 0.0) + (29.92 - qnh_inhg) * 1000.0

def derate_level_from_mode(mode: str) -> int:
    m = (mode or "").upper().strip()
    if "D-TO2" in m: return 3
    if "D-TO1" in m: return 2
    if "D-TO"  in m or "FLEX" in m: return 1
    return 0

# ------------------------- Load Calibrations -------------------------
CALIB_PATH = Path(__file__).with_name("calibrations.json")

_MINIMAL_FALLBACK = {
    "engines": {
        "generic": { "a": 95.0, "b_temp": -0.06, "c_pa": 0.24, "d_derate": -1.5, "w_ref": 300.0, "e_wt": 0.018 }
    },
    "airframes": {
        "generic": { "engine_id": "generic", "brand": "boeing",
                     "mtow_klb": 300.0, "n1_max_pct": 101.5,
                     "detents": ["0","5","15","25","30"],
                     "baseline_trim": 8.0, "baseline_flaps": 5, "cg_ref": 25.0,
                     "if_trim_default_pct": 12.0 }
    }
}

def load_calibrations_unified() -> Dict[str, Any]:
    try:
        raw = CALIB_PATH.read_text(encoding="utf-8")
        if not raw.strip().startswith("{"):
            raise ValueError("calibrations.json is not valid JSON (bad start).")
        return json.loads(raw)
    except FileNotFoundError:
        st.warning(f"`{CALIB_PATH.name}` not found. Using minimal built-in calibration.")
        return _MINIMAL_FALLBACK
    except json.JSONDecodeError as e:
        st.error(f"`{CALIB_PATH.name}` invalid JSON at line {e.lineno}, col {e.colno}: {e.msg}")
        return _MINIMAL_FALLBACK
    except Exception as e:
        st.error(f"Could not read `{CALIB_PATH.name}`: {e}")
        return _MINIMAL_FALLBACK

CALIB = load_calibrations_unified()
ENGINES = CALIB.get("engines", {})
AIRFRAMES = CALIB.get("airframes", {})

def get_flap_detents(brand: str, series: str) -> List[str]:
    """Return canonical, correctly ordered flap detents for the airframe/brand."""
    af = AIRFRAMES.get(series, {})
    det = af.get("detents")

    if (brand or "").lower() == "airbus":
        # Canonical Airbus set/order
        canon = ["0","1","1+F","2","3","FULL"]
        if not det:
            return canon
        # Keep only known Airbus labels, in canonical order
        have = {d.upper(): d for d in det}
        return [have.get(lbl, lbl) for lbl in canon if lbl in have]
    else:
        # Boeing: numeric detents sorted ascending; fall back to a common set
        if not det:
            det = ["0","1","2","5","10","15","25","30","40"]
        # Normalize & sort numerically
        def to_num(s: str) -> float:
            s = (s or "").upper().replace("°","").strip()
            m = re.search(r"-?\d+(\.\d+)?", s)
            return float(m.group()) if m else 0.0
        # Deduplicate while preserving original strings
        uniq = []
        seen = set()
        for d in det:
            u = d.upper()
            if u not in seen:
                uniq.append(d)
                seen.add(u)
        uniq.sort(key=lambda x: to_num(x))
        return uniq
      
# Not strictly required anymore, but can be handy if you need per-brand ordering elsewhere
def canonicalize_detents(brand: str, detents: List[str]) -> List[str]:
    return get_flap_detents(brand, "unused")


def choose_selected_label(detents: List[str], numeric_flaps: int, brand: str) -> str:
    if (brand or "").lower() == "airbus":
        return "1+F" if "1+F" in detents else ("1" if "1" in detents else detents[0])
    pref = 5
    if numeric_flaps in [1,5,10,15,25,30,40]: pref = numeric_flaps
    return str(pref) if str(pref) in detents else detents[0]

def detent_numeric_value(label: str, brand: str) -> float:
    lab = (label or "").upper()
    if (brand or "").lower() == "airbus":
        return {"0":0,"1":1,"1+F":1,"2":2,"3":3,"FULL":4}.get(lab, 0.0)
    try:
        return float(lab.replace("°",""))
    except:
        return 0.0

# ------------------------- Overrides & Estimators -------------------------
def apply_n1_overrides(series: str, flaps_label: str, tow_klb: float, n1_est: float) -> float:
    af = AIRFRAMES.get(series, {})
    overrides = af.get("n1_overrides", [])
    if not overrides: return n1_est
    mtow = float(af.get("mtow_klb", 0) or 0)
    if mtow <= 0: return n1_est
    wf = max(0.0, min(1.0, (tow_klb or 0.0)/mtow))
    cands = [ov for ov in overrides if (ov.get("flaps_label","").upper() == (flaps_label or "").upper())]
    if not cands: return n1_est
    best = min(cands, key=lambda ov: abs(wf - float(ov.get("weight_frac", 0.6))))
    try:
        target = float(best["n1_target"])
        return 0.55*n1_est + 0.45*target
    except:
        return n1_est

def estimate_n1(engine_id: str, oat_c: float, sel_temp_c: float, qnh_inhg: float,
                elev_ft: float, tow_klb: float, bleeds_on: bool, anti_ice_on: bool,
                thrust_mode: str, brand: str, flaps: Optional[int],
                header_text: str, series: str) -> Dict[str, Any]:
    """
    Heuristic N1 model for Infinite Flight (not for real-world ops).
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
    sel_val = detent_numeric_value(sel_label, brand)
    base_val = 1.0 if (brand or "").lower()=="airbus" else 5.0

    n1 = a + b*deltaT + c*pa_kft + d*derate_steps + e*wt_term
    if bleeds_on: n1 += 0.2
    if anti_ice_on: n1 += 0.5
    n1 += 0.35 * max(0.0, sel_val - base_val)

    af = AIRFRAMES.get(series, {})
    mtow = float(af.get("mtow_klb", 0) or 0)
    wf = max(0.0, min(1.0, (tow_klb or 0.0) / mtow)) if mtow > 0 else 0.6
    n1 += 0.8 * derate_steps * (wf - 0.6)

    weight_ratio = max(0.8, (tow_klb or w_ref) / w_ref)
    n1_floor = 88.0 + 5.5*weight_ratio + 0.35*pa_kft + 0.25*max(0.0, sel_val - base_val)
    n1 = max(n1, n1_floor)

    n1_cap = float(af.get("n1_max_pct", 103.0))
    n1 = min(n1, n1_cap)

    n1 = apply_n1_overrides(series, sel_label or "", tow_klb, n1)

    conf_pm = 0.6 + 0.06*pa_kft + 0.15*derate_steps
    return {"n1": n1, "conf_pm": conf_pm, "pa_ft": pa_ft, "flaps_label": sel_label}

def estimate_trim_if(series: str, brand: str, flaps_label: str, tow_klb: float) -> float:
    """
    Infinite Flight trim % (−100..+100), nose-up positive.
    Uses per-airframe defaults + small weight & flap shaping; blends overrides.
    """
    af = AIRFRAMES.get(series, {})
    base_map = {"boeing": 12.0, "airbus": 8.0}
    base = float(af.get("if_trim_default_pct", base_map.get((brand or "").lower(), 10.0)))

    mtow = float(af.get("mtow_klb", 0) or 0)
    wf = max(0.0, min(1.0, (tow_klb or 0.0) / mtow)) if mtow > 0 else 0.6
    k_w = 22.0 if (brand or "").lower() == "boeing" else 16.0
    base += k_w * (wf - 0.6)

    def detent_numeric(lab: str) -> float:
        lab = (lab or "").upper()
        if (brand or "").lower() == "airbus":
            return {"FULL":4.0,"3":3.0,"2":2.0,"1+F":1.0,"1":1.0}.get(lab, 0.0)
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
            except: pass

    return max(-100.0, min(100.0, round(base, 1)))

# ----------------------------- Drawing -----------------------------
def draw_n1_dial_boeing(n1_percent: float, max_n1_pct: float = 102.0):
    """Boeing dial: thin bezel, sharp yellow chevron, square box, red end tick, 0→10 clockwise."""
    def pct_to_scale(p): return max(0.0, min(10.0, (p/100.0)*10.0))
    n1_sc  = pct_to_scale(n1_percent)
    max_sc = pct_to_scale(max_n1_pct)

    start_deg = 0.0
    end_deg   = -225.0 * min(1.0, max_sc/10.0)
    to_rad    = np.deg2rad

    bg="#0a0f14"; white="#ffffff"; red="#d84b4b"; chevron="#ffd21f"
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    ax.set_aspect("equal"); ax.axis("off")
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)

    R_ring=0.98; ring_lw=2.6
    tick_len_major=0.11; tick_len_minor=0.06
    font_small=9; font_big=15

    def scale_to_angle(v):
        frac = max(0.0, min(1.0, v/max(1e-6, max_sc)))
        return to_rad(start_deg + frac*(end_deg - start_deg))

    # Bezel
    theta = np.linspace(to_rad(start_deg), to_rad(end_deg), 720)
    ax.plot(R_ring*np.cos(theta), R_ring*np.sin(theta),
            color=white, linewidth=ring_lw, solid_capstyle='round', zorder=2)

    # Internal ticks & labels
    for v in np.arange(0, 10.1, 1.0):
        if v > max_sc + 1e-6: continue
        ang=scale_to_angle(v)
        is_major=(int(v)%2==0)
        tlen=tick_len_major if is_major else tick_len_minor
        x1,y1 = R_ring*np.cos(ang), R_ring*np.sin(ang)
        x2,y2 = (R_ring - tlen)*np.cos(ang), (R_ring - tlen)*np.sin(ang)
        ax.plot([x1,x2],[y1,y2], color=white, linewidth=2 if is_major else 1.2, zorder=3)
        if is_major:
            rl=R_ring - tlen - 0.08
            ax.text(rl*np.cos(ang), rl*np.sin(ang), f"{int(v)}",
                    ha="center", va="center", fontsize=font_small, color=white, zorder=4)

    # Red max tick
    ang_max=scale_to_angle(max_sc)
    x1m,y1m=R_ring*np.cos(ang_max), R_ring*np.sin(ang_max)
    x2m,y2m=(R_ring - tick_len_major - 0.02)*np.cos(ang_max), (R_ring - tick_len_major - 0.02)*np.sin(ang_max)
    ax.plot([x1m,x2m],[y1m,y2m], color=red, linewidth=3, zorder=4)

    # Yellow chevron (sharp), inward
    ang_n1=scale_to_angle(min(max(n1_sc,0.0), max_sc))
    tip_r = R_ring - 0.008
    base_r= R_ring + 0.045
    spread= math.radians(3.0)
    tip   = np.array([tip_r*np.cos(ang_n1),  tip_r*np.sin(ang_n1)])
    left  = np.array([base_r*np.cos(ang_n1 + spread), base_r*np.sin(ang_n1 + spread)])
    right = np.array([base_r*np.cos(ang_n1 - spread), base_r*np.sin(ang_n1 - spread)])
    ax.plot([left[0], tip[0]],  [left[1], tip[1]],  color=chevron, linewidth=2.4, solid_capstyle='round', zorder=6)
    ax.plot([right[0], tip[0]], [right[1], tip[1]], color=chevron, linewidth=2.4, solid_capstyle='round', zorder=6)

    # Square digital box at 0 mark, lifted
    ang0=scale_to_angle(0.0)
    anchor_r=R_ring + 0.34
    anchor_x,anchor_y = anchor_r*np.cos(ang0), anchor_r*np.sin(ang0)
    box_w,box_h = 0.60, 0.26
    ll=(anchor_x - box_w/2, anchor_y)
    rect=patches.Rectangle(ll, box_w, box_h, linewidth=2, edgecolor=white, facecolor=(0,0,0,0.0), zorder=10)
    ax.add_patch(rect)
    cx,cy = (anchor_x, anchor_y + box_h/2)
    ax.text(cx, cy + 0.01, f"{n1_sc:.1f}", ha="center", va="center",
            fontsize=font_big, fontweight="bold", color=white, zorder=11)

        # tighten layout
    try:
        fig.tight_layout(pad=0.2)
    except Exception:
        pass
    return fig

def draw_n1_dial_airbus(n1_percent: float, max_n1_pct: float = 100.0):
    """Airbus dial: thin bezel, inner ticks, small white needle, red overboost, stacked green readout."""
    def pct_to_scale(p): return max(0.0, min(10.0, (p/100.0)*10.0))
    n1_sc=pct_to_scale(n1_percent); max_sc=pct_to_scale(max_n1_pct)

    start_deg=-225.0; end_deg=-45.0; to_rad=np.deg2rad
    bg="#0a0f14"; white="#ffffff"; green="#7CFF5A"; red="#d84b4b"

    fig, ax = plt.subplots(figsize=(3.6,3.6))
    ax.set_aspect("equal"); ax.axis("off")
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)

    R_outer=1.00; ring_lw=2.2; tick_len=0.05
    def scale_to_angle(v):
        frac=max(0.0,min(1.0,v/10.0)); return to_rad(start_deg + frac*(end_deg-start_deg))

    theta=np.linspace(to_rad(start_deg), to_rad(end_deg), 720)
    ax.plot(R_outer*np.cos(theta), R_outer*np.sin(theta),
            color=white, linewidth=ring_lw, solid_capstyle='round', zorder=2)

    for v in np.arange(0,10.1,1.0):
        ang=scale_to_angle(v)
        x1,y1=R_outer*np.cos(ang), R_outer*np.sin(ang)
        x2,y2=(R_outer - tick_len)*np.cos(ang), (R_outer - tick_len)*np.sin(ang)
        ax.plot([x1,x2],[y1,y2], color=white, linewidth=1.2, zorder=3)
        if abs(v-5.0)<0.01 or abs(v-10.0)<0.01:
            rl=R_outer - tick_len - 0.10
            ax.text(rl*np.cos(ang), rl*np.sin(ang), f"{int(v)}",
                    ha="center", va="center", fontsize=9, color=white, zorder=4)

    over_lo=max(0.0,9.3); over_hi=min(10.0,max_sc)
    if over_hi>over_lo:
        t_over=np.linspace(scale_to_angle(over_lo), scale_to_angle(over_hi), 60)
        ax.plot(R_outer*np.cos(t_over), R_outer*np.sin(t_over),
                color=red, linewidth=ring_lw+0.5, solid_capstyle='butt', zorder=5)

    ang_n1=scale_to_angle(n1_sc)
    needle_r1=0.78; needle_r2=R_outer - 0.04
    ax.plot([needle_r1*np.cos(ang_n1), needle_r2*np.cos(ang_n1)],
            [needle_r1*np.sin(ang_n1), needle_r2*np.sin(ang_n1)],
            color=white, linewidth=2.0, solid_capstyle='round', zorder=6)

    ax.text(0, 0.09, "N1%", ha="center", va="center", fontsize=11, color=green, zorder=7)
    ax.text(0, -0.01, f"{n1_percent:.1f}", ha="center", va="center",
            fontsize=17, fontweight="bold", color=green, zorder=7)
    ax.text(0, -0.12, "%", ha="center", va="center", fontsize=11, color=green, zorder=7)
        # tighten layout
    try:
        fig.tight_layout(pad=0.2)
    except Exception:
        pass
    return fig


def draw_trim_bar(trim_pct: float):
    """Compact vertical trim bar −100..+100% with neutral marker."""
    trim_pct = max(-100.0, min(100.0, float(trim_pct or 0)))
    fig, ax = plt.subplots(figsize=(2.0, 3.0))
    ax.set_facecolor("#0a0f14"); fig.patch.set_facecolor("#0a0f14")
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xlim(0,1); ax.set_ylim(-100,100); ax.get_xaxis().set_visible(False)
    ax.plot([0.5,0.5],[-100,100], color="#ffffff", linewidth=2.2)
    for t in range(-100,101,20):
        ax.plot([0.42,0.58],[t,t], color="#ffffff", linewidth=1.2)
    ax.plot([0.38,0.62],[0,0], color="#ffffff", linewidth=2.0)
    ax.plot([0.5],[trim_pct], marker="o", markersize=8,
            markerfacecolor="#ffd21f", markeredgecolor="#ffd21f")
    ax.text(0.5,105,"TRIM", ha="center", va="bottom", fontsize=10, color="#ffffff", fontweight="bold")
    ax.text(0.5,-115,f"{trim_pct:.0f}%", ha="center", va="top", fontsize=10, color="#ffffff")
    ax.set_yticks([])
        # tighten layout
    try:
        fig.tight_layout(pad=0.2)
    except Exception:
        pass
    return fig


def draw_flap_detents_small(brand: str, detents: List[str], selected_label: str):
    """
    Compact flap ladder:
      • AIRBUS: 0 (top) → 1 → 1+F → 2 → 3 → FULL (bottom)
      • BOEING: 0 (top) → ... → 40 (bottom)
    Labels on the LEFT; highlight hugs ladder segment (no text overlap).
    """
    fig, ax = plt.subplots(figsize=(2.2, 3.0))
    bg = "#0a0f14"; white="#ffffff"; yellow="#ffd21f"
    ax.set_facecolor(bg); fig.patch.set_facecolor(bg)
    for s in ax.spines.values(): s.set_visible(False)
    ax.get_xaxis().set_visible(False)

    # Ensure correct set & order by brand
    detents_draw = get_flap_detents(brand, series="unused")
    # If the caller passed a specific list (from airframe), merge with canonical order:
    if detents:
        det_u = {d.upper(): d for d in detents}
        detents_draw = [det_u.get(lbl, lbl) for lbl in detents_draw if lbl.upper() in det_u]

    n = len(detents_draw)
    ax.set_xlim(0,1)
    # Invert Y so index 0 renders at TOP (0/top → FULL/bottom)
    ax.set_ylim(n-0.5, -0.5)

    x_ladder = 0.62
    ax.plot([x_ladder, x_ladder], [0, n-1], color=white, linewidth=2.0)

    for i, lab in enumerate(detents_draw):
        ax.plot([x_ladder-0.06, x_ladder+0.06], [i, i], color=white, linewidth=1.6)
        ax.text(x_ladder-0.10, i, lab, va="center", ha="right", fontsize=10, color=white)

    # Selection highlight: rectangle around the ladder segment only
    if selected_label:
        try:
            # Match case-insensitively
            idx = next(i for i, l in enumerate(detents_draw) if l.upper() == selected_label.upper())
            ax.add_patch(plt.Rectangle((x_ladder-0.08, idx-0.35), 0.16, 0.7,
                                       fill=False, edgecolor=yellow, linewidth=2.0))
        except StopIteration:
            pass

    ax.text(0.5, -0.15, "FLAPS", ha="center", va="top",
            fontsize=10, color=white, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 1.08, f"{selected_label or ''}", ha="center", va="bottom",
            fontsize=10, color=white, transform=ax.transAxes)

    ax.set_yticks([])
    try:
        fig.tight_layout(pad=0.2)
    except Exception:
        pass
    return fig



# ----------------------------- Parser -----------------------------
PASTE_KEYS = {
    "OAT": r"\bOAT\s+(-?\d+)",
    "QNH": r"\bQNH\s+(\d{2}\.\d{2}|\d{3,4})",
    "ELEV": r"\bELEV\s+(-?\d+)",
    "WEIGHT": r"\bWEIGHT\s+(\d{3}\.\d|\d{2,3})",
    "SEL": r"\bSEL\s+TEMP\s+(\d{1,2})|\bSEL\s+(\d{1,2})",
    "THRUST": r"\bTHRUST\s+([A-Z0-9\-+ ]+)",
    "FLAPS_OUT": r"\bFLAPS\s+(\d{1,2})\b|\bFLAPS\s+(FULL|1\+F|1|2|3)",
    "BLEEDS": r"\bBLEEDS\s+(ON|OFF)",
    "AICE": r"\bA/ICE\s+(ON|OFF)",
    "APT": r"\bAPT\s+([A-Z]{4})\/?([A-Z]{3})?",
    "RWY": r"\bRWY\s+([0-9]{2}[LRC]?)/\+\d|\bRWY\s+([0-9]{2}[LRC]?)",
    # was: "ENGINE_LINE": r"TAKEOFF PERFORMANCE.*\n([N0-9A-Z\- ]+?)\n",
    "ENGINE_LINE": r"TAKEOFF PERFORMANCE[^\n]*\n([^\n]+)\n",
    "TYPE": r"N\d+\s+([A-Z0-9\- ]+?)\s+(LEAP|CFM|PW|GE|TRENT|XWB|CFM56|GE90|PW20|LEAP-1B|LEAP-1A)",
    # V-speeds (and related)
    "V1": r"\bV1\s+(\d{2,3})\b",
    "VR": r"\bVR\s+(\d{2,3})\b",
    "V2": r"\bV2\s+(\d{2,3})\b",
    "VREF": r"\bVREF(?:30|40)?\s+(\d{2,3})\b",
    "GREENDOT": r"\bGREEN\s+DOT\s+(\d{2,3})\b"
}

def parse_paste(text: str) -> Dict[str, Any]:
    t = text.replace("\u00a0"," ")
    out: Dict[str, Any] = {}
    for k, pat in PASTE_KEYS.items():
        m = re.search(pat, t, re.IGNORECASE)
        if not m: continue
        if k == "SEL":
            out["SEL"] = float(next(g for g in m.groups() if g))
        elif k == "FLAPS_OUT":
            g = next((g for g in m.groups() if g), None)
            out["FLAPS_OUT"] = (g.upper() if g else None)
        elif k == "TYPE":
            out["TYPE"] = m.group(1).strip().upper()
        elif k == "ENGINE_LINE":
            out["ENGINE_LINE"] = m.group(1).strip().upper()
        elif k in ("OAT","ELEV","WEIGHT","V1","VR","V2","VREF","GREENDOT"):
            out[k] = float(m.group(1))
        elif k == "QNH":
            q = m.group(1)
            out["QNH"] = float(q) if "." in q else round(float(q)/33.8639, 2)
        else:
            out[k] = m.group(1).strip().upper()
    return out

def guess_series_and_brand(parsed: Dict[str, Any]) -> tuple[str, str]:
    """
    Robustly infer airframe series + brand from TYPE and ENGINE_LINE.
    Handles slashes (e.g., CFM56-5B3/P) and hyphens (e.g., A321-200).
    """
    line = f"{parsed.get('TYPE','')} {parsed.get('ENGINE_LINE','')}".upper()
    # Normalize separators
    norm = line.replace("/", " ").replace("-", " ")
    tokens = norm.split()

    joined = " " .join(tokens)

    # Airbus first (specific → general)
    if "A380" in joined: return "airbus_a380", "airbus"
    if "A350" in joined: return "airbus_a350", "airbus"
    if "A321" in joined: return "airbus_a321", "airbus"
    if "A320" in joined: return "airbus_a320", "airbus"
    if "A330" in joined: return "airbus_a330", "airbus"

    # Boeing (specific → general)
    if any(k in joined for k in ["737 MAX", "B737 MAX", "737 8", "737-8".replace("-"," ")]):
        return "boeing_737_max8", "boeing"
    if any(k in joined for k in ["777 200", "B777 200", "B777", "777"]):
        return "boeing_777", "boeing"
    if "757" in joined: return "boeing_757", "boeing"
    if "787" in joined: return "boeing_787", "boeing"

    # Fallback by explicit brand words, if present
    if "AIRBUS" in joined: return "generic", "airbus"
    if "BOEING" in joined: return "generic", "boeing"

    # Last resort: infer from engine families (very weak heuristic)
    eng_hint = joined
    if any(k in eng_hint for k in ["LEAP 1A", "CFM56 5B", "TRENT XWB", "TRENT 970"]):
        return "generic", "airbus"
    if any(k in eng_hint for k in ["LEAP 1B", "GE90", "PW20", "CFM56 7"]):
        return "generic", "boeing"

    # Safe default
    return "generic", "boeing"


# ----------------------------- UI -----------------------------
st.title("SimBrief Takeoff Performance to Infinite Flight Converter Tool")

txt = st.text_area(
    "Paste your SimBrief-style TAKEOFF PERFORMANCE block:",
    height=220,
    placeholder="Paste the block here..."
)

mid = st.columns([1,4,1])[1]
with mid:
    go = st.button("Estimate N1%")

if go:
    data = parse_paste(txt or "")
    missing = []
    for need in ("OAT","QNH","ELEV","WEIGHT","SEL","THRUST"):
        if need not in data: missing.append(need)
    if missing:
        st.error("Some inputs are missing. Ensure your paste includes OAT, QNH, ELEV, WEIGHT, SEL TEMP, THRUST.")
        st.stop()

    series, brand = guess_series_and_brand(data)
    af = AIRFRAMES.get(series, {})
    engine_id = af.get("engine_id","generic")

    oat = data["OAT"]; sel = data["SEL"]; qnh = data["QNH"]; elev = data["ELEV"]
    tow_klb = data["WEIGHT"]; thrust_mode = data.get("THRUST","")
    bleeds_on = (data.get("BLEEDS","ON") == "ON"); anti_ice_on = (data.get("AICE","OFF") == "ON")

    res = estimate_n1(engine_id=engine_id, oat_c=oat, sel_temp_c=sel, qnh_inhg=qnh, elev_ft=elev,
                      tow_klb=tow_klb, bleeds_on=bleeds_on, anti_ice_on=anti_ice_on,
                      thrust_mode=thrust_mode, brand=brand, flaps=0,
                      header_text=data.get("ENGINE_LINE",""), series=series)
    n1 = float(res["n1"]); pa_ft = float(res["pa_ft"])
    selected_flaps_label = res["flaps_label"]

    trim_pct = estimate_trim_if(series=series, brand=brand,
                                flaps_label=selected_flaps_label, tow_klb=tow_klb)

   # -------- Metrics strip (top) – brighter, larger, clearer --------
    v1 = data.get("V1"); vr = data.get("VR"); v2 = data.get("V2")
    vref = data.get("VREF"); gdot = data.get("GREENDOT")

    metrics_html = f'''
    <div class="info-bar">
    <div class="info-kv"><span class="k">N1</span><span class="v">{n1:.1f}%</span></div>
    <div class="info-kv"><span class="k">Flaps</span><span class="v">{selected_flaps_label}</span></div>
    <div class="info-kv"><span class="k">Trim</span><span class="v">{trim_pct:.0f}%</span></div>
    <div class="info-kv"><span class="k">PA</span><span class="v">{pa_ft:.0f} ft</span></div>
    '''
    if v1:  metrics_html += f'<div class="info-kv"><span class="k">V1</span><span class="v">{int(v1)}</span></div>'
    if vr:  metrics_html += f'<div class="info-kv"><span class="k">VR</span><span class="v">{int(vr)}</span></div>'
    if v2:  metrics_html += f'<div class="info-kv"><span class="k">V2</span><span class="v">{int(v2)}</span></div>'
    if vref: metrics_html += f'<div class="info-kv"><span class="k">VREF</span><span class="v">{int(vref)}</span></div>'
    if gdot: metrics_html += f'<div class="info-kv"><span class="k">Green Dot</span><span class="v">{int(gdot)}</span></div>'
    metrics_html += "</div>"

    st.markdown(metrics_html, unsafe_allow_html=True)


    # -------- Diagram row --------
    # -------- Diagram row (consistent spacing) --------
col_dial, col_right = st.columns([3, 2], gap="small")

with col_dial:
    n1_cap = af.get("n1_max_pct", 102.0)
    if (brand or "").lower() == "boeing":
        fig = draw_n1_dial_boeing(n1_percent=n1, max_n1_pct=n1_cap)
    else:
        fig = draw_n1_dial_airbus(n1_percent=n1, max_n1_pct=n1_cap)
    # Wrap to help CSS trim extra spacing
    with st.container():
        st.pyplot(fig, use_container_width=False)

with col_right:
    s1, s2 = st.columns(2, gap="small")
    with s1:
        detents = get_flap_detents(brand, series)
        fig_f = draw_flap_detents_small(brand, detents, selected_flaps_label)
        with st.container():
            st.pyplot(fig_f, use_container_width=False)
    with s2:
        fig_t = draw_trim_bar(trim_pct)
        with st.container():
            st.pyplot(fig_t, use_container_width=False)


    # -------- Performance card (bottom) --------
    perf = f"""
    <div style="margin-top:10px; padding:14px; border:1px solid #2a3542; border-radius:12px; background:#111824; color:#e9f0f6;">
      <div style="font-size:1.08rem; font-weight:700; margin-bottom:6px;">Performance Summary</div>
      <div>Series: <b style="color:#fff;">{series.replace('_',' ').title()}</b> &nbsp;|&nbsp; Brand: <b style="color:#fff;">{brand.title()}</b> &nbsp;|&nbsp; Thrust: <b style="color:#fff;">{thrust_mode}</b></div>
      <div>SEL Temp: <b style="color:#fff;">{sel:.0f}°C</b> &nbsp;|&nbsp; OAT: <b style="color:#fff;">{oat:.0f}°C</b> &nbsp;|&nbsp; QNH: <b style="color:#fff;">{qnh:.2f} inHg</b> &nbsp;|&nbsp; Field Elev: <b style="color:#fff;">{elev:.0f} ft</b> &nbsp;|&nbsp; PA: <b style="color:#fff;">{pa_ft:.0f} ft</b></div>
    """

    if any([v1, vr, v2, vref, gdot]):
        perf += "<div>"
        if v1:  perf += f"V1: <b>{int(v1)}</b> &nbsp; "
        if vr:  perf += f"VR: <b>{int(vr)}</b> &nbsp; "
        if v2:  perf += f"V2: <b>{int(v2)}</b> &nbsp; "
        if vref: perf += f"VREF: <b>{int(vref)}</b> &nbsp; "
        if gdot: perf += f"Green Dot: <b>{int(gdot)}</b>"
        perf += "</div>"
    perf += "</div>"
    st.markdown(perf, unsafe_allow_html=True)

    st.caption("Note: Heuristic estimator for Infinite Flight only. Not for real-world flight operations.")
