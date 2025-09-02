# app.py
# Infinite Flight - N1% Estimator (Sim-use only)
# UX tweaks: better engine detection label, larger top section, smaller sensitivity plots, seaborn styling.

import re
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # ← NEW
import math

def _rgba_to_mpl(rgba_str, fallback="#cccccc", alpha_fallback=0.2):
    """
    Convert 'rgba(r,g,b,a)' to (r/255,g/255,b/255,a). If parsing fails, fallback.
    """
    try:
        import re
        m = re.match(r"rgba\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([0-9.]+)\)", rgba_str.strip(), flags=re.I)
        if not m: return fallback
        r,g,b,a = m.groups()
        return (int(r)/255.0, int(g)/255.0, int(b)/255.0, float(a))
    except Exception:
        return (0.8, 0.8, 0.8, alpha_fallback)


# ----- Seaborn global style (cleaner visuals) -----
sns.set_theme(style="whitegrid", context="talk")  # slightly larger fonts overall

# --------------------------
# Config / Data Loading
# --------------------------

CAL_PATH = Path(__file__).with_name("engine_calibrations.json")

def load_calibrations() -> Dict[str, Any]:
    if CAL_PATH.exists():
        with open(CAL_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "leap-1b28": {"a": 93.5, "b_temp": -0.10, "c_pa": 0.25, "d_derate": -2.0, "w_ref": 160.0, "e_wt": 0.01},
        "generic":   {"a": 92.0, "b_temp": -0.10, "c_pa": 0.20, "d_derate": -2.0, "w_ref": 300.0, "e_wt": 0.00}
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

# Short, human-friendly labels
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

# --- MFD Style Palettes ---
MFD_STYLES = {
    "boeing": {
        "bg": "#0a0f14",           # deep gray-blue (PFD/ND look)
        "tick": "#d8e6ff",         # off-white ticks
        "label": "#d8e6ff",        # off-white labels
        "needle": "#ffd21f",       # Boeing-ish yellow bug
        "band": "rgba(255,210,31,0.20)",  # semi transparent yellow band (we'll parse)
        "green": "rgba(60, 190, 120, 0.25)",  # normal band
        "hub": "#d8e6ff",
        "title": "#d8e6ff",
        "grid_alpha": 0.0
    },
    "airbus": {
        "bg": "#0b0b0b",           # darker charcoal
        "tick": "#bfe3ff",         # light cyan-ish Airbus glass look
        "label": "#bfe3ff",
        "needle": "#00ffd0",       # Airbus cyan cue
        "band": "rgba(0,255,208,0.18)",
        "green": "rgba(100, 220, 140, 0.25)",
        "hub": "#bfe3ff",
        "title": "#bfe3ff",
        "grid_alpha": 0.0
    }
}
def detect_engine_in_text(txt: str) -> Tuple[str, Optional[str]]:
    """
    Scan the entire pasted text for any known engine token.
    Return (engine_id, matched_token or None).
    """
    for token, eng_id in ENGINE_ALIASES.items():
        if re.search(rf"\b{re.escape(token)}\b", txt, flags=re.I):
            return eng_id, token
    return "generic", None

def pressure_altitude_ft(field_elev_ft: float, qnh_inhg: float) -> float:
    return float(field_elev_ft) + (29.92 - float(qnh_inhg)) * 1000.0

def parse_takeoff_text(txt: str) -> Dict[str, Any]:
    d = {}

    # Keep original header (for the card title), but DO NOT use it to decide engine label
    header_line = next((l for l in txt.splitlines() if re.search(r"\b[A-Z0-9]{3,}.*", l)), "")
    d["header"] = header_line

    # Airport / runway (optional; used on the card)
    apt = re.search(r"APT\s+([A-Z]{4})/([A-Z]{3})", txt)
    d["apt_icao"] = apt.group(1) if apt else None
    rwy = re.search(r"RWY\s+(\d+[LRC]?)/\+?0?", txt)
    d["runway"] = rwy.group(1) if rwy else None

    # Weather
    m = re.search(r"OAT\s+(-?\d+)", txt);        d["oat_c"] = float(m.group(1)) if m else None
    m = re.search(r"QNH\s+(\d{2}\.\d{2})", txt); d["qnh_inhg"] = float(m.group(1)) if m else None
    m = re.search(r"ELEV\s+(-?\d+)", txt);       d["elev_ft"] = float(m.group(1)) if m else None

    # Inputs / outputs
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

    # Engine detection (robust): scan whole text for known tokens
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

def draw_n1_gauge(n1: float, conf_pm: float):
    # Larger title for emphasis (bigger top section)
    fig, ax = plt.subplots(figsize=(8.5, 1.5))  # slightly wider & taller than before
    ax.set_xlim(80, 102)
    ax.set_ylim(0, 1)
    ax.axvspan(max(80, n1-conf_pm), min(102, n1+conf_pm), alpha=0.15)
    ax.axvline(n1, linewidth=3)
    ax.set_yticks([])
    ax.set_xlabel("N1 (%)")
    ax.set_title(f"N1 target ≈ {n1:.1f}%  (±{conf_pm:.1f}%)", fontsize=18)  # bigger title
    return fig

def draw_n1_dial(n1: float, conf_pm: float, style: str = "boeing"):
    """
    Compact MFD-style N1 dial with Boeing/Airbus themes.
    Arc 70–110%, ticks q5%, labels q10%, green normal band, target bug, ± band.
    """
    # Style picks
    S = MFD_STYLES.get(style.lower(), MFD_STYLES["boeing"])
    bg = S["bg"]
    c_tick = S["tick"]
    c_label = S["label"]
    c_needle = S["needle"]
    c_band = _rgba_to_mpl(S["band"])
    c_green = _rgba_to_mpl(S["green"])
    c_hub = S["hub"]
    c_title = S["title"]

    min_n1, max_n1 = 70, 110
    start_deg, end_deg = -210, 30
    span_deg = end_deg - start_deg

    def n1_to_angle(v):
        v = max(min(v, max_n1), min_n1)
        frac = (v - min_n1) / (max_n1 - min_n1)
        return math.radians(start_deg + frac * span_deg)

    fig, ax = plt.subplots(figsize=(3.8, 3.8))
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    R_outer = 1.0
    R_inner = 0.74

    # Dial face
    theta = np.linspace(math.radians(start_deg), math.radians(end_deg), 300)
    x_outer = R_outer * np.cos(theta); y_outer = R_outer * np.sin(theta)
    x_inner = R_inner * np.cos(theta[::-1]); y_inner = R_inner * np.sin(theta[::-1])
    ax.fill(np.concatenate([x_outer, x_inner]),
            np.concatenate([y_outer, y_inner]),
            color=(1,1,1,0.04))

    # Green "normal" band (80–102%)
    g0, g1 = 80, 102
    t_band = np.linspace(n1_to_angle(g0), n1_to_angle(g1), 120)
    xb_o = R_outer * np.cos(t_band); yb_o = R_outer * np.sin(t_band)
    xb_i = R_inner * np.cos(t_band[::-1]); yb_i = R_inner * np.sin(t_band[::-1])
    ax.fill(np.concatenate([xb_o, xb_i]),
            np.concatenate([yb_o, yb_i]),
            color=c_green)

    # ± band around target
    band_lo, band_hi = n1 - conf_pm, n1 + conf_pm
    t_pm = np.linspace(n1_to_angle(band_lo), n1_to_angle(band_hi), 80)
    xp_o = R_outer * np.cos(t_pm); yp_o = R_outer * np.sin(t_pm)
    xp_i = (R_outer - 0.05) * np.cos(t_pm[::-1]); yp_i = (R_outer - 0.05) * np.sin(t_pm[::-1])
    ax.fill(np.concatenate([xp_o, xp_i]),
            np.concatenate([yp_o, yp_i]),
            color=c_band)

    # Tick marks & labels
    for val in range(70, 111, 5):
        ang = n1_to_angle(val)
        is_major = (val % 10 == 0)
        r2 = R_inner - (0.06 if is_major else 0.03)
        x1, y1 = R_outer * np.cos(ang), R_outer * np.sin(ang)
        x2, y2 = r2 * np.cos(ang), r2 * np.sin(ang)
        ax.plot([x1, x2], [y1, y2], color=c_tick, linewidth=2)

        if is_major:
            rl = r2 - 0.10
            xl, yl = rl * np.cos(ang), rl * np.sin(ang)
            ax.text(xl, yl, f"{val}", ha="center", va="center",
                    fontsize=9, color=c_label)

    # Target “bug” (needle)
    ang_tgt = n1_to_angle(n1)
    xh, yh = (R_inner - 0.18) * np.cos(ang_tgt), (R_inner - 0.18) * np.sin(ang_tgt)
    ax.plot([0, xh], [0, yh], color=c_needle, linewidth=3)
    hub = plt.Circle((0, 0), 0.05, color=c_hub)
    ax.add_artist(hub)

    # Readouts (Boeing/Airbus-ish)
    ax.text(0, -0.55, f"N1  {n1:.1f}%", ha="center", va="center",
            fontsize=14, fontweight="bold", color=c_title)
    ax.text(0, -0.72, f"±{conf_pm:.1f}%", ha="center", va="center",
            fontsize=9, color=c_label)

    return fig



def draw_flap_schematic(flaps_value: int, airframe_hint: str = "", style: str = "boeing"):
    """
    Simple wing planform; theme colors follow dial style.
    """
    S = MFD_STYLES.get(style.lower(), MFD_STYLES["boeing"])
    bg = S["bg"]
    ink = S["label"]

    fig, ax = plt.subplots(figsize=(4.0, 2.4))
    ax.axis("off")
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)

    # Wing body
    ax.add_patch(plt.Rectangle((1, 2.0), 8, 0.40, linewidth=1.4, fill=False, edgecolor=ink))

    # Slats (leading edge)
    slat_defl = min(max(int(flaps_value or 0), 0), 5) * 0.06
    ax.add_patch(plt.Rectangle((1, 2.45), 8, 0.12, linewidth=1, fill=True, color=(1,1,1,0.06), edgecolor=None))
    ax.plot([1, 9], [2.57, 2.57 + slat_defl], linewidth=3, color=ink, alpha=0.7)

    # Flaps (trailing edge)
    flap_defl = min(max(int(flaps_value or 0), 0), 25) * 0.02
    segments = [(1.5, 1.9, 2.8), (4.0, 1.9, 2.8), (6.5, 1.9, 2.8)]
    for (sx, sy, w) in segments:
        ax.add_patch(plt.Rectangle((sx, sy), w, 0.10, linewidth=1, fill=True, color=(1,1,1,0.06), edgecolor=None))
        ax.plot([sx, sx + w], [sy, sy - flap_defl], linewidth=3, color=ink, alpha=0.7)

    # Label
    family = "Airbus" if "A3" in airframe_hint.upper() else ("Boeing" if "B7" in airframe_hint.upper() else "")
    label = f"Flaps {flaps_value}" + (f" • {family}" if family else "")
    ax.text(5, 0.35, label, ha="center", va="center", fontsize=11, fontweight="bold", color=ink)

    return fig


def draw_perf_card(meta: Dict[str, Any], result: Dict[str, Any]):
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.axis("off")

    # Header
    title = (meta.get("header") or "TAKEOFF PERFORMANCE").strip()
    ax.text(0.02, 0.92, title, fontsize=13, fontweight="bold")

    # Left column (aircraft/config)
    left_y = 0.82
    def L(label, value):
        nonlocal left_y
        ax.text(0.02, left_y, f"{label}: {value}", fontsize=10)
        left_y -= 0.06

    L("Airport", f'{meta.get("apt_icao") or "-"} / RWY {meta.get("runway") or "-"} / {meta.get("rwy_cond","-")}')
    L("Engine",  meta.get("engine_pretty", "Generic"))
    L("Flaps",   meta.get("flaps") or "-")
    L("Thrust",  meta.get("thrust_mode") or "-")
    L("SEL TEMP (°C)", meta.get("sel_temp_c") or "-")
    L("Bleeds / A-I", f'{"ON" if meta.get("bleeds_on") else "OFF"} / {"ON" if meta.get("anti_ice_on") else "OFF"}')
    L("TOW (k lb)", meta.get("tow_klb") or "-")

    # Middle column (env)
    mid_y = 0.82
    def M(label, value):
        nonlocal mid_y
        ax.text(0.42, mid_y, f"{label}: {value}", fontsize=10)
        mid_y -= 0.06

    M("OAT (°C)", meta.get("oat_c") or "-")
    M("QNH (inHg)", meta.get("qnh_inhg") or "-")
    M("Field Elev (ft)", meta.get("elev_ft") or "-")
    M("Press Alt (ft)", f'{result.get("pa_ft",0):,.0f}')
    if meta.get("v1") and meta.get("vr") and meta.get("v2"):
        M("V1 / VR / V2", f'{meta["v1"]} / {meta["vr"]} / {meta["v2"]}')

    # Right column (N1)
    ax.text(0.72, 0.82, "N1 Target", fontsize=10, fontweight="bold")
    ax.text(0.72, 0.76, f'{result["n1"]:.1f}% ±{result["conf_pm"]:.1f}%', fontsize=16)
    # Mini gauge
    g_left, g_bottom, g_width, g_height = 0.68, 0.18, 0.28, 0.16
    ax_in = fig.add_axes([g_left, g_bottom, g_width, g_height])
    ax_in.set_xlim(80, 102)
    ax_in.set_ylim(0, 1)
    ax_in.axvspan(max(80, result["n1"]-result["conf_pm"]), min(102, result["n1"]+result["conf_pm"]), alpha=0.15)
    ax_in.axvline(result["n1"], linewidth=3)
    ax_in.set_yticks([])
    ax_in.set_xlabel("N1 (%)", fontsize=8)

    ax.text(0.02, 0.05, "Simulation aid only • Heuristic estimate", fontsize=8)
    return fig

# --------------------------
# UI
# --------------------------

st.set_page_config(page_title="IF Takeoff N1 Estimator", page_icon="🛫", layout="wide")

# Slightly larger main title (bigger top section)
st.markdown("<h1 style='font-size:2.1rem; margin-bottom:0;'>🛫 Infinite Flight – Takeoff N1% Estimator</h1>", unsafe_allow_html=True)
st.caption("Paste your TAKEOFF PERFORMANCE text → get an estimated N1% and a Takeoff Performance Card. **Sim-use only; not for real-world operations.**")

txt = st.text_area(
    "Paste your TAKEOFF PERFORMANCE text:",
    height=280,
    placeholder="Paste the whole block here…"
)

go = st.button("Estimate N1%")

st.markdown("---")

if go and txt.strip():
    data = parse_takeoff_text(txt)

    # Clean engine label (no “from header”, never show TAKEOFF PERFORMANCE)
    st.write(f"**Detected engine:** {data['engine_pretty']}")

    # Compute
    if None in (data["oat_c"], data["sel_temp_c"], data["qnh_inhg"], data["elev_ft"], data["tow_klb"]):
        st.warning("Some inputs are missing. Make sure your paste includes OAT, QNH, Elevation, Weight, and SEL TEMP.")
    else:
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

        # Top metrics — Streamlit metrics are fixed size, so we emphasize via bigger page title (above),
        # and a larger gauge title below. Also group metrics in wider columns to feel "larger".
        m1, m2, m3 = st.columns([1.2, 1, 1])
        with m1: st.metric("Estimated N1%", f"{n1:.1f}%", f"±{conf:.1f}%")
        with m2: st.metric("Pressure Altitude", f'{res["pa_ft"]:,.0f} ft')
        with m3:
            if data["v1"] and data["vr"] and data["v2"]:
                st.metric("V1 / VR / V2", f'{data["v1"]} / {data["vr"]} / {data["v2"]}')

        # Bigger “Takeoff Thrust Target” section (gauge has larger title + figure size)
        st.markdown("<h3 style='margin-top:0.5rem;'>Takeoff Thrust Target</h3>", unsafe_allow_html=True)
        # OLD: fig_g = draw_n1_gauge(n1, conf)
        # st.pyplot(fig_g, use_container_width=True)

        # Style picker (Boeing vs Airbus)
        style = st.segmented_control("MFD style", options=["Boeing", "Airbus"], default="Boeing")

        st.markdown("<h3 style='margin-top:0.5rem;'>Takeoff Thrust Target</h3>", unsafe_allow_html=True)
        c_g1, c_g2 = st.columns([1, 1], gap="large")

        with c_g1:
            fig_dial = draw_n1_dial(n1, conf, style=style.lower())
            st.pyplot(fig_dial, use_container_width=False)

        with c_g2:
            airframe_hint = data.get("header", "")
            fig_flaps = draw_flap_schematic(int(data.get("flaps") or 0), airframe_hint=airframe_hint, style=style.lower())
            st.pyplot(fig_flaps, use_container_width=False)


        # Performance Card
        st.subheader("Takeoff Performance Card")
        fig_card = draw_perf_card(data, res)
        st.pyplot(fig_card, use_container_width=True)

        # Smaller sensitivity plots: put them side-by-side with smaller fig sizes
        st.subheader("Sensitivity")
        c1, c2 = st.columns(2, gap="large")

        with c1:
            st.caption("N1 vs Selected Temperature")
            sel_range = np.arange(max(0, data["sel_temp_c"]-12), data["sel_temp_c"]+13, 1)
            n1_curve = []
            for s in sel_range:
                tmp = estimate_n1(data["engine_id"], data["oat_c"], float(s), data["qnh_inhg"], data["elev_ft"],
                                  data["tow_klb"], data["bleeds_on"], data["anti_ice_on"], data["thrust_mode"])
                n1_curve.append(tmp["n1"])
            fig2, ax2 = plt.subplots(figsize=(4.6, 2.6))  # ← smaller
            ax2.plot(sel_range, n1_curve, linewidth=2)
            ax2.axvline(data["sel_temp_c"], linestyle="--")
            ax2.set_xlabel("Selected Temperature (°C)")
            ax2.set_ylabel("Estimated N1 (%)")
            st.pyplot(fig2, use_container_width=True)

        with c2:
            st.caption("N1 vs Field Elevation (Pressure Altitude Effect)")
            elev_demo = np.arange(0, 9001, 500)
            pa_curve = []
            for e_ft in elev_demo:
                tmp = estimate_n1(data["engine_id"], data["oat_c"], data["sel_temp_c"], data["qnh_inhg"], float(e_ft),
                                  data["tow_klb"], data["bleeds_on"], data["anti_ice_on"], data["thrust_mode"])
                pa_curve.append(tmp["n1"])
            fig3, ax3 = plt.subplots(figsize=(4.6, 2.6))  # ← smaller
            ax3.plot(elev_demo, pa_curve, linewidth=2)
            ax3.axvline(float(data["elev_ft"]), linestyle="--")
            ax3.set_xlabel("Field Elevation (ft) at same QNH")
            ax3.set_ylabel("Estimated N1 (%)")
            st.pyplot(fig3, use_container_width=True)
