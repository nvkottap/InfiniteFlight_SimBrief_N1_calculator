# app.py
# Infinite Flight - N1% Estimator (Sim-use only)
# Paste TAKEOFF PERFORMANCE text → estimate N1% + show a Takeoff Performance Card.
# Edits: removed live engine tweaks, button now "Estimate N1%", added performance card.

import re
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Config / Data Loading
# --------------------------

CAL_PATH = Path(__file__).with_name("engine_calibrations.json")

def load_calibrations() -> Dict[str, Any]:
    if CAL_PATH.exists():
        with open(CAL_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    # Minimal fallback if file missing (should not happen in normal use)
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

def detect_engine_id(raw_engine_line: str) -> Tuple[str, str]:
    pretty = raw_engine_line.strip()
    for key, eng_id in ENGINE_ALIASES.items():
        if key.lower() in pretty.lower():
            return eng_id, pretty
    return "generic", pretty

def pressure_altitude_ft(field_elev_ft: float, qnh_inhg: float) -> float:
    # Approximation: PA = Elev + (29.92 - QNH)*1000
    return float(field_elev_ft) + (29.92 - float(qnh_inhg)) * 1000.0

def parse_takeoff_text(txt: str) -> Dict[str, Any]:
    """
    Parse the pasted TAKEOFF PERFORMANCE block.
    """
    d = {}

    # Header: e.g., "N808SB B737 MAX 8 LEAP-1B28"
    header_line = next((l for l in txt.splitlines() if re.search(r"\b[A-Z0-9]{3,}.*", l)), "")
    d["header"] = header_line
    d["engine_id"], d["engine_name"] = detect_engine_id(header_line)

    # Airport / runway (optional; used only for the card)
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

    return d

def derate_level_from_mode(mode: str) -> int:
    if not mode:
        return 0
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
      N1 = a + b*(SEL - OAT) + c*(PA_kft) + d*(derate_steps) + e*(TOW - w_ref) + bleed/anti-ice taps
    """
    cal = CAL.get(engine_id, CAL.get("generic"))
    a = cal["a"]; b = cal["b_temp"]; c = cal["c_pa"]; d = cal["d_derate"]; w_ref = cal["w_ref"]; e = cal["e_wt"]

    deltaT = (sel_temp_c - oat_c) if (sel_temp_c is not None and oat_c is not None) else 0.0
    pa_ft = pressure_altitude_ft(elev_ft or 0.0, qnh_inhg or 29.92)
    pa_kft = max(pa_ft, 0.0) / 1000.0

    derate_steps = derate_level_from_mode(thrust_mode or "")
    wt_term = ((tow_klb or w_ref) - w_ref)

    n1 = a + b*deltaT + c*pa_kft + d*derate_steps + e*wt_term
    if bleeds_on:
        n1 += 0.2
    if anti_ice_on:
        n1 += 0.5

    conf_pm = 0.4 + 0.05*pa_kft + 0.1*derate_steps
    return {"n1": n1, "conf_pm": conf_pm, "pa_ft": pa_ft}

def draw_n1_gauge(n1: float, conf_pm: float):
    fig, ax = plt.subplots(figsize=(7, 1.2))
    ax.set_xlim(80, 102)
    ax.set_ylim(0, 1)
    ax.axvspan(max(80, n1-conf_pm), min(102, n1+conf_pm), alpha=0.15)
    ax.axvline(n1, linewidth=3)
    ax.set_yticks([])
    ax.set_xlabel("N1 (%)")
    ax.set_title(f"N1 target ≈ {n1:.1f}%  (±{conf_pm:.1f}%)")
    return fig

def draw_perf_card(meta: Dict[str, Any], result: Dict[str, Any]):
    """
    Renders a simple Takeoff Performance Card as a matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.axis("off")

    # Header
    title = (meta.get("header") or "TAKEOFF PERFORMANCE").strip()
    ax.text(0.02, 0.92, title, fontsize=12, fontweight="bold")

    # Left column (aircraft/config)
    left_y = 0.82
    def L(label, value):
        nonlocal left_y
        ax.text(0.02, left_y, f"{label}: {value}", fontsize=10)
        left_y -= 0.06

    L("Airport", f'{meta.get("apt_icao") or "-"} / RWY {meta.get("runway") or "-"} / {meta.get("rwy_cond","-")}')
    L("Flaps", meta.get("flaps") or "-")
    L("Thrust", meta.get("thrust_mode") or "-")
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

    # Footer
    ax.text(0.02, 0.05, "Simulation aid only • Heuristic estimate", fontsize=8)
    return fig

# --------------------------
# UI
# --------------------------

st.set_page_config(page_title="IF Takeoff N1 Estimator", page_icon="🛫", layout="wide")

st.title("🛫 Infinite Flight – Takeoff N1% Estimator")
st.caption("Paste your TAKEOFF PERFORMANCE text → get an estimated N1% and a Takeoff Performance Card. **Sim-use only; not for real-world operations.**")

txt = st.text_area(
    "Paste your TAKEOFF PERFORMANCE text:",
    height=280,
    placeholder="Paste the whole block here…"
)

go = st.button("Estimate N1%")  # (2) Button label updated

st.markdown("---")

if go and txt.strip():
    data = parse_takeoff_text(txt)

    # Engine autodetect (fallback to generic if missing)
    detected_engine = data["engine_id"]
    pretty_engine = data["engine_name"] or "Engine not found in header"
    st.write(f"**Detected engine (from header):** {pretty_engine}")

    # Compute
    if None in (data["oat_c"], data["sel_temp_c"], data["qnh_inhg"], data["elev_ft"], data["tow_klb"]):
        st.warning("Some inputs are missing. Make sure your paste includes OAT, QNH, Elevation, Weight, and SEL TEMP.")
    else:
        res = estimate_n1(
            engine_id=detected_engine,
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
        m1, m2, m3 = st.columns(3)
        with m1: st.metric("Estimated N1%", f"{n1:.1f}%", f"±{conf:.1f}%")
        with m2: st.metric("Pressure Altitude", f'{res["pa_ft"]:,.0f} ft')
        with m3:
            if data["v1"] and data["vr"] and data["v2"]:
                st.metric("V1 / VR / V2", f'{data["v1"]} / {data["vr"]} / {data["v2"]}')

        # Gauge
        st.subheader("Takeoff Thrust Target")
        fig_g = draw_n1_gauge(n1, conf)
        st.pyplot(fig_g, use_container_width=True)

        # (3) Takeoff Performance Card
        st.subheader("Takeoff Performance Card")
        fig_card = draw_perf_card(data, res)
        st.pyplot(fig_card, use_container_width=True)

        # Sensitivities (kept—useful for tuning FLEX choices)
        st.subheader("Sensitivity: N1 vs Selected Temperature")
        sel_range = np.arange(max(0, data["sel_temp_c"]-12), data["sel_temp_c"]+13, 1)
        n1_curve = []
        for s in sel_range:
            tmp = estimate_n1(detected_engine, data["oat_c"], float(s), data["qnh_inhg"], data["elev_ft"],
                              data["tow_klb"], data["bleeds_on"], data["anti_ice_on"], data["thrust_mode"])
            n1_curve.append(tmp["n1"])
        fig2, ax2 = plt.subplots(figsize=(7,3))
        ax2.plot(sel_range, n1_curve, linewidth=2)
        ax2.axvline(data["sel_temp_c"], linestyle="--")
        ax2.set_xlabel("Selected Temperature (°C)")
        ax2.set_ylabel("Estimated N1 (%)")
        st.pyplot(fig2, use_container_width=True)

        st.subheader("Sensitivity: N1 vs Field Elevation (Pressure Altitude Effect)")
        elev_demo = np.arange(0, 9001, 500)
        pa_curve = []
        for e_ft in elev_demo:
            tmp = estimate_n1(detected_engine, data["oat_c"], data["sel_temp_c"], data["qnh_inhg"], float(e_ft),
                              data["tow_klb"], data["bleeds_on"], data["anti_ice_on"], data["thrust_mode"])
            pa_curve.append(tmp["n1"])
        fig3, ax3 = plt.subplots(figsize=(7,3))
        ax3.plot(elev_demo, pa_curve, linewidth=2)
        ax3.axvline(float(data["elev_ft"]), linestyle="--")
        ax3.set_xlabel("Field Elevation (ft) at same QNH")
        ax3.set_ylabel("Estimated N1 (%)")
        st.pyplot(fig3, use_container_width=True)
