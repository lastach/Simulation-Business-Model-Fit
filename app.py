# app.py
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Simulation #3 — Business Model Fit (ThermaLoop)",
    page_icon="📈",
    layout="wide",
)

# ======================================================================================
# DATA MODELS & CONSTANTS
# ======================================================================================

SEGMENTS = {
    "homeowner": {
        "name": "Homeowners (B2C self-serve)",
        # Base funnel (per 10k monthly visits; we’ll scale by channel volume)
        "base_conv": {
            "visit_to_trial": 0.06,
            "trial_to_activation": 0.35,
            "activation_to_paid": 0.20,
            "retention_month_1": 0.86,  # 1-month retention prob
        },
        # Price elasticity (ARPU sensitivity); 1.0 = neutral
        "price_elasticity": 1.15,
        # Paywall friction baseline (higher = more drop between trial→activation)
        "paywall_friction": 1.10,
        # Churn drivers
        "churn_sensitivity": {
            "time_to_value": 1.10,  # slower TTV => more churn
            "price": 1.05,          # higher price => more churn
        },
        # ARPU base (paid users)
        "arpu_base": 11.0,  # $/mo
        # Whether sales-led lifts retention/ACV if used (mostly B2B, so False here)
        "sales_led_benefit": False,
    },
    "landlord": {
        "name": "Small Landlords (B2B light-sales)",
        "base_conv": {
            "visit_to_trial": 0.04,
            "trial_to_activation": 0.42,
            "activation_to_paid": 0.35,
            "retention_month_1": 0.90,
        },
        "price_elasticity": 0.9,     # less elastic (can bear more price)
        "paywall_friction": 0.95,    # less sensitive to paywalls
        "churn_sensitivity": {
            "time_to_value": 1.00,
            "price": 1.02,
        },
        "arpu_base": 18.0,
        "sales_led_benefit": True,
    },
    "installer": {
        "name": "HVAC Installers (B2B2C pro tool)",
        "base_conv": {
            "visit_to_trial": 0.03,
            "trial_to_activation": 0.48,
            "activation_to_paid": 0.40,
            "retention_month_1": 0.92,
        },
        "price_elasticity": 0.85,
        "paywall_friction": 0.90,
        "churn_sensitivity": {
            "time_to_value": 0.95,  # less sensitive; workflow value is sticky
            "price": 1.00,
        },
        "arpu_base": 24.0,
        "sales_led_benefit": True,
    },
}

DELIVERY_LEVELS = {
    # impacts COGS per paid unit, Time-to-Value (TTV), Support load
    "Manual": {"cogs_mult": 1.00, "ttv_days": 5, "support": "High"},
    "Hybrid": {"cogs_mult": 0.75, "ttv_days": 3, "support": "Medium"},
    "Automated": {"cogs_mult": 0.55, "ttv_days": 1, "support": "Low"},
}

UNIT_OF_VALUE = ["per home", "per rental unit", "per installer seat/kit"]

MODELS = {
    "Hardware + Subscription": {"note": "Device margin + monthly SaaS. Higher upfront CAC acceptable."},
    "Subscription only": {"note": "Pure SaaS. Lower upfront value proof; must nail activation."},
    "Usage-based": {"note": "Metered by units/automation minutes. Aligns price to value."},
    "Tiered": {"note": "Good/Better/Best features; segments by willingness-to-pay."},
    "Freemium": {"note": "High top-of-funnel; conversion relies on aha+paywall."},
    "Annual Contract": {"note": "Higher ACV and commitment; longer cycles and CAC."},
}

CHANNELS = {
    "Content/SEO": {
        "ramp": "slow",
        "fit": {"homeowner": 0.9, "landlord": 0.7, "installer": 0.6},
        "cac_mean": 35,
        "cac_std": 10,
        "ceiling": 0.35,  # share of volume before diminishing returns
    },
    "Referral/Partner": {
        "ramp": "medium",
        "fit": {"homeowner": 0.8, "landlord": 1.0, "installer": 1.1},
        "cac_mean": 50,
        "cac_std": 15,
        "ceiling": 0.40,
    },
    "Outbound/Email": {
        "ramp": "medium",
        "fit": {"homeowner": 0.7, "landlord": 1.1, "installer": 1.0},
        "cac_mean": 90,
        "cac_std": 25,
        "ceiling": 0.30,
    },
    "Paid Social/Search": {
        "ramp": "fast",
        "fit": {"homeowner": 1.1, "landlord": 0.8, "installer": 0.7},
        "cac_mean": 120,
        "cac_std": 35,
        "ceiling": 0.50,
    },
    "Events/Webinars": {
        "ramp": "slow",
        "fit": {"homeowner": 0.6, "landlord": 1.1, "installer": 1.1},
        "cac_mean": 140,
        "cac_std": 40,
        "ceiling": 0.25,
    },
    "App Directory/Marketplace": {
        "ramp": "fast",
        "fit": {"homeowner": 0.9, "landlord": 1.0, "installer": 1.0},
        "cac_mean": 80,
        "cac_std": 20,
        "ceiling": 0.30,
    },
}

RAMP_MULT = {"slow": 0.6, "medium": 0.8, "fast": 1.0}

# Random market events (stochastic, mild)
MARKET_EVENTS = [
    {"name": "Seasonality dip", "mult_visits": 0.90, "mult_cac": 1.05, "desc": "Lower inbound; ads a bit pricier."},
    {"name": "Partner boost", "mult_visits": 1.12, "mult_cac": 0.95, "desc": "Referral spike; slightly cheaper CAC."},
    {"name": "Ad inventory shock", "mult_visits": 1.00, "mult_cac": 1.15, "desc": "Ads cost more; similar traffic."},
    {"name": "Rebate surge", "mult_visits": 1.08, "mult_cac": 0.92, "desc": "More interest from incentives."},
    {"name": "No major shocks", "mult_visits": 1.00, "mult_cac": 1.00, "desc": "Steady state."},
]

# Scoring weights
SCORES = {
    "Model Coherence": 30,
    "Unit Economics": 30,
    "GTM Fit & Focus": 20,
    "Iteration Quality": 10,
    "Evidence Plan": 10,
}

# ======================================================================================
# STATE
# ======================================================================================
def init_state():
    ss = st.session_state
    ss.setdefault("stage", "intro")
    ss.setdefault("segment", None)
    ss.setdefault("secondary_segment", None)

    ss.setdefault("value_prop", {
        "core_job": "",
        "outcome": "",
        "proof": "",
        "promise": "Standard",
    })
    ss.setdefault("delivery", "Hybrid")
    ss.setdefault("unit_value", "per home")
    ss.setdefault("model", "Tiered")

    # Pricing tiers, can be interpreted per selected unit
    ss.setdefault("pricing", {"good": 9.0, "better": 15.0, "best": 25.0})
    # Paywall strictness: 0 (generous) to 2 (strict)
    ss.setdefault("paywall", 1)

    # Channels → tokens (0..100 total recommended)
    ss.setdefault("channels", {ch: 0 for ch in CHANNELS})
    ss.setdefault("sales_led", False)  # toggle for B2B flavors

    # Dev tokens (reduce COGS / TTV)
    ss.setdefault("dev_tokens", {"onboarding": 0, "automation": 0})  # 0..10 each

    # Results by quarter
    ss.setdefault("quarters", {})  # "Q1": {...}, "Q2": {...}, ...
    ss.setdefault("q_list", [])

    # Decisions / notes
    ss.setdefault("final", {"next1": "", "next2": ""})

init_state()

# ======================================================================================
# HELPERS
# ======================================================================================

def stepper():
    steps = [
        "Intro",
        "Segment",
        "Value Prop",
        "Pricing/Model",
        "Channels",
        "Run Q1",
        "Adjust & Run Q2",
        "Decide",
        "Results",
    ]
    order = {
        "intro": 0, "segment": 1, "value_prop": 2, "pricing": 3,
        "channels": 4, "run_q1": 5, "adjust_q2": 6, "decide": 7, "results": 8
    }
    active = order.get(st.session_state.stage, 0)
    cols = st.columns(len(steps))
    for i, c in enumerate(cols):
        with c:
            style = (
                "background:#eef6ff;border:1px solid #cde;"
                if i == active else "background:#f7f7f9;border:1px solid #eee;"
            )
            st.markdown(
                f"<div style='{style};padding:8px 10px;border-radius:10px;text-align:center;min-height:52px'>{steps[i]}</div>",
                unsafe_allow_html=True,
            )

def goto(stage: str):
    st.session_state.stage = stage

def channel_allocation_summary(alloc: Dict[str, int]) -> Tuple[int, Dict[str, float]]:
    total = sum(alloc.values())
    if total <= 0:
        return 0, {ch: 0.0 for ch in alloc}
    mix = {ch: alloc[ch] / total for ch in alloc}
    return total, mix

def nudge(text: str, variant="info"):
    if variant == "warn":
        st.warning(text)
    elif variant == "success":
        st.success(text)
    elif variant == "error":
        st.error(text)
    else:
        st.info(text)

# --------------------------------------------------------------------------------------
# Cohort & economics simulation
# --------------------------------------------------------------------------------------
@dataclass
class QuarterOutcome:
    visits: int
    trials: int
    activations: int
    paid: int
    retained: int
    arpu: float
    gross_margin: float
    cac_effective: float
    ltv: float
    ltv_cac: float
    payback_months: float
    churn: float
    cogs_per_unit: float
    sales_cycle_days: int
    cash_burn: float
    event: str
    notes: List[str]

def simulate_quarter(
    qname: str,
    segment_key: str,
    delivery: str,
    model: str,
    pricing: Dict[str, float],
    paywall_strictness: int,
    channels: Dict[str, int],
    sales_led: bool,
    dev_tokens: Dict[str, int],
    prev_q: QuarterOutcome | None = None,
) -> QuarterOutcome:
    seg = SEGMENTS[segment_key]
    base = seg["base_conv"]
    # Visits scaling from channels
    total_tokens, mix = channel_allocation_summary(channels)
    # If nothing allocated, give a minimal trickle (to avoid divide-by-zero)
    base_traffic = 10000 if total_tokens > 0 else 800

    # Channel ramp (we apply ramp-weighted visits and CAC)
    visit_multiplier = 0.0
    cac_components = []
    for ch, share in mix.items():
        if share <= 0:
            continue
        ch_def = CHANNELS[ch]
        # diminishing returns after ceiling:
        ceiling = ch_def["ceiling"]
        dim = 1.0 if share <= ceiling else max(0.6, 1.0 - (share - ceiling) * 1.25)
        ramp = RAMP_MULT[ch_def["ramp"]]
        fit = ch_def["fit"][segment_key]
        ch_visits = base_traffic * share * ramp * fit * dim
        visit_multiplier += ch_visits / base_traffic
        # CAC samples
        sample_cac = np.random.normal(ch_def["cac_mean"], ch_def["cac_std"])
        sample_cac = max(20, float(sample_cac * (1.0 / fit) * (1.0 / ramp)))  # better fit & ramp => lower CAC
        cac_components.append(sample_cac * share)

    # Market event
    ev = random.choice(MARKET_EVENTS)
    event_vis_mult = ev["mult_visits"]
    event_cac_mult = ev["mult_cac"]

    visits = int(base_traffic * max(0.2, visit_multiplier) * event_vis_mult)

    # Paywall & elasticity effects
    # Delivery impacts time-to-value & COGS
    dlv = DELIVERY_LEVELS[delivery]
    ttv_days = dlv["ttv_days"] - dev_tokens.get("onboarding", 0) * 0.3
    ttv_days = max(0.7, ttv_days)
    cogs_mult = dlv["cogs_mult"] - dev_tokens.get("automation", 0) * 0.02
    cogs_mult = max(0.45, cogs_mult)

    # Price anchor → choose middle tier as central ARPU anchor
    price_point = max(pricing.get("better", 0.0), 0.0)
    # Segment ARPU base adjusted by price elasticity
    arpu = seg["arpu_base"] * (price_point / max(1.0, seg["arpu_base"])) ** (1.0 / seg["price_elasticity"])
    # Freemium or tiered may alter ARPU slightly (simple rules)
    if model == "Freemium":
        arpu *= 0.85  # more free usage → lower early ARPU
    elif model == "Annual Contract" and seg["sales_led_benefit"]:
        arpu *= 1.15  # bigger ACV/ARPU for B2B under annuals

    # Funnel
    v2t = base["visit_to_trial"]
    t2a = base["trial_to_activation"]
    a2p = base["activation_to_paid"]
    # Apply paywall friction (strictness 0..2)
    paywall_fric = seg["paywall_friction"] * (1.0 + 0.08 * (paywall_strictness - 1))
    t2a_adj = max(0.05, t2a / paywall_fric)

    # Sales-led boosts activation & paid for B2B; slows cycle
    sales_cycle_days = 7
    if sales_led and seg["sales_led_benefit"]:
        t2a_adj *= 1.08
        a2p *= 1.12
        sales_cycle_days = 18

    # Time-to-value effect on activation
    ttv_effect = max(0.8, 1.2 - 0.05 * ttv_days)  # faster TTV => >1.0 multiplier
    t2a_adj *= ttv_effect

    trials = int(visits * v2t)
    activations = int(trials * t2a_adj)
    paid = int(activations * a2p)

    # Retention month-1 and churn
    # Baseline churn = 1 - retention_month_1
    base_ret = base["retention_month_1"]
    churn = 1.0 - base_ret

    # Adjust churn for time-to-value & price
    churn *= seg["churn_sensitivity"]["time_to_value"] ** max(0, (ttv_days - 2.0) / 3.0)
    churn *= seg["churn_sensitivity"]["price"] ** max(0, (price_point - seg["arpu_base"]) / 10.0)

    # Sales-led retention lift
    if sales_led and seg["sales_led_benefit"]:
        churn *= 0.94

    churn = min(max(churn, 0.02), 0.25)
    retained = int(paid * (1.0 - churn))

    # COGS per paid unit
    # For Hardware+Subscription, assume initial device costs amortized; keep it simple:
    base_cogs = 4.5  # $/paid-unit-month baseline
    if model == "Hardware + Subscription":
        base_cogs += 2.2
    cogs_per_unit = base_cogs * cogs_mult

    # CAC (effective): mix-weighted with event multiplier, plus sales-led overhead
    if cac_components:
        cac_raw = sum(cac_components) / sum(mix.values())
    else:
        cac_raw = 120.0
    if sales_led and seg["sales_led_benefit"]:
        cac_raw *= 1.25
    cac_effective = cac_raw * event_cac_mult

    # LTV & payback
    # LTV (very simplified): ARPU * (1/churn) * gross margin
    # Gross margin per paid unit:
    gross_margin = max(0.10, (arpu - cogs_per_unit) / max(0.01, arpu))
    ltv = max(0.0, (arpu * (1.0 / max(churn, 0.02))) * gross_margin)
    ltv_cac = ltv / max(1.0, cac_effective)
    payback_months = max(1.0, cac_effective / max(0.01, (arpu * gross_margin)))

    # Cash burn (toy model): CAC spend for new paid users + fixed ops – gross profit
    fixed_ops = 25000 if sales_led else 16000
    cac_spend = paid * cac_effective
    gross_profit = max(0.0, (arpu - cogs_per_unit) * paid)
    cash_burn = max(0.0, fixed_ops + cac_spend - gross_profit)

    # Notes / nudges
    notes = []
    if payback_months > (9 if segment_key == "homeowner" else 15):
        notes.append("Payback is high for this segment. Consider higher price, better conversion, or lower CAC.")
    if model == "Freemium" and (paid / max(1, activations)) < 0.02:
        notes.append("Freemium conversion is low (<2%). Paywall too generous or aha too late?")
    if gross_margin < 0.60:
        notes.append("Gross margin below 60%. Consider hybrid/automation or raise price.")
    return QuarterOutcome(
        visits=visits,
        trials=trials,
        activations=activations,
        paid=paid,
        retained=retained,
        arpu=round(arpu, 2),
        gross_margin=round(gross_margin, 2),
        cac_effective=round(cac_effective, 2),
        ltv=round(ltv, 2),
        ltv_cac=round(ltv_cac, 2),
        payback_months=round(payback_months, 1),
        churn=round(churn, 3),
        cogs_per_unit=round(cogs_per_unit, 2),
        sales_cycle_days=sales_cycle_days,
        cash_burn=round(cash_burn, 2),
        event=f"{ev['name']} — {ev['desc']}",
        notes=notes,
    )

# --------------------------------------------------------------------------------------
# Scoring & feedback
# --------------------------------------------------------------------------------------
def score_sim(qs: Dict[str, QuarterOutcome]) -> Tuple[int, Dict[str, int], Dict[str, str]]:
    weights = SCORES.copy()
    # Model Coherence (30): alignment of delivery, pricing, channels with segment
    coh = 0
    seg = st.session_state.segment
    deliv = st.session_state.delivery
    model = st.session_state.model
    channels = st.session_state.channels
    _, mix = channel_allocation_summary(channels)

    # Coherence heuristics:
    if seg == "homeowner":
        # self-serve: prefer Automated/Hybrid, Paid Social/Search + Content
        if deliv in ("Hybrid", "Automated"):
            coh += 10
        if mix.get("Paid Social/Search", 0) >= 0.25:
            coh += 8
        if mix.get("Content/SEO", 0) >= 0.15:
            coh += 6
        if model in ("Tiered", "Freemium", "Subscription only"):
            coh += 6
    elif seg in ("landlord", "installer"):
        # B2B: sales-led OK; Referral/Partner, Outbound, Events strong
        if deliv in ("Hybrid", "Automated"):
            coh += 8
        if st.session_state.sales_led:
            coh += 8
        if mix.get("Referral/Partner", 0) >= 0.20:
            coh += 6
        if mix.get("Outbound/Email", 0) >= 0.15 or mix.get("Events/Webinars", 0) >= 0.15:
            coh += 4
        if model in ("Hardware + Subscription", "Annual Contract", "Tiered"):
            coh += 4
    coh = min(weights["Model Coherence"], coh)

    # Unit Economics (30): from Q2 if available else Q1
    last = qs.get("Q2", qs.get("Q1"))
    econ = 0
    if last:
        if last.ltv_cac >= 3.0:
            econ += 12
        elif last.ltv_cac >= 2.0:
            econ += 8
        else:
            econ += 4

        if last.gross_margin >= 0.60:
            econ += 8
        elif last.gross_margin >= 0.50:
            econ += 5
        else:
            econ += 2

        threshold = 9 if st.session_state.segment == "homeowner" else 15
        if last.payback_months <= threshold:
            econ += 6
        elif last.payback_months <= threshold + 4:
            econ += 4
        else:
            econ += 2

        # churn band
        if last.churn <= 0.06:
            econ += 4
        elif last.churn <= 0.10:
            econ += 2
    econ = min(weights["Unit Economics"], econ)

    # GTM Fit & Focus (20): segment-fit and no channel >60% by Q2
    gtm = 0
    # segment-fit: sum of (mix * fit)
    fit_score = 0.0
    for ch, share in mix.items():
        fit_score += share * CHANNELS[ch]["fit"][st.session_state.segment]
    if fit_score >= 0.95:
        gtm += 10
    elif fit_score >= 0.80:
        gtm += 7
    else:
        gtm += 4
    # concentration
    if mix:
        max_share = max(mix.values())
        if max_share <= 0.60:
            gtm += 10
        elif max_share <= 0.75:
            gtm += 6
        else:
            gtm += 3
    gtm = min(weights["GTM Fit & Focus"], gtm)

    # Iteration Quality (10): Q2 improves Q1 on binding constraint (payback or LTV/CAC or churn)
    it = 0
    if "Q1" in qs and "Q2" in qs:
        q1, q2 = qs["Q1"], qs["Q2"]
        gains = 0
        if q2.payback_months < q1.payback_months:
            gains += 1
        if q2.ltv_cac > q1.ltv_cac:
            gains += 1
        if q2.churn < q1.churn:
            gains += 1
        if q2.gross_margin > q1.gross_margin:
            gains += 1
        it = [0, 4, 7, 9, 10][min(gains, 4)]
    it = min(weights["Iteration Quality"], it)

    # Evidence Plan (10): presence & specificity of next tests
    ev = 0
    n1 = st.session_state.final.get("next1", "").strip()
    n2 = st.session_state.final.get("next2", "").strip()
    if n1 and n2:
        ev += 6
        # crude specificity test: look for a number or %
        for n in (n1, n2):
            if any(ch in n for ch in ["%", "≥", ">", "<", "pp", "target"]):
                ev += 2
    elif n1 or n2:
        ev += 3
    ev = min(weights["Evidence Plan"], ev)

    breakdown = {
        "Model Coherence": coh,
        "Unit Economics": econ,
        "GTM Fit & Focus": gtm,
        "Iteration Quality": it,
        "Evidence Plan": ev,
    }
    total = sum(breakdown.values())
    # Reasons
    reasons = {
        "Model Coherence": "Delivery, pricing model, and channels align with the chosen segment.",
        "Unit Economics": "Assessed LTV/CAC, gross margin, churn band, and payback vs. segment targets.",
        "GTM Fit & Focus": "Evaluated channel/segment fit and concentration (no single channel >60%).",
        "Iteration Quality": "Checked Q2 vs Q1 improvements on payback, LTV/CAC, churn, and margin.",
        "Evidence Plan": "Credited presence/specificity of two next tests with clear targets.",
    }
    return total, breakdown, reasons

# ======================================================================================
# SCREENS
# ======================================================================================

def screen_intro():
    stepper()
    st.title("Simulation #3 — Business Model Fit (ThermaLoop)")
    st.write(
        "Translate your early validation into a coherent business model.\n"
        "You’ll choose a segment, craft a value prop, pick a revenue model and pricing, "
        "allocate GTM channels, simulate Q1–Q2, iterate, and see your score."
    )
    st.markdown("**You will:**")
    st.markdown(
        "- Choose **ICP (segment)** and optional secondary\n"
        "- Build a **value proposition** and **delivery level** (Manual/Hybrid/Automated)\n"
        "- Select **pricing/revenue model** and **paywall strictness**\n"
        "- Allocate **GTM tokens** across channels + optional **Dev tokens**\n"
        "- **Run Q1**, adjust, **Run Q2**, then finalize decisions and get feedback"
    )
    if st.button("Start Simulation", type="primary"):
        goto("segment")

def screen_segment():
    stepper()
    st.header("1) Choose Segment")
    c1, c2 = st.columns([0.6, 0.4])
    with c1:
        st.radio(
            "Primary ICP / beachhead:",
            ["homeowner", "landlord", "installer"],
            index=["homeowner", "landlord", "installer"].index(st.session_state.segment) if st.session_state.segment else 0,
            key="segment",
            format_func=lambda k: SEGMENTS[k]["name"],
        )
        st.checkbox("Add secondary segment (optional)", key="add_sec")
        if st.session_state.get("add_sec"):
            st.selectbox(
                "Secondary (optional):",
                ["—", "homeowner", "landlord", "installer"],
                index=0,
                key="secondary_segment",
                format_func=lambda k: "None" if k == "—" else SEGMENTS[k]["name"],
            )
    with c2:
        st.markdown("**Baseline funnel (per segment)**")
        seg = SEGMENTS[st.session_state.segment]
        base = seg["base_conv"]
        df = pd.DataFrame(
            {
                "Stage": ["Visit→Trial", "Trial→Activation", "Activation→Paid", "Retention M1"],
                "Rate": [
                    f"{int(base['visit_to_trial']*100)}%",
                    f"{int(base['trial_to_activation']*100)}%",
                    f"{int(base['activation_to_paid']*100)}%",
                    f"{int(base['retention_month_1']*100)}%",
                ],
            }
        )
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.caption("These are baseline tendencies; your choices will shift these.")

    st.divider()
    st.button("Next: Value Proposition", type="primary", on_click=lambda: goto("value_prop"))

def screen_value_prop():
    stepper()
    st.header("2) Value Proposition & Delivery")
    c1, c2, c3 = st.columns([0.45, 0.30, 0.25])
    with c1:
        st.text_input("Core Job (what you help them do)", key="vp_core_job", value=st.session_state.value_prop["core_job"])
        st.text_input("Key Outcome (metric they care about)", key="vp_outcome", value=st.session_state.value_prop["outcome"])
        st.text_input("Proof Element (pilot, quote, rebate)", key="vp_proof", value=st.session_state.value_prop["proof"])
        st.selectbox("Promise strength", ["Conservative", "Standard", "Bold"], key="vp_promise", index=["Conservative", "Standard", "Bold"].index(st.session_state.value_prop["promise"]))
    with c2:
        st.selectbox("Delivery level", list(DELIVERY_LEVELS.keys()), key="delivery", index=list(DELIVERY_LEVELS.keys()).index(st.session_state.delivery))
        st.selectbox("Unit of value", UNIT_OF_VALUE, key="unit_value", index=UNIT_OF_VALUE.index(st.session_state.unit_value))
        st.checkbox("Sales-led motion (applies best to B2B)", key="sales_led", value=st.session_state.sales_led)
    with c3:
        st.markdown("**Delivery impact**")
        d = DELIVERY_LEVELS[st.session_state.delivery]
        st.write(f"- COGS multiplier: **×{d['cogs_mult']}**")
        st.write(f"- Time-to-Value: **~{d['ttv_days']} days**")
        st.write(f"- Support load: **{d['support']}**")

    # persist
    st.session_state.value_prop = {
        "core_job": st.session_state.vp_core_job,
        "outcome": st.session_state.vp_outcome,
        "proof": st.session_state.vp_proof,
        "promise": st.session_state.vp_promise,
    }

    st.divider()
    st.button("Next: Pricing & Model", type="primary", on_click=lambda: goto("pricing"))

def screen_pricing():
    stepper()
    st.header("3) Revenue Model & Pricing")
    c1, c2 = st.columns([0.55, 0.45])
    with c1:
        st.selectbox("Business model", list(MODELS.keys()), key="model", index=list(MODELS.keys()).index(st.session_state.model))
        st.caption(MODELS[st.session_state.model]["note"])
        st.slider("Paywall strictness (0 = generous, 2 = strict)", 0, 2, key="paywall", value=st.session_state.paywall)
        st.markdown("#### Price anchors (monthly)")
        st.number_input("Good", min_value=0.0, value=float(st.session_state.pricing["good"]), key="p_good", format="%.2f")
        st.number_input("Better", min_value=0.0, value=float(st.session_state.pricing["better"]), key="p_better", format="%.2f")
        st.number_input("Best", min_value=0.0, value=float(st.session_state.pricing["best"]), key="p_best", format="%.2f")
        st.caption("The **Better** tier is used as the central ARPU anchor in the model.")
    with c2:
        st.markdown("**Freemium gravity**")
        st.write("- High top-of-funnel; lower early ARPU. Conversion depends on paywall and time-to-value.")
        st.markdown("**Annual contracts**")
        st.write("- Higher ACV + retention in B2B; slower cycles and higher CAC.")

    # persist
    st.session_state.pricing = {"good": st.session_state.p_good, "better": st.session_state.p_better, "best": st.session_state.p_best}

    st.divider()
    st.button("Next: Channels", type="primary", on_click=lambda: goto("channels"))

def screen_channels():
    stepper()
    st.header("4) Channels & GTM Allocation")
    st.caption("Allocate GTM tokens across channels (suggested total ≈ 100). Diminishing returns apply after each channel’s ceiling.")
    cols = st.columns(3)
    keys = list(CHANNELS.keys())
    for i, ch in enumerate(keys):
        with cols[i % 3]:
            st.slider(ch, 0, 100, key=f"ch_{i}", value=st.session_state.channels[ch])

    # persist channels from sliders
    for i, ch in enumerate(keys):
        st.session_state.channels[ch] = st.session_state.get(f"ch_{i}", 0)

    total, mix = channel_allocation_summary(st.session_state.channels)
    st.write(f"**Total GTM tokens:** {total}")
    if total == 0:
        nudge("You have 0 tokens allocated — you will get only a trickle of traffic.", "warn")

    st.markdown("#### Optional: Dev tokens")
    c1, c2 = st.columns(2)
    with c1:
        st.slider("Onboarding polish (reduces TTV)", 0, 10, key="dev_onboarding", value=st.session_state.dev_tokens["onboarding"])
    with c2:
        st.slider("Automation investment (reduces COGS)", 0, 10, key="dev_automation", value=st.session_state.dev_tokens["automation"])
    st.session_state.dev_tokens = {"onboarding": st.session_state.dev_onboarding, "automation": st.session_state.dev_automation}

    st.divider()
    st.button("Run Quarter 1", type="primary", on_click=lambda: goto("run_q1"))

def screen_run_q1():
    stepper()
    st.header("5) Run Quarter 1")
    if st.button("Simulate Q1", type="primary"):
        q1 = simulate_quarter(
            "Q1",
            st.session_state.segment,
            st.session_state.delivery,
            st.session_state.model,
            st.session_state.pricing,
            st.session_state.paywall,
            st.session_state.channels,
            st.session_state.sales_led,
            st.session_state.dev_tokens,
            None,
        )
        st.session_state.quarters["Q1"] = q1
        if "Q1" not in st.session_state.q_list:
            st.session_state.q_list.append("Q1")
        goto("adjust_q2")

    st.caption("We’ll compute funnel, ARPU, churn, CAC, LTV/CAC, payback, gross margin, and cash burn with a mild market event.")

def show_quarter(qname: str, q: QuarterOutcome):
    st.subheader(f"{qname} Results")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Visits", f"{q.visits:,}")
        st.metric("Trials", f"{q.trials:,}")
    with c2:
        st.metric("Activations", f"{q.activations:,}")
        st.metric("Paid", f"{q.paid:,}")
    with c3:
        st.metric("ARPU ($/mo)", f"{q.arpu:.2f}")
        st.metric("Gross Margin", f"{int(q.gross_margin*100)}%")
    with c4:
        st.metric("LTV/CAC", f"{q.ltv_cac:.2f}")
        st.metric("Payback (mo)", f"{q.payback_months:.1f}")
    st.markdown(
        f"- **Churn (M1):** {int(q.churn*100)}%  \n"
        f"- **Retained (M1):** {q.retained:,}  \n"
        f"- **CAC (effective):** ${q.cac_effective:.0f}  \n"
        f"- **COGS per unit:** ${q.cogs_per_unit:.2f}  \n"
        f"- **Sales cycle:** ~{q.sales_cycle_days} days  \n"
        f"- **Cash burn (qtr):** ${q.cash_burn:,.0f}  \n"
        f"- **Event:** {q.event}"
    )
    if q.notes:
        st.markdown("**Nudges:**")
        for n in q.notes:
            nudge(f"- {n}", "warn")

def screen_adjust_q2():
    stepper()
    st.header("6) Adjust & Run Quarter 2")
    q1 = st.session_state.quarters.get("Q1")
    if not q1:
        nudge("No Q1 results found. Please run Quarter 1 first.", "error")
        return

    show_quarter("Q1", q1)

    st.markdown("#### Tweak your model/pricing/channels and run Q2")
    c1, c2, c3 = st.columns([0.4, 0.35, 0.25])
    with c1:
        st.selectbox("Delivery level", list(DELIVERY_LEVELS.keys()), key="delivery", index=list(DELIVERY_LEVELS.keys()).index(st.session_state.delivery))
        st.checkbox("Sales-led motion (B2B)", key="sales_led", value=st.session_state.sales_led)
        st.slider("Paywall strictness (0 = generous, 2 = strict)", 0, 2, key="paywall", value=st.session_state.paywall)
    with c2:
        st.selectbox("Business model", list(MODELS.keys()), key="model", index=list(MODELS.keys()).index(st.session_state.model))
        st.number_input("Better tier (central anchor)", min_value=0.0, key="p_better2", value=float(st.session_state.pricing["better"]), format="%.2f")
        st.number_input("Good tier", min_value=0.0, key="p_good2", value=float(st.session_state.pricing["good"]), format="%.2f")
        st.number_input("Best tier", min_value=0.0, key="p_best2", value=float(st.session_state.pricing["best"]), format="%.2f")
    with c3:
        st.slider("Onboarding polish (reduces TTV)", 0, 10, key="dev_onboarding2", value=st.session_state.dev_tokens["onboarding"])
        st.slider("Automation investment (reduces COGS)", 0, 10, key="dev_automation2", value=st.session_state.dev_tokens["automation"])

    st.markdown("#### Channels (re-allocate)")
    cols = st.columns(3)
    for i, ch in enumerate(CHANNELS):
        with cols[i % 3]:
            st.slider(ch, 0, 100, key=f"ch2_{i}", value=st.session_state.channels[ch])

    # Persist adjustments
    for i, ch in enumerate(CHANNELS):
        st.session_state.channels[ch] = st.session_state.get(f"ch2_{i}", 0)
    st.session_state.pricing = {"good": st.session_state.p_good2, "better": st.session_state.p_better2, "best": st.session_state.p_best2}
    st.session_state.dev_tokens = {"onboarding": st.session_state.dev_onboarding2, "automation": st.session_state.dev_automation2}

    if st.button("Simulate Q2", type="primary"):
        q2 = simulate_quarter(
            "Q2",
            st.session_state.segment,
            st.session_state.delivery,
            st.session_state.model,
            st.session_state.pricing,
            st.session_state.paywall,
            st.session_state.channels,
            st.session_state.sales_led,
            st.session_state.dev_tokens,
            prev_q=q1,
        )
        st.session_state.quarters["Q2"] = q2
        if "Q2" not in st.session_state.q_list:
            st.session_state.q_list.append("Q2")
        goto("decide")

def screen_decide():
    stepper()
    st.header("7) Decide & Document")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Your selections**")
        st.write(f"- **Segment:** {SEGMENTS[st.session_state.segment]['name']}")
        sec = st.session_state.secondary_segment
        if sec and sec != "—":
            st.write(f"- **Secondary:** {SEGMENTS[sec]['name']}")
        vp = st.session_state.value_prop
        st.write(f"- **Value prop:** {vp['core_job']} → {vp['outcome']} (proof: {vp['proof']}, promise: {vp['promise']})")
        st.write(f"- **Delivery:** {st.session_state.delivery}")
        st.write(f"- **Unit of value:** {st.session_state.unit_value}")
        st.write(f"- **Model:** {st.session_state.model}")
        st.write(f"- **Pricing (G/B/B):** ${st.session_state.pricing['good']:.2f} / ${st.session_state.pricing['better']:.2f} / ${st.session_state.pricing['best']:.2f}")
        tot, mix = channel_allocation_summary(st.session_state.channels)
        if tot > 0:
            mix_str = ", ".join([f"{k} {int(v*100)}%" for k, v in mix.items() if v > 0.01])
        else:
            mix_str = "No allocation"
        st.write(f"- **Channels:** {mix_str}")
        st.write(f"- **Dev tokens:** Onboarding {st.session_state.dev_tokens['onboarding']}, Automation {st.session_state.dev_tokens['automation']}")
    with c2:
        st.text_input("Next test #1 (metric + target)", key="next1", value=st.session_state.final.get("next1", "Paywall test: move comfort map behind paywall; goal free→paid ≥ 4%."))
        st.text_input("Next test #2 (metric + target)", key="next2", value=st.session_state.final.get("next2", "Partner pilot: 3 HVAC partners; goal CAC −20% vs paid social."))
        st.session_state.final = {"next1": st.session_state.next1, "next2": st.session_state.next2}

    st.divider()
    st.button("Finish & See Results", type="primary", on_click=lambda: goto("results"))

def screen_results():
    stepper()
    st.header("Results & Feedback")

    # Show quarter tables
    if not st.session_state.quarters:
        nudge("No results — please run Q1 (and Q2) first.", "error")
        return

    # Display Q1 and Q2
    for qn in st.session_state.q_list:
        show_quarter(qn, st.session_state.quarters[qn])

    st.divider()
    total, breakdown, reasons = score_sim(st.session_state.quarters)
    st.metric("Total Score (0–100)", total)

    # Category table with reasons
    ordered = ["Model Coherence", "Unit Economics", "GTM Fit & Focus", "Iteration Quality", "Evidence Plan"]
    rows = []
    for cat in ordered:
        raw = breakdown.get(cat, 0)
        out100 = round(100 * raw / SCORES[cat])
        rows.append({"Category": cat, "Score (/100)": out100, "Why": reasons.get(cat, "—")})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # Quick deltas if Q2 exists
    if "Q1" in st.session_state.quarters and "Q2" in st.session_state.quarters:
        q1, q2 = st.session_state.quarters["Q1"], st.session_state.quarters["Q2"]
        st.markdown("### Q2 vs Q1 — Key Deltas")
        delta_rows = [
            {"Metric": "LTV/CAC", "Q1": f"{q1.ltv_cac:.2f}", "Q2": f"{q2.ltv_cac:.2f}", "Δ": f"{q2.ltv_cac - q1.ltv_cac:+.2f}"},
            {"Metric": "Payback (mo)", "Q1": f"{q1.payback_months:.1f}", "Q2": f"{q2.payback_months:.1f}", "Δ": f"{q2.payback_months - q1.payback_months:+.1f}"},
            {"Metric": "Churn", "Q1": f"{q1.churn:.3f}", "Q2": f"{q2.churn:.3f}", "Δ": f"{q2.churn - q1.churn:+.3f}"},
            {"Metric": "Gross Margin", "Q1": f"{q1.gross_margin:.2f}", "Q2": f"{q2.gross_margin:.2f}", "Δ": f"{q2.gross_margin - q1.gross_margin:+.2f}"},
            {"Metric": "Cash Burn ($)", "Q1": f"{q1.cash_burn:,.0f}", "Q2": f"{q2.cash_burn:,.0f}", "Δ": f"{q2.cash_burn - q1.cash_burn:,+.0f}"},
        ]
        st.dataframe(pd.DataFrame(delta_rows), hide_index=True, use_container_width=True)

    st.divider()
    st.markdown("### Coaching Notes")
    coach = []
    # Add targeted, concrete advice
    last = st.session_state.quarters.get("Q2", st.session_state.quarters.get("Q1"))
    if last:
        if last.ltv_cac < 3.0:
            coach.append("Improve LTV/CAC: raise ARPU via better tier value or reduce CAC by shifting mix to higher-fit channels.")
        if last.payback_months > (9 if st.session_state.segment == "homeowner" else 15):
            coach.append("Payback is long for this segment — revisit price ladder and top-of-funnel conversion to speed recovery.")
        if last.gross_margin < 0.60:
            coach.append("Gross margin below target — consider moving from Manual→Hybrid or investing more in automation.")
        if last.churn > 0.08:
            coach.append("Churn is high — accelerate time-to-value (onboarding polish), and ensure paywall aligns with the aha moment.")
    _, mix = channel_allocation_summary(st.session_state.channels)
    if mix and max(mix.values()) > 0.60:
        coach.append("Your channel mix is concentrated — diversify to reduce volatility and improve resilience.")
    if st.session_state.model == "Freemium" and last and (last.paid / max(1, last.activations)) < 0.02:
        coach.append("Freemium conversion is weak — tighten paywall and shorten the path to aha.")
    if not coach:
        coach.append("Coherent choices and solid early economics — nice! Next: validate with a paywall A/B and a partner pilot.")
    for c in coach:
        nudge(f"- {c}")

    st.success("End of Simulation 3. Refresh to try a different segment, model, and mix.")

# ======================================================================================
# ROUTER
# ======================================================================================
def router():
    s = st.session_state.stage
    if s == "intro":
        screen_intro()
    elif s == "segment":
        screen_segment()
    elif s == "value_prop":
        screen_value_prop()
    elif s == "pricing":
        screen_pricing()
    elif s == "channels":
        screen_channels()
    elif s == "run_q1":
        screen_run_q1()
    elif s == "adjust_q2":
        screen_adjust_q2()
    elif s == "decide":
        screen_decide()
    elif s == "results":
        screen_results()
    else:
        screen_intro()

# ======================================================================================
# RUN
# ======================================================================================
router()
