# app_sim3.py
# Startup Simulation #3 — Business Model Fit (Streamlit MVP)
# Run: streamlit run app_sim3.py

import math
import random
from copy import deepcopy
from typing import Dict, List, Any

import streamlit as st

random.seed(7)
st.set_page_config(page_title="Simulation #3 — Business Model Fit", page_icon="📈", layout="wide")

TITLE = "Simulation #3 — Business Model Fit"
SUB = "Align segment, value prop, channels, and revenue mechanics through trade-offs"

# ──────────────────────────────────────────────────────────────────────────────
# Seed content
# ──────────────────────────────────────────────────────────────────────────────
MARKET_BRIEF = (
    "Independent fitness market with varied studio types. Many classes have uneven attendance; "
    "owners balance acquisition cost, retention, and staff load. Early signals show interest in "
    "‘keep classes full’ outcomes, but willingness to pay and delivery cost vary by segment."
)

SEGMENTS = {
    # elasticities: how sensitive conversions/churn are to price, paywall, time-to-value
    "Specialty Studio (e.g., yoga, boxing)": {
        "base_ARPU": 85, "price_elasticity": 0.9, "paywall_friction": 0.7, "ttv_sensitivity": 0.8,
        "b2b": False, "churn_base": 0.035, "size_index": 1.0
    },
    "Premium Amenities Studio": {
        "base_ARPU": 129, "price_elasticity": 0.6, "paywall_friction": 0.4, "ttv_sensitivity": 0.6,
        "b2b": False, "churn_base": 0.025, "size_index": 0.8
    },
    "Budget Studio": {
        "base_ARPU": 59, "price_elasticity": 1.2, "paywall_friction": 0.9, "ttv_sensitivity": 1.0,
        "b2b": False, "churn_base": 0.045, "size_index": 1.3
    },
    "Corporate Wellness Partner": {
        "base_ARPU": 240, "price_elasticity": 0.4, "paywall_friction": 0.5, "ttv_sensitivity": 0.5,
        "b2b": True, "churn_base": 0.018, "size_index": 0.6
    },
}

VALUE_OUTCOMES = [
    "Keep classes ≥80% full",
    "Cut staff scheduling time",
    "Reduce churn",
    "Grow high-value members",
    "Automate promo timing",
]
PROOF_ELEMENTS = ["Customer quotes", "Before/After metrics", "Production demo", "ROI calculator", "Third-party review"]
PROMISE = ["Conservative", "Balanced", "Bold"]
DELIVERY = {
    "Manual": {"COGS_pct": 0.45, "ttv_days": 14},
    "Hybrid": {"COGS_pct": 0.30, "ttv_days": 7},
    "Automated": {"COGS_pct": 0.18, "ttv_days": 3},
}
UNIT_OF_VALUE = ["Per location", "Per staff seat", "Per active member"]

REVENUE_MODELS = {
    "Subscription":      {"trial": True,  "paywall": "Core value behind paywall"},
    "One-time Purchase": {"trial": False, "paywall": "Updates/support separate"},
    "Usage-based":       {"trial": True,  "paywall": "Pay per action/seat"},
    "Freemium":          {"trial": False, "paywall": "Basic free; value gates paid"},
    "Tiered Access":     {"trial": True,  "paywall": "Good/Better/Best features"},
    "Contracts":         {"trial": True,  "paywall": "Annual B2B with pilot"},
}

CHANNELS = {
    # cost=avg CAC at steady state, ramp months (to reach 100%), ceiling=volume scaler, fit boosts by segment type
    "Content/SEO":      {"cost": 70,  "ramp": 4, "ceiling": 0.7, "fit": {"b2b": 0.2, "b2c": 0.6}},
    "Referral/Partner": {"cost": 55,  "ramp": 2, "ceiling": 0.8, "fit": {"b2b": 0.7, "b2c": 0.4}},
    "Outbound/Email":   {"cost": 120, "ramp": 1, "ceiling": 0.5, "fit": {"b2b": 0.8, "b2c": 0.2}},
    "Paid Social":      {"cost": 110, "ramp": 1, "ceiling": 1.0, "fit": {"b2b": 0.3, "b2c": 0.9}},
    "Paid Search":      {"cost": 95,  "ramp": 1, "ceiling": 0.9, "fit": {"b2b": 0.5, "b2c": 0.7}},
    "Events/Webinars":  {"cost": 180, "ramp": 2, "ceiling": 0.4, "fit": {"b2b": 0.8, "b2c": 0.2}},
    "App Directory":    {"cost": 60,  "ramp": 2, "ceiling": 0.5, "fit": {"b2b": 0.4, "b2c": 0.6}},
}

ACRONYM_HELP = "ARPU = Average Revenue per Unit, CAC = Customer Acquisition Cost, LTV = Lifetime Value."

# ──────────────────────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────────────────────
def init_state():
    st.session_state.s3 = {
        "stage": "intro",
        "segment": None,
        "segment_secondary": None,
        "value_outcome": VALUE_OUTCOMES[0],
        "proof": [PROOF_ELEMENTS[0]],
        "promise": PROMISE[1],
        "delivery": "Hybrid",
        "unit_value": UNIT_OF_VALUE[0],
        "model": "Subscription",
        "price_tier": {"Good": 79, "Better": 119, "Best": 149},
        "paywall_note": "Core automation gated at paid tiers.",
        "gtm_tokens": 12,
        "dev_tokens": 4,
        "gtm_alloc": {k: 0 for k in CHANNELS.keys()},
        "history": {},  # quarter -> results dict
        "cash": 200_000,  # starting cash
        "notes_tests": {"t1": "", "t2": ""},
        "score": None,
        "coach": None,
    }

if "s3" not in st.session_state:
    init_state()
S = st.session_state.s3

# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────
def clamp(x, lo, hi): return max(lo, min(hi, x))

def ramp_factor(ramp_months, quarter_index):
    # Q1,Q2,Q3,Q4  -> fraction of maturity based on ramp
    months = quarter_index * 3
    return clamp(months / max(1, ramp_months * 3), 0.25, 1.0)

def price_effect(price, base_ARPU, elasticity):
    # <1 boosts conv; >1 hurts conv; symmetrical damped
    ratio = price / base_ARPU
    return clamp(math.exp(-elasticity * (ratio - 1.0)), 0.5, 1.2)

def ttv_effect(ttv_days, sens):
    # faster time-to-value improves activation/retention
    return clamp(1.1 - sens * (ttv_days / 30.0) * 0.4, 0.7, 1.15)

def model_paywall_effect(model, paywall_friction, b2b):
    # Freemium high signup/low convert; Contracts better convert but slow
    if model == "Freemium":
        return 1.2, clamp(1.0 - 0.3 * paywall_friction, 0.6, 0.95), 0.9
    if model == "Contracts":
        return 0.8, 1.1 if b2b else 0.9, 0.85
    if model == "Usage-based":
        return 1.0, 1.05, 1.0
    if model == "One-time Purchase":
        return 0.9, 0.95, 1.05
    if model == "Tiered Access":
        return 1.0, 1.0, 1.0
    return 1.0, 1.0, 1.0  # Subscription default

def channel_volume_and_cac(seg_key, quarter_idx) -> (float, float, Dict[str, Dict[str, float]]):
    seg = SEGMENTS[seg_key]
    b2b = seg["b2b"]
    details = {}
    total_visits = 0.0
    weighted_cac_cost = 0.0
    for ch, tokens in S["gtm_alloc"].items():
        if tokens <= 0: 
            continue
        p = CHANNELS[ch]
        fit = p["fit"]["b2b" if b2b else "b2c"]
        maturity = ramp_factor(p["ramp"], quarter_idx)
        diminishing = clamp(1 - 0.06 * max(0, tokens - 4), 0.6, 1.0)
        volume = tokens * p["ceiling"] * fit * maturity * diminishing * 250 * seg["size_index"]
        cac = p["cost"] / clamp(0.7 + 0.06 * tokens, 0.8, 1.3)  # a bit better with more tokens, up to a point
        total_visits += volume
        weighted_cac_cost += cac * volume
        details[ch] = {"visits": volume, "cac": cac, "maturity": maturity}
    avg_cac = (weighted_cac_cost / total_visits) if total_visits > 0 else 0
    return total_visits, avg_cac, details

def quarterly_events(q):
    events = []
    mul = 1.0
    # Simple seasonality: Q3 dip, Q1 slight rise for consumer; B2B steady except Q4 slows
    if q == 3:
        events.append("Seasonality dip (summer attendance)"); mul *= 0.92
    if q == 1:
        events.append("New year bump"); mul *= 1.06
    # Random:
    roll = random.random()
    if roll < 0.15:
        events.append("Partner boost"); mul *= 1.08
    elif roll < 0.30:
        events.append("Ad inventory shock"); mul *= 0.93
    return mul, events

def simulate_quarter(q_index: int, seg_key: str, price_choice: int, delivery_key: str, model_key: str) -> Dict[str, Any]:
    seg = SEGMENTS[seg_key]
    b2b = seg["b2b"]
    # price selection (Good/Better/Best)
    tier_price = [S["price_tier"]["Good"], S["price_tier"]["Better"], S["price_tier"]["Best"]][price_choice]
    # Effects
    price_fx = price_effect(tier_price, seg["base_ARPU"], seg["price_elasticity"])
    ttv_fx = ttv_effect(DELIVERY[delivery_key]["ttv_days"], seg["ttv_sensitivity"])
    signup_fx, convert_fx, cycle_fx = model_paywall_effect(model_key, seg["paywall_friction"], b2b)
    # Channels
    visits, avg_cac, channel_detail = channel_volume_and_cac(seg_key, q_index)
    season_mul, events = quarterly_events(q_index)

    # Funnel baseline
    visit_to_trial = 0.06 * signup_fx * season_mul
    trial_to_activation = 0.33 * ttv_fx * season_mul
    activation_to_paid = 0.45 * convert_fx * season_mul
    if model_key == "Freemium":
        visit_to_trial *= 1.3
        activation_to_paid *= 0.55
    if model_key == "Contracts" and b2b:
        activation_to_paid *= 1.15
        avg_cac *= 1.3  # sales-led effect
    # Conversions shaped by price
    activation_to_paid *= clamp(0.7 + 0.4*(price_fx-0.8), 0.5, 1.2)

    trials = visits * visit_to_trial
    activations = trials * trial_to_activation
    new_paid = activations * activation_to_paid

    # Revenue & COGS (per unit)
    cogs_pct = DELIVERY[delivery_key]["COGS_pct"]
    arpu = tier_price if model_key != "Usage-based" else max(49, tier_price*0.8 + 30)
    gross_margin = clamp(1 - cogs_pct, 0.3, 0.9)

    # Churn (monthly) → convert to quarter
    monthly_churn = seg["churn_base"] * (1.05 - 0.1*ttv_fx) * (1.02 + 0.06*(price_fx < 0.9))
    q_churn = 1 - (1 - monthly_churn)**3

    # Existing base carries over from prior quarter
    prev_paid = S["history"].get(q_index-1, {}).get("ending_paid", 0)
    churned = prev_paid * q_churn
    ending_paid = prev_paid - churned + new_paid

    # Economics
    revenue = ending_paid * arpu * 3  # per quarter
    cogs = revenue * (1 - gross_margin)
    # CAC spend approx = new_paid * avg_cac (guard div-by-zero)
    cac_spend = new_paid * avg_cac
    # Simple OPEX from channels (+ dev tokens amortized)
    opex = 30_000 + sum(S["gtm_alloc"].values())*2_000 + (max(0, 4 - S["dev_tokens"]))*1_500
    cash_delta = revenue - (cogs + cac_spend + opex)
    cash_end = S["cash"] + cash_delta

    # KPI
    payback_months = (avg_cac / max(1, arpu * gross_margin)) if new_paid > 0 else float("inf")
    ltv = (arpu * gross_margin) / max(1e-6, monthly_churn)
    ltv_cac = (ltv / avg_cac) if avg_cac > 0 else float("inf")

    # Warnings (nudges)
    nudges = []
    if payback_months > (15 if b2b else 9):
        nudges.append("Payback too long — consider lowering CAC (channel mix) or raising price/value.")
    if model_key == "Freemium" and activation_to_paid < 0.02:
        nudges.append("Freemium conversion very low — paywall may be too generous.")
    if gross_margin < 0.60:
        nudges.append("COGS high for price — consider Hybrid/Automated delivery or higher pricing.")
    if sum(S["gtm_alloc"].values()) and max(S["gtm_alloc"].values()) > 0.6 * sum(S["gtm_alloc"].values()):
        nudges.append("GTM heavily concentrated — reduce reliance on a single channel.")

    result = dict(
        q=q_index,
        visits=int(visits),
        trials=int(trials),
        activations=int(activations),
        new_paid=int(new_paid),
        ending_paid=int(ending_paid),
        arpu=round(arpu, 2),
        gross_margin=round(gross_margin, 2),
        churn_q=round(q_churn, 3),
        avg_cac=round(avg_cac, 2),
        ltv=round(ltv, 2) if math.isfinite(ltv) else float("inf"),
        ltv_cac=round(ltv_cac, 2) if math.isfinite(ltv_cac) else float("inf"),
        payback_m=round(payback_months, 1) if math.isfinite(payback_months) else float("inf"),
        revenue=int(revenue),
        cogs=int(cogs),
        cac_spend=int(cac_spend),
        opex=int(opex),
        cash_delta=int(cash_delta),
        cash_end=int(cash_end),
        events=events,
        nudges=nudges,
        channel_detail=channel_detail,
        b2b=b2b
    )
    return result

def score_run(history: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    # Coherence: model vs segment vs delivery vs channels (quick heuristics)
    seg = SEGMENTS[S["segment"]]
    b2b = seg["b2b"]
    channels_used = {k:v for k,v in S["gtm_alloc"].items() if v>0}
    big_channel = max(channels_used.values()) if channels_used else 0
    b2b_fit = (channels_used.get("Outbound/Email",0) + channels_used.get("Events/Webinars",0)) >= 4 if b2b else (channels_used.get("Paid Social",0) + channels_used.get("Paid Search",0)) >= 4
    delivery_ok = DELIVERY[S["delivery"]]["COGS_pct"] <= 0.4 or (S["price_tier"]["Better"] >= 119)

    coherence = clamp((0.5 + 0.2*b2b_fit + 0.3*delivery_ok - 0.1*(big_channel > 0.6*sum(S["gtm_alloc"].values()))), 0, 1)

    # Unit economics from Q4 (or last quarter run)
    last_q = max(history.keys()) if history else 1
    last = history[last_q]
    econ_ok = (
        (last["ltv_cac"] >= 3.0) and
        (last["gross_margin"] >= 0.60) and
        (last["payback_m"] <= (15 if b2b else 9))
    )
    unit_econ = clamp(0.3 + 0.5*(last["ltv_cac"]/4) + 0.2*(1 if econ_ok else 0), 0, 1)

    # GTM fit & focus: by Q2, no channel >60% volume and segment fit is used
    q2 = history.get(2, last)
    focus_penalty = 0.0
    if sum(S["gtm_alloc"].values())>0 and max(S["gtm_alloc"].values()) > 0.6*sum(S["gtm_alloc"].values()):
        focus_penalty = 0.2
    gtm_fit = clamp(0.7 - focus_penalty, 0, 1)

    # Iteration quality: did Q2 improve binding constraint vs Q1?
    iter_score = 0.5
    if 1 in history and 2 in history:
        if history[2]["payback_m"] < history[1]["payback_m"]: iter_score += 0.15
        if history[2]["ltv_cac"] > history[1]["ltv_cac"]: iter_score += 0.15
        if history[2]["ending_paid"] > history[1]["ending_paid"]: iter_score += 0.1
    iter_score = clamp(iter_score, 0, 1)

    # Evidence plan: two tests provided and specific
    t1, t2 = S["notes_tests"]["t1"], S["notes_tests"]["t2"]
    def is_specific(s): 
        return any(x in s.lower() for x in ["target", "%", "pp", ">=", "<=", " by "]) and len(s.strip())>8
    evidence = clamp(0.4 + 0.3*is_specific(t1) + 0.3*is_specific(t2), 0, 1)

    total = round(100*(0.30*coherence + 0.30*unit_econ + 0.20*gtm_fit + 0.10*iter_score + 0.10*evidence))

    reasons = {
        "Model Coherence": ("Excellent" if coherence>=0.8 else "Good" if coherence>=0.6 else "Needs work") +
            f" — Segment {'B2B' if b2b else 'B2C'} with channels {'well aligned' if b2b_fit else 'partly aligned'}; "
            f"delivery {'supports' if delivery_ok else 'pressures'} margin; "
            f\"concentration={'high' if big_channel > 0.6*sum(S['gtm_alloc'].values()) else 'balanced'}.\",
        "Unit Economics": (
            ("Excellent" if unit_econ>=0.8 else "Good" if unit_econ>=0.6 else "Needs work") +
            f" — LTV/CAC={history[last_q]['ltv_cac']}, GM={int(history[last_q]['gross_margin']*100)}%, "
            f"Payback={history[last_q]['payback_m']} mo."
        ),
        "GTM Fit & Focus": ("Excellent" if gtm_fit>=0.8 else "Good" if gtm_fit>=0.6 else "Needs work") +
            ("" if focus_penalty==0 else " — Heavy reliance on one channel (>60%) by Q2."),
        "Iteration Quality": ("Excellent" if iter_score>=0.8 else "Good" if iter_score>=0.6 else "Needs work") +
            (" — Q2 improved payback/LTV/CAC vs Q1." if (1 in history and 2 in history and (history[2]['payback_m']<history[1]['payback_m'] or history[2]['ltv_cac']>history[1]['ltv_cac'])) else " — Limited measurable improvement."),
        "Evidence Plan": ("Excellent" if evidence>=0.8 else "Good" if evidence>=0.6 else "Needs work") +
            (" — Both tests include targets." if (is_specific(t1) and is_specific(t2)) else " — Add clear targets/thresholds."),
    }

    score = {
        "total": total,
        "components": {
            "Model Coherence": round(coherence,2),
            "Unit Economics": round(unit_econ,2),
            "GTM Fit & Focus": round(gtm_fit,2),
            "Iteration Quality": round(iter_score,2),
            "Evidence Plan": round(evidence,2),
        },
        "reasons": reasons
    }
    return score

# ──────────────────────────────────────────────────────────────────────────────
# UI Blocks
# ──────────────────────────────────────────────────────────────────────────────
def header():
    st.title(TITLE)
    st.caption(SUB)

def page_intro():
    st.markdown(f"**Market Brief:** {MARKET_BRIEF}")
    st.markdown("""
**What you'll do (60–90 min):**  
1) Pick a **segment**, 2) assemble a **value proposition & delivery**,  
3) choose a **revenue model & pricing**, 4) allocate **go-to-market (GTM) tokens** (+ optional **Dev tokens**),  
5) **run quarters** to see funnel & financials, 6) **adjust** and run again,  
7) capture your **Business Model Snapshot** + **next two tests**.
""")
    if st.button("Start"):
        S["stage"] = "segment"; st.rerun()

def page_segment():
    st.subheader("Segment focus")
    st.write("Choose a primary segment and optional secondary (ICP vs beachhead).")
    c1, c2 = st.columns(2)
    with c1:
        S["segment"] = st.selectbox("Primary segment", list(SEGMENTS.keys()), index=0)
    with c2:
        S["segment_secondary"] = st.selectbox("Secondary (optional)", ["—"] + list(SEGMENTS.keys()), index=0)

    seg = SEGMENTS[S["segment"]]
    with st.expander("Baseline profile & funnel assumptions"):
        st.write(f"- **ARPU anchor:** ${seg['base_ARPU']}  \n- **Price elasticity:** {seg['price_elasticity']}  \n"
                 f"- **Paywall friction:** {seg['paywall_friction']}  \n- **Time-to-value sensitivity:** {seg['ttv_sensitivity']}  \n"
                 f"- **Churn base (monthly):** {seg['churn_base']:.3f}  \n- **B2B:** {seg['b2b']}")

    if st.button("Next: Value Prop & Offer"):
        S["stage"] = "value"; st.rerun()

def page_value():
    st.subheader("Value Prop & Offer Sketch")
    c1, c2, c3 = st.columns(3)
    with c1:
        S["value_outcome"] = st.selectbox("Core outcome", VALUE_OUTCOMES, index=0)
        S["promise"] = st.selectbox("Promise strength", PROMISE, index=1)
    with c2:
        S["proof"] = st.multiselect("Proof elements", PROOF_ELEMENTS, default=[PROOF_ELEMENTS[0]])
        S["unit_value"] = st.selectbox("Unit of value", UNIT_OF_VALUE, index=0)
    with c3:
        S["delivery"] = st.selectbox("Delivery level", list(DELIVERY.keys()), index=1)
        st.caption(f"COGS≈{int(DELIVERY[S['delivery']]['COGS_pct']*100)}%, Time-to-value≈{DELIVERY[S['delivery']]['ttv_days']} days")

    if st.button("Next: Pricing & Model"):
        S["stage"] = "pricing"; st.rerun()

def page_pricing():
    st.subheader("Revenue model & price point")
    S["model"] = st.selectbox("Revenue model", list(REVENUE_MODELS.keys()), index=0,
                              help="Subscription, One-time Purchase, Usage-based, Freemium, Tiered Access, Contracts")
    cols = st.columns(3)
    S["price_tier"]["Good"] = cols[0].number_input("Good $", min_value=19, max_value=499, value=S["price_tier"]["Good"], step=5)
    S["price_tier"]["Better"] = cols[1].number_input("Better $", min_value=19, max_value=799, value=S["price_tier"]["Better"], step=5)
    S["price_tier"]["Best"] = cols[2].number_input("Best $", min_value=19, max_value=999, value=S["price_tier"]["Best"], step=5)

    S["paywall_note"] = st.text_input("Paywall note (what’s paid vs free)", value=S["paywall_note"])
    st.info("The sim computes COGS, gross margin, and payback, and applies price sensitivity based on your chosen segment.")

    if st.button("Next: Channels & GTM mix"):
        S["stage"] = "channels"; st.rerun()

def page_channels():
    st.subheader("Channel & GTM Mix")
    st.caption("Allocate GTM tokens across channels. You also have optional Dev tokens to implement fixes that reduce COGS or time-to-value.")
    left, right = st.columns([2,1])
    with left:
        total = 0
        for ch in CHANNELS.keys():
            S["gtm_alloc"][ch] = st.slider(ch, 0, 6, S["gtm_alloc"][ch])
            total += S["gtm_alloc"][ch]
        st.write(f"**Allocated:** {total} / {S['gtm_tokens']} GTM tokens")
        if total > S["gtm_tokens"]:
            st.error("You’ve allocated more than your GTM tokens. Reduce some sliders.")
    with right:
        S["dev_tokens"] = st.slider("Development tokens", 0, 6, S["dev_tokens"])
        st.caption("Dev tokens reduce delivery cost/time-to-value penalties behind the scenes.")

    if st.button("Simulate Q1"):
        if total <= S["gtm_tokens"]:
            S["stage"] = "run"; S["history"].clear(); st.rerun()
        else:
            st.warning("Fix token over-allocation before continuing.")

def page_run():
    st.subheader("Run Quarters & Adjust")
    st.caption(ACRONYM_HELP)

    # Controls for run
    c1, c2, c3, c4 = st.columns(4)
    price_choice = c1.selectbox("Price anchor", ["Good", "Better", "Best"], index=1)
    price_idx = ["Good","Better","Best"].index(price_choice)
    delivery = c2.selectbox("Delivery", list(DELIVERY.keys()), index=list(DELIVERY.keys()).index(S["delivery"]))
    model = c3.selectbox("Model", list(REVENUE_MODELS.keys()), index=list(REVENUE_MODELS.keys()).index(S["model"]))
    q_to_run = c4.selectbox("Quarter to run now", ["Q1","Q2","Q3","Q4"], index=len(S["history"]))

    q_index = int(q_to_run[-1])  # 1..4
    if st.button(f"Run {q_to_run}"):
        res = simulate_quarter(q_index, S["segment"], price_idx, delivery, model)
        S["history"][q_index] = res
        S["cash"] = res["cash_end"]
        st.rerun()

    # Show latest
    if S["history"]:
        last_q = max(S["history"].keys())
        res = S["history"][last_q]
        st.markdown(f"### Results — Q{last_q}")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Visits", res["visits"])
        k2.metric("Trials", res["trials"])
        k3.metric("New Paid", res["new_paid"])
        k4.metric("Ending Paid", res["ending_paid"])
        k5.metric("ARPU ($/mo)", res["arpu"])

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Gross Margin", f"{int(res['gross_margin']*100)}%")
        m2.metric("Avg CAC ($)", res["avg_cac"])
        m3.metric("LTV/CAC", res["ltv_cac"] if math.isfinite(res["ltv_cac"]) else "—")
        m4.metric("Payback (mo)", res["payback_m"] if math.isfinite(res["payback_m"]) else "—")
        m5.metric("Cash Δ ($)", res["cash_delta"])

        st.markdown("**Financials (quarter):** "
                    f"Revenue ${res['revenue']:,} • COGS ${res['cogs']:,} • CAC spend ${res['cac_spend']:,} • "
                    f"OPEX ${res['opex']:,} • Cash end ${res['cash_end']:,}")
        if res["events"]:
            st.info("Events: " + " · ".join(res["events"]))
        if res["nudges"]:
            for n in res["nudges"]:
                st.warning(n)

    c_l, c_r = st.columns(2)
    if c_l.button("Adjust mix (back)"):
        S["stage"] = "channels"; st.rerun()
    if c_r.button("Next: Decide & Document"):
        S["stage"] = "decide"; st.rerun()

def page_decide():
    st.subheader("Decide & Document — Business Model Snapshot")
    st.write("Summarize your choices and jot two evidence tests with clear targets.")
    with st.form("snapshot"):
        st.markdown(f"**ICP (segment):** {S['segment']}")
        st.text_input("Value proposition (one sentence)",
                      value=f"{S['value_outcome']} with a {S['promise'].lower()} promise, delivered {S['delivery'].lower()}.")
        st.text_input("Model & pricing",
                      value=f"{S['model']} — Good/Better/Best: ${S['price_tier']['Good']}/{S['price_tier']['Better']}/{S['price_tier']['Best']}.")
        st.text_input("Channels (GTM summary)",
                      value=", ".join([f"{k}:{v}" for k,v in S["gtm_alloc"].items() if v>0]) or "None")
        st.text_input("Expected funnel metric call-outs",
                      value="Aim LTV/CAC ≥ 3; payback ≤ 9mo (B2C) or ≤ 15mo (B2B); GM ≥ 60%.")
        S["notes_tests"]["t1"] = st.text_input("Next test #1",
                      value=S["notes_tests"]["t1"] or "Paywall A/B: move feature X behind paywall; target free→paid ≥ 4%.")
        S["notes_tests"]["t2"] = st.text_input("Next test #2",
                      value=S["notes_tests"]["t2"] or "Partner pilot: 3 partners, 4 weeks; target ≥3 weekly shares and CAC ↓ 20%.")
        submitted = st.form_submit_button("Submit & Score")
    if submitted:
        S["score"] = score_run(S["history"])
        # brief coaching
        reasons = S["score"]["reasons"]
        S["coach"] = [
            "When payback is long, change the binding constraint first: price/ARPU, CAC/channel mix, or churn.",
            "Freemium only works if the aha is fast and the paywall captures true value; otherwise pick trial→subscription.",
            "Delivery level must match price tier; if COGS drags GM below 60%, move toward Hybrid/Automated or increase price.",
            "Avoid GTM monocultures; one channel >60% makes you fragile. Mix 2–3 that fit your segment.",
        ]
        S["stage"] = "results"; st.rerun()

def page_results():
    st.subheader("Results & Coaching")
    sc = S["score"]
    st.metric("Total Score", f"{sc['total']}/100")
    st.markdown("#### Category Scores")
    for k,v in sc["components"].items():
        label = "Excellent" if v>=0.8 else ("Good" if v>=0.6 else "Needs work")
        st.write(f"- **{k}:** {int(v*100)}/100 — {label}")
        st.caption(sc["reasons"][k])

    st.markdown("#### Key Lessons")
    for tip in S["coach"]:
        st.write(f"- {tip}")

    st.markdown("#### Deliverables")
    st.write(f"- **Business Model Snapshot:** ICP={S['segment']}; Model={S['model']}; "
             f"Price Good/Better/Best=${S['price_tier']['Good']}/{S['price_tier']['Better']}/{S['price_tier']['Best']}; "
             f"Channels=" + (", ".join([f"{k}:{v}" for k,v in S["gtm_alloc"].items() if v>0]) or "None"))
    st.write(f"- **Next tests:** 1) {S['notes_tests']['t1']}  2) {S['notes_tests']['t2']}")

    if st.button("Restart simulation"):
        init_state(); st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# Flow
# ──────────────────────────────────────────────────────────────────────────────
def main():
    header()
    stage = S["stage"]
    steps = ["intro","segment","value","pricing","channels","run","decide","results"]
    st.progress((steps.index(stage)+1)/len(steps))
    if stage == "intro":
        page_intro()
    elif stage == "segment":
        page_segment()
    elif stage == "value":
        page_value()
    elif stage == "pricing":
        page_pricing()
    elif stage == "channels":
        page_channels()
    elif stage == "run":
        page_run()
    elif stage == "decide":
        page_decide()
    else:
        page_results()

main()
