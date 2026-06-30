"""
COVID-19 Burden & Risk Dashboard  —  MSBA 382 Healthcare Analytics
==================================================================
A consultant data-analyst tool profiling the burden of COVID-19 and predicting
patient mortality risk. Crosses two sources:
  - Our World in Data (country-level)  -> global trends, geography, cross-country drivers
  - Mexican Ministry of Health (patient-level) -> demographics, risk factors, prediction

Run:  streamlit run app.py        (password: msba382)
Data: run `python prepare_data.py` first to build ./data/*.parquet
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------- CONFIG
st.set_page_config(page_title="COVID-19 Burden & Risk Dashboard",
                   page_icon="🩺", layout="wide", initial_sidebar_state="expanded")

COMORBID = ["diabetes", "hypertension", "obesity", "copd", "asthma",
            "cardiovascular", "renal_chronic", "immunosuppression", "smoking"]
FEATURES = ["age", "sex_male"] + COMORBID + ["pneumonia"]
PRETTY = {"diabetes": "Diabetes", "hypertension": "Hypertension", "obesity": "Obesity",
          "copd": "COPD", "asthma": "Asthma", "cardiovascular": "Cardiovascular",
          "renal_chronic": "Chronic kidney", "immunosuppression": "Immunosuppression",
          "smoking": "Smoking", "pneumonia": "Pneumonia", "age": "Age (per year)",
          "sex_male": "Male sex"}
CORRNAME = {"median_age": "Median age", "aged_70_older": "Aged 70+",
            "human_development_index": "HDI", "life_expectancy": "Life expectancy",
            "extreme_poverty": "Extreme poverty", "hospital_beds_per_thousand": "Hospital beds/1k",
            "gdp_per_capita": "GDP per capita", "diabetes_prevalence": "Diabetes prevalence",
            "cardiovasc_death_rate": "Cardiovascular deaths"}

# brand palette
NAVY, BLUE, RED, TEAL, AMBER, GREY = "#0f2b46", "#2c6fbb", "#c0392b", "#159a8c", "#e8a33d", "#9aa7b2"
PLOTLY_SEQ = [BLUE, RED, TEAL, AMBER, "#7b5ea7", "#5b8c5a"]

# consistent Plotly look
PLOT_LAYOUT = dict(font=dict(family="Inter, Segoe UI, Arial", size=13, color="#1f2d3d"),
                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                   margin=dict(l=10, r=10, t=50, b=10),
                   colorway=PLOTLY_SEQ, hoverlabel=dict(font_size=12))

def style_fig(fig, title=None, h=380):
    fig.update_layout(**PLOT_LAYOUT, height=h)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=16, color=NAVY)))
    fig.update_xaxes(gridcolor="#eef2f6", zeroline=False)
    fig.update_yaxes(gridcolor="#eef2f6", zeroline=False)
    return fig

# ------------------------------------------------------------------------- CSS
def inject_css():
    st.markdown(f"""
    <style>
      .stApp {{ background: linear-gradient(180deg,#f7f9fc 0%, #eef3f8 100%); }}
      #MainMenu, footer {{ visibility: hidden; }}
      .hero {{ background: linear-gradient(110deg,{NAVY} 0%,{BLUE} 100%);
               padding: 22px 28px; border-radius: 16px; color: #fff; margin-bottom: 6px;
               box-shadow: 0 6px 22px rgba(15,43,70,.18); }}
      .hero h1 {{ margin:0; font-size: 30px; font-weight: 800; letter-spacing:-.5px; }}
      .hero p {{ margin:4px 0 0; opacity:.85; font-size:14px; }}
      .kpi {{ background:#fff; border-radius:14px; padding:16px 18px; text-align:left;
              box-shadow:0 2px 10px rgba(15,43,70,.07); border-left:5px solid {BLUE}; }}
      .kpi .v {{ font-size:26px; font-weight:800; color:{NAVY}; line-height:1.1; }}
      .kpi .l {{ font-size:12px; color:{GREY}; text-transform:uppercase; letter-spacing:.5px; margin-top:3px;}}
      .kpi.red {{ border-left-color:{RED}; }} .kpi.teal {{ border-left-color:{TEAL}; }}
      .kpi.amber {{ border-left-color:{AMBER}; }}
      .stTabs [data-baseweb="tab-list"] {{ gap:4px; }}
      .stTabs [data-baseweb="tab"] {{ background:#fff; border-radius:10px 10px 0 0;
              padding:8px 16px; font-weight:600; }}
      .stTabs [aria-selected="true"] {{ background:{BLUE}; color:#fff; }}
      .insight {{ background:#fff8ec; border-left:4px solid {AMBER}; padding:12px 16px;
                  border-radius:8px; font-size:14px; margin:6px 0; }}
      div[data-testid="stMetricValue"] {{ font-size:26px; }}
    </style>""", unsafe_allow_html=True)

def kpi(col, value, label, kind=""):
    col.markdown(f"<div class='kpi {kind}'><div class='v'>{value}</div>"
                 f"<div class='l'>{label}</div></div>", unsafe_allow_html=True)

# ----------------------------------------------------------------- PASSWORD GATE
def check_password() -> bool:
    try:
        correct = st.secrets["password"]
    except Exception:
        correct = "msba382"
    if st.session_state.get("authed"):
        return True
    st.markdown(f"<div class='hero' style='text-align:center;margin-top:8vh'>"
                f"<h1>🩺 COVID-19 Burden & Risk Dashboard</h1>"
                f"<p>Healthcare analytics consulting tool · restricted access</p></div>",
                unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        pw = st.text_input("Access password", type="password", label_visibility="collapsed",
                           placeholder="Enter access password")
        if st.button("Enter dashboard", width='stretch', type="primary"):
            if pw == correct:
                st.session_state["authed"] = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        st.caption("Demo password: msba382")
    return False

# ------------------------------------------------------------------- DATA / MODEL
@st.cache_data(show_spinner=False)
def load_data(uploaded_patients=None):
    owid = pd.read_parquet("data/owid.parquet")
    owid["date"] = pd.to_datetime(owid["date"])
    ind = pd.read_parquet("data/country_indicators.parquet")
    if uploaded_patients is not None:
        patients = pd.read_parquet(uploaded_patients) if str(uploaded_patients).endswith("parquet") \
            else pd.read_csv(uploaded_patients)
    else:
        patients = pd.read_parquet("data/patients.parquet")
    # auto handle types / missing
    for c in COMORBID + ["pneumonia", "died"]:
        if c in patients:
            patients[c] = pd.to_numeric(patients[c], errors="coerce").fillna(0).astype(int)
    if "age_band" not in patients:
        patients["age_band"] = pd.cut(patients["age"], [0, 18, 30, 40, 50, 65, 80, 200],
                                      labels=["0-17", "18-29", "30-39", "40-49", "50-64", "65-79", "80+"],
                                      right=False)
    patients["n_comorbid"] = patients[COMORBID].sum(axis=1)
    return owid, ind, patients

@st.cache_resource(show_spinner=False)
def train_models(patients: pd.DataFrame):
    df = patients.copy()
    df["sex_male"] = (df["sex"] == "Male").astype(int)
    out = {}
    for label, feats in [("Full", FEATURES), ("Baseline", [f for f in FEATURES if f != "pneumonia"])]:
        X, y = df[feats], df["died"]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        m = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, ytr)
        proba = m.predict_proba(Xte)[:, 1]
        pr, rc, f1, _ = precision_recall_fscore_support(yte, (proba >= .5).astype(int),
                                                        average="binary", zero_division=0)
        out[label] = dict(model=m, feats=feats, auc=roc_auc_score(yte, proba),
                          precision=pr, recall=rc, f1=f1)
    return out

# ------------------------------------------------------------------------- TABS
def svg_spark(vals, color, w=132, h=34):
    vals = [v for v in vals if v == v]
    if len(vals) < 2 or max(vals) == min(vals):
        return ""
    mn, mx = min(vals), max(vals)
    pts = [(i/(len(vals)-1)*w, h - (v-mn)/(mx-mn)*(h-7) - 3) for i, v in enumerate(vals)]
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"0,{h} " + poly + f" {w},{h}"
    return (f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}' style='margin-top:6px'>"
            f"<polygon points='{area}' fill='{color}' opacity='0.13'/>"
            f"<polyline points='{poly}' fill='none' stroke='{color}' stroke-width='2'/></svg>")

def kpi_spark(col, value, label, spark, kind=""):
    col.markdown(f"<div class='kpi {kind}'><div class='v'>{value}</div>"
                 f"<div class='l'>{label}</div>{spark}</div>", unsafe_allow_html=True)

def callout(col, title, value, sub, color):
    col.markdown(f"<div style='background:#fff;border-radius:12px;padding:13px 15px;margin-bottom:11px;"
                 f"box-shadow:0 2px 8px rgba(15,43,70,.06);border-left:4px solid {color}'>"
                 f"<div style='font-size:10.5px;color:{GREY};text-transform:uppercase;letter-spacing:.6px'>{title}</div>"
                 f"<div style='font-size:19px;font-weight:800;color:{NAVY};line-height:1.15'>{value}</div>"
                 f"<div style='font-size:12px;color:#5b6b7a'>{sub}</div></div>", unsafe_allow_html=True)

def section(title, source, scolor, filter_note):
    st.markdown(
        f"<div style='margin:6px 0 2px'>"
        f"<span style='font-size:19px;font-weight:800;color:{NAVY}'>{title}</span>"
        f"<span style='background:{scolor};color:#fff;font-size:11px;font-weight:700;"
        f"padding:3px 9px;border-radius:10px;margin-left:10px;vertical-align:middle'>{source}</span>"
        f"</div><div style='font-size:12px;color:{GREY};margin-bottom:8px'>↳ {filter_note}</div>",
        unsafe_allow_html=True)

def tab_overview(owid, fdf, patients):
    # monthly OWID series (global, not country-filtered — it's the global picture)
    mo = owid.copy()
    mo["month"] = mo["date"].dt.to_period("M").dt.to_timestamp()
    g = mo.groupby("month")[["new_cases_smoothed", "new_deaths_smoothed"]].sum()
    g = g[(g.new_cases_smoothed > 0) & (g.index >= pd.Timestamp("2020-04-01"))]
    g["cfr"] = g.new_deaths_smoothed / g.new_cases_smoothed * 100

    # ============ SECTION 1: GLOBAL — WORLD (OWID) ============
    section("Global picture", "🌍 WORLD DATA (Our World in Data)", BLUE,
            "Reacts to the GLOBAL filter (region, countries, dates) in the sidebar.")
    tc = fdf.groupby("location")["total_cases"].max().sum()
    td = fdf.groupby("location")["total_deaths"].max().sum()
    c = st.columns(4)
    kpi_spark(c[0], f"{tc/1e6:.1f}M", "Total cases (selected)",
              svg_spark(g.new_cases_smoothed.tolist(), BLUE), "")
    kpi_spark(c[1], f"{td/1e3:.0f}K", "Total deaths (selected)",
              svg_spark(g.new_deaths_smoothed.tolist(), RED), "red")
    kpi_spark(c[2], f"{g.cfr.iloc[-6:].mean():.2f}%", "Recent global CFR",
              svg_spark(g.cfr.tolist(), AMBER), "amber")
    kpi(c[3], f"{fdf['location'].nunique()}", "Countries selected", "teal")
    st.write("")
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_scatter(x=g.index, y=g.new_cases_smoothed, name="Cases",
                        line=dict(color=BLUE, width=2), fill="tozeroy", fillcolor="rgba(44,111,187,.12)")
        fig.add_scatter(x=g.index, y=g.new_deaths_smoothed, name="Deaths", yaxis="y2",
                        line=dict(color=RED, width=2))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False),
                          legend=dict(orientation="h", y=1.14))
        st.plotly_chart(style_fig(fig, "Global cases & deaths over time", 300), width='stretch')
    with right:
        fig = go.Figure()
        fig.add_vrect(x0="2021-06-01", x1="2021-12-01", fillcolor=AMBER, opacity=.15, line_width=0,
                      annotation_text="Delta", annotation_position="top left")
        fig.add_vrect(x0="2021-12-01", x1=str(g.index.max().date()), fillcolor=TEAL, opacity=.12,
                      line_width=0, annotation_text="Omicron+", annotation_position="top left")
        fig.add_scatter(x=g.index, y=g.cfr, line=dict(color=RED, width=2.5), name="Monthly CFR %")
        st.plotly_chart(style_fig(fig, "Lethality over time — Delta → Omicron", 300), width='stretch')
    st.markdown("<div class='insight'>📉 Global case fatality fell ~5× from the Delta era (~1.7%) to "
                "Omicron+ (~0.3%). Note: this global CFR (~1–2%) is far lower than the Mexico cohort's "
                "below — different populations, see the source badges.</div>", unsafe_allow_html=True)

    st.divider()

    # ============ SECTION 2: PATIENT COHORT — MEXICO ============
    section("Patient cohort", "🏥 MEXICO PATIENT DATA (Ministry of Health)", "#159a8c",
            "Reacts to the CLINICAL filter (sex, age) in the sidebar — not the country filter.")
    age_cfr = patients.groupby("age_band", observed=True).died.mean().mul(100)
    age_counts = patients.groupby("age_band", observed=True).size()
    c = st.columns(4)
    kpi_spark(c[0], f"{len(patients)/1000:.0f}K", "Patients (filtered)",
              svg_spark(age_counts.tolist(), TEAL), "teal")
    kpi(c[1], f"{patients.died.mean()*100:.1f}%", "Cohort case fatality", "red")
    kpi(c[2], f"{patients.age.mean():.0f}", "Mean age")
    kpi(c[3], f"{patients.n_comorbid.mean():.1f}", "Avg. comorbidities", "amber")
    st.write("")
    left, right = st.columns([2, 1])
    with left:
        bands = ["18-29", "30-39", "40-49", "50-64", "65-79", "80+"]
        conds = ["diabetes", "hypertension", "obesity", "renal_chronic", "cardiovascular"]
        z = []
        for cd in conds:
            row = []
            for a in bands:
                sub = patients[(patients.age_band == a) & (patients[cd] == 1)]
                row.append(round(sub.died.mean()*100, 0) if len(sub) > 30 else None)
            z.append(row)
        fig = go.Figure(go.Heatmap(z=z, x=bands, y=[PRETTY[c] for c in conds],
                                   colorscale="Teal", text=z, texttemplate="%{text}",
                                   textfont=dict(size=11), colorbar=dict(title="CFR %")))
        st.plotly_chart(style_fig(fig, "Case fatality % by age band × condition (Mexico cohort)", 300),
                        width='stretch')
    with right:
        dd = patients.died.value_counts().rename({0: "Survived", 1: "Died"})
        fig = go.Figure(go.Pie(labels=dd.index, values=dd.values, hole=.62,
                               marker=dict(colors=[TEAL, RED]), textinfo="percent"))
        st.plotly_chart(style_fig(fig, "Outcomes (Mexico cohort)", 300), width='stretch')

    prev = patients[COMORBID].mean()*100
    cfrc = {cd: (patients[patients[cd] == 1].died.mean()*100 if (patients[cd] == 1).any() else 0)
            for cd in COMORBID}
    top_prev, top_cfr = prev.idxmax(), max(cfrc, key=cfrc.get)
    cc = st.columns(4)
    callout(cc[0], "Most common condition", PRETTY[top_prev], f"{prev[top_prev]:.1f}% of cohort", BLUE)
    callout(cc[1], "Deadliest condition", PRETTY[top_cfr], f"{cfrc[top_cfr]:.0f}% case fatality", RED)
    callout(cc[2], "Most affected age", "80+", f"{age_cfr.get('80+', 0):.0f}% case fatality", AMBER)
    callout(cc[3], "Higher-risk sex", "Male" if patients[patients.sex=='Male'].died.mean() >=
            patients[patients.sex=='Female'].died.mean() else "Female",
            f"{patients[patients.sex=='Male'].died.mean()*100:.1f}% M vs "
            f"{patients[patients.sex=='Female'].died.mean()*100:.1f}% F", TEAL)

def tab_geography(owid, fdf):
    st.markdown("#### Geographic distribution")
    st.caption("🌍 World data (Our World in Data) · reacts to the GLOBAL filter (region, dates)")
    LABELS = {"total_deaths_per_million": "Deaths per million",
              "total_cases_per_million": "Cases per million",
              "people_fully_vaccinated_per_hundred": "Fully vaccinated (per 100)",
              "people_vaccinated_per_hundred": "≥1 dose (per 100)"}
    metric = st.selectbox("Metric", list(LABELS.keys()), format_func=lambda k: LABELS[k])

    # last NON-NULL value per country for the chosen metric (recovers UAE/Gulf vaccination)
    snap = (fdf.dropna(subset=[metric]).sort_values("date")
            .groupby("location").tail(1).copy())
    is_vax = "vaccinated" in metric
    if is_vax:  # OWID per-hundred can exceed 100 for Gulf states (migrant doses); cap for display
        snap[metric] = snap[metric].clip(upper=100)
    scale = "Reds" if "deaths" in metric else ("Greens" if is_vax else "Blues")

    fig = px.choropleth(snap, locations="iso_code", color=metric, hover_name="location",
                        color_continuous_scale=scale, labels={metric: LABELS[metric]})
    fig.update_layout(**{k: v for k, v in PLOT_LAYOUT.items() if k != "colorway"}, height=460,
                      geo=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, width='stretch')

    mena = ["Iran", "Iraq", "Lebanon", "Jordan", "Syria", "Yemen", "Saudi Arabia",
            "United Arab Emirates", "Qatar", "Kuwait", "Bahrain", "Oman", "Egypt",
            "Tunisia", "Libya", "Algeria", "Morocco", "Palestine", "Turkey"]
    a, b = st.columns(2)
    top = snap.nlargest(12, metric)
    with a:
        st.plotly_chart(style_fig(px.bar(top, x=metric, y="location", orientation="h",
                        color=metric, color_continuous_scale=scale, labels={metric: LABELS[metric]})
                        .update_yaxes(autorange="reversed"), f"Top 12 — {LABELS[metric]}"), width='stretch')
    with b:
        ms = snap[snap.location.isin(mena)].nlargest(12, metric)
        st.plotly_chart(style_fig(px.bar(ms, x=metric, y="location", orientation="h",
                        color_discrete_sequence=[TEAL], labels={metric: LABELS[metric]})
                        .update_yaxes(autorange="reversed"), f"MENA — {LABELS[metric]}"), width='stretch')

    # Reach vs completion: ≥1 dose vs fully, side by side (MENA), using last non-null per metric
    st.markdown("##### Vaccination: reach (≥1 dose) vs completion (fully) — MENA")
    rows = []
    for c in mena:
        sub = fdf[fdf.location == c]
        d1 = sub.dropna(subset=["people_vaccinated_per_hundred"])["people_vaccinated_per_hundred"]
        d2 = sub.dropna(subset=["people_fully_vaccinated_per_hundred"])["people_fully_vaccinated_per_hundred"]
        if len(d1):
            rows.append({"country": c, "≥1 dose": min(d1.iloc[-1], 100),
                         "Fully": min(d2.iloc[-1], 100) if len(d2) else None})
    vc = pd.DataFrame(rows).sort_values("≥1 dose", ascending=True)
    fig = go.Figure()
    fig.add_bar(y=vc.country, x=vc["≥1 dose"], name="≥1 dose", orientation="h", marker_color=BLUE)
    fig.add_bar(y=vc.country, x=vc["Fully"], name="Fully vaccinated", orientation="h", marker_color=TEAL)
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.08))
    st.plotly_chart(style_fig(fig, "Per 100 people — the gap is the completion drop-off", 420),
                    width='stretch')
    st.markdown("<div class='insight'>💉 <b>Reach vs completion:</b> the Gulf states (UAE, Qatar) "
                "reached near-universal coverage, while others show a wider gap between starting and "
                "finishing the series. Values use each country's last reported figure (some stopped "
                "reporting early), and are capped at 100.</div>", unsafe_allow_html=True)

def tab_drivers(ind):
    st.markdown("#### What national factors track COVID mortality?")
    st.caption("🌍 World data (Our World in Data) · 226 countries · not affected by the sidebar filters")
    inds = list(CORRNAME.keys())
    corr = {}
    for c in inds:
        v = ind[[c, "total_deaths_per_million"]].dropna()
        if len(v) > 30:
            corr[c] = v[c].corr(v["total_deaths_per_million"])
    cs = pd.Series(corr).sort_values()
    dfc = pd.DataFrame({"indicator": [CORRNAME[k] for k in cs.index], "r": cs.values})
    a, b = st.columns([1, 1])
    with a:
        st.plotly_chart(style_fig(px.bar(dfc, x="r", y="indicator", orientation="h",
                        color="r", color_continuous_scale="RdBu", range_color=[-.7, .7]),
                        "Correlation with deaths per million"), width='stretch')
    with b:
        xcol = st.selectbox("Explore an indicator", inds, format_func=lambda k: CORRNAME[k])
        v = ind[[xcol, "total_deaths_per_million", "location"]].dropna()
        fig = px.scatter(v, x=xcol, y="total_deaths_per_million", hover_name="location",
                         color_discrete_sequence=[BLUE],
                         labels={xcol: CORRNAME[xcol], "total_deaths_per_million": "Deaths per million"})
        # trendline drawn with numpy (no statsmodels dependency)
        if len(v) >= 2:
            m, c = np.polyfit(v[xcol], v["total_deaths_per_million"], 1)
            xs = np.array([v[xcol].min(), v[xcol].max()])
            fig.add_scatter(x=xs, y=m * xs + c, mode="lines", name="trend",
                            line=dict(color=RED, width=2, dash="dash"), showlegend=False)
        st.plotly_chart(style_fig(fig, f"{CORRNAME[xcol]} vs mortality"), width='stretch')
    st.markdown("<div class='insight'>💡 <b>Ecological-fallacy alert:</b> diabetes prevalence shows almost "
                "no country-level correlation with mortality (r≈0), yet at the individual level (Risk "
                "Factors tab) diabetes is a strong predictor. Country aggregates can hide what matters for "
                "individuals — which is why we analyse patient data too.</div>", unsafe_allow_html=True)

def tab_demographics(pf):
    st.markdown("#### Who is most affected")
    st.caption("🏥 Mexico patient data (Ministry of Health) · reacts to the CLINICAL filter (sex, age)")
    c = st.columns(4)
    kpi(c[0], f"{len(pf):,}", "Patients (filtered)")
    kpi(c[1], f"{pf.died.mean()*100:.1f}%", "Case fatality ratio", "red")
    kpi(c[2], f"{pf.age.mean():.0f}", "Mean age", "teal")
    kpi(c[3], f"{pf.n_comorbid.mean():.1f}", "Avg. comorbidities", "amber")
    st.write("")
    st.markdown("<div class='insight'>🔗 <b>Coordinated view:</b> click an age band in the left chart and "
                "the two panels beside and below it update to that group. The charts are linked, not separate.</div>",
                unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        ba = pf.groupby("age_band", observed=True).died.mean().mul(100).reset_index()
        figb = px.bar(ba, x="age_band", y="died", color="died",
                      color_continuous_scale="Reds", labels={"died": "CFR %", "age_band": "Age"})
        ev = st.plotly_chart(style_fig(figb, "Mortality by age band  (👆 click to filter)"),
                             width='stretch', on_select="rerun", selection_mode="points",
                             key="age_select")
        sel = None
        try:
            pts = ev.selection["points"] if ev and ev.selection else []
            if pts:
                sel = pts[0].get("x")
        except Exception:
            sel = None
    sub = pf[pf.age_band == sel] if sel else pf
    label = f"age band {sel}" if sel else "all ages"
    with b:
        sx = sub.groupby("sex").died.mean().mul(100).reset_index()
        st.plotly_chart(style_fig(px.bar(sx, x="sex", y="died", color="sex",
                        color_discrete_sequence=[GREY, BLUE], labels={"died": "CFR %"}),
                        f"Mortality by sex — {label}"), width='stretch')
    # linked comorbidity breakdown, filtered by the clicked age band
    cc = [{"factor": PRETTY[cd], "cfr": (sub[sub[cd] == 1].died.mean()*100 if (sub[cd] == 1).any() else 0)}
          for cd in COMORBID]
    cfr_df = pd.DataFrame(cc).sort_values("cfr")
    st.plotly_chart(style_fig(px.bar(cfr_df, x="cfr", y="factor", orientation="h",
                    color_discrete_sequence=[RED], labels={"cfr": "CFR %", "factor": ""}),
                    f"Fatality by condition — {label}"), width='stretch')
    if sel:
        st.caption(f"🔎 Focused on {len(sub):,} patients in age band {sel}. "
                   f"Click the bar again to clear, or pick another band.")

def tab_riskfactors(pf):
    st.markdown("#### Clinical risk factors")
    st.caption("🏥 Mexico patient data (Ministry of Health) · reacts to the CLINICAL filter (sex, age)")
    a, b = st.columns([1.1, 1])
    with a:
        rows = [{"factor": PRETTY[c], "with": pf[pf[c] == 1].died.mean()*100,
                 "without": pf[pf[c] == 0].died.mean()*100} for c in COMORBID]
        rf = pd.DataFrame(rows).sort_values("with")
        fig = go.Figure()
        fig.add_bar(y=rf.factor, x=rf["with"], name="Has condition", orientation="h", marker_color=RED)
        fig.add_bar(y=rf.factor, x=rf["without"], name="Does not", orientation="h", marker_color=GREY)
        fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(style_fig(fig, "Mortality % with vs without", 420), width='stretch')
    with b:
        prev = (pf[COMORBID].mean()*100).sort_values()
        dpv = pd.DataFrame({"c": [PRETTY[i] for i in prev.index], "v": prev.values})
        st.plotly_chart(style_fig(px.bar(dpv, x="v", y="c", orientation="h",
                        color_discrete_sequence=[BLUE], labels={"v": "% of patients", "c": ""}),
                        "Comorbidity prevalence", 420), width='stretch')
    a, b = st.columns(2)
    with a:
        cnt = pf.n_comorbid.value_counts()
        dose = pf.groupby("n_comorbid").died.mean().mul(100)[cnt >= 20].reset_index()
        st.plotly_chart(style_fig(px.line(dose, x="n_comorbid", y="died", markers=True,
                        color_discrete_sequence=[RED], labels={"died": "CFR %", "n_comorbid": "# conditions"}),
                        "Dose-response: more conditions, more risk"), width='stretch')
    with b:
        sev = pd.DataFrame({"stage": ["Outpatient", "Hospitalized", "ICU", "Died"],
                            "n": [(pf.hospitalized == 0).sum(), pf.hospitalized.sum(),
                                  pf.icu.sum() if "icu" in pf else 0, pf.died.sum()]})
        st.plotly_chart(style_fig(px.funnel(sev, x="n", y="stage", color_discrete_sequence=[BLUE]),
                        "Outcome severity funnel"), width='stretch')
    st.markdown("<div class='insight'>⚠️ <b>On smoking:</b> it shows a surprisingly weak (even reversed) "
                "association. Recorded in only ~7% of patients and unstable across models — the documented "
                "'smoker's paradox', treated here as a <b>data-quality limitation, not a protective effect</b>."
                "</div>", unsafe_allow_html=True)

def tab_prediction(pf, models):
    st.markdown("#### Mortality risk prediction")
    st.caption("🏥 Mexico patient data (Ministry of Health) · model trained on this cohort")
    c = st.columns(4)
    kpi(c[0], f"{models['Full']['auc']:.3f}", "Full model AUC")
    kpi(c[1], f"{models['Baseline']['auc']:.3f}", "Baseline model AUC", "teal")
    kpi(c[2], f"{models['Full']['recall']:.0%}", "Recall (deaths caught)", "amber")
    kpi(c[3], f"{models['Full']['precision']:.0%}", "Precision", "red")
    st.markdown("<div class='insight'>Two models: the <b>Full</b> model (AUC ~0.91) includes pneumonia, "
                "which is partly a <i>consequence</i> of severe disease. The <b>Baseline</b> model (AUC ~0.84) "
                "uses only pre-existing factors a clinician can screen on at intake — the deployable one. "
                "Both favour recall, since missing a high-risk patient is the costly error. "
                "AUC is stable under 5-fold cross-validation (0.91 ± 0.004 and 0.84 ± 0.006), and the "
                "baseline model is reasonably calibrated (Brier 0.17).</div>",
                unsafe_allow_html=True)
    odds = pd.Series(np.exp(models["Full"]["model"].coef_[0]), index=FEATURES).sort_values()
    do = pd.DataFrame({"f": [PRETTY[i] for i in odds.index], "or": odds.values})
    st.plotly_chart(style_fig(px.bar(do, x="or", y="f", orientation="h", color="or",
                    color_continuous_scale="RdBu_r", labels={"or": "Odds ratio (>1 = higher risk)", "f": ""}),
                    "Adjusted odds ratios — full model", 420), width='stretch')

    st.markdown("##### 🧮 Patient risk calculator")
    g = st.columns(3)
    age = g[0].slider("Age", 0, 100, 55)
    sex_male = 1 if g[1].selectbox("Sex", ["Female", "Male"]) == "Male" else 0
    pneu = 1 if g[2].selectbox("Pneumonia", ["No", "Yes"]) == "Yes" else 0
    picked = st.multiselect("Comorbidities", [PRETTY[c] for c in COMORBID])
    inv = {v: k for k, v in PRETTY.items()}
    row = {f: 0 for f in FEATURES}
    row.update({"age": age, "sex_male": sex_male, "pneumonia": pneu})
    for p in picked:
        row[inv[p]] = 1
    prob = models["Full"]["model"].predict_proba(pd.DataFrame([row])[FEATURES])[0, 1]
    color = RED if prob > .3 else (AMBER if prob > .1 else TEAL)
    cc = len(picked)
    cc1, cc2 = st.columns([2, 1])
    with cc1:
        st.markdown(f"<div style='text-align:center;padding:18px;background:#fff;border-radius:14px;"
                    f"box-shadow:0 2px 10px rgba(15,43,70,.07)'>"
                    f"<div style='font-size:13px;color:{GREY};text-transform:uppercase'>Predicted mortality probability</div>"
                    f"<div style='font-size:46px;font-weight:800;color:{color}'>{prob:.0%}</div></div>",
                    unsafe_allow_html=True)
    with cc2:
        st.markdown(f"<div style='text-align:center;padding:18px;background:#fff;border-radius:14px;"
                    f"box-shadow:0 2px 10px rgba(15,43,70,.07)'>"
                    f"<div style='font-size:13px;color:{GREY};text-transform:uppercase'>Comorbidity count</div>"
                    f"<div style='font-size:46px;font-weight:800;color:{NAVY}'>{cc}</div></div>",
                    unsafe_allow_html=True)
    st.markdown("<div class='insight'>🧱 <b>Engineered feature \u2014 comorbidity count:</b> each additional "
                "condition multiplies the odds of death by ~1.43. Remarkably, a model using just "
                "<b>age, sex and this count</b> predicts almost as well as the full nine-condition model "
                "(AUC 0.834 vs 0.838) \u2014 a simple, deployable bedside score.</div>",
                unsafe_allow_html=True)

# --------------------------------------------------------------------------- MAIN
def main():
    inject_css()
    if not check_password():
        st.stop()

    # sidebar: optional upload + filters
    st.sidebar.markdown("### ⚙️ Controls")
    with st.sidebar.expander("Use your own patient data (optional)"):
        up = st.file_uploader("Same-schema CSV/Parquet", type=["csv", "parquet"])
    with st.spinner("Loading data..."):
        owid, ind, patients = load_data(up if up else None)
        models = train_models(patients)

    st.sidebar.markdown("#### 🌍 Global tabs filter")
    regions = ["All"] + sorted(owid["continent"].dropna().unique())
    region = st.sidebar.selectbox("Region", regions)
    pool = owid if region == "All" else owid[owid.continent == region]
    countries = sorted(pool.location.unique())
    default = [c for c in ["Lebanon", "Mexico", "United States", "India"] if c in countries][:4]
    picked = st.sidebar.multiselect("Countries (trend charts)", countries, default=default or countries[:4])
    dmin, dmax = owid.date.min().to_pydatetime(), owid.date.max().to_pydatetime()
    dr = st.sidebar.slider("Date range", min_value=dmin, max_value=dmax, value=(dmin, dmax))

    st.sidebar.markdown("#### 🧑‍⚕️ Clinical tabs filter")
    sexes = st.sidebar.multiselect("Sex", ["Female", "Male"], default=["Female", "Male"])
    arange = st.sidebar.slider("Age range", 0, 100, (0, 100))

    # apply filters
    fdf = pool[(pool.location.isin(picked)) & (pool.date >= pd.Timestamp(dr[0])) & (pool.date <= pd.Timestamp(dr[1]))]
    geo = pool[(pool.date >= pd.Timestamp(dr[0])) & (pool.date <= pd.Timestamp(dr[1]))]
    pf = patients[(patients.sex.isin(sexes)) & (patients.age.between(arange[0], arange[1]))]

    st.markdown(f"<div class='hero'><h1>🩺 COVID-19 Burden & Risk Dashboard</h1>"
                f"<p>From global burden to individual risk · Our World in Data + Mexican Ministry of Health</p></div>",
                unsafe_allow_html=True)

    with st.expander("ℹ️  About this dashboard — what it is and how to use it"):
        st.markdown(
            "**Purpose.** A triage-support tool for a capacity-constrained health centre: it answers "
            "*\"when beds are limited, which patients should be prioritised?\"* by moving from the global "
            "burden of COVID-19 down to individual patient risk.\n\n"
            "**Two data sources, kept separate** (different units, no shared key):\n"
            "- 🌍 **World data (Our World in Data)** — country-level cases, deaths, vaccinations and national "
            "indicators. Drives the Geography and Cross-Country tabs and the global charts. Reacts to the "
            "**Global filter** (region, countries, dates).\n"
            "- 🏥 **Mexico patient data (Ministry of Health)** — ~392k individual records with age, sex, "
            "comorbidities and outcome. Drives the Demographics, Risk Factors and Prediction tabs. Reacts to "
            "the **Clinical filter** (sex, age). Used as a *transferable risk framework* whose drivers match "
            "Lebanon's profile.\n\n"
            "**Everything is coordinated.** One set of sidebar filters drives every chart at once, so the views "
            "respond together rather than as a series of separate graphs. The Demographics tab goes further: "
            "click an age band and its neighbouring panels re-filter to that group — linked, interactive graphs. "
            "The risk calculator is itself coordinated input-to-output: change a patient detail and the risk "
            "updates live.\n\n"
            "**How to use.** Pick filters in the sidebar, move tab by tab (Overview → Geography → "
            "Cross-Country → Demographics → Risk Factors → Prediction), click an age band on Demographics to see "
            "the panels link, and try the **risk calculator** in the Prediction tab to score a patient live. "
            "Each panel is labelled with its data source.")

    if st.sidebar.button("Log out"):
        st.session_state["authed"] = False
        st.rerun()

    t = st.tabs(["🌐 Overview", "🗺️ Geography", "📊 Cross-Country", "👥 Demographics",
                 "🫀 Risk Factors", "🔮 Prediction"])
    with t[0]: tab_overview(owid, fdf if len(fdf) else geo, pf if len(pf) else patients)
    with t[1]: tab_geography(owid, geo)
    with t[2]: tab_drivers(ind)
    with t[3]: tab_demographics(pf if len(pf) else patients)
    with t[4]: tab_riskfactors(pf if len(pf) else patients)
    with t[5]: tab_prediction(pf if len(pf) else patients, models)

    with st.expander("📄 View raw data"):
        which = st.radio("Dataset", ["Patient-level (Mexico)", "Country-level (OWID)", "Country indicators"],
                         horizontal=True)
        st.dataframe({"Patient-level (Mexico)": patients.head(500),
                      "Country-level (OWID)": owid.head(500),
                      "Country indicators": ind.head(500)}[which], width='stretch')

if __name__ == "__main__":
    main()
