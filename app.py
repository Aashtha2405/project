"""
ClimateCrop - Climate Change Impact on Crop Yield (Streamlit front end)
Run:  streamlit run app.py
Needs (same folder): crop_yield_model.pkl, portable_model.py   (+ crop_data_clean.csv.gz for the charts)
All three are created by the notebook.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from portable_model import portable_predict

st.set_page_config(page_title="ClimateCrop", page_icon="🌾", layout="wide", initial_sidebar_state="expanded")

HERE = Path(__file__).parent
MODEL_ORDER = ["XGBoost", "Random Forest", "Decision Tree", "SVM (SVR)", "Linear Regression", "Gradient Descent"]
MODEL_COLORS = {"Linear Regression": "#2a78d6", "Gradient Descent": "#eb6834", "Decision Tree": "#1baf7a",
                "Random Forest": "#eda100", "SVM (SVR)": "#e87ba4", "XGBoost": "#2d6a4f"}
DARK, MID, LIGHT = "#1b4332", "#2d6a4f", "#b7e4c7"

# ============================================================================== THEME (CSS)
st.markdown(
    """
<style>
.stApp { background: linear-gradient(135deg, #eaf6ec 0%, #d9ecdb 100%) !important; }
.block-container { padding-top: 1.6rem; max-width: 1250px; }
section[data-testid="stSidebar"] { background: #f4faf5 !important; border-right: 1px solid #c8e6c9; }
section[data-testid="stSidebar"] h3 { color: #1b4332; font-size: 18px; margin-top: .6rem; }

.hero { background: linear-gradient(120deg, #081c15 0%, #1b4332 55%, #2d6a4f 100%); border-radius: 18px;
        padding: 2rem 2rem 1.6rem 2rem; margin-bottom: 1.4rem; box-shadow: 0 10px 28px rgba(8,28,21,.25); }
.hero-title { font-size: 40px; font-weight: 800; color: #fff; letter-spacing: -.5px; }
.hero-sub { font-size: 18px; color: #b7e4c7; margin-top: 2px; font-weight: 500; }
.hero-tags { margin-top: 14px; }
.hero-tags span { display: inline-block; background: rgba(255,255,255,.12); color: #d8f3dc; border: 1px solid rgba(255,255,255,.25);
                  padding: 4px 12px; border-radius: 999px; font-size: 13px; margin-right: 8px; }

.section-title { font-size: 22px; font-weight: 700; color: #1b4332; margin: .6rem 0 .9rem 0; padding-bottom: 6px; border-bottom: 2px solid #a3b18a; }

div[data-testid="stMetric"] { background: #fff !important; border-radius: 14px !important; padding: 1rem 1.2rem !important;
        border-left: 6px solid #1b4332 !important; box-shadow: 0 4px 15px rgba(0,0,0,.05) !important; transition: transform .2s ease; }
div[data-testid="stMetric"]:hover { transform: translateY(-3px); box-shadow: 0 8px 20px rgba(0,0,0,.10) !important; }
div[data-testid="stMetric"] label, div[data-testid="stMetric"] label p { color: #2d6a4f !important; font-weight: 700 !important; }
div[data-testid="stMetricValue"] { color: #1b4332 !important; font-weight: 800 !important; }

.result-card { padding: 1.8rem 1.5rem; border-radius: 18px; background: linear-gradient(135deg, #1b4332 0%, #081c15 100%);
               color: #fff; text-align: center; box-shadow: 0 10px 25px rgba(8,28,21,.3); margin: .4rem 0 1.2rem 0; }
.result-card h2 { font-size: 18px; font-weight: 600; color: #b7e4c7; margin: 0 0 .3rem 0; }
.result-card h1 { font-size: 52px; font-weight: 800; color: #fff; margin: .2rem 0; line-height: 1.1; }
.result-card h1 small { font-size: 18px; font-weight: 500; color: #b7e4c7; }
.result-card p { font-size: 15px; color: #d8f3dc; margin: .4rem 0 0 0; }

.chip { display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: 13px; font-weight: 700; margin: 4px 6px 0 0; }
.chip-good { background: #d8f3dc; color: #1b4332; }
.chip-warn { background: #fff0cc; color: #7a5200; }
.chip-bad  { background: #ffd9d2; color: #8a2a17; }
.chip-dark { background: rgba(255,255,255,.14); color: #fff; border: 1px solid rgba(255,255,255,.3); }

.note { background: #fff; border-left: 5px solid #f4a261; border-radius: 10px; padding: .8rem 1rem; color: #3d3d3d; font-size: 14px;
        box-shadow: 0 2px 10px rgba(0,0,0,.04); margin: .6rem 0; }
.note b { color: #1b4332; }

button[data-baseweb="tab"] { font-weight: 700; font-size: 16px; }
button[data-baseweb="tab"][aria-selected="true"] { color: #1b4332 !important; }
div[data-baseweb="tab-highlight"] { background-color: #1b4332 !important; }

div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { border-radius: 8px !important; border-color: #95c9a5 !important; background: #fff !important; }
label, label p { color: #1b4332 !important; font-weight: 600 !important; }
div.stButton > button { background: linear-gradient(135deg, #1b4332, #081c15) !important; color: #fff !important; font-weight: 700 !important;
        border: none !important; border-radius: 10px !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================================== LOAD
PKL_PATH = HERE / "crop_yield_model.pkl"
if not PKL_PATH.exists():
    st.error("`crop_yield_model.pkl` not found next to app.py. Run the notebook (last section) first.")
    st.stop()


@st.cache_resource(show_spinner="Loading trained models…")
def load_artifacts():
    return joblib.load(PKL_PATH)


@st.cache_data(show_spinner=False)
def load_data():
    path = HERE / "crop_data_clean.csv.gz"
    return pd.read_csv(path) if path.exists() else None


A = load_artifacts()
DF = load_data()
opt, dflt, models = A["options"], A["defaults"], A["models"]
AVAILABLE = [m for m in MODEL_ORDER if m in models]
best = A["best_model"]
metrics = pd.DataFrame(A["metrics"]).set_index("Model")
climate = A.get("climate")
y_min, y_max = A["year_range"]


# ============================================================================== PREDICTION HELPERS
def build_features(crop, state, season, soil, year, area, temp, wind, rain, hum, n, p, k):
    # same pipeline as the notebook: log(area), one-hot, standardise numeric columns
    row = pd.DataFrame(0.0, index=[0], columns=A["columns"])
    row.loc[0, A["num_features"]] = [year, np.log1p(area), temp, wind, rain, hum, n, p, k]
    for col in (f"crop_names_{crop}", f"state_names_{state}", f"season_names_{season}", f"soil_type_{soil}"):
        if col in row.columns:
            row.loc[0, col] = 1.0
    row[A["num_features"]] = (row[A["num_features"]].values - A["scaler_mean"]) / A["scaler_scale"]
    return row.values.astype(float)


def predict_yield(name, X):
    log_pred = portable_predict(models[name], X)
    return float(np.expm1(max(float(np.ravel(log_pred)[0]), 0.0)))      # models predict log(1 + yield)


def style_fig(fig, height=380):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=55, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff", font=dict(color=DARK), title_font_size=16)
    return fig


def climate_effect(d_temp, d_hum):
    # % change in yield for a climate change (statistical estimate from the notebook, section 8) -> (estimate, low, high)
    d = np.array([d_temp, d_hum, 0.0])
    mean, sd = float(climate["beta"] @ d), float(np.sqrt(d @ climate["cov"] @ d))
    return tuple(float(np.exp(mean + k * sd) - 1) for k in (0, -1.96, 1.96))


# ============================================================================== SIDEBAR (inputs)
with st.sidebar:
    st.markdown("### 🌱 Field details")
    state = st.selectbox("📍 State", opt["states"], index=opt["states"].index("Uttar Pradesh") if "Uttar Pradesh" in opt["states"] else 0)
    crop = st.selectbox("🌾 Crop", opt["crops"], index=opt["crops"].index("Rice") if "Rice" in opt["crops"] else 0)
    season = st.selectbox("🗓️ Season", opt["seasons"], index=opt["seasons"].index("Kharif") if "Kharif" in opt["seasons"] else 0)
    soil = st.selectbox("🪨 Soil type", opt["soils"], index=opt["soils"].index("clay") if "clay" in opt["soils"] else 0)
    year = st.number_input("📅 Year", min_value=y_min, max_value=2100, value=y_max, step=1)
    area = st.number_input("📐 Area (hectares)", min_value=1.0, value=float(round(dflt["area"])), step=10.0)

    npk = A["crop_npk"].get(crop, {"N": dflt["N"], "P": dflt["P"], "K": dflt["K"]})
    st.markdown("**🧪 Fertiliser N · P · K**")
    c1, c2, c3 = st.columns(3)
    n = c1.number_input("N", min_value=0.0, value=float(npk["N"]), key=f"N_{crop}")
    p = c2.number_input("P", min_value=0.0, value=float(npk["P"]), key=f"P_{crop}")
    k = c3.number_input("K", min_value=0.0, value=float(npk["K"]), key=f"K_{crop}")
    st.caption("Pre-filled with the typical value for the chosen crop.")

    st.markdown("### 🌦️ Current climate")
    temp = st.slider("🌡️ Temperature (°C)", 5.0, 40.0, float(round(dflt["temperature"], 1)), 0.5)
    hum = st.slider("💧 Humidity (%)", 10.0, 100.0, float(round(dflt["humidity"])), 1.0)
    wind = st.slider("🌬️ Wind speed", 0.0, 5.0, float(round(dflt["wind_speed"], 1)), 0.1)
    rain = dflt["precipitation"]

    st.markdown("### 🤖 Model")
    selected = st.selectbox("Prediction model", AVAILABLE, index=AVAILABLE.index(best) if best in AVAILABLE else 0,
                            help=f"{best} was the most accurate model on the test data.")

inputs = dict(crop=crop, state=state, season=season, soil=soil, year=year, area=area, temp=temp, wind=wind, rain=rain, hum=hum, n=n, p=p, k=k)
X_now = build_features(**inputs)
all_preds = {m: predict_yield(m, X_now) for m in AVAILABLE}
yhat = all_preds[selected]

# ============================================================================== HEADER + KPIs
st.markdown(
    "<div class='hero'><div class='hero-title'>🌾 ClimateCrop</div>"
    "<div class='hero-sub'>Climate Change Impact on Crop Yield Prediction</div>"
    "<div class='hero-tags'><span>6 machine-learning models</span><span>Climate-change scenarios</span><span>Data insights</span></div></div>",
    unsafe_allow_html=True,
)
k1, k2, k3, k4 = st.columns(4)
k1.metric("📊 Records analysed", f"{A.get('meta', {}).get('n_records', A['trained_on_rows']):,}")
k2.metric("🌾 Crops", len(opt["crops"]))
k3.metric("📍 States", len(opt["states"]))
k4.metric(f"🏆 Best model: {best}", f"R² {metrics.loc[best, 'Test R2']:.3f}")
st.write("")

tab_pred, tab_climate, tab_data, tab_models, tab_about = st.tabs(
    ["🔮 Predict yield", "🌡️ Climate impact", "📊 Data insights", "🏆 Models", "ℹ️ About"])

# ============================================================================== TAB 1: PREDICT
with tab_pred:
    rmse = float(metrics.loc[selected, "Test RMSE"])
    low, high = yhat * np.exp(-rmse), yhat * np.exp(rmse)
    stats = A["crop_yield_stats"].get(crop)

    chips = f"<span class='chip chip-dark'>Likely range {low:,.2f} – {high:,.2f}</span>"
    if stats:
        if stats[0.25] <= yhat <= stats[0.75]:
            chips += "<span class='chip chip-good'>Within the typical range for this crop</span>"
        elif yhat > stats[0.75]:
            chips += "<span class='chip chip-warn'>Above the typical range for this crop</span>"
        else:
            chips += "<span class='chip chip-warn'>Below the typical range for this crop</span>"
    st.markdown(
        f"<div class='result-card'><h2>Predicted yield · {selected}</h2>"
        f"<h1>{yhat:,.2f} <small>per hectare</small></h1>"
        f"<p>📍 {state} &nbsp;|&nbsp; 🌾 {crop} &nbsp;|&nbsp; 🗓️ {season} {year} &nbsp;|&nbsp; 🪨 {soil}</p>{chips}</div>",
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("🚜 Estimated total production", f"{yhat * area:,.0f}", help="Predicted yield × area")
    if stats:
        m2.metric(f"📈 Typical {crop} yield (median)", f"{stats[0.5]:,.2f}")
        m3.metric("↔️ Compared with typical", f"{(yhat / stats[0.5] - 1) * 100:+.0f}%")
    if year > y_max:
        st.markdown(f"<div class='note'>⚠️ The training data ends in <b>{y_max}</b>; predictions for {year} are extrapolations.</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>What does each algorithm predict for this field?</div>", unsafe_allow_html=True)
    cmp = pd.DataFrame({"Model": list(all_preds), "Yield": list(all_preds.values())}).sort_values("Yield")
    fig = go.Figure(go.Bar(x=cmp["Yield"], y=cmp["Model"], orientation="h", marker_color=[MODEL_COLORS[m] for m in cmp["Model"]],
                           text=[f"{v:,.2f}" for v in cmp["Yield"]], textposition="outside"))
    if stats:
        fig.add_vline(x=stats[0.5], line_dash="dash", line_color="#52514e", annotation_text=f"typical {crop}", annotation_position="top")
    fig.update_layout(title="Predicted yield by model", xaxis_title="Yield per hectare", showlegend=False)
    st.plotly_chart(style_fig(fig, 360))
    spread = (max(all_preds.values()) - min(all_preds.values())) / max(np.mean(list(all_preds.values())), 1e-9)
    st.caption(f"Models differ by {spread:.0%} of the average prediction. A small difference means the models agree and the result is more reliable.")

# ============================================================================== TAB 2: CLIMATE IMPACT
with tab_climate:
    st.markdown("<div class='section-title'>🌡️ What if the climate changes?</div>", unsafe_allow_html=True)
    if not climate:
        st.info("Climate analysis is missing from the model file. Re-run the notebook (section 8 and the last section).")
    else:
        PRESETS = {"Custom": None, "Moderate warming (+1.5 °C)": (1.5, 0.0), "High warming (+3 °C, drier air)": (3.0, -3.0),
                   "Extreme warming (+5 °C, drier air)": (5.0, -5.0), "Cooler & more humid (−1 °C, +5 %)": (-1.0, 5.0)}
        st.session_state.setdefault("d_temp", 3.0)
        st.session_state.setdefault("d_hum", -3.0)

        def apply_preset():
            values = PRESETS[st.session_state["preset"]]
            if values:
                st.session_state["d_temp"], st.session_state["d_hum"] = values

        st.selectbox("Choose a climate scenario", list(PRESETS), index=2, key="preset", on_change=apply_preset)
        s1, s2 = st.columns(2)
        d_temp = s1.slider("🌡️ Temperature change (°C)", -3.0, 6.0, step=0.5, key="d_temp")
        d_hum = s2.slider("💧 Humidity change (%-points)", -20.0, 20.0, step=1.0, key="d_hum")

        est, lo, hi = climate_effect(d_temp, d_hum)
        baseline = yhat
        scenario, scen_lo, scen_hi = baseline * (1 + est), baseline * (1 + lo), baseline * (1 + hi)

        c1, c2, c3 = st.columns(3)
        c1.metric("🌾 Yield today", f"{baseline:,.2f}", help=f"Predicted by {selected} for the inputs in the sidebar")
        c2.metric("🌡️ Yield in the new climate", f"{scenario:,.2f}")
        c3.metric("📉 Change in yield", f"{est * 100:+.1f}%", delta=f"{est * 100:+.1f}%")

        clear = (lo > 0) or (hi < 0)
        verdict = ("<span class='chip chip-bad'>Clear negative effect</span>" if clear and est < 0 else
                   "<span class='chip chip-good'>Clear positive effect</span>" if clear else
                   "<span class='chip chip-warn'>No clear effect: the 95 % range includes 0</span>")
        part_t, part_h = climate_effect(d_temp, 0.0)[0], climate_effect(0.0, d_hum)[0]
        st.markdown(f"{verdict}<span class='chip chip-dark' style='color:#1b4332;border-color:#95c9a5'>95 % range: {lo * 100:+.1f}% to {hi * 100:+.1f}% "
                    f"→ yield between {scen_lo:,.2f} and {scen_hi:,.2f}</span><br>"
                    f"<span class='chip chip-dark' style='color:#1b4332;border-color:#95c9a5'>🌡️ from temperature: {part_t * 100:+.1f}%</span>"
                    f"<span class='chip chip-dark' style='color:#1b4332;border-color:#95c9a5'>💧 from humidity: {part_h * 100:+.1f}%</span>", unsafe_allow_html=True)

        # warming curve
        dT = np.linspace(-3, 6, 91)
        curve = np.array([climate_effect(t, d_hum) for t in dT]) * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dT, y=curve[:, 2], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=dT, y=curve[:, 1], fill="tonexty", fillcolor="rgba(82,183,136,.30)", line=dict(width=0), name="95 % range", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=dT, y=curve[:, 0], line=dict(color=DARK, width=3), name="Estimated effect"))
        fig.add_trace(go.Scatter(x=[d_temp], y=[est * 100], mode="markers", marker=dict(size=14, color="#e76f51", line=dict(color="#fff", width=2)), name="Your scenario"))
        fig.add_hline(y=0, line_color="#52514e", line_width=1)
        fig.update_layout(title="How yield responds to warming (humidity change as selected)", xaxis_title="Temperature change (°C)",
                          yaxis_title="Change in yield (%)", legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(style_fig(fig, 400))

        st.markdown(
            f"<div class='note'><b>How this is calculated.</b> The effect comes from a statistical analysis of the <b>{climate['n_rows']:,} records with real weather data</b> "
            f"({climate['n_states']} states, {climate['n_crops']} crops), comparing the same crop, state, season and year trend. It is applied on top of the "
            f"{selected} prediction. <b>Rainfall cannot be analysed</b>: the rainfall/precipitation column in the dataset is almost constant (~1013 in every row). "
            f"Treat the result as an indication, not a forecast.</div>", unsafe_allow_html=True)

        names = {"temperature": "🌡️ Temperature (+1 °C)", "humidity": "💧 Humidity (+1 %-point)", "wind_speed": "🌬️ Wind speed (+1 unit)"}
        rows = []
        for i, key in enumerate(climate["names"]):
            b, s = float(climate["beta"][i]), float(np.sqrt(climate["cov"][i, i]))
            rows.append({"Climate factor": names.get(key, key), "Effect on yield": f"{(np.exp(b) - 1) * 100:+.2f}%",
                         "95 % range": f"{(np.exp(b - 1.96 * s) - 1) * 100:+.2f}% to {(np.exp(b + 1.96 * s) - 1) * 100:+.2f}%",
                         "Clear effect?": "✅ Yes" if abs(b / s) > 1.96 else "➖ No (range includes 0)"})
        st.markdown("<div class='section-title'>Sensitivity of yield to each climate factor</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(rows), hide_index=True)

        with st.expander("🔬 What do the raw machine-learning models say? (and why we do not rely on them for climate impact)"):
            raw = []
            for m in AVAILABLE:
                new = predict_yield(m, build_features(**{**inputs, "temp": temp + d_temp, "hum": float(np.clip(hum + d_hum, 0, 100))}))
                raw.append({"Model": m, "Yield today": round(all_preds[m], 2), "Yield in new climate": round(new, 2),
                            "Change": f"{(new / all_preds[m] - 1) * 100:+.1f}%" if all_preds[m] else "-"})
            st.dataframe(pd.DataFrame(raw), hide_index=True)
            st.caption("The models disagree (different signs and sizes) because 94 % of the weather values in the training data are one identical filler value, "
                       "so they cannot learn weather effects reliably. That is why the climate impact above uses the statistical analysis instead.")

# ============================================================================== TAB 3: DATA INSIGHTS
with tab_data:
    if DF is None:
        st.info("`crop_data_clean.csv.gz` not found next to app.py. Run the notebook (last section) to create it, then the charts appear here.")
    else:
        st.markdown(f"<div class='section-title'>📊 What the data shows (highlighting: {crop})</div>", unsafe_allow_html=True)
        a, b = st.columns(2)
        with a:
            trend = DF[DF["crop_names"] == crop].groupby("crop_year")["yield"].agg(["median", "size"]).query("size >= 5").reset_index().rename(columns={"median": "yield"})
            fig = px.line(trend, x="crop_year", y="yield", markers=True, title=f"{crop}: median yield over the years",
                          color_discrete_sequence=[MID])
            fig.update_layout(xaxis_title="Year", yaxis_title="Median yield")
            st.plotly_chart(style_fig(fig))
        with b:
            top = DF.groupby("crop_names")["yield"].median().nlargest(12).reset_index().sort_values("yield")
            fig = go.Figure(go.Bar(x=top["yield"], y=top["crop_names"], orientation="h",
                                   marker_color=[DARK if c == crop else LIGHT for c in top["crop_names"]]))
            fig.update_layout(title="Top 12 crops by median yield", xaxis_title="Median yield", yaxis_title="")
            st.plotly_chart(style_fig(fig))

        c, d = st.columns(2)
        with c:
            real = DF[DF["real_weather"]].copy()
            real["Group"] = np.where(real["crop_names"] == crop, crop, "Other crops")
            fig = px.scatter(real, x="temperature", y="yield", color="Group", log_y=True, opacity=0.55,
                             color_discrete_map={crop: "#e76f51", "Other crops": "#95c9a5"},
                             title="Temperature vs yield (records with real weather)")
            fig.update_layout(xaxis_title="Temperature (°C)", yaxis_title="Yield (log scale)", legend_title="")
            st.plotly_chart(style_fig(fig))
        with d:
            share = DF["real_weather"].value_counts(normalize=True)
            fig = go.Figure(go.Pie(labels=["Filler (identical) weather value", "Real weather measurement"],
                                   values=[share.get(False, 0), share.get(True, 0)], hole=0.6,
                                   marker_colors=["#d8d8d0", MID], textinfo="percent"))
            fig.update_layout(title="How much of the weather data is real?", legend=dict(orientation="h", y=-0.1))
            st.plotly_chart(style_fig(fig))

        e, f = st.columns(2)
        with e:
            sub = DF[DF["crop_names"] == crop]
            fig = px.box(sub, x="season_names", y="yield", log_y=True, color_discrete_sequence=[MID], title=f"{crop}: yield by season")
            fig.update_layout(xaxis_title="", yaxis_title="Yield (log scale)")
            st.plotly_chart(style_fig(fig))
        with f:
            st_med = DF[DF["crop_names"] == crop].groupby("state_names")["yield"].agg(["median", "size"]).query("size >= 5")
            st_med = st_med.nlargest(10, "median").reset_index().sort_values("median")
            fig = go.Figure(go.Bar(x=st_med["median"], y=st_med["state_names"], orientation="h",
                                   marker_color=[DARK if s == state else LIGHT for s in st_med["state_names"]]))
            fig.update_layout(title=f"{crop}: best states (median yield)", xaxis_title="Median yield", yaxis_title="")
            st.plotly_chart(style_fig(fig))

# ============================================================================== TAB 4: MODELS
with tab_models:
    st.markdown("<div class='section-title'>🏆 Which algorithm is the most accurate? (test data)</div>", unsafe_allow_html=True)
    mt = metrics.copy().sort_values("Test R2", ascending=False)
    a, b = st.columns(2)
    with a:
        fig = go.Figure(go.Bar(x=mt["Test R2"][::-1], y=mt.index[::-1], orientation="h", marker_color=[MODEL_COLORS[m] for m in mt.index[::-1]],
                               text=[f"{v:.3f}" for v in mt["Test R2"][::-1]], textposition="outside"))
        fig.update_layout(title="Accuracy: test R² (higher is better)", xaxis=dict(range=[0, 1.05]))
        st.plotly_chart(style_fig(fig, 340))
    with b:
        fig = go.Figure(go.Bar(x=mt["Test RMSE"][::-1], y=mt.index[::-1], orientation="h", marker_color=[MODEL_COLORS[m] for m in mt.index[::-1]],
                               text=[f"{v:.3f}" for v in mt["Test RMSE"][::-1]], textposition="outside"))
        fig.update_layout(title="Error: test RMSE (lower is better)")
        st.plotly_chart(style_fig(fig, 340))
    show = [c for c in ["Train R2", "Test R2", "Test RMSE", "Test MAE", "Overfit gap", "Time (s)"] if c in mt.columns]
    st.dataframe(mt[show].round(3))
    st.caption(f"🏆 Best model: **{best}**. Models were trained on {A['trained_on_rows']:,} records. "
               "R² = share of the variation explained (1 = perfect). RMSE is on the log scale, so 0.24 means a typical error of about ±27 %.")

# ============================================================================== TAB 5: ABOUT
with tab_about:
    st.markdown("<div class='section-title'>ℹ️ About this project</div>", unsafe_allow_html=True)
    st.markdown(f"""
**Goal:** predict crop yield (production per hectare) and understand how climate change could affect it.

**1. Yield prediction (machine learning).** Six algorithms (Linear Regression, Gradient Descent, Decision Tree, Random Forest, SVM, XGBoost) were trained on
{A.get('meta', {}).get('n_records', 0):,} records of crop, state, season, soil, fertiliser (N, P, K), area and weather. **{best}** was the most accurate.

**2. Climate impact (statistics).** The weather columns of the dataset are mostly one identical filler value, so ML models cannot learn weather effects reliably.
The climate-impact tab therefore uses a regression on the records that have real weather, controlling for crop, state, season and a time trend,
and reports the effect with a 95 % confidence range.

**Limitations**
- Only ~6 % of the records have real weather, from a few states; rainfall cannot be analysed (the column is almost constant).
- Yield units differ between crops (e.g. coconut is counted in nuts), so compare a prediction with the typical yield *of the same crop*.
- Predictions are a decision aid based on historical data up to {y_max}, not a guarantee.
""")

st.divider()
st.markdown("<div style='text-align:center;color:#1b4332;padding:.6rem 0'>🌾 <b>ClimateCrop</b><br>"
            "<span style='font-size:.9em'>ML-based crop yield prediction &amp; climate change impact analysis</span></div>", unsafe_allow_html=True)
