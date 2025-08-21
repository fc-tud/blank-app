import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


# --- Page setup ---
st.set_page_config(page_title="Pareto Front Optimization Demo", layout="wide")

# --- Domain-specific parameters and surrogate models ---
# Five inputs with realistic ranges/units
PARAMS = {
    "cement": {"label": "Cement content (kg/m^3)", "min": 200.0, "max": 600.0, "step": 5.0, "default": 380.0},
    "w_c": {"label": "Water-to-cement ratio (w/c)", "min": 0.25, "max": 0.60, "step": 0.01, "default": 0.40},
    "textile": {"label": "Textile amount (vol.-%)", "min": 0.0, "max": 10.0, "step": 0.1, "default": 3.0},
    "flyash": {"label": "Fly ash content (% of binder)", "min": 0.0, "max": 40.0, "step": 1.0, "default": 15.0},
    "silica": {"label": "Silica fume content (% of binder)", "min": 0.0, "max": 15.0, "step": 0.5, "default": 6.0},
}

# Surrogate objective functions capturing common trends
# GWP (lower is better): clinker >> silica fume > textile >> fly ash ~ water
# Resistance (higher is better): peaks around w/c 0.35-0.40; increases with textile; silica helps; excessive fly ash slightly reduces

def evaluate_phys(cement, w_c, textile, flyash, silica):
    # Effective binder split (simplified): fly ash and silica are percentages of binder referenced to cement
    # Approximate masses for SCMs relative to cement amount
    m_flyash = cement * (flyash / 100.0)
    m_silica = cement * (silica / 100.0)
    # Assume SCMs replace part of clinker fraction (not strictly mass-balanced; surrogate only)
    m_clinker = cement * (1.0 - 0.7 * (flyash / 100.0))  # fly ash substitution reduces clinker mass

    # Embodied carbon factors (relative, demo values)
    EF_CLINKER = 0.90   # kg CO2 per kg
    EF_FLYASH  = 0.05   # allocated low
    EF_SILICA  = 0.60
    EF_TEXTILE = 2.00   # per vol.-% unit (proxy)
    EF_WATER   = 0.001  # negligible but nonzero

    # Water mass ~ w/c * cement
    m_water = w_c * cement

    # GWP surrogate
    gwp = (
        EF_CLINKER * m_clinker
        + EF_FLYASH * m_flyash
        + EF_SILICA * m_silica
        + EF_TEXTILE * textile
        + EF_WATER * m_water
    )

    # Resistance surrogate
    # Strength vs. w/c: bell-shaped peak near 0.38
    wc_peak = 0.38
    wc_k = 40.0  # sharpness
    f_wc = np.exp(-wc_k * (w_c - wc_peak) ** 2)

    # Base contribution from binder (diminishing returns with sqrt)
    base = 0.08 * np.sqrt(max(cement, 1.0)) * (1.0 + 0.2 * (silica / 10.0)) * f_wc

    # Textile contribution with mild saturation; penalize very high textile for workability/fiber efficiency
    txt_gain = 2.2 * (1.0 - np.exp(-0.35 * textile))
    txt_penalty = 0.04 * max(textile - 6.0, 0.0) ** 2

    # Fly ash slight early-age reduction
    flyash_penalty = 0.02 * (flyash / 10.0)

    resistance = 10.0 * base + txt_gain - txt_penalty - flyash_penalty

    return float(gwp), float(resistance)

# --- Pareto utilities ---
def pareto_upper_envelope(points: np.ndarray) -> np.ndarray:
    order = np.argsort(points[:, 0])  # sort by GWP
    pts = points[order]
    keep = []
    best_res = -np.inf
    for i, (_, r) in enumerate(pts):
        if r > best_res:
            keep.append(order[i])
            best_res = r
    front = points[keep]
    front = front[np.argsort(front[:, 0])]
    return front

# Distance from a point to a polyline (the ordered Pareto front)
def point_to_polyline_distance(p: np.ndarray, poly: np.ndarray):
    if len(poly) == 0:
        return None, None
    if len(poly) == 1:
        return float(np.linalg.norm(p - poly[0])), poly[0]

    def seg_dist(px, a, b):
        ab = b - a
        ap = px - a
        denom = np.dot(ab, ab)
        t = 0.0 if denom == 0 else np.clip(np.dot(ap, ab) / denom, 0.0, 1.0)
        proj = a + t * ab
        return float(np.linalg.norm(px - proj)), proj

    best_d, best_proj = float("inf"), None
    for i in range(len(poly) - 1):
        d, proj = seg_dist(p, poly[i], poly[i + 1])
        if d < best_d:
            best_d, best_proj = d, proj
    return best_d, best_proj

# --- Sample design space and compute objectives (only keep the Pareto front) ---
rng = np.random.default_rng(42)
N_SAMPLES = 50000  # dense sampling, but we will keep only the front

if "front" not in st.session_state:
    X_sampled = np.column_stack([
        rng.uniform(PARAMS["cement"]["min"], PARAMS["cement"]["max"], N_SAMPLES),
        rng.uniform(PARAMS["w_c"]["min"], PARAMS["w_c"]["max"], N_SAMPLES),
        rng.uniform(PARAMS["textile"]["min"], PARAMS["textile"]["max"], N_SAMPLES),
        rng.uniform(PARAMS["flyash"]["min"], PARAMS["flyash"]["max"], N_SAMPLES),
        rng.uniform(PARAMS["silica"]["min"], PARAMS["silica"]["max"], N_SAMPLES)
    ])


    # --- Add design space extrema (all min/max combinations) ---
    from itertools import product
    minmax_values = [[PARAMS["cement"]["min"], PARAMS["cement"]["max"]], # cement
                     [PARAMS["w_c"]["min"], PARAMS["w_c"]["max"]], # wc
                     [PARAMS["textile"]["min"], PARAMS["textile"]["max"]], # textile
                     [PARAMS["flyash"]["min"], PARAMS["flyash"]["max"]], # flyash
                     [PARAMS["silica"]["min"],  PARAMS["silica"]["max"]] # silica
                     ]
    extrema = np.array(list(product(*minmax_values)))

    # Combine sampled points with extrema
    X_full = np.vstack([X_sampled, extrema])
    results = np.array([evaluate_phys(*x) for x in X_full])
    st.session_state.front = pareto_upper_envelope(results)
    st.session_state.results = results
    st.session_state.X_full = X_full

# --- App UI ---
st.title("Pareto Front Optimization Demo")

st.markdown(
    """
    ### Instructions
    This app demonstrates **multi-objective optimization** for textile-reinforced concrete (TRC) design with
    five input parameters and two competing objectives:
    - **GWP (Global Warming Potential)** - lower is better (minimize).
    - **Impact Resistance** - higher is better (maximize).

    The **Pareto front** is the set of nondominated trade-offs. Adjust the parameters to move your design towards this front.

    **Plot legend**
    - Blue star: your current design.
    - Navy points: Pareto-optimal front.
    - Green dot & dashed line: closest point and distance from your design to the Pareto front.
    """
)

# Two-column layout (inputs on the left, plot on the right)
col_inputs, col_plot = st.columns([1, 2.2])

with col_inputs:
    st.header("Input Parameters")
    cement = st.slider(PARAMS["cement"]["label"], PARAMS["cement"]["min"], PARAMS["cement"]["max"], PARAMS["cement"]["default"], PARAMS["cement"]["step"])
    w_c = st.slider(PARAMS["w_c"]["label"], PARAMS["w_c"]["min"], PARAMS["w_c"]["max"], PARAMS["w_c"]["default"], PARAMS["w_c"]["step"])
    textile = st.slider(PARAMS["textile"]["label"], PARAMS["textile"]["min"], PARAMS["textile"]["max"], PARAMS["textile"]["default"], PARAMS["textile"]["step"])
    flyash = st.slider(PARAMS["flyash"]["label"], PARAMS["flyash"]["min"], PARAMS["flyash"]["max"], PARAMS["flyash"]["default"], PARAMS["flyash"]["step"])
    silica = st.slider(PARAMS["silica"]["label"], PARAMS["silica"]["min"], PARAMS["silica"]["max"], PARAMS["silica"]["default"], PARAMS["silica"]["step"])

# Evaluate user selection
user_gwp, user_res = evaluate_phys(cement, w_c, textile, flyash, silica)
user_pt = np.array([user_gwp, user_res])

# Distance to Pareto front
dist_to_front, proj_pt = point_to_polyline_distance(user_pt, st.session_state.front)


with col_plot:
    fig = go.Figure()
    fig.add_scatter(x=st.session_state.front[:, 0], y=st.session_state.front[:, 1], name="Pareto Front", mode="markers", marker=dict(color="#00305e", size=6))
    fig.add_scatter(x=[user_gwp], y=[user_res], mode="markers", name="Your design", marker=dict(color="blue", size=12, symbol="star"))
    if proj_pt is not None:
        fig.add_scatter(x=[proj_pt[0]], y=[proj_pt[1]], mode="markers", name="Nearest on front", marker=dict(color="green", size=10))
        fig.add_scatter(x=[user_gwp, proj_pt[0]], y=[user_res, proj_pt[1]], mode="lines", name="Distance", line=dict(dash="dash", color="green"))
    fig.update_layout(
        xaxis_title="Global Warming Potential, GWP [kg CO2-eq/m^3] (minimize)",
        yaxis_title="Impact Resistance [a.u.] (maximize)"
    )

    # Button to show Pareto-optimal solutions with inputs
    if "show_pareto_inputs" not in st.session_state:
        st.session_state.show_pareto_inputs = False
        
        
    # Initialize once Button 
    if "show_pareto_inputs" not in st.session_state:
        st.session_state.show_pareto_inputs = False
    # Switch behavior
    if st.button("Show Pareto-optimal input parameters"):
        st.session_state.show_pareto_inputs = not st.session_state.show_pareto_inputs

    if st.session_state.show_pareto_inputs:
        if "pareto_mask" not in st.session_state:
            # indices of points in X_full that belong to the front
            st.session_state.pareto_mask = np.isin(st.session_state.results, st.session_state.front).all(axis=1)
            st.session_state.custom_data = st.session_state.X_full[st.session_state.pareto_mask]

        fig.add_scatter(
        x=st.session_state.front[:, 0], y=st.session_state.front[:, 1],
        mode="markers",
        showlegend=False,  # hide from legend
        marker=dict(color="#00305e", size=6),
        customdata=st.session_state.custom_data,
        hovertemplate=(
        "GWP: %{x:.2f}<br>"
        "Resistance: %{y:.2f}<br>"
        "Cement: %{customdata[0]:.1f} kg/m^3<br>"
        "w/c: %{customdata[1]:.2f}<br>"
        "Textile: %{customdata[2]:.2f} %<br>"
        "Fly ash: %{customdata[3]:.2f} %<br>"
        "Silica: %{customdata[4]:.2f} %<extra></extra>"
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    
    if st.session_state.show_pareto_inputs and proj_pt is not None:
        # find index of nearest point on front
        idx = np.argmin(np.linalg.norm(st.session_state.front - np.array([user_gwp, user_res]), axis=1))
        nearest_inputs = st.session_state.X_full[st.session_state.pareto_mask][idx]
        # round to 2 decimals
        nearest_inputs = np.round(nearest_inputs, 2)
        
        # create a readable dict
        input_dict = {
            "Cement [kg/m^3]": f"{nearest_inputs[0]:.2f}",
            "w/c ratio": f"{nearest_inputs[1]:.2f}",
            "Textile [%]": f"{nearest_inputs[2]:.2f}",
            "Fly ash [%]": f"{nearest_inputs[3]:.2f}",
            "Silica [%]": f"{nearest_inputs[4]:.2f}",
        }
        
        st.markdown("**Nearest Pareto-optimal input parameters:**")
        st.table(input_dict)
            
    st.markdown(
        f" - **Your design:** GWP = {user_gwp:.2f}, Impact Resistance = {user_res:.2f}" +
        (f" \n - **Distance to Pareto front:** {dist_to_front:.2f}" if dist_to_front is not None else "")
    )