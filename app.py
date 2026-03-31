# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

from peltierlab.core.simulator import Simulator
from peltierlab.core.simulator_hysteresis_real import SimulatorHysteresisReal

# -------------------------------
# Config
# -------------------------------
st.set_page_config(page_title="PeltierLab Simulator", layout="wide")

# -------------------------------
# Session state
# -------------------------------
if "running" not in st.session_state:
    st.session_state.running = False

if "idx" not in st.session_state:
    st.session_state.idx = 1  # evitar gráfico vacío

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.markdown("## ❄️ PeltierLab Simulator")
st.sidebar.markdown("---")

sim_type = st.sidebar.radio(
    "Mode",
    ["Dynamic", "Static"],
    horizontal=True
)

mode = st.sidebar.selectbox("Control mode", ["PID", "FOPID", "Hysteresis"])
duration = st.sidebar.slider("Simulation duration [s]", 100, 500, 300, step=10)

# -------------------------------
# Buttons
# -------------------------------
if sim_type == "Dynamic":
    col1, col2 = st.sidebar.columns(2)

    if col1.button("Start"):
        st.session_state.running = True

    if col2.button("Stop"):
        st.session_state.running = False
else:
    st.sidebar.info("Static mode: instant response")

# -------------------------------
# Defaults
# -------------------------------
best_params = [1.9507, 2.4906, 36.4772, 0.4806, 14.0687, 2.2298, 66.5757, 11.8439]
T_start = 19.0

Kp_default = 58.93
Ki_default = 3.91
Kd_default = 2.66
lambda_default = 0.67
mu_default = 1.47

# -------------------------------
# Controls
# -------------------------------
with st.sidebar.expander("Control Parameters", expanded=True):

    lam = lambda_default
    mu = mu_default

    if mode in ["PID", "FOPID"]:
        T_set = st.slider("Setpoint [°C]", 5.0, 18.0, 12.0, 0.1)
        bias = st.slider("Bias [°C]", -2.0, 2.0, 0.0, 0.1)
        Kp = st.slider("Kp", 0, 200, int(Kp_default), 1)
        Ki = st.slider("Ki", 0.0, 50.0, Ki_default, 0.1)
        Kd = st.slider("Kd", 0.0, 50.0, Kd_default, 0.1)

        if mode == "FOPID":
            lam = st.slider("Lambda (λ)", 0.1, 2.0, lambda_default, 0.01)
            mu = st.slider("Mu (μ)", 0.1, 2.0, mu_default, 0.01)

    else:
        T_set = st.slider("Setpoint [°C]", 10.0, 18.0, 12.0, 0.1)
        dT1 = st.slider("Upper band (dT1)", 0.1, 1.0, 0.5, 0.1)
        dT2 = st.slider("Lower band (dT2)", 0.1, 1.0, 0.5, 0.1)

# -------------------------------
# Simulation
# -------------------------------
t_full = np.linspace(0, duration, duration + 1)

if mode in ["PID", "FOPID"]:
    sim = Simulator(best_params, T_start=T_start)

    Tc_full, pwm_full = sim.simulate_3nodes_FOPID(
        t_custom=t_full,
        T_set=T_set,
        Kp=Kp, Ki=Ki, Kd=Kd,
        bias=bias,
        lam=lam,
        mu=mu
    )

else:
    sim = SimulatorHysteresisReal(best_params, T_start=T_start)

    Tc_full, _, _, pwm_full = sim.simulate(
        t_custom=t_full,
        T_set=T_set,
        dT1=dT1,
        dT2=dT2,
        P_max=5.0
    )

# -------------------------------
# Plot
# -------------------------------
st.subheader(f"{mode} Simulation")

# -------------------------------
# STATIC
# -------------------------------
if sim_type == "Static":

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(t_full, Tc_full, lw=1.5)
    ax.axhline(T_set, linestyle="--")

    ax.set_xlim(0, duration)
    ax.set_ylim(0, 20)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [°C]")
    ax.grid(True)

    st.pyplot(fig)

# -------------------------------
# DYNAMIC
# -------------------------------
else:

    step = 1

    if st.session_state.running:
        st.session_state.idx = min(
            st.session_state.idx + step,
            len(t_full) - 1
        )

    i = max(1, st.session_state.idx)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(t_full[:i], Tc_full[:i], lw=1.5)
    ax.axhline(T_set, linestyle="--")

    ax.set_xlim(0, duration)
    ax.set_ylim(0, 20)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [°C]")
    ax.grid(True)

    st.pyplot(fig)

    if st.session_state.running:
        time.sleep(0.05)
        st.rerun()

# -------------------------------
# Metrics (CORRECTED)
# -------------------------------
error = Tc_full - T_set

ss_error = np.mean(error[-50:])
rmse = np.sqrt(np.mean(error**2))

overshoot = 0.0
cross_idx = None

for k in range(len(Tc_full)):
    if Tc_full[k] <= T_set:
        cross_idx = k
        break

if cross_idx is not None:
    post = Tc_full[cross_idx:]
    if len(post) > 0:
        overshoot = max(0.0, np.max(post) - T_set)

settling_time = None
for j in range(len(Tc_full)):
    if np.all(np.abs(error[j:]) <= 0.5):
        settling_time = t_full[j]
        break

# -------------------------------
# Sidebar metrics
# -------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### Metrics & Status")

st.sidebar.markdown(f"**Time:** {t_full[i]:.1f} s")
st.sidebar.markdown(f"**PWM:** {pwm_full[i]:.1f}")
st.sidebar.progress(int(pwm_full[i] / 255 * 100))

col1, col2, col3, col4 = st.sidebar.columns(4)
col1.metric("SS Error", f"{ss_error:.2f} °C")
col2.metric("RMSE", f"{rmse:.2f}")
col3.metric("Overshoot", f"{overshoot:.2f} °C")
col4.metric("Settling Time", f"{settling_time if settling_time else '—'} s")

# -------------------------------
# Recommendations
# -------------------------------
recs = []

if mode in ["PID", "FOPID"]:
    if abs(ss_error) > 0.5:
        recs.append("Increase Ki")

    if overshoot > 1.0:
        recs.append("Reduce Kp or increase Kd")

    if settling_time is None or settling_time > 150:
        recs.append("Increase Kp")

    if rmse > 1.0:
        recs.append("Tune gains globally")

else:
    if overshoot > 1.0:
        recs.append("Reduce hysteresis band")

    if settling_time is None:
        recs.append("Increase hysteresis band")

    if rmse > 1.0:
        recs.append("Adjust thresholds")

if recs:
    st.sidebar.markdown("### 🧠 Recommendations")
    for r in recs:
        st.sidebar.markdown(f"- {r}")
