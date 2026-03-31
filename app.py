# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

from peltierlab.core.simulator import Simulator
from peltierlab.core.simulator_hysteresis_real import SimulatorHysteresisReal

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="PeltierLab Simulator",
    layout="wide"
)

# -------------------------------
# Global parameters
# -------------------------------
best_params = [1.9507, 2.4906, 36.4772, 0.4806, 14.0687, 2.2298, 66.5757, 11.8439]
T_start = 19.0

Kp_default = 58.93
Ki_default = 3.91
Kd_default = 2.66
lambda_default = 0.67
mu_default = 1.47

# -------------------------------
# Sidebar Title
# -------------------------------
st.sidebar.title("❄️ PeltierLab Interactive Simulator")
st.sidebar.markdown(
    "Thermoelectric system simulation using PID, FOPID, and Hysteresis controls."
)

# -------------------------------
# Sidebar controls
# -------------------------------
mode = st.sidebar.selectbox("Control mode", ["PID", "FOPID", "Hysteresis"])

duration = st.sidebar.slider("Simulation duration [s]", 100, 500, 300, step=10)

# Dynamic / Static side by side
col1, col2 = st.sidebar.columns(2)
dynamic = col1.radio("Mode", ["Dynamic", "Static"], index=0)

start = col2.button("Start")
stop = col2.button("Stop")

# -------------------------------
# Control sliders
# -------------------------------
with st.sidebar.expander("Control Parameters", expanded=True):
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
        dT1 = st.slider("Upper band (dT1) [°C]", 0.1, 1.0, 0.5, 0.1)
        dT2 = st.slider("Lower band (dT2) [°C]", 0.1, 1.0, 0.5, 0.1)

# -------------------------------
# Session state initialization
# -------------------------------
if 'running' not in st.session_state:
    st.session_state.running = False
if 'y_data' not in st.session_state or len(st.session_state.y_data)==0:
    st.session_state.y_data = []
if 't_data' not in st.session_state or len(st.session_state.t_data)==0:
    st.session_state.t_data = []
if 'idx' not in st.session_state:
    st.session_state.idx = 0

# -------------------------------
# Start / Stop handling
# -------------------------------
if start:
    st.session_state.running = True
    st.session_state.idx = 0
    st.session_state.y_data = []
    st.session_state.t_data = []

if stop:
    st.session_state.running = False

# -------------------------------
# Prepare simulation data
# -------------------------------
t_full = np.linspace(0, duration, duration+1)
if mode in ["PID", "FOPID"]:
    sim = Simulator(best_params, T_start=T_start)
    if mode == "PID":
        Tc_full, pwm_full = sim.simulate_3nodes_FOPID(
            t_custom=t_full, T_set=T_set, Kp=Kp, Ki=Ki, Kd=Kd,
            bias=bias, lam=lambda_default, mu=mu_default
        )
    else:
        Tc_full, pwm_full = sim.simulate_3nodes_FOPID(
            t_custom=t_full, T_set=T_set, Kp=Kp, Ki=Ki, Kd=Kd,
            bias=bias, lam=lam, mu=mu
        )
else:
    sim = SimulatorHysteresisReal(best_params, T_start=T_start)
    Tc_full, Tm_full, Th_full, pwm_full = sim.simulate(
        t_custom=t_full, T_set=T_set, dT1=dT1, dT2=dT2, P_max=5.0
    )

# -------------------------------
# Initialize data for plotting
# -------------------------------
if len(st.session_state.y_data) == 0:
    st.session_state.y_data.append(Tc_full[0])
    st.session_state.t_data.append(t_full[0])
    st.session_state.idx = 1

# -------------------------------
# Plot setup
# -------------------------------
fig, ax = plt.subplots(figsize=(8,4))
line, = ax.plot(st.session_state.t_data, st.session_state.y_data, lw=1.5, color='blue', label="Temperature")
ax.axhline(T_set, color="red", linestyle="--", label="Setpoint")
ax.set_xlim(0, duration)
ax.set_ylim(0, 20)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Temperature [°C]")
ax.tick_params(axis='both', labelsize=8)
ax.grid(True, lw=0.5)
ax.legend(fontsize=8)
plot_placeholder = st.pyplot(fig)

# -------------------------------
# Sidebar metrics
# -------------------------------
st.sidebar.markdown("---")
elapsed_placeholder = st.sidebar.empty()
pwm_placeholder = st.sidebar.empty()
pwm_bar = st.sidebar.empty()

# -------------------------------
# Recommendations below sidebar
# -------------------------------
info_expander = st.expander("Metrics & Recommendations", expanded=True)
metrics_text = info_expander.empty()

# -------------------------------
# Simulation update
# -------------------------------
if st.session_state.running and st.session_state.idx < len(t_full):
    idx = st.session_state.idx
    st.session_state.y_data.append(Tc_full[idx])
    st.session_state.t_data.append(t_full[idx])

    # Update plot
    line.set_data(st.session_state.t_data, st.session_state.y_data)
    ax.set_xlim(0, duration)
    plot_placeholder.pyplot(fig)

    # Update sidebar metrics
    elapsed_placeholder.markdown(f"**Time:** {int(t_full[idx])} s")
    pwm_placeholder.markdown(f"**PWM:** {pwm_full[idx]:.1f}")
    pwm_bar.progress(int(pwm_full[idx]/255*100))

    # Metrics
    error = np.array(st.session_state.y_data) - T_set
    ss_error = np.mean(error[-50:]) if len(error)>50 else np.mean(error)
    rmse = np.sqrt(np.mean(error**2))
    overshoot = max(st.session_state.y_data) - T_set
    settling_time = next((t_full[j] for j in range(idx, len(t_full)) if np.all(np.abs(error[j:])<=0.5)), None)
    recs = []
    if mode in ["PID","FOPID"]:
        if ss_error > 0.5:
            recs.append("Increase Ki to reduce steady-state error")
        if settling_time is None or settling_time>150:
            recs.append("Increase Kp for faster settling")
        if overshoot > 1.0:
            recs.append("Decrease Kd to reduce overshoot")
    metrics_text.markdown(
        f"**Steady-state error:** {ss_error:.2f} °C  \n"
        f"**RMSE:** {rmse:.2f}  \n"
        f"**Overshoot:** {overshoot:.2f} °C  \n"
        f"**Settling time:** {settling_time if settling_time else 'Not reached'} s  \n"
        + ("\n".join(f"- {r}" for r in recs))
    )

    st.session_state.idx += 1
