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

# Default controller parameters
Kp_default = 58.93
Ki_default = 3.91
Kd_default = 2.66
lambda_default = 0.67
mu_default = 1.47

# -------------------------------
# Title
# -------------------------------
st.title("❄️ PeltierLab Interactive Simulator")
st.markdown(
    "Explore thermoelectric system behavior using PID, FOPID, and Hysteresis control strategies."
)

# -------------------------------
# Sidebar (Settings & Controls)
# -------------------------------
st.sidebar.header("Settings")

# Control mode
mode = st.sidebar.selectbox("Control mode", ["PID", "FOPID", "Hysteresis"])

# Temperature ambient editable
T_start = st.sidebar.slider("Ambient temperature [°C]", 15.0, 25.0, T_start, 0.1)

# Simulation duration
duration = st.sidebar.slider("Simulation duration [s]", 100, 500, 300, step=10)

# Controller parameters
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
    elif mode == "Hysteresis":
        T_set = st.slider("Setpoint [°C]", 10.0, 18.0, 12.0, 0.1)
        dT1 = st.slider("Upper band (dT1) [°C]", 0.1, 1.0, 0.5, 0.1)
        dT2 = st.slider("Lower band (dT2) [°C]", 0.1, 1.0, 0.5, 0.1)

# Start / Pause / Stop buttons in one row
col1, col2, col3 = st.sidebar.columns(3)
start_clicked = col1.button("Start")
pause_clicked = col2.button("Pause")
stop_clicked = col3.button("Stop")

# -------------------------------
# Session state for simulation
# -------------------------------
if 'running' not in st.session_state:
    st.session_state['running'] = False
if 'paused' not in st.session_state:
    st.session_state['paused'] = False
if 't_data' not in st.session_state:
    st.session_state['t_data'] = []
if 'y_data' not in st.session_state:
    st.session_state['y_data'] = []
if 'pwm_data' not in st.session_state:
    st.session_state['pwm_data'] = []

# Handle buttons
if start_clicked:
    st.session_state['running'] = True
    st.session_state['paused'] = False
if pause_clicked:
    st.session_state['paused'] = True
if stop_clicked:
    st.session_state['running'] = False
    st.session_state['paused'] = False
    st.session_state['t_data'] = []
    st.session_state['y_data'] = []
    st.session_state['pwm_data'] = []

# -------------------------------
# Prepare simulation data (once)
# -------------------------------
t_full = np.linspace(0, duration, duration + 1)

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
elif mode == "Hysteresis":
    sim = SimulatorHysteresisReal(best_params, T_start=T_start)
    Tc_full, Tm_full, Th_full, pwm_full = sim.simulate(
        t_custom=t_full, T_set=T_set, dT1=dT1, dT2=dT2, P_max=5.0
    )

# -------------------------------
# Plot placeholder (once)
# -------------------------------
st.subheader(f"Results: {mode}")
fig, ax = plt.subplots(figsize=(7, 3.5))
line, = ax.plot([], [], lw=1.5, color='blue', label="Temperature")
ax.axhline(T_set, color="red", linestyle="--", label="Setpoint")
ax.set_xlim(0, duration)
ax.set_ylim(0, 20)
ax.set_xlabel("Time [s]", fontsize=8)
ax.set_ylabel("Temperature [°C]", fontsize=8)
ax.tick_params(axis='both', labelsize=7)
ax.grid(True, lw=0.5)
ax.legend(fontsize=7)
plot_placeholder = st.pyplot(fig)

# -------------------------------
# Metrics & Reference
# -------------------------------
info_expander = st.expander("Model Information & Metrics", expanded=True)
metrics_text = info_expander.empty()

# Reference optimal controller (inside metrics)
if mode == "PID":
    ref_text = "**Reference optimal PID (PSO):**\nKp = 58.93, Ki = 3.91, Kd = 2.66"
else:
    ref_text = "**Reference optimal FOPID (PSO):**\nKp = 58.93, Ki = 3.91, Kd = 2.66, λ = 0.67, μ = 1.47"

metrics_text.markdown(ref_text)

# -------------------------------
# Simulation loop
# -------------------------------
if st.session_state['running']:
    for i in range(len(t_full)):
        if st.session_state['paused']:
            break  # freeze loop, line stays
        if i < len(st.session_state['t_data']):
            continue  # skip already simulated points

        st.session_state['t_data'].append(t_full[i])
        st.session_state['y_data'].append(Tc_full[i])
        st.session_state['pwm_data'].append(pwm_full[i])

        # Update line
        line.set_data(st.session_state['t_data'], st.session_state['y_data'])
        ax.set_xlim(0, duration)
        plot_placeholder.pyplot(fig)

        # Update metrics (compact)
        elapsed = st.session_state['t_data'][-1]
        pwm_now = st.session_state['pwm_data'][-1]
        error = np.array(st.session_state['y_data']) - T_set
        ss_error = np.mean(error[-50:]) if len(error) > 50 else np.mean(error)
        rmse = np.sqrt(np.mean(error**2))
        settling_time = next((st.session_state['t_data'][j] for j in range(len(st.session_state['y_data'])) if np.all(np.abs(error[j:]) <= 0.5)), None)

        # Recommendations based on parameters
        recs = []
        if mode in ["PID", "FOPID"]:
            if Kp < 50:
                recs.append("Kp low → response may be slow, increase Kp for faster settling.")
            if Ki < 2:
                recs.append("Ki low → may have steady-state error, increase Ki to reduce error.")
            if Kd > 5:
                recs.append("Kd high → may cause overshoot, reduce Kd for smoother response.")
            if mode == "FOPID":
                if lam < 0.5:
                    recs.append("λ low → integral effect weak, may increase steady-state error.")
                if lam > 1.5:
                    recs.append("λ high → may cause overshoot.")
                if mu < 0.5:
                    recs.append("μ low → derivative effect weak, slower damping.")
                if mu > 1.5:
                    recs.append("μ high → may cause aggressive control.")
        else:
            recs.append("Hysteresis control → adjust dT1/dT2 for tighter/looser band.")

        metrics_text.markdown(
            f"**Time elapsed:** {int(elapsed)} s  \n"
            f"**PWM:** {pwm_now:.1f}  \n"
            f"**Steady-state error:** {ss_error:.3f} °C  \n"
            f"**RMSE:** {rmse:.3f}  \n"
            f"**Settling time:** {settling_time if settling_time else 'Not reached'} s  \n"
            + ("\n".join(f"- {r}" for r in recs))
        )

        time.sleep(0.25)  # ~4 FPS
