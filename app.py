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
    page_title="PeltierLab Dashboard",
    layout="wide"
)

# -------------------------------
# Global parameters
# -------------------------------
best_params = [1.9507, 2.4906, 36.4772, 0.4806, 14.0687, 2.2298, 66.5757, 11.8439]
T_start = 19.0

# Default controller params (PSO reference)
PID_ref = {"Kp": 58.93, "Ki": 3.91, "Kd": 2.66}
FOPID_ref = {"Kp": 58.93, "Ki": 3.91, "Kd": 2.66, "lam": 0.67, "mu": 1.47}

# -------------------------------
# Grid layout: 5x4
# -------------------------------
grid = st.container()
cols = grid.columns([1,1,1,1,1])  # 5 columnas

# Title + Mode selector (fila 0)
with cols[0]:
    st.markdown("## PeltierLab Dashboard")
with cols[1]:
    mode = st.selectbox("Control Mode", ["PID", "FOPID", "Hysteresis"], index=1)

# -------------------------------
# Left controls panel (cols 0-1, filas 1-3)
# -------------------------------
control_cols = st.columns([1,1])
with control_cols[0]:
    st.markdown("### Controller Parameters")
    T_set = st.slider("Setpoint [°C]", 5.0, 18.0, 12.0, 0.1)
    T_amb = st.slider("Ambient Temp [°C]", 15.0, 30.0, 20.0, 0.1)
    Kp = st.number_input("Kp", value=PID_ref["Kp"] if mode=="PID" else FOPID_ref["Kp"])
    Ki = st.number_input("Ki", value=PID_ref["Ki"] if mode=="PID" else FOPID_ref["Ki"])
    Kd = st.number_input("Kd", value=PID_ref["Kd"] if mode=="PID" else FOPID_ref["Kd"])
    if mode=="FOPID":
        lam = st.number_input("Lambda (λ)", value=FOPID_ref["lam"])
        mu = st.number_input("Mu (μ)", value=FOPID_ref["mu"])

with control_cols[1]:
    st.markdown("### Simulation Control")
    col_btns = st.columns(3)
    if "running" not in st.session_state:
        st.session_state.running = False
        st.session_state.paused = False
    start = col_btns[0].button("Start")
    pause = col_btns[1].button("Pause")
    stop = col_btns[2].button("Stop")

    duration = st.number_input("Duration [s]", value=300, min_value=50, max_value=600, step=10)

# -------------------------------
# Handle simulation state
# -------------------------------
if start:
    st.session_state.running = True
    st.session_state.paused = False
if pause:
    st.session_state.paused = True
if stop:
    st.session_state.running = False
    st.session_state.paused = False
    st.session_state.t_data = []
    st.session_state.y_data = []

# -------------------------------
# Prepare simulation data
# -------------------------------
if "t_data" not in st.session_state:
    st.session_state.t_data = []
    st.session_state.y_data = []

sim = None
t_full = np.linspace(0, duration, duration + 1)

if mode in ["PID", "FOPID"]:
    sim = Simulator(best_params, T_start=T_start)
    if mode=="PID":
        Tc_full, pwm_full = sim.simulate_3nodes_FOPID(
            t_custom=t_full, T_set=T_set, Kp=Kp, Ki=Ki, Kd=Kd,
            bias=0.0, lam=1.0, mu=1.0
        )
    else:
        Tc_full, pwm_full = sim.simulate_3nodes_FOPID(
            t_custom=t_full, T_set=T_set, Kp=Kp, Ki=Ki, Kd=Kd,
            bias=0.0, lam=lam, mu=mu
        )
elif mode=="Hysteresis":
    sim = SimulatorHysteresisReal(best_params, T_start=T_start)
    Tc_full, Tm_full, Th_full, pwm_full = sim.simulate(
        t_custom=t_full, T_set=T_set, dT1=0.5, dT2=0.5, P_max=5.0
    )

# -------------------------------
# Center Graph (cols 2-3, filas 1-2)
# -------------------------------
graph_cols = st.columns([1,1])
with graph_cols[0]:
    st.markdown("### Temperature Response")
    fig, ax = plt.subplots(figsize=(6,4))
    ax.set_xlim(0, duration)
    ax.set_ylim(0, 25)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temp [°C]")
    ax.grid(True, lw=0.5)
    line, = ax.plot(st.session_state.t_data, st.session_state.y_data, color='blue', lw=2)
    ax.axhline(T_set, color='red', linestyle='--', label="Setpoint")
    ax.legend()
    st.pyplot(fig, clear_figure=False)

# -------------------------------
# Right metrics panel (cols 3-4)
# -------------------------------
metrics_cols = st.columns(1)
with metrics_cols[0]:
    st.markdown("### Metrics & Reference")
    if mode=="PID":
        st.markdown("**Reference optimal PID (PSO):**")
        st.text(f"Kp = {PID_ref['Kp']}, Ki = {PID_ref['Ki']}, Kd = {PID_ref['Kd']}")
    else:
        st.markdown("**Reference optimal FOPID (PSO):**")
        st.text(f"Kp = {FOPID_ref['Kp']}, Ki = {FOPID_ref['Ki']}, Kd = {FOPID_ref['Kd']}, λ = {FOPID_ref['lam']}, μ = {FOPID_ref['mu']}")

    # Real-time metrics
    if len(st.session_state.y_data)>0:
        error = np.array(st.session_state.y_data) - T_set
        rmse = np.sqrt(np.mean(error**2))
        ss_error = np.mean(error[-50:]) if len(error)>50 else np.mean(error)
        settling_time = next((t_full[i] for i in range(len(error)) if np.all(np.abs(error[i:])<=0.5)), None)
        st.text(f"Steady-state error: {ss_error:.3f} °C")
        st.text(f"RMSE: {rmse:.3f}")
        st.text(f"Settling time: {settling_time if settling_time else 'Not reached'} s")

# -------------------------------
# Update simulation if running
# -------------------------------
if st.session_state.running and not st.session_state.paused:
    for i in range(len(t_full)):
        st.session_state.t_data.append(t_full[i])
        st.session_state.y_data.append(Tc_full[i])
        time.sleep(0.01)
        st.experimental_rerun()  # update in real-time
