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
    layout="centered"
)

# -------------------------------
# Global parameters
# -------------------------------
best_params = [1.9507, 2.4906, 36.4772, 0.4806, 14.0687, 2.2298, 66.5757, 11.8439]
T_start_default = 19.0

pid_ref = [58.93, 3.91, 2.66]
fopid_ref = [58.93, 3.91, 2.66, 0.67, 1.47]

# -------------------------------
# TITLE
# -------------------------------
st.markdown("### PeltierLab Thermal Simulator")
st.markdown("*Interactive PID / FOPID / Hysteresis control*")

# -------------------------------
# CONTROLS
# -------------------------------
st.markdown("**Settings:**")

col1, col2, col3, col4 = st.columns([1,1,1,2])
with col1:
    mode = st.selectbox("Control mode", ["PID", "FOPID", "Hysteresis"])
with col2:
    start_btn = st.button("Start")
with col3:
    pause_btn = st.button("Pause")
with col4:
    stop_btn = st.button("Stop")

# Parameters sliders
st.markdown("**Parameters:**")
T_start = st.number_input("Ambient Temperature [°C]", min_value=0.0, max_value=40.0, value=T_start_default, step=0.1)

if mode in ["PID", "FOPID"]:
    T_set = st.slider("Setpoint [°C]", 5.0, 18.0, 12.0, 0.1)
    bias = st.slider("Bias [°C]", -2.0, 2.0, 0.0, 0.1)
    Kp = st.slider("Kp", 0, 200, 80, 1)
    Ki = st.slider("Ki", 0.0, 50.0, 3.91, 0.1)
    Kd = st.slider("Kd", 0.0, 50.0, 2.66, 0.1)
    if mode == "FOPID":
        lam = st.slider("Lambda (λ)", 0.1, 2.0, 0.67, 0.01)
        mu = st.slider("Mu (μ)", 0.1, 2.0, 1.47, 0.01)
elif mode == "Hysteresis":
    T_set = st.slider("Setpoint [°C]", 10.0, 18.0, 12.0, 0.1)
    dT1 = st.slider("Upper band dT1 [°C]", 0.1, 1.0, 0.5, 0.1)
    dT2 = st.slider("Lower band dT2 [°C]", 0.1, 1.0, 0.5, 0.1)

# -------------------------------
# SESSION STATE INIT
# -------------------------------
if 'running' not in st.session_state:
    st.session_state.running = False
if 'paused' not in st.session_state:
    st.session_state.paused = False
if 't_data' not in st.session_state:
    st.session_state.t_data = []
if 'y_data' not in st.session_state:
    st.session_state.y_data = []
if 'pwm_data' not in st.session_state:
    st.session_state.pwm_data = []
if 'sim_index' not in st.session_state:
    st.session_state.sim_index = 0

# Handle buttons
if start_btn:
    st.session_state.running = True
    st.session_state.paused = False
if pause_btn:
    st.session_state.paused = True
    st.session_state.running = False
if stop_btn:
    st.session_state.running = False
    st.session_state.paused = False
    st.session_state.t_data = []
    st.session_state.y_data = []
    st.session_state.pwm_data = []
    st.session_state.sim_index = 0

# -------------------------------
# SIMULATION PREP
# -------------------------------
duration = 300
t_full = np.linspace(0, duration, duration+1)

if mode in ["PID", "FOPID"]:
    sim = Simulator(best_params, T_start=T_start)
    Tc_full, pwm_full = sim.simulate_3nodes_FOPID(
        t_custom=t_full,
        T_set=T_set, Kp=Kp, Ki=Ki, Kd=Kd,
        bias=bias,
        lam=lam if mode=="FOPID" else 0.67,
        mu=mu if mode=="FOPID" else 1.47
    )
elif mode=="Hysteresis":
    sim = SimulatorHysteresisReal(best_params, T_start=T_start)
    Tc_full, _, _, pwm_full = sim.simulate(
        t_custom=t_full, T_set=T_set, dT1=dT1, dT2=dT2, P_max=5.0
    )

# -------------------------------
# PLOT
# -------------------------------
st.markdown("**Simulation:**")
fig, ax = plt.subplots(figsize=(6,3))
line, = ax.plot(st.session_state.t_data, st.session_state.y_data, lw=1.5, color='blue')
ax.axhline(T_set, color='red', linestyle='--')
ax.set_xlim(0, duration)
ax.set_ylim(0, 20)
ax.set_xlabel("Time [s]", fontsize=8)
ax.set_ylabel("Temperature [°C]", fontsize=8)
ax.grid(True, lw=0.5)
plot_placeholder = st.pyplot(fig, clear_figure=True)

# -------------------------------
# METRICS & REFERENCE
# -------------------------------
st.markdown("**Metrics:**")
if st.session_state.y_data:
    y_arr = np.array(st.session_state.y_data)
    error = y_arr - T_set
    ss_error = np.mean(error[-50:]) if len(error)>50 else np.mean(error)
    rmse = np.sqrt(np.mean(error**2))
    settling_time = next((st.session_state.t_data[j] for j in range(len(y_arr)) if np.all(np.abs(error[j:])<=0.5)), None)
else:
    ss_error = 0
    rmse = 0
    settling_time = None

st.markdown(f"**Time elapsed:** {int(st.session_state.t_data[-1]) if st.session_state.t_data else 0} s")
st.markdown(f"**Steady-state error:** {ss_error:.3f} °C")
st.markdown(f"**RMSE:** {rmse:.3f}")
st.markdown(f"**Settling time:** {settling_time if settling_time else 'Not reached'} s")

# Reference optimal
if mode=="PID":
    ref_txt = f"**Reference optimal PID (PSO):**\nKp = {pid_ref[0]}, Ki = {pid_ref[1]}, Kd = {pid_ref[2]}"
else:
    ref_txt = f"**Reference optimal FOPID (PSO):**\nKp = {fopid_ref[0]}, Ki = {fopid_ref[1]}, Kd = {fopid_ref[2]}, λ = {fopid_ref[3]}, μ = {fopid_ref[4]}"
st.markdown(ref_txt)

# -------------------------------
# SIMULATION LOOP
# -------------------------------
if st.session_state.running and st.session_state.sim_index < len(t_full):
    for i in range(st.session_state.sim_index, len(t_full)):
        if st.session_state.paused:
            break
        st.session_state.t_data.append(t_full[i])
        st.session_state.y_data.append(Tc_full[i])
        st.session_state.pwm_data.append(pwm_full[i])
        st.session_state.sim_index += 1

        # Update plot
        line.set_data(st.session_state.t_data, st.session_state.y_data)
        plot_placeholder.pyplot(fig, clear_figure=True)
