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
# Title
# -------------------------------
st.title("❄️ PeltierLab Interactive Simulator")
st.markdown(
    "Explore thermoelectric system behavior using PID, FOPID, and Hysteresis control strategies."
)

# -------------------------------
# Sidebar controls
# -------------------------------
st.sidebar.header("Settings")
mode = st.sidebar.selectbox("Control mode", ["PID", "FOPID", "Hysteresis"])
duration = st.sidebar.slider("Simulation duration [s]", 100, 500, 300, step=10)

# Start/Pause & Stop buttons
col1, col2 = st.sidebar.columns(2)
start_pause = col1.button("Start / Pause")
stop_button = col2.button("Stop")

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

    else:  # Hysteresis
        T_set = st.slider("Setpoint [°C]", 10.0, 18.0, 12.0, 0.1)
        dT1 = st.slider("Upper band (dT1) [°C]", 0.1, 1.0, 0.5, 0.1)
        dT2 = st.slider("Lower band (dT2) [°C]", 0.1, 1.0, 0.5, 0.5, 0.1)

# -------------------------------
# Initialize session state
# -------------------------------
if 'sim_state' not in st.session_state or stop_button:
    st.session_state.sim_state = {
        'running': False,
        't_index': 0,
        't_full': np.linspace(0, duration, duration+1),
        'Tc_full': np.zeros(duration+1),
        'pwm_full': np.zeros(duration+1)
    }
    st.session_state.sim_state['running'] = False
    # Prepare simulation arrays
    if mode in ["PID", "FOPID"]:
        sim = Simulator(best_params, T_start=T_start)
        if mode == "PID":
            Tc_full, pwm_full = sim.simulate_3nodes_FOPID(
                t_custom=st.session_state.sim_state['t_full'],
                T_set=T_set, Kp=Kp, Ki=Ki, Kd=Kd,
                bias=bias, lam=lambda_default, mu=mu_default
            )
        else:
            Tc_full, pwm_full = sim.simulate_3nodes_FOPID(
                t_custom=st.session_state.sim_state['t_full'],
                T_set=T_set, Kp=Kp, Ki=Ki, Kd=Kd,
                bias=bias, lam=lam, mu=mu
            )
        st.session_state.sim_state['Tc_full'] = Tc_full
        st.session_state.sim_state['pwm_full'] = pwm_full
    else:  # Hysteresis
        sim = SimulatorHysteresisReal(best_params, T_start=T_start)
        Tc_full, Tm_full, Th_full, pwm_full = sim.simulate(
            t_custom=st.session_state.sim_state['t_full'],
            T_set=T_set, dT1=dT1, dT2=dT2, P_max=5.0
        )
        st.session_state.sim_state['Tc_full'] = Tc_full
        st.session_state.sim_state['pwm_full'] = pwm_full

# -------------------------------
# Toggle running
# -------------------------------
if start_pause:
    st.session_state.sim_state['running'] = not st.session_state.sim_state['running']

# -------------------------------
# Plot placeholders
# -------------------------------
plot_placeholder = st.empty()
elapsed_placeholder = st.empty()
pwm_placeholder = st.empty()
pwm_bar = st.empty()

# -------------------------------
# Simulation step
# -------------------------------
sim_state = st.session_state.sim_state
if sim_state['running'] and sim_state['t_index'] < len(sim_state['t_full']):
    # Advance one step
    i = sim_state['t_index']
    sim_state['t_index'] += 1

# -------------------------------
# Draw plot
# -------------------------------
y_data = sim_state['Tc_full'][:sim_state['t_index']]
t_data = sim_state['t_full'][:sim_state['t_index']]

fig, ax = plt.subplots(figsize=(8,4))
ax.plot(t_data, y_data, color='blue', lw=2, label="Temperature")
ax.axhline(T_set, color='red', linestyle='--', label="Setpoint")
ax.set_xlim(0, duration)
ax.set_ylim(0, 20)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Temperature [°C]")
ax.grid(True, lw=0.5)
ax.legend()
plot_placeholder.pyplot(fig)

# -------------------------------
# Metrics & PWM
# -------------------------------
if len(y_data) > 0:
    elapsed_placeholder.markdown(f"**Time elapsed:** {int(t_data[-1])} s")
    pwm_placeholder.markdown(f"**PWM:** {sim_state['pwm_full'][sim_state['t_index']-1]:.1f}")
    pwm_bar.progress(int(sim_state['pwm_full'][sim_state['t_index']-1]/255*100))
