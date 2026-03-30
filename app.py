# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time

from peltierlab.core.simulator import Simulator
from peltierlab.core.simulator_hysteresis_real import SimulatorHysteresisReal

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="PeltierLab Simulator", layout="wide")

# -------------------------------
# Mini title in sidebar
# -------------------------------
st.sidebar.markdown("### ❄️ PeltierLab Simulator")
st.sidebar.markdown(
    "<span style='font-size:10px'>Interactive simulation of thermoelectric systems using PID, FOPID, and Hysteresis control.</span>",
    unsafe_allow_html=True
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
# Sidebar settings
# -------------------------------
st.sidebar.header("Settings")
mode = st.sidebar.selectbox("Control mode", ["PID", "FOPID", "Hysteresis"], index=0)
duration = st.sidebar.slider("Simulation duration [s]", 100, 500, 300, step=10)

# Compact sliders
if mode in ["PID", "FOPID"]:
    st.sidebar.subheader("Control Parameters")
    T_set = st.sidebar.slider("Setpoint [°C]", 5.0, 20.0, 12.0, step=0.1)
    bias = st.sidebar.slider("Bias [°C]", -2.0, 2.0, 0.0, step=0.1)
    Kp = st.sidebar.slider("Kp", 0, 200, int(Kp_default), step=1)
    Ki = st.sidebar.slider("Ki", 0.0, 50.0, Ki_default, step=0.1)
    Kd = st.sidebar.slider("Kd", 0.0, 50.0, Kd_default, step=0.1)
    if mode == "FOPID":
        st.sidebar.subheader("Fractional Parameters")
        lam = st.sidebar.slider("Lambda (λ)", 0.1, 2.0, lambda_default, step=0.01)
        mu = st.sidebar.slider("Mu (μ)", 0.1, 2.0, mu_default, step=0.01)
elif mode == "Hysteresis":
    st.sidebar.subheader("ON/OFF Control")
    T_set = st.sidebar.slider("Setpoint [°C]", 5.0, 20.0, 12.0, step=0.1)
    dT1 = st.sidebar.slider("Upper band (dT1) [°C]", 0.1, 1.0, 0.5, step=0.1)
    dT2 = st.sidebar.slider("Lower band (dT2) [°C]", 0.1, 1.0, 0.5, step=0.1)

# -------------------------------
# Start/Stop button & elapsed
# -------------------------------
if "running" not in st.session_state:
    st.session_state.running = False
if "start_time" not in st.session_state:
    st.session_state.start_time = None

start_stop_col = st.sidebar.container()
start_button = start_stop_col.button("Start" if not st.session_state.running else "Stop")
time_placeholder = start_stop_col.empty()

if start_button:
    st.session_state.running = not st.session_state.running
    if st.session_state.running:
        st.session_state.start_time = time.time()
    else:
        st.session_state.start_time = None

# -------------------------------
# Simulation initialization
# -------------------------------
if mode in ["PID", "FOPID"]:
    sim = Simulator(best_params, T_start=T_start)
elif mode == "Hysteresis":
    sim = SimulatorHysteresisReal(best_params, T_start=T_start)

# Placeholders
sim_placeholder = st.empty()
metrics_placeholder = st.empty()
reco_placeholder = st.empty()

# -------------------------------
# Figure with 2/3 Temp + 1/3 PWM
# -------------------------------
fig = plt.figure(figsize=(6,4))
gs = GridSpec(3,1, figure=fig)
ax_temp = fig.add_subplot(gs[:2,0])
ax_pwm = fig.add_subplot(gs[2,0])

# Temperature plot
ax_temp.set_xlim(0, duration)
ax_temp.set_ylim(5, 20)
ax_temp.set_xlabel("Time [s]", fontsize=8)
ax_temp.set_ylabel("Temp [°C]", fontsize=8)
ax_temp.tick_params(axis='both', labelsize=7)
ax_temp.grid(True, linestyle='--', alpha=0.5)
ax_temp.axhline(T_set, color="red", linestyle="--", label="Setpoint")
line_temp, = ax_temp.plot([], [], color="blue", linewidth=1.2, label="Temperature")
ax_temp.legend(fontsize=7)

# PWM plot
ax_pwm.set_xlim(0, duration)
ax_pwm.set_ylim(0, 255)
ax_pwm.set_xlabel("Time [s]", fontsize=7)
ax_pwm.set_ylabel("PWM", fontsize=7)
ax_pwm.tick_params(axis='both', labelsize=6)
ax_pwm.grid(True, linestyle='--', alpha=0.3)
line_pwm, = ax_pwm.plot([], [], color="green", linewidth=1.2, label="PWM")

# Render initial empty plot
sim_placeholder.pyplot(fig)

# -------------------------------
# Real-time simulation loop (solo si Start)
# -------------------------------
if st.session_state.running:
    t_vals, y_vals, pwm_vals = [], [], []
    fps = 4
    dt = 1/fps
    total_steps = int(duration/dt)

    for step in range(total_steps):
        if not st.session_state.running:
            break

        t_current = step*dt

        # Compute temp & pwm
        if mode == "PID":
            Tc, pwm = sim.simulate_3nodes_FOPID(np.array([0,t_current]), T_set=T_set, Kp=Kp, Ki=Ki, Kd=Kd,
                                                bias=bias, lam=lambda_default, mu=mu_default)
            y_new = Tc[-1]
            pwm_new = pwm[-1]
        elif mode == "FOPID":
            Tc, pwm = sim.simulate_3nodes_FOPID(np.array([0,t_current]), T_set=T_set, Kp=Kp, Ki=Ki, Kd=Kd,
                                                bias=bias, lam=lam, mu=mu)
            y_new = Tc[-1]
            pwm_new = pwm[-1]
        else:
            Tc, Tm, Th, pwm = sim.simulate(np.array([0,t_current]), T_set=T_set, dT1=dT1, dT2=dT2, P_max=5.0)
            y_new = Tc[-1]
            pwm_new = pwm[-1]

        t_vals.append(t_current)
        y_vals.append(y_new)
        pwm_vals.append(pwm_new)

        # Update plots
        line_temp.set_data(t_vals, y_vals)
        line_pwm.set_data(t_vals, pwm_vals)
        sim_placeholder.pyplot(fig)

        # -------------------------------
        # Update metrics (real-time)
        error = np.array(y_vals)-T_set
        ss_error = np.mean(error[-min(len(error), int(fps*10)):])
        rmse = np.sqrt(np.mean(error**2))
        settling_time = next((t_vals[i] for i in range(len(y_vals)) if np.all(np.abs(error[i:]) <= 0.5)), None)

        metrics_placeholder.markdown(
            f"**Metrics (real-time):**\n- Settling time: {settling_time if settling_time else 'Not reached'} s\n"
            f"- Steady-state error: {ss_error:.2f} °C\n- RMSE: {rmse:.2f} °C"
        )

        # -------------------------------
        # Recommendations
        reco_text = ""
        if mode in ["PID","FOPID"]:
            if Kp>100: reco_text+="- Kp alto → overshoot, reducir Kp\n"
            if Kp<20: reco_text+="- Kp bajo → respuesta lenta, aumentar Kp\n"
            if Ki>25: reco_text+="- Ki alto → puede generar oscilaciones\n"
            if Ki<2: reco_text+="- Ki bajo → error persistente\n"
            if mode=="FOPID":
                if lam>1.2: reco_text+="- λ alto → mayor precisión\n"
                if lam<0.5: reco_text+="- λ bajo → respuesta más lenta\n"
                if mu>1.5: reco_text+="- μ alto → posible overshoot\n"
                if mu<0.5: reco_text+="- μ bajo → respuesta lenta\n"
        reco_placeholder.markdown("**Recommendations:**\n"+reco_text)

        # -------------------------------
        # Update elapsed time
        if st.session_state.start_time:
            elapsed = time.time()-st.session_state.start_time
            time_placeholder.markdown(f"Time elapsed: {int(elapsed)} s")

        # Real-time frame
        time.sleep(dt)
