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

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("Settings")

# 🔥 NEW: Dynamic / Static
sim_type = st.sidebar.radio(
    "Simulation mode",
    ["Dynamic", "Static"]
        horizontal=True
)

mode = st.sidebar.selectbox("Control mode", ["PID", "FOPID", "Hysteresis"])
duration = st.sidebar.slider("Simulation duration [s]", 100, 500, 300, step=10)

# -------------------------------
# Start / Stop ONLY for Dynamic
# -------------------------------
if 'running_state' not in st.session_state:
    st.session_state['running_state'] = False

if sim_type == "Dynamic":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Start"):
            st.session_state['running_state'] = True
    with col2:
        if st.button("Stop"):
            st.session_state['running_state'] = False
else:
    st.sidebar.info("Static mode: updates instantly")

running = st.session_state['running_state']

# -------------------------------
# Placeholders
# -------------------------------
elapsed_placeholder = st.sidebar.empty()
pwm_placeholder = st.sidebar.empty()
pwm_bar = st.sidebar.empty()

# -------------------------------
# Controls
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
# Prepare simulation
# -------------------------------
t_full = np.linspace(0, duration, duration + 1)

if mode in ["PID", "FOPID"]:
    sim = Simulator(best_params, T_start=T_start)
else:
    sim = SimulatorHysteresisReal(best_params, T_start=T_start)

# -------------------------------
# STATIC MODE 🚀
# -------------------------------
if sim_type == "Static":

    if mode == "PID":
        Tc_full, pwm_full = sim.simulate_3nodes_FOPID(
            t_custom=t_full, T_set=T_set,
            Kp=Kp, Ki=Ki, Kd=Kd,
            bias=bias,
            lam=lambda_default, mu=mu_default
        )

    elif mode == "FOPID":
        Tc_full, pwm_full = sim.simulate_3nodes_FOPID(
            t_custom=t_full, T_set=T_set,
            Kp=Kp, Ki=Ki, Kd=Kd,
            bias=bias,
            lam=lam, mu=mu
        )

    else:
        Tc_full, _, _, pwm_full = sim.simulate(
            t_custom=t_full,
            T_set=T_set,
            dT1=dT1, dT2=dT2,
            P_max=5.0
        )

    # Plot completo
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_full, Tc_full, lw=2, label="Temperature")
    ax.axhline(T_set, color="red", linestyle="--", label="Setpoint")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title(f"{mode} - Static Simulation")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)

# -------------------------------
# DYNAMIC MODE ⏱️
# -------------------------------
else:

    st.subheader(f"{mode} - Dynamic Simulation")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    line, = ax.plot([], [], lw=1.5, label="Temperature")
    ax.axhline(T_set, color="red", linestyle="--", label="Setpoint")
    ax.set_xlim(0, duration)
    ax.set_ylim(0, 20)
    ax.grid(True)
    ax.legend()
    plot_placeholder = st.pyplot(fig)

    if running:

        # Run simulation first
        if mode == "PID":
            Tc_full, pwm_full = sim.simulate_3nodes_FOPID(
                t_custom=t_full, T_set=T_set,
                Kp=Kp, Ki=Ki, Kd=Kd,
                bias=bias,
                lam=lambda_default, mu=mu_default
            )

        elif mode == "FOPID":
            Tc_full, pwm_full = sim.simulate_3nodes_FOPID(
                t_custom=t_full, T_set=T_set,
                Kp=Kp, Ki=Ki, Kd=Kd,
                bias=bias,
                lam=lam, mu=mu
            )

        else:
            Tc_full, _, _, pwm_full = sim.simulate(
                t_custom=t_full,
                T_set=T_set,
                dT1=dT1, dT2=dT2,
                P_max=5.0
            )

        y_data = []
        t_data = []
        start_time = time.time()

        for i in range(len(t_full)):

            if not st.session_state['running_state']:
                break

            if time.time() - start_time < t_full[i]:
                time.sleep(t_full[i] - (time.time() - start_time))

            y_data.append(Tc_full[i])
            t_data.append(t_full[i])

            line.set_data(t_data, y_data)
            plot_placeholder.pyplot(fig)

            elapsed_placeholder.markdown(f"**Time:** {int(t_full[i])} s")
            pwm_placeholder.markdown(f"**PWM:** {pwm_full[i]:.1f}")
            pwm_bar.progress(int(pwm_full[i] / 255 * 100))
