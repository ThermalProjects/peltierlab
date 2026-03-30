# app_real_time.py
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
st.title("❄️ PeltierLab Interactive Simulator (Real-Time)")
st.markdown(
    "Explore thermoelectric system behavior using PID, FOPID, and Hysteresis control strategies."
)

# -------------------------------
# SIDEBAR (controls)
# -------------------------------
st.sidebar.header("Settings")

mode = st.sidebar.selectbox("Control mode", ["PID", "FOPID", "Hysteresis"])
duration = st.sidebar.slider("Simulation duration [s]", 100, 500, 300, step=10)

# -------------------------------
# Dynamic sliders
# -------------------------------
if mode in ["PID", "FOPID"]:
    st.sidebar.subheader("Control Parameters")
    T_set = st.sidebar.slider("Setpoint [°C]", 5.0, 18.0, 12.0, 0.1)
    bias = st.sidebar.slider("Bias [°C]", -2.0, 2.0, 0.0, 0.1)
    Kp = st.sidebar.slider("Kp", 0, 200, int(Kp_default), 1)
    Ki = st.sidebar.slider("Ki", 0.0, 50.0, Ki_default, 0.1)
    Kd = st.sidebar.slider("Kd", 0.0, 50.0, Kd_default, 0.1)

    if mode == "FOPID":
        st.sidebar.subheader("Fractional Parameters")
        lam = st.sidebar.slider("Lambda (λ)", 0.1, 2.0, lambda_default, 0.01)
        mu = st.sidebar.slider("Mu (μ)", 0.1, 2.0, mu_default, 0.01)

elif mode == "Hysteresis":
    st.sidebar.subheader("ON/OFF Control")
    T_set = st.sidebar.slider("Setpoint [°C]", 10.0, 18.0, 12.0, 0.1)
    dT1 = st.sidebar.slider("Upper band (dT1) [°C]", 0.1, 1.0, 0.5, 0.1)
    dT2 = st.sidebar.slider("Lower band (dT2) [°C]", 0.1, 1.0, 0.5, 0.1)

# -------------------------------
# Start button
# -------------------------------
start_sim = st.sidebar.button("Start Simulation")

# Placeholder for real-time plot
plot_placeholder = st.empty()

if start_sim:
    st.subheader(f"Results: {mode} (Real-Time)")

    # Time vector
    t_new = np.linspace(0, duration, duration)
    
    # Initialize simulator
    if mode in ["PID", "FOPID"]:
        sim = Simulator(best_params, T_start=T_start)
        Tc_sim, pwm_sim = sim.simulate_3nodes_FOPID(
            t_custom=t_new,
            T_set=T_set,
            Kp=Kp,
            Ki=Ki,
            Kd=Kd,
            bias=bias,
            lam=lam if mode=="FOPID" else lambda_default,
            mu=mu if mode=="FOPID" else mu_default
        )
        y = Tc_sim
    else:
        sim = SimulatorHysteresisReal(best_params, T_start=T_start)
        Tc, Tm, Th, pwm = sim.simulate(
            t_custom=t_new,
            T_set=T_set,
            dT1=dT1,
            dT2=dT2,
            P_max=5.0
        )
        y = Tc

    # Real-time plotting
    fig, ax = plt.subplots(figsize=(8,4))
    ax.set_xlim(0, duration)
    ax.set_ylim(min(y)-1, max(y)+1)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title(f"{mode} Control Simulation")
    ax.axhline(T_set, color="red", linestyle="--", label="Setpoint")
    ax.grid(True)
    line, = ax.plot([], [], color="blue", linewidth=2, label="Temperature")
    ax.legend(fontsize=9)

    temp_data = []
    for i in range(len(y)):
        temp_data.append(y[i])
        line.set_data(t_new[:i+1], temp_data)
        plot_placeholder.pyplot(fig)
        time.sleep(0.02)  # pausa para animación (~50fps)
    
    # -------------------------------
    # Metrics after simulation
    # -------------------------------
    error = y - T_set
    settling_time = next((t_new[i] for i in range(len(y)) if np.all(np.abs(y[i:] - T_set) <= 0.5)), None)
    ss_error = np.mean(error[-50:])
    rmse = np.sqrt(np.mean(error**2))

    with st.expander("Model Information & Metrics", expanded=True):
        st.markdown("### Metrics")
        st.write(f"Settling time: {settling_time:.2f} s" if settling_time else "Settling time: Not reached")
        st.write(f"Steady-state error: {ss_error:.3f} °C")
        st.write(f"RMSE: {rmse:.3f}")
