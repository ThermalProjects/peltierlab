# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

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
st.title("🔥 PeltierLab Interactive Simulator")
st.markdown(
    "Explore thermoelectric system behavior using **PID, FOPID, and Hysteresis control strategies**."
)

# -------------------------------
# SIDEBAR (controls)
# -------------------------------
st.sidebar.header("⚙️ Settings")

mode = st.sidebar.selectbox(
    "Control mode",
    ["PID", "FOPID", "Hysteresis"]
)

# -------------------------------
# Dynamic sliders
# -------------------------------
if mode in ["PID", "FOPID"]:
    st.sidebar.subheader("🎯 Control Parameters")

    T_set = st.sidebar.slider("Setpoint [°C]", 5.0, 25.0, 12.0, 0.1)
    bias = st.sidebar.slider("Bias [°C]", -2.0, 2.0, 0.0, 0.1)

    Kp = st.sidebar.slider("Kp", 0, 200, int(Kp_default), 1)
    Ki = st.sidebar.slider("Ki", 0.0, 50.0, Ki_default, 0.1)
    Kd = st.sidebar.slider("Kd", 0.0, 50.0, Kd_default, 0.1)

    if mode == "FOPID":
        st.sidebar.subheader("🔬 Fractional Parameters")
        lam = st.sidebar.slider("Lambda (λ)", 0.1, 2.0, lambda_default, 0.01)
        mu = st.sidebar.slider("Mu (μ)", 0.1, 2.0, mu_default, 0.01)

elif mode == "Hysteresis":
    st.sidebar.subheader("🎯 ON/OFF Control")

    T_set = st.sidebar.slider("Setpoint [°C]", 10.0, 20.0, 12.0, 0.1)
    dT1 = st.sidebar.slider("Upper band (dT1) [°C]", 0.1, 2.0, 0.5, 0.1)
    dT2 = st.sidebar.slider("Lower band (dT2) [°C]", 0.1, 2.0, 0.5, 0.1)

# -------------------------------
# Simulation (auto-run)
# -------------------------------
st.subheader(f"📈 Results: {mode}")

# PID / FOPID
if mode in ["PID", "FOPID"]:
    t_new = np.linspace(0, 300, 500)
    sim = Simulator(best_params, T_start=T_start)

    if mode == "PID":
        Tc_sim, pwm_sim = sim.simulate_3nodes_FOPID(
            t_custom=t_new,
            T_set=T_set,
            Kp=Kp,
            Ki=Ki,
            Kd=Kd,
            bias=bias,
            lam=lambda_default,
            mu=mu_default
        )
    else:
        Tc_sim, pwm_sim = sim.simulate_3nodes_FOPID(
            t_custom=t_new,
            T_set=T_set,
            Kp=Kp,
            Ki=Ki,
            Kd=Kd,
            bias=bias,
            lam=lam,
            mu=mu
        )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_new, Tc_sim, linewidth=2, label="Temperature")
    ax.axhline(T_set, linestyle="--", label="Setpoint")

# Hysteresis
elif mode == "Hysteresis":
    t_new = np.linspace(0, 600, 1200)
    sim = SimulatorHysteresisReal(best_params, T_start=T_start)

    Tc, Tm, Th, pwm = sim.simulate(
        t_custom=t_new,
        T_set=T_set,
        dT1=dT1,
        dT2=dT2,
        P_max=5.0
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_new, Tc, linewidth=2, label="Temperature")
    ax.axhline(T_set, linestyle="--", label="Setpoint")

# -------------------------------
# Common formatting
# -------------------------------
ax.set_xlabel("Time [s]")
ax.set_ylabel("Temperature [°C]")
ax.set_title(f"{mode} Control Simulation")
ax.grid(True)
ax.legend()

st.pyplot(fig)

# -------------------------------
# Extra info panel
# -------------------------------
with st.expander("📊 Model Information"):
    st.write(f"**Control mode:** {mode}")
    st.write(f"**Setpoint:** {T_set} °C")
