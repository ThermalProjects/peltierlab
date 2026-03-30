# app_real_time_compact.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

from peltierlab.core.simulator import Simulator
from peltierlab.core.simulator_hysteresis_real import SimulatorHysteresisReal

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="PeltierLab Simulator", layout="wide")

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
# Session state
# -------------------------------
if "running" not in st.session_state:
    st.session_state.running = False
if "t_index" not in st.session_state:
    st.session_state.t_index = 0

# -------------------------------
# Title
# -------------------------------
st.title("❄️ PeltierLab Real-Time Simulator")
st.markdown("Simulate thermoelectric control with PID, FOPID, and Hysteresis in real-time (1:1).")

# -------------------------------
# Sidebar (compact)
# -------------------------------
with st.sidebar:
    st.header("Settings")
    mode = st.selectbox("Control mode", ["PID", "FOPID", "Hysteresis"])
    duration = st.slider("Duration [s]", 50, 500, 300, step=10)

    if mode in ["PID", "FOPID"]:
        st.subheader("Control Params")
        T_set = st.slider("Setpoint [°C]", 5.0, 18.0, 12.0, 0.1)
        bias = st.slider("Bias [°C]", -2.0, 2.0, 0.0, 0.1)
        Kp = st.slider("Kp", 0, 200, int(Kp_default), 1)
        Ki = st.slider("Ki", 0.0, 50.0, Ki_default, 0.1)
        Kd = st.slider("Kd", 0.0, 50.0, Kd_default, 0.1)
        if mode == "FOPID":
            st.subheader("Fractional Params")
            lam = st.slider("Lambda (λ)", 0.1, 2.0, lambda_default, 0.01)
            mu = st.slider("Mu (μ)", 0.1, 2.0, mu_default, 0.01)
    else:
        st.subheader("Hysteresis Control")
        T_set = st.slider("Setpoint [°C]", 10.0, 18.0, 12.0, 0.1)
        dT1 = st.slider("Upper band dT1 [°C]", 0.1, 1.0, 0.5, 0.1)
        dT2 = st.slider("Lower band dT2 [°C]", 0.1, 1.0, 0.5, 0.1)

    # Start/Stop button
    def toggle_run():
        st.session_state.running = not st.session_state.running
        if not st.session_state.running:
            st.session_state.t_index = 0

    st.button("Start/Stop", on_click=toggle_run)
    st.markdown("**Time elapsed:**")
    time_placeholder = st.empty()

# -------------------------------
# Prepare simulation
# -------------------------------
t_sim = np.arange(0, duration + 1)
if mode in ["PID", "FOPID"]:
    sim = Simulator(best_params, T_start=T_start)
    Tc_full, _ = sim.simulate_3nodes_FOPID(
        t_custom=t_sim,
        T_set=T_set,
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        bias=bias,
        lam=lam if mode=="FOPID" else lambda_default,
        mu=mu if mode=="FOPID" else mu_default
    )
else:
    sim = SimulatorHysteresisReal(best_params, T_start=T_start)
    Tc_full, _, _, _ = sim.simulate(
        t_custom=t_sim,
        T_set=T_set,
        dT1=dT1,
        dT2=dT2,
        P_max=5.0
    )

# -------------------------------
# Placeholder plot
# -------------------------------
plot_placeholder = st.empty()

fig, ax = plt.subplots(figsize=(7,3))  # más compacto
ax.set_xlim(0, duration)
ax.set_ylim(0, 20)
ax.set_xlabel("Time [s]", fontsize=10)
ax.set_ylabel("Temperature [°C]", fontsize=10)
ax.set_title(f"{mode} Control Simulation", fontsize=12)
ax.axhline(T_set, color="red", linestyle="--", label="Setpoint")
ax.grid(True)
line, = ax.plot([], [], color="blue", linewidth=2, label="Temperature")
ax.legend(fontsize=9)
plot_placeholder.pyplot(fig)

# -------------------------------
# Real-time loop
# -------------------------------
if st.session_state.running:
    temps = []
    for i in range(st.session_state.t_index, len(t_sim)):
        if not st.session_state.running:
            break
        temps.append(Tc_full[i])
        line.set_data(t_sim[:i+1], Tc_full[:i+1])
        plot_placeholder.pyplot(fig)

        # Update time below button
        time_placeholder.markdown(f"**Time elapsed:** {i} s / {duration} s")

        st.session_state.t_index = i + 1
        time.sleep(1)

# -------------------------------
# Metrics debajo de la gráfica
# -------------------------------
error = Tc_full[:st.session_state.t_index] - T_set
if len(error) > 0:
    ss_error = np.mean(error[-50:]) if len(error) >= 50 else np.mean(error)
    rmse = np.sqrt(np.mean(error**2))
    settling_time = next((t_sim[i] for i in range(len(error)) if np.all(np.abs(error[i:]) <= 0.5)), None)
else:
    ss_error = rmse = settling_time = None

with st.expander("Metrics & Quick Analysis", expanded=True):
    st.write(f"Control mode: {mode}")
    st.write(f"Setpoint: {T_set:.2f} °C")
    st.write(f"Settling time: {settling_time:.2f} s" if settling_time else "Settling time: Not reached")
    st.write(f"Steady-state error: {ss_error:.3f} °C" if ss_error is not None else "Steady-state error: N/A")
    st.write(f"RMSE: {rmse:.3f}" if rmse is not None else "RMSE: N/A")
