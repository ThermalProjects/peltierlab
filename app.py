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
# Title
# -------------------------------
st.title("❄️ PeltierLab Interactive Simulator")
st.markdown("Explore thermoelectric system behavior using PID, FOPID, and Hysteresis control strategies.")

# -------------------------------
# Sidebar (compact)
# -------------------------------
st.sidebar.header("Settings")
st.sidebar.markdown("<style>div.row-widget.stSlider{margin-bottom: 0.5rem;}</style>", unsafe_allow_html=True)

mode = st.sidebar.selectbox("Control mode", ["PID", "FOPID", "Hysteresis"])
duration = st.sidebar.slider("Simulation duration [s]", 50, 500, 300, 10)

if mode in ["PID", "FOPID"]:
    st.sidebar.subheader("Control Parameters")
    T_set = st.sidebar.slider("Setpoint [°C]", 0.0, 20.0, 12.0, 0.1)
    bias = st.sidebar.slider("Bias [°C]", -2.0, 2.0, 0.0, 0.1)
    Kp = st.sidebar.slider("Kp", 0, 200, int(Kp_default), 1)
    Ki = st.sidebar.slider("Ki", 0.0, 50.0, Ki_default, 0.1)
    Kd = st.sidebar.slider("Kd", 0.0, 50.0, Kd_default, 0.1)
    if mode == "FOPID":
        st.sidebar.subheader("Fractional Parameters")
        lam = st.sidebar.slider("Lambda (λ)", 0.1, 2.0, lambda_default, 0.01)
        mu = st.sidebar.slider("Mu (μ)", 0.1, 2.0, mu_default, 0.01)
else:
    st.sidebar.subheader("ON/OFF Control")
    T_set = st.sidebar.slider("Setpoint [°C]", 0.0, 20.0, 12.0, 0.1)
    dT1 = st.sidebar.slider("Upper band (dT1) [°C]", 0.1, 1.0, 0.5, 0.1)
    dT2 = st.sidebar.slider("Lower band (dT2) [°C]", 0.1, 1.0, 0.5, 0.1)

# -------------------------------
# Start/Stop button & elapsed time
# -------------------------------
if 'running' not in st.session_state:
    st.session_state.running = False
    st.session_state.time_elapsed = 0

col1, col2 = st.columns([1,1])
with col1:
    start_stop_btn = st.button("Start" if not st.session_state.running else "Stop")
with col2:
    time_text = st.empty()

if start_stop_btn:
    st.session_state.running = not st.session_state.running
    if not st.session_state.running:
        st.session_state.time_elapsed = 0

# -------------------------------
# Simulation (precompute)
# -------------------------------
t_new = np.linspace(0, duration, duration*4 + 1)  # 4 fps
if mode in ["PID", "FOPID"]:
    sim = Simulator(best_params, T_start=T_start)
    Tc_sim, _ = sim.simulate_3nodes_FOPID(
        t_custom=t_new, T_set=T_set, Kp=Kp, Ki=Ki, Kd=Kd,
        bias=bias, lam=lambda_default if mode=="PID" else lam,
        mu=mu_default if mode=="PID" else mu
    )
    y_sim = Tc_sim
else:
    sim = SimulatorHysteresisReal(best_params, T_start=T_start)
    Tc, _, _, _ = sim.simulate(
        t_custom=t_new, T_set=T_set, dT1=dT1, dT2=dT2, P_max=5.0
    )
    y_sim = Tc

# -------------------------------
# Placeholder for matplotlib figure
# -------------------------------
fig_placeholder = st.empty()
fig, ax = plt.subplots(figsize=(8,4))
ax.set_xlim(0, duration)
ax.set_ylim(0, 20)
ax.set_xlabel("Time [s]", fontsize=10)
ax.set_ylabel("Temperature [°C]", fontsize=10)
ax.tick_params(axis='both', labelsize=8)
ax.grid(True, linestyle='--', alpha=0.5)
line, = ax.plot([], [], color='blue', linewidth=1.5, label="Temperature")
ax.axhline(T_set, color="red", linestyle="--", linewidth=1, label="Setpoint")
ax.legend(fontsize=8)
fig_placeholder.pyplot(fig)

# -------------------------------
# Real-time simulation loop
# -------------------------------
x_data, y_data = [], []

if st.session_state.running:
    for i in range(len(t_new)):
        if not st.session_state.running:
            break
        x_data.append(t_new[i])
        y_data.append(y_sim[i])
        line.set_data(x_data, y_data)
        ax.set_xlim(0, max(duration, t_new[i]+1))
        fig_placeholder.pyplot(fig)
        st.session_state.time_elapsed = t_new[i]
        time_text.text(f"Time elapsed: {t_new[i]:.1f} s")
        time.sleep(0.25)  # 4 frames per second

# -------------------------------
# Metrics & Recommendations
# -------------------------------
with st.expander("Model Information & Metrics", expanded=True):
    st.markdown("### Simulation Details")
    st.write(f"Control mode: {mode}")
    st.write(f"Setpoint: {T_set:.2f} °C")

    if y_data:
        error = np.array(y_data) - T_set
        ss_error = np.mean(error[-50:]) if len(error) >= 50 else None
        rmse = np.sqrt(np.mean(error**2)) if len(error) > 0 else None
        settling_time = next((x_data[i] for i in range(len(error)) if np.all(np.abs(error[i:]) <= 0.5)), None)
    else:
        ss_error = rmse = settling_time = None

    st.markdown("### Metrics")
    st.write(f"Settling time: {settling_time:.2f} s" if settling_time else "Settling time: Not reached")
    st.write(f"Steady-state error: {ss_error:.3f} °C" if ss_error is not None else "Steady-state error: N/A")
    st.write(f"RMSE: {rmse:.3f}" if rmse is not None else "RMSE: N/A")

    st.markdown("### Quick Recommendations")
    if ss_error is not None and abs(ss_error) > 0.5:
        st.warning("High steady-state error → consider adjusting Ki (or λ for FOPID).")
    if settling_time is not None and settling_time > duration/2:
        st.info("Slow settling time → consider increasing Kp or adjusting Ki/Kd.")
    if rmse is not None and rmse < 1:
        st.success("Overall accuracy is good (low RMSE).")
