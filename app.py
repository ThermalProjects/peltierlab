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
# SIDEBAR (controls)
# -------------------------------
st.sidebar.header("Settings")
mode = st.sidebar.selectbox("Control mode", ["PID", "FOPID", "Hysteresis"])
duration = st.sidebar.slider("Simulation duration [s]", 100, 500, 300, step=10)

# Control Parameters
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
        dT2 = st.slider("Lower band (dT2) [°C]", 0.1, 1.0, 0.5, 0.1)

# -------------------------------
# Session state initialization
# -------------------------------
if "running_state" not in st.session_state:
    st.session_state.running_state = False
if "current_index" not in st.session_state:
    st.session_state.current_index = 0

# -------------------------------
# Buttons Start/Pause & Stop in same row
# -------------------------------
col1, col2 = st.sidebar.columns([1, 1])
with col1:
    start_pause = st.button("▶️ Start/Pause")
with col2:
    stop = st.button("⏹ Stop")

# -------------------------------
# Stop resets simulation
# -------------------------------
if stop:
    st.session_state.running_state = False
    st.session_state.current_index = 0

# -------------------------------
# Prepare simulation arrays only once
# -------------------------------
if "t_full" not in st.session_state:
    t_full = np.linspace(0, duration, duration + 1)
    if mode in ["PID", "FOPID"]:
        sim = Simulator(best_params, T_start=T_start)
        if mode == "PID":
            Tc_full, pwm_full = sim.simulate_3nodes_FOPID(
                t_custom=t_full, T_set=T_set,
                Kp=Kp, Ki=Ki, Kd=Kd,
                bias=bias, lam=lambda_default, mu=mu_default
            )
        else:
            Tc_full, pwm_full = sim.simulate_3nodes_FOPID(
                t_custom=t_full, T_set=T_set,
                Kp=Kp, Ki=Ki, Kd=Kd,
                bias=bias, lam=lam, mu=mu
            )
        st.session_state["Tc_full"] = Tc_full
        st.session_state["pwm_full"] = pwm_full
    else:  # Hysteresis
        sim = SimulatorHysteresisReal(best_params, T_start=T_start)
        Tc_full, Tm_full, Th_full, pwm_full = sim.simulate(
            t_custom=t_full, T_set=T_set, dT1=dT1, dT2=dT2, P_max=5.0
        )
        st.session_state["Tc_full"] = Tc_full
        st.session_state["pwm_full"] = pwm_full
    st.session_state["t_full"] = t_full

# -------------------------------
# Toggle running state
# -------------------------------
if start_pause:
    st.session_state.running_state = not st.session_state.running_state

# -------------------------------
# Plot placeholders
# -------------------------------
st.subheader(f"Results: {mode}")
fig, ax = plt.subplots(figsize=(10, 4))
line, = ax.plot([], [], lw=2, color="blue", label="Temperature")
ax.axhline(T_set, color="red", linestyle="--", label="Setpoint")
ax.set_xlim(0, duration)
ax.set_ylim(0, 20)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Temperature [°C]")
ax.grid(True)
ax.legend()
plot_placeholder = st.pyplot(fig)

# Metrics placeholders
elapsed_placeholder = st.empty()
pwm_placeholder = st.empty()
pwm_bar = st.empty()
metrics_text = st.empty()

# -------------------------------
# Simulation loop (non-blocking, step-by-step)
# -------------------------------
if st.session_state.running_state:
    idx = st.session_state.current_index
    if idx < len(st.session_state["t_full"]):
        # Update data up to current index
        t_data = st.session_state["t_full"][:idx+1]
        y_data = st.session_state["Tc_full"][:idx+1]
        pwm_data = st.session_state["pwm_full"][:idx+1]

        line.set_data(t_data, y_data)
        ax.set_xlim(0, duration)
        plot_placeholder.pyplot(fig)

        # Update metrics
        elapsed_placeholder.markdown(f"**Time elapsed:** {t_data[-1]:.0f} s")
        pwm_placeholder.markdown(f"**PWM:** {pwm_data[-1]:.1f}")
        pwm_bar.progress(int(pwm_data[-1]/255*100))

        error = np.array(y_data) - T_set
        ss_error = np.mean(error[-50:]) if len(error) > 50 else np.mean(error)
        rmse = np.sqrt(np.mean(error**2))
        settling_time = next((t_data[j] for j in range(len(y_data)) if np.all(np.abs(error[j:]) <= 0.5)), None)
        recs = []
        if abs(ss_error) > 0.5:
            recs.append("Consider tuning controller to reduce steady-state error.")
        if settling_time is None or settling_time > 150:
            recs.append("Slow response → consider increasing gains.")
        metrics_text.markdown(
            f"**Steady-state error:** {ss_error:.3f} °C  \n"
            f"**RMSE:** {rmse:.3f}  \n"
            f"**Settling time:** {settling_time if settling_time else 'Not reached'} s  \n"
            + ("\n".join(f"- {r}" for r in recs))
        )

        # Advance index
        st.session_state.current_index += 1
        # Auto-refresh page to animate
        st.experimental_rerun()
