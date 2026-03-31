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
# Sidebar: Title + Settings
# -------------------------------
st.sidebar.title("❄️ PeltierLab Interactive Simulator")
st.sidebar.header("Settings")

mode = st.sidebar.selectbox("Control mode", ["PID", "FOPID", "Hysteresis"])
duration = st.sidebar.slider("Simulation duration [s]", 100, 500, 300, step=10)

# Dynamic / Static toggle (mutually exclusive)
col1, col2 = st.sidebar.columns(2)
if 'dynamic_mode' not in st.session_state:
    st.session_state.dynamic_mode = True
if 'static_mode' not in st.session_state:
    st.session_state.static_mode = False

dynamic_checked = col1.checkbox("Dynamic", value=st.session_state.dynamic_mode)
static_checked = col2.checkbox("Static", value=st.session_state.static_mode)

# Exclusivity
if dynamic_checked:
    st.session_state.dynamic_mode = True
    st.session_state.static_mode = False
if static_checked:
    st.session_state.static_mode = True
    st.session_state.dynamic_mode = False

start_button = st.sidebar.button("Start")
stop_button = st.sidebar.button("Stop")

# Time & PWM placeholders
elapsed_placeholder = st.sidebar.empty()
pwm_placeholder = st.sidebar.empty()
pwm_bar = st.sidebar.empty()

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
        dT2 = st.slider("Lower band (dT2) [°C]", 0.1, 1.0, 0.5, 0.1)

# -------------------------------
# Initialize session state
# -------------------------------
if 'running' not in st.session_state:
    st.session_state.running = False
if 'idx' not in st.session_state:
    st.session_state.idx = 0

# Start / Stop buttons
if start_button:
    st.session_state.running = True
    st.session_state.idx = 0  # reset to start
if stop_button:
    st.session_state.running = False

# -------------------------------
# Prepare simulation
# -------------------------------
if mode in ["PID", "FOPID"]:
    sim = Simulator(best_params, T_start=T_start)
    t_full = np.linspace(0, duration, duration + 1)
    if mode == "PID":
        Tc_full, pwm_full = sim.simulate_3nodes_FOPID(
            t_custom=t_full, T_set=T_set, Kp=Kp, Ki=Ki, Kd=Kd,
            bias=bias, lam=lambda_default, mu=mu_default
        )
    else:
        Tc_full, pwm_full = sim.simulate_3nodes_FOPID(
            t_custom=t_full, T_set=T_set, Kp=Kp, Ki=Ki, Kd=Kd,
            bias=bias, lam=lam, mu=mu
        )
elif mode == "Hysteresis":
    sim = SimulatorHysteresisReal(best_params, T_start=T_start)
    t_full = np.linspace(0, duration, duration + 1)
    Tc_full, Tm_full, Th_full, pwm_full = sim.simulate(
        t_custom=t_full, T_set=T_set, dT1=dT1, dT2=dT2, P_max=5.0
    )

# -------------------------------
# Plot setup
# -------------------------------
fig, ax = plt.subplots(figsize=(7,3.5))
line, = ax.plot([], [], lw=1.5, color='blue', label="Temperature")
ax.axhline(T_set, color="red", linestyle="--", label="Setpoint")
ax.set_xlim(0, duration)
ax.set_ylim(0, 20)
ax.set_xlabel("Time [s]", fontsize=8)
ax.set_ylabel("Temperature [°C]", fontsize=8)
ax.tick_params(axis='both', labelsize=7)
ax.grid(True, lw=0.5)
ax.legend(fontsize=7)
plot_placeholder = st.pyplot(fig)

# -------------------------------
# Metrics & recommendations
# -------------------------------
metrics_expander = st.expander("Metrics & Recommendations", expanded=True)
metrics_placeholder = metrics_expander.empty()

# -------------------------------
# Simulation update
# -------------------------------
def update_dynamic():
    if st.session_state.running and st.session_state.idx < len(t_full)-1:
        st.session_state.idx += 1

    y_data = Tc_full[:st.session_state.idx+1]
    t_data = t_full[:st.session_state.idx+1]

    line.set_data(t_data, y_data)
    ax.set_xlim(0, duration)
    plot_placeholder.pyplot(fig)

    # Time & PWM
    elapsed_placeholder.markdown(f"**Time elapsed:** {int(t_data[-1])} s")
    pwm_placeholder.markdown(f"**PWM:** {pwm_full[st.session_state.idx]:.1f}")
    pwm_bar.progress(int(pwm_full[st.session_state.idx]/255*100))

    # Metrics
    error = np.array(y_data) - T_set
    ss_error = np.mean(error[-50:]) if len(error) > 50 else np.mean(error)
    rmse = np.sqrt(np.mean(error**2))
    overshoot = max(error) if len(error) > 0 else 0
    settling_time = next((t_data[j] for j in range(len(y_data)) if np.all(np.abs(error[j:]) <= 0.5)), None)

    recs = []
    if mode != "Hysteresis":
        if ss_error > 0.5: recs.append("Decrease Kp to reduce steady-state error.")
        if ss_error < -0.5: recs.append("Increase Kp to reduce steady-state error.")
        if settling_time is None or settling_time > 150: recs.append("Increase Kp or Ki for faster settling.")

    metrics_placeholder.markdown(
        f"**Steady-state error:** {ss_error:.2f} °C  \n"
        f"**RMSE:** {rmse:.2f}  \n"
        f"**Overshoot:** {overshoot:.2f} °C  \n"
        f"**Settling time:** {settling_time if settling_time else 'Not reached'} s  \n"
        + ("\n".join(f"- {r}" for r in recs))
    )

if st.session_state.dynamic_mode:
    update_dynamic()
    st.experimental_rerun()  # auto-refresh while running

# -------------------------------
# Static mode: full simulation
# -------------------------------
if st.session_state.static_mode:
    line.set_data(t_full, Tc_full)
    ax.set_xlim(0, duration)
    plot_placeholder.pyplot(fig)

    elapsed_placeholder.markdown(f"**Time elapsed:** {int(duration)} s")
    pwm_placeholder.markdown(f"**PWM:** {np.max(pwm_full):.1f}")
    pwm_bar.progress(int(np.max(pwm_full)/255*100))

    error = Tc_full - T_set
    ss_error = np.mean(error[-50:])
    rmse = np.sqrt(np.mean(error**2))
    overshoot = np.max(error)
    settling_time = next((t_full[j] for j in range(len(error)) if np.all(np.abs(error[j:]) <= 0.5)), None)

    recs = []
    if mode != "Hysteresis":
        if ss_error > 0.5: recs.append("Decrease Kp to reduce steady-state error.")
        if ss_error < -0.5: recs.append("Increase Kp to reduce steady-state error.")
        if settling_time is None or settling_time > 150: recs.append("Increase Kp or Ki for faster settling.")

    metrics_placeholder.markdown(
        f"**Steady-state error:** {ss_error:.2f} °C  \n"
        f"**RMSE:** {rmse:.2f}  \n"
        f"**Overshoot:** {overshoot:.2f} °C  \n"
        f"**Settling time:** {settling_time if settling_time else 'Not reached'} s  \n"
        + ("\n".join(f"- {r}" for r in recs))
    )
