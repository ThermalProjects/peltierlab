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
# SIDEBAR (title + description)
# -------------------------------
st.sidebar.title("❄️ PeltierLab Interactive Simulator")
st.sidebar.markdown(
    "Explore thermoelectric system behavior using PID, FOPID, and Hysteresis control strategies."
)

# -------------------------------
# Control mode
# -------------------------------
mode = st.sidebar.selectbox("Control mode", ["PID", "FOPID", "Hysteresis"])

# -------------------------------
# Real-time / Static switch
# -------------------------------
col1, col2 = st.sidebar.columns(2)
dynamic = col1.button("Dynamic")
static = col2.button("Static")

if 'dynamic_mode' not in st.session_state:
    st.session_state['dynamic_mode'] = True

if dynamic:
    st.session_state['dynamic_mode'] = True
if static:
    st.session_state['dynamic_mode'] = False

# -------------------------------
# Simulation duration & start/stop
# -------------------------------
duration = st.sidebar.slider("Simulation duration [s]", 100, 500, 300, step=10)
start = st.sidebar.button("Start")
stop = st.sidebar.button("Stop")

if 'running' not in st.session_state:
    st.session_state['running'] = False

if start:
    st.session_state['running'] = True
if stop:
    st.session_state['running'] = False

# -------------------------------
# Compact sliders
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
    elif mode == "Hysteresis":
        T_set = st.slider("Setpoint [°C]", 10.0, 18.0, 12.0, 0.1)
        dT1 = st.slider("Upper band (dT1) [°C]", 0.1, 1.0, 0.5, 0.1)
        dT2 = st.slider("Lower band (dT2) [°C]", 0.1, 1.0, 0.5, 0.1)

# -------------------------------
# Prepare simulation data
# -------------------------------
t_full = np.linspace(0, duration, duration + 1)

if mode in ["PID", "FOPID"]:
    sim = Simulator(best_params, T_start=T_start)
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
    Tc_full, Tm_full, Th_full, pwm_full = sim.simulate(
        t_custom=t_full, T_set=T_set, dT1=dT1, dT2=dT2, P_max=5.0
    )

# -------------------------------
# Plot placeholders
# -------------------------------
st.subheader(f"Results: {mode}")
fig, ax = plt.subplots(figsize=(7, 3.5))
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
# Time & PWM in sidebar
# -------------------------------
elapsed_placeholder = st.sidebar.empty()
pwm_placeholder = st.sidebar.empty()
pwm_bar = st.sidebar.empty()

# -------------------------------
# Metrics & Recommendations in sidebar
# -------------------------------
metrics_text = st.sidebar.empty()

# -------------------------------
# Simulation loop
# -------------------------------
y_data = []
t_data = []

if st.session_state['dynamic_mode'] and st.session_state['running']:
    fps = 4
    interval = 1.0 / fps
    start_time = time.time()

    for i in range(len(t_full)):
        current_time = time.time()
        elapsed_real = current_time - start_time
        if elapsed_real < t_full[i]:
            time.sleep(t_full[i] - elapsed_real)

        y_data.append(Tc_full[i])
        t_data.append(t_full[i])

        # Update plot
        line.set_data(t_data, y_data)
        ax.set_xlim(0, duration)
        plot_placeholder.pyplot(fig)

        # Update time & PWM
        elapsed_placeholder.markdown(f"**Time elapsed:** {int(t_full[i])} s")
        pwm_placeholder.markdown(f"**PWM:** {pwm_full[i]:.1f}")
        pwm_bar.progress(int(pwm_full[i]/255*100))

        # -----------------------
        # Metrics (including overshoot corrected)
        # -----------------------
        error = np.array(y_data) - T_set
        ss_error = np.mean(error[-50:]) if len(error) > 50 else np.mean(error)
        rmse = np.sqrt(np.mean(error**2))
        # Overshoot only after crossing twice
        crossed = np.where(np.diff(np.sign(error)))[0]
        if len(crossed) >= 2:
            overshoot = max(y_data[:crossed[1]+1]) - T_set
        else:
            overshoot = '-'

        metrics_text.markdown(
            f"**Steady-state error:** {ss_error:.3f} °C  \n"
            f"**RMSE:** {rmse:.3f}  \n"
            f"**Overshoot:** {overshoot if overshoot != '-' else '-'} °C"
        )

# -------------------------------
# Static mode
# -------------------------------
if not st.session_state['dynamic_mode']:
    y_data = Tc_full
    t_data = t_full
    line.set_data(t_data, y_data)
    ax.set_xlim(0, duration)
    plot_placeholder.pyplot(fig)

    # Update metrics
    error = np.array(y_data) - T_set
    ss_error = np.mean(error[-50:]) if len(error) > 50 else np.mean(error)
    rmse = np.sqrt(np.mean(error**2))
    crossed = np.where(np.diff(np.sign(error)))[0]
    if len(crossed) >= 2:
        overshoot = max(y_data[:crossed[1]+1]) - T_set
    else:
        overshoot = '-'

    metrics_text.markdown(
        f"**Steady-state error:** {ss_error:.3f} °C  \n"
        f"**RMSE:** {rmse:.3f}  \n"
        f"**Overshoot:** {overshoot if overshoot != '-' else '-'} °C"
    )

    # Show time & PWM
    elapsed_placeholder.markdown(f"**Time elapsed:** {duration} s")
    pwm_placeholder.markdown(f"**PWM:** {pwm_full[-1]:.1f}")
    pwm_bar.progress(int(pwm_full[-1]/255*100))
