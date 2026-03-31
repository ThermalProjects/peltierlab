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
# SIDEBAR
# -------------------------------
st.sidebar.title("❄️ PeltierLab Interactive Simulator")

mode = st.sidebar.selectbox("Control mode", ["PID", "FOPID", "Hysteresis"])
duration = st.sidebar.slider("Simulation duration [s]", 100, 500, 300, step=10)

# Botones Start / Stop uno al lado del otro
col1, col2 = st.sidebar.columns([1,1])
start = col1.button("Start")
stop  = col2.button("Stop")

# Time y PWM justo debajo de los botones
elapsed_placeholder = st.sidebar.empty()
pwm_placeholder     = st.sidebar.empty()
pwm_bar             = st.sidebar.empty()

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
# Metrics & recommendations
# -------------------------------
info_expander = st.expander("Model Information & Metrics", expanded=True)
metrics_text = st.empty()

# -------------------------------
# Simulation loop (4 FPS)
# -------------------------------
if 'running' not in st.session_state:
    st.session_state.running = False
if 'y_data' not in st.session_state or len(st.session_state.y_data) == 0:
    st.session_state.y_data = []
if 't_data' not in st.session_state or len(st.session_state.t_data) == 0:
    st.session_state.t_data = []
if 'idx' not in st.session_state:
    st.session_state.idx = 0

# Start reinicia desde cero correctamente
if start:
    st.session_state.running = True
    st.session_state.idx = 0
    st.session_state.y_data = [Tc_full[0]]
    st.session_state.t_data = [t_full[0]]

# Stop solo pausa
if stop:
    st.session_state.running = False

if st.session_state.running:
    fps = 4
    interval = 1.0 / fps
    start_time = time.time() - (st.session_state.t_data[-1] if len(st.session_state.t_data)>0 else 0)

    for i in range(st.session_state.idx, len(t_full)):
        current_time = time.time()
        elapsed_real = current_time - start_time
        if elapsed_real < t_full[i]:
            time.sleep(t_full[i] - elapsed_real)

        st.session_state.y_data.append(Tc_full[i])
        st.session_state.t_data.append(t_full[i])
        st.session_state.idx = i

        # Update line
        line.set_data(st.session_state.t_data, st.session_state.y_data)
        ax.set_xlim(0, duration)
        plot_placeholder.pyplot(fig)

        # Update time elapsed
        elapsed_placeholder.markdown(f"**Time elapsed:** {int(st.session_state.t_data[-1])} s")

        # Update PWM
        pwm_placeholder.markdown(f"**PWM:** {pwm_full[i]:.1f}")
        pwm_bar.progress(int(pwm_full[i]/255*100))

        # Update metrics
        error = np.array(st.session_state.y_data) - T_set
        ss_error = np.mean(error[-50:]) if len(error) > 50 else np.mean(error)
        rmse = np.sqrt(np.mean(error**2))
        overshoot = max(st.session_state.y_data) - T_set
        settling_time = next((st.session_state.t_data[j] for j in range(len(st.session_state.y_data)) if np.all(np.abs(error[j:]) <= 0.5)), None)

        recs = []
        if mode != "Hysteresis":
            if abs(ss_error) > 0.5:
                recs.append("Consider tuning controller to reduce steady-state error.")
            if settling_time is None or settling_time > 150:
                recs.append("Slow response → consider increasing gains for faster settling.")
            if overshoot > 1:
                recs.append("Overshoot high → consider lowering Kp or Kd.")
        metrics_text.markdown(
            f"**Steady-state error:** {ss_error:.3f} °C  \n"
            f"**RMSE:** {rmse:.3f}  \n"
            f"**Overshoot:** {overshoot:.3f} °C  \n"
            f"**Settling time:** {settling_time if settling_time else 'Not reached'} s  \n"
            + ("\n".join(f"- {r}" for r in recs))
        )
