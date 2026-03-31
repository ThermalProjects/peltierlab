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

# Default controller parameters
PID_optimal = {"Kp": 58.93, "Ki": 3.91, "Kd": 2.66}
FOPID_optimal = {"Kp": 58.93, "Ki": 3.91, "Kd": 2.66, "λ": 0.67, "μ": 1.47}

# -------------------------------
# Title
# -------------------------------
st.title("❄️ PeltierLab Interactive Simulator")
st.markdown(
    "Explore thermoelectric system behavior using PID, FOPID, and Hysteresis control strategies."
)

# -------------------------------
# Sidebar controls
# -------------------------------
st.sidebar.header("Settings")

mode = st.sidebar.selectbox("Control mode", ["PID", "FOPID", "Hysteresis"])
duration = st.sidebar.slider("Simulation duration [s]", 100, 500, 300, step=10)
start_stop = st.sidebar.button("Start/Stop")

# Elapsed time and PWM placeholders
elapsed_placeholder = st.sidebar.empty()
pwm_placeholder = st.sidebar.empty()
pwm_bar = st.sidebar.empty()

# -------------------------------
# Control parameter sliders
# -------------------------------
with st.sidebar.expander("Control Parameters", expanded=True):
    if mode in ["PID", "FOPID"]:
        T_set = st.slider("Setpoint [°C]", 5.0, 18.0, 12.0, 0.1)
        bias = st.slider("Bias [°C]", -2.0, 2.0, 0.0, 0.1)
        Kp = st.slider("Kp", 0, 200, int(PID_optimal["Kp"]), 1)
        Ki = st.slider("Ki", 0.0, 50.0, PID_optimal["Ki"], 0.1)
        Kd = st.slider("Kd", 0.0, 50.0, PID_optimal["Kd"], 0.1)

        if mode == "FOPID":
            lam = st.slider("Lambda (λ)", 0.1, 2.0, FOPID_optimal["λ"], 0.01)
            mu = st.slider("Mu (μ)", 0.1, 2.0, FOPID_optimal["μ"], 0.01)

    elif mode == "Hysteresis":
        T_set = st.slider("Setpoint [°C]", 10.0, 18.0, 12.0, 0.1)
        dT1 = st.slider("Upper band (dT1) [°C]", 0.1, 1.0, 0.5, 0.1)
        dT2 = st.slider("Lower band (dT2) [°C]", 0.1, 1.0, 0.5, 0.1)

# -------------------------------
# Prepare simulation
# -------------------------------
t_full = np.linspace(0, duration, duration + 1)
if mode in ["PID", "FOPID"]:
    sim = Simulator(best_params, T_start=T_start)
    if mode == "PID":
        Tc_full, pwm_full = sim.simulate_3nodes_FOPID(
            t_custom=t_full, T_set=T_set, Kp=Kp, Ki=Ki, Kd=Kd,
            bias=bias, lam=FOPID_optimal["λ"], mu=FOPID_optimal["μ"]
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
# Plot setup
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
# Metrics & reference tuning
# -------------------------------
info_expander = st.expander("Model Information & Metrics", expanded=True)
metrics_text = info_expander.empty()
tuning_text = info_expander.empty()

# -------------------------------
# Simulation control
# -------------------------------
if 'running_state' not in st.session_state:
    st.session_state['running_state'] = False

# Toggle running only on button click
if start_stop:
    st.session_state['running_state'] = not st.session_state['running_state']

if st.session_state['running_state']:
    y_data = []
    t_data = []
    fps = 4
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

        # Update elapsed time & PWM
        elapsed_placeholder.markdown(f"**Time elapsed:** {int(t_full[i])} s")
        pwm_placeholder.markdown(f"**PWM:** {pwm_full[i]:.1f}")
        pwm_bar.progress(int(pwm_full[i]/255*100))

        # -------------------------------
        # Metrics
        # -------------------------------
        error = np.array(y_data) - T_set
        ss_error = np.mean(error[-50:]) if len(error) > 50 else np.mean(error)
        rmse = np.sqrt(np.mean(error**2))
        settling_time = next((t_data[j] for j in range(len(y_data)) if np.all(np.abs(error[j:]) <= 0.5)), None)

        # -------------------------------
        # Recommendations
        # -------------------------------
        recs = []
        if mode == "PID":
            tuning_text.markdown(
                f"**Reference optimal PID (PSO):**  \n"
                f"Kp = {PID_optimal['Kp']}, Ki = {PID_optimal['Ki']}, Kd = {PID_optimal['Kd']}"
            )
            if abs(ss_error) > 0.5:
                recs.append("High steady-state error → increase Ki to reduce it.")
            if settling_time is None or settling_time > 150:
                recs.append("Slow response → increase Kp to speed up settling.")
            if rmse > 1.0:
                recs.append("Oscillations → reduce Kd or slightly decrease Kp.")

        elif mode == "FOPID":
            tuning_text.markdown(
                f"**Reference optimal FOPID (PSO):**  \n"
                f"Kp = {FOPID_optimal['Kp']}, Ki = {FOPID_optimal['Ki']}, Kd = {FOPID_optimal['Kd']}, "
                f"λ = {FOPID_optimal['λ']}, μ = {FOPID_optimal['μ']}"
            )
            if abs(ss_error) > 0.5:
                recs.append("High steady-state error → increase Ki or μ slightly.")
            if settling_time is None or settling_time > 150:
                recs.append("Slow response → increase Kp or λ to speed up settling.")
            if rmse > 1.0:
                recs.append("Oscillations → reduce Kd, decrease λ or μ slightly.")

        elif mode == "Hysteresis":
            recs.append("Hysteresis control → tune dT1/dT2 for desired amplitude and overshoot.")

        metrics_text.markdown(
            f"**Steady-state error:** {ss_error:.3f} °C  \n"
            f"**RMSE:** {rmse:.3f}  \n"
            f"**Settling time:** {settling_time if settling_time else 'Not reached'} s  \n"
            + ("\n".join(f"- {r}" for r in recs))
        )
