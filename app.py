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
# Sidebar - compacta
# -------------------------------
st.sidebar.markdown("<h3 style='font-size:20px'>❄️ PeltierLab Simulator</h3>", unsafe_allow_html=True)

mode = st.sidebar.selectbox("Control mode", ["PID", "FOPID", "Hysteresis"])

if mode in ["PID", "FOPID"]:
    T_set = st.sidebar.slider("Setpoint [°C]", 5.0, 18.0, 12.0, 0.1)
    bias = st.sidebar.slider("Bias [°C]", -2.0, 2.0, 0.0, 0.1)
    Kp = st.sidebar.slider("Kp", 0, 200, int(Kp_default), 1)
    Ki = st.sidebar.slider("Ki", 0.0, 50.0, Ki_default, 0.1)
    Kd = st.sidebar.slider("Kd", 0.0, 50.0, Kd_default, 0.1)
    if mode == "FOPID":
        lam = st.sidebar.slider("Lambda (λ)", 0.1, 2.0, lambda_default, 0.01)
        mu = st.sidebar.slider("Mu (μ)", 0.1, 2.0, mu_default, 0.01)

elif mode == "Hysteresis":
    T_set = st.sidebar.slider("Setpoint [°C]", 10.0, 18.0, 12.0, 0.1)
    dT1 = st.sidebar.slider("Upper band (dT1) [°C]", 0.1, 1.0, 0.5, 0.1)
    dT2 = st.sidebar.slider("Lower band (dT2) [°C]", 0.1, 1.0, 0.5, 0.1)

duration = st.sidebar.slider("Simulation duration [s]", 50, 500, 300, 10)

start_button = st.sidebar.button("Start")
stop_placeholder = st.sidebar.empty()  # se actualizará con Stop
time_elapsed_placeholder = st.sidebar.empty()

# Firma abajo
st.sidebar.markdown(
    "<span style='font-size:10px'>Interactive simulation of thermoelectric systems using PID, FOPID, and Hysteresis control.</span>",
    unsafe_allow_html=True
)

# -------------------------------
# Main title
# -------------------------------
st.title("❄️ PeltierLab Interactive Simulator")
st.markdown("Interactive simulation of thermoelectric systems using PID, FOPID, and Hysteresis control.")

# -------------------------------
# Figure setup
# -------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(0, duration)
ax.set_ylim(0, 20)
ax.set_xlabel("Time [s]", fontsize=8)
ax.set_ylabel("Temperature [°C]", fontsize=8)
ax.set_title(f"{mode} Control Simulation", fontsize=10)
ax.grid(True)
ax.axhline(T_set, color="red", linestyle="--", label="Setpoint")
line_temp, = ax.plot([], [], color='blue', linewidth=1)
line_pwm, = ax.plot([], [], color='green', linewidth=1)
st.pyplot(fig, clear_figure=True)

# -------------------------------
# Simulation setup
# -------------------------------
if mode in ["PID", "FOPID"]:
    sim = Simulator(best_params, T_start=T_start)
else:
    sim = SimulatorHysteresisReal(best_params, T_start=T_start)

y_vals, pwm_vals, t_vals = [], [], []
t_current = 0.0
dt = 0.25  # 4 FPS
running = False

# -------------------------------
# Metrics and recommendations placeholder
# -------------------------------
metrics_placeholder = st.empty()
recommend_placeholder = st.empty()

# -------------------------------
# Run simulation loop
# -------------------------------
while True:
    if start_button:
        running = True
        start_button = False  # no reinicia cada loop
    if not running:
        time.sleep(0.1)
        continue

    # Stop button
    if stop_placeholder.button("Stop"):
        running = False
        continue

    # Step simulation
    if mode in ["PID", "FOPID"]:
        Tc, pwm = sim.step(T_set, Kp, Ki, Kd,
                           lam if mode=="FOPID" else lambda_default,
                           mu if mode=="FOPID" else mu_default,
                           bias, dt)
    else:
        Tc, pwm = sim.step(T_set, dT1, dT2, dt)

    t_current += dt
    y_vals.append(Tc)
    pwm_vals.append(pwm)
    t_vals.append(t_current)

    # Update plot
    line_temp.set_data(t_vals, y_vals)
    line_pwm.set_data(t_vals, pwm_vals)
    ax.set_xlim(0, max(duration, t_current))
    st.pyplot(fig, clear_figure=True)

    # Update metrics
    error = np.array(y_vals) - T_set
    ss_error = np.mean(error[-int(1/dt):])
    rmse = np.sqrt(np.mean(error**2))
    settling_time = next((t_vals[i] for i in range(len(y_vals)) if np.all(np.abs(error[i:]) <= 0.5)), None)

    metrics_placeholder.markdown(f"""
    **Metrics**  
    Settling time: {settling_time:.2f} s  
    Steady-state error: {ss_error:.3f} °C  
    RMSE: {rmse:.3f}  
    """)
    
    # Recommendations
    rec_text = ""
    if Kp > Kp_default: rec_text += "➡️ Kp demasiado alto → overshoot mayor\n"
    elif Kp < Kp_default: rec_text += "➡️ Kp demasiado bajo → respuesta lenta\n"
    if mode=="FOPID":
        if lam > lambda_default: rec_text += "➡️ Lambda mayor → más precisión\n"
        if mu > mu_default: rec_text += "➡️ Mu mayor → respuesta más agresiva\n"
    recommend_placeholder.markdown(rec_text)

    # Update elapsed time
    time_elapsed_placeholder.markdown(f"Time elapsed: {t_current:.1f} s")

    if t_current >= duration:
        running = False
        break

    time.sleep(dt)
