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
# Global default parameters
# -------------------------------
best_params = [1.9507, 2.4906, 36.4772, 0.4806, 14.0687, 2.2298, 66.5757, 11.8439]

# PID / FOPID defaults
Kp_default = 58.93
Ki_default = 3.91
Kd_default = 2.66
lambda_default = 0.67
mu_default = 1.47

# -------------------------------
# Sidebar controls
# -------------------------------
st.sidebar.header("Simulation Settings")

# Ambient temperature
T_start = st.sidebar.slider("Ambient temperature [°C]", 15.0, 25.0, 19.0, 0.1)

# Control mode
mode = st.sidebar.selectbox("Control mode", ["PID", "FOPID", "Hysteresis"])

# Simulation duration
duration = st.sidebar.slider("Simulation duration [s]", 100, 500, 300, step=10)

# -------------------------------
# Controller parameters
# -------------------------------
with st.sidebar.expander("Controller Parameters", expanded=True):
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
# Session state initialization
# -------------------------------
if 'sim_state' not in st.session_state:
    st.session_state['sim_state'] = {
        'running': False,
        'paused': False,
        'index': 0,
        't_data': [],
        'y_data': [],
        'pwm_data': [],
    }
sim_state = st.session_state['sim_state']

# -------------------------------
# Buttons in a single row
# -------------------------------
cols = st.sidebar.columns(3)
start_btn = cols[0].button("Start")
pause_btn = cols[1].button("Pause/Resume")
stop_btn = cols[2].button("Stop")

# Button actions
if start_btn:
    sim_state['running'] = True
    sim_state['paused'] = False
    sim_state['index'] = 0
    sim_state['t_data'] = []
    sim_state['y_data'] = []
    sim_state['pwm_data'] = []

if pause_btn:
    sim_state['paused'] = not sim_state['paused']  # toggle pause

if stop_btn:
    sim_state['running'] = False
    sim_state['paused'] = False
    sim_state['index'] = 0
    sim_state['t_data'] = []
    sim_state['y_data'] = []
    sim_state['pwm_data'] = []

# -------------------------------
# Prepare simulation
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
# Title & plots
# -------------------------------
st.title("❄️ PeltierLab Interactive Simulator")
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
with info_expander:
    metrics_text = st.empty()

    # Reference optimal display inside metrics
    if mode == "PID":
        metrics_text.markdown("**Reference optimal PID (PSO):**")
        metrics_text.markdown(f"Kp = {Kp_default}, Ki = {Ki_default}, Kd = {Kd_default}")
    elif mode == "FOPID":
        metrics_text.markdown("**Reference optimal FOPID (PSO):**")
        metrics_text.markdown(f"Kp = {Kp_default}, Ki = {Ki_default}, Kd = {Kd_default}, λ = {lambda_default}, μ = {mu_default}")

# -------------------------------
# Simulation loop
# -------------------------------
if sim_state['running']:
    y_data = sim_state['y_data']
    t_data = sim_state['t_data']
    pwm_data = sim_state['pwm_data']

    fps = 4
    interval = 1.0 / fps
    start_time = time.time()

    for i in range(sim_state['index'], len(t_full)):
        if not sim_state['running']:
            break
        if sim_state['paused']:
            time.sleep(0.1)
            continue  # freeze plot and metrics

        # wait to match simulation time
        current_time = time.time()
        elapsed_real = current_time - start_time
        if elapsed_real < t_full[i]:
            time.sleep(t_full[i] - elapsed_real)

        # append data
        y_data.append(Tc_full[i])
        t_data.append(t_full[i])
        pwm_data.append(pwm_full[i])
        sim_state['index'] += 1

        # update line
        line.set_data(t_data, y_data)
        ax.set_xlim(0, duration)
        plot_placeholder.pyplot(fig)

        # update sidebar info (compact)
        elapsed_placeholder = st.sidebar.empty()
        pwm_placeholder = st.sidebar.empty()
        pwm_bar = st.sidebar.empty()
        elapsed_placeholder.markdown(f"**Time elapsed:** {int(t_full[i])} s")
        pwm_placeholder.markdown(f"**PWM:** {pwm_full[i]:.1f}")
        pwm_bar.progress(int(pwm_full[i]/255*100))

        # compute error metrics
        error = np.array(y_data) - T_set
        ss_error = np.mean(error[-50:]) if len(error) > 50 else np.mean(error)
        rmse = np.sqrt(np.mean(error**2))
        settling_time = next((t_data[j] for j in range(len(y_data))
                              if np.all(np.abs(error[j:]) <= 0.5)), None)

        # recommendations
        recs = []
        if mode in ["PID", "FOPID"]:
            if Kp < Kp_default:
                recs.append("Kp low → slower response, larger overshoot.")
            else:
                recs.append("Kp high → faster response, smaller settling time but risk of overshoot.")
            if Ki < Ki_default:
                recs.append("Ki low → larger steady-state error.")
            else:
                recs.append("Ki high → reduces steady-state error, may increase overshoot.")
            if Kd < Kd_default:
                recs.append("Kd low → less damping, oscillatory response.")
            else:
                recs.append("Kd high → more damping, slower response.")
            if mode == "FOPID":
                if lam < lambda_default:
                    recs.append("λ low → reduces integral action, may increase error.")
                else:
                    recs.append("λ high → stronger integral effect, reduces error but can overshoot.")
                if mu < mu_default:
                    recs.append("μ low → derivative effect weaker, slower damping.")
                else:
                    recs.append("μ high → stronger derivative effect, more damping.")

        metrics_text.markdown(
            f"**Steady-state error:** {ss_error:.3f} °C  \n"
            f"**RMSE:** {rmse:.3f}  \n"
            f"**Settling time:** {settling_time if settling_time else 'Not reached'} s  \n"
            + ("\n".join(f"- {r}" for r in recs))
        )
