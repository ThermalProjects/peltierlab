# app_static.py
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
# CSS para compactar todo y quitar scroll
# -------------------------------
st.markdown("""
<style>
/* Texto pequeño en todos los controles */
div.stSlider label, div.stSelectbox label, div.stButton label, div.stNumberInput label {
    font-size: 0.75rem !important;
    line-height: 0.85rem !important;
    padding: 1px 2px !important;
}
div.stSlider, div.stSelectbox, div.stButton, div.stNumberInput {
    max-height: none !important;
    overflow: visible !important;
}
/* Panel de métricas */
div.stExpander {
    font-size: 0.75rem !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Global parameters
# -------------------------------
best_params = [1.9507, 2.4906, 36.4772, 0.4806, 14.0687, 2.2298, 66.5757, 11.8439]
T_start = 19.0

# PSO defaults
Kp_default = 58.93
Ki_default = 3.91
Kd_default = 2.66
lambda_default = 0.67
mu_default = 1.47

# -------------------------------
# Title and Mode
# -------------------------------
st.title("PeltierLab Interactive Simulator")
mode = st.selectbox("Control mode", ["PID", "FOPID", "Hysteresis"])

# -------------------------------
# Controls Grid
# -------------------------------
col1, col2 = st.columns([1,1])

with col1:
    duration = st.slider("Simulation duration [s]", 100, 500, 300, step=10)
    T_set = st.slider("Setpoint [°C]", 5.0, 18.0, 12.0, 0.1)
    if mode in ["PID", "FOPID"]:
        Kp = st.slider("Kp", 0, 200, int(Kp_default), 1)
        Ki = st.slider("Ki", 0.0, 50.0, Ki_default, 0.1)
        Kd = st.slider("Kd", 0.0, 50.0, Kd_default, 0.1)
        if mode == "FOPID":
            lam = st.slider("Lambda (λ)", 0.1, 2.0, lambda_default, 0.01)
            mu = st.slider("Mu (μ)", 0.1, 2.0, mu_default, 0.01)
    elif mode == "Hysteresis":
        dT1 = st.slider("Upper band (dT1) [°C]", 0.1, 1.0, 0.5, 0.1)
        dT2 = st.slider("Lower band (dT2) [°C]", 0.1, 1.0, 0.5, 0.1)

    # Buttons
    col_buttons = st.columns(3)
    if col_buttons[0].button("Start"):
        st.session_state['running_state'] = True
    if col_buttons[1].button("Pause"):
        st.session_state['running_state'] = False
    if col_buttons[2].button("Stop"):
        st.session_state['running_state'] = False
        st.session_state['reset'] = True

# -------------------------------
# Graph and Metrics Grid
# -------------------------------
col3, col4 = st.columns([2,1])

with col3:
    fig, ax = plt.subplots(figsize=(6,3))
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

with col4:
    metrics_exp = st.expander("Model Information & Metrics", expanded=True)
    metrics_text = metrics_exp.empty()

# -------------------------------
# Initialize session state
# -------------------------------
if 'running_state' not in st.session_state:
    st.session_state['running_state'] = False
if 'reset' not in st.session_state:
    st.session_state['reset'] = True
if 'y_data' not in st.session_state or st.session_state['reset']:
    st.session_state['y_data'] = []
    st.session_state['t_data'] = []
    st.session_state['reset'] = False

# -------------------------------
# Prepare simulation data
# -------------------------------
sim = Simulator(best_params, T_start=T_start) if mode != "Hysteresis" else SimulatorHysteresisReal(best_params, T_start=T_start)
t_full = np.linspace(0, duration, duration + 1)
if mode == "PID":
    Tc_full, pwm_full = sim.simulate_3nodes_FOPID(t_custom=t_full, T_set=T_set, Kp=Kp, Ki=Ki, Kd=Kd, bias=0, lam=lambda_default, mu=mu_default)
elif mode == "FOPID":
    Tc_full, pwm_full = sim.simulate_3nodes_FOPID(t_custom=t_full, T_set=T_set, Kp=Kp, Ki=Ki, Kd=Kd, bias=0, lam=lam, mu=mu)
else:
    Tc_full, Tm_full, Th_full, pwm_full = sim.simulate(t_custom=t_full, T_set=T_set, dT1=dT1, dT2=dT2, P_max=5.0)

# -------------------------------
# Update loop (only when running)
# -------------------------------
if st.session_state['running_state']:
    for i in range(len(t_full)):
        if not st.session_state['running_state']:
            break

        st.session_state['y_data'].append(Tc_full[i])
        st.session_state['t_data'].append(t_full[i])

        # Update plot
        line.set_data(st.session_state['t_data'], st.session_state['y_data'])
        ax.set_xlim(0, duration)
        plot_placeholder.pyplot(fig)

        # Update metrics
        error = np.array(st.session_state['y_data']) - T_set
        ss_error = np.mean(error[-50:]) if len(error) > 50 else np.mean(error)
        rmse = np.sqrt(np.mean(error**2))
        settling_time = next((st.session_state['t_data'][j] for j in range(len(st.session_state['y_data'])) 
                              if np.all(np.abs(error[j:]) <= 0.5)), None)
        recs = []
        if abs(ss_error) > 0.5:
            recs.append("Steady-state error high → consider adjusting Kp/Ki/Kd (or λ/μ for FOPID).")
        if settling_time is None or settling_time > 150:
            recs.append("Slow response → increase Kp or λ to reduce settling time.")
        metrics_text.markdown(
            f"**Time elapsed:** {len(st.session_state['t_data'])} s  \n"
            f"**Steady-state error:** {ss_error:.3f} °C  \n"
            f"**RMSE:** {rmse:.3f}  \n"
            f"**Settling time:** {settling_time if settling_time else 'Not reached'} s  \n"
            + ("\n".join(f"- {r}" for r in recs))
        )
