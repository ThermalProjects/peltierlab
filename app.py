import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

from peltierlab.core.simulator import Simulator
from peltierlab.core.simulator_fopid import SimulatorFOPID
from peltierlab.core.simulator_hysteresis_real import SimulatorHysteresisReal

# ------------------------------
# CONFIGURACIÓN
# ------------------------------
st.set_page_config(page_title="❄️ PeltierLab Interactive Simulator", layout="wide")

# Barra lateral
with st.sidebar:
    st.title("❄️ PeltierLab Interactive Simulator")
    st.write("Explore thermoelectric system behavior using PID, FOPID, and Hysteresis control strategies.")

    # Control type
    control_type = st.selectbox("Control Type", ["PID", "FOPID", "Hysteresis"])

    # Dynamic / Static mode
    mode = st.radio("Mode", ["Dynamic", "Static"], horizontal=True)

    # Start / Stop
    col1, col2, col3, col4 = st.columns([1,1,1,1])
    with col1:
        start_btn = st.button("Start")
    with col2:
        stop_btn = st.button("Stop")
    with col3:
        time_display = st.empty()
    with col4:
        pwm_display = st.empty()

    # Recommendations (moved to bottom)
    st.write("---")
    st.subheader("Recommendations")
    recommendations = st.empty()

# ------------------------------
# SIMULADOR
# ------------------------------
simulator_map = {
    "PID": Simulator,
    "FOPID": SimulatorFOPID,
    "Hysteresis": SimulatorHysteresisReal
}

sim = simulator_map[control_type]()

running = False
y_data = []
t_data = []

T_set = 50.0  # referencia de temperatura
dt = 0.1

undershoot = 0.0  # inicializamos

while True:
    if start_btn:
        running = True
    if stop_btn:
        running = False

    if running:
        t_next = t_data[-1] + dt if t_data else 0.0
        y_next = sim.step(T_set)
        y_data.append(y_next)
        t_data.append(t_next)

        # ------------------------------
        # Calculo undershoot (FIX: no falla si array vacío)
        # ------------------------------
        transitory_end = int(len(t_data)/2)  # ejemplo de final de transitorio
        if len(y_data[:transitory_end]) > 0:
            Tmin = np.min(np.array(y_data[:transitory_end]))
            undershoot = max(0.0, (T_set - Tmin) / T_set * 100.0)
        else:
            undershoot = 0.0

    # ------------------------------
    # Mostrar gráficos y métricas
    # ------------------------------
    fig, ax = plt.subplots()
    ax.plot(t_data, y_data, label="Tsup")
    ax.axhline(T_set, color="r", linestyle="--", label="Tref")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    st.pyplot(fig)

    # Mostrar métricas
    time_display.text(f"Time: {t_data[-1]:.2f} s" if t_data else "Time: 0.00 s")
    pwm_display.text(f"PWM: {sim.get_pwm():.2f}" if hasattr(sim, "get_pwm") else "PWM: -")
    recommendations.text(f"Undershoot: {undershoot:.2f}%")

    time.sleep(0.05)
