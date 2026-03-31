# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

from peltierlab.core.simulator import Simulator  # Corregido: solo existe Simulator
from peltierlab.core.simulator_hysteresis_real import SimulatorHysteresisReal

# ------------------------------
# Configuración inicial
# ------------------------------
st.set_page_config(layout="wide")

st.sidebar.title("❄️ PeltierLab Interactive Simulator")
st.sidebar.markdown(
    "Explore thermoelectric system behavior using PID, FOPID, and Hysteresis control strategies."
)

# Controles
control_type = st.sidebar.selectbox("Control Type", ["PID", "FOPID", "Hysteresis"])
dynamic_mode = st.sidebar.checkbox("Dynamic", value=True)
start_button = st.sidebar.button("Start")
stop_button = st.sidebar.button("Stop")

# Variables de simulación
time_elapsed = 0.0
pwm_value = 0.0
overshoot = "-"
settling_time = "-"
error_value = "-"
recommendations = ""

# Datos de la gráfica
x_data = []
y_data = []

# Crear simulador
if control_type in ["PID", "FOPID"]:
    simulator = Simulator()
else:
    simulator = SimulatorHysteresisReal()

# ------------------------------
# Loop de simulación
# ------------------------------
running = False
if start_button:
    running = True
if stop_button:
    running = False

if running:
    # Obtener datos simulados
    t, T = simulator.step()  # Método ficticio: devuelve un solo paso
    time_elapsed += t
    x_data.append(time_elapsed)
    y_data.append(T)

    # Calcular métricas (ejemplo: undershoot)
    Tmin = np.min(y_data)
    overshoot = max(0.0, (simulator.Tref - Tmin))  # Ajuste: usamos undershoot real
    settling_time = simulator.calculate_settling_time()  # Método interno
    error_value = simulator.Tref - y_data[-1]

# ------------------------------
# Gráfica
# ------------------------------
fig, ax = plt.subplots()
ax.plot(x_data, y_data)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Temperature (°C)")
ax.grid(True)
st.pyplot(fig)

# ------------------------------
# Métricas
# ------------------------------
st.sidebar.markdown("### Metrics")
st.sidebar.write(f"Time elapsed: {time_elapsed:.2f} s")
st.sidebar.write(f"PWM: {pwm_value:.2f}")
st.sidebar.write(f"Undershoot: {overshoot}")
st.sidebar.write(f"Settling time: {settling_time}")
st.sidebar.write(f"Error: {error_value}")

# Recomendaciones
st.sidebar.markdown("### Recommendations")
st.sidebar.write(recommendations)
