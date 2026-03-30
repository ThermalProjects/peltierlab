# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from peltierlab.core.simulator import Simulator
from peltierlab.core.simulator_hysteresis_real import SimulatorHysteresisReal

# -------------------------------
# Parámetros globales
# -------------------------------
best_params = [1.9507, 2.4906, 36.4772, 0.4806, 14.0687, 2.2298, 66.5757, 11.8439]
T_start = 19.0

Kp_default = 58.93
Ki_default = 3.91
Kd_default = 2.66
lambda_default = 0.67
mu_default = 1.47

# -------------------------------
# Título y descripción
# -------------------------------
st.title("Simulador de Sistema Peltier")
st.markdown("""
Selecciona un modo de simulación y ajusta los parámetros con los sliders.
Los resultados se mostrarán en la gráfica.
""")

# -------------------------------
# Selección de modo
# -------------------------------
mode = st.selectbox(
    "Selecciona el modo de simulación:",
    ["PID", "FOPID", "Hysteresis"]
)

# -------------------------------
# Panel de sliders dinámico según modo
# -------------------------------
if mode in ["PID", "FOPID"]:
    T_set = st.slider("Setpoint [°C]", 5.0, 25.0, 12.0, 0.1)
    bias = st.slider("Bias [°C]", -2.0, 2.0, 0.0, 0.1)
    Kp = st.slider("Kp", 0, 200, int(Kp_default), 1)
    Ki = st.slider("Ki", 0.0, 50.0, Ki_default, 0.1)
    Kd = st.slider("Kd", 0.0, 50.0, Kd_default, 0.1)

    if mode == "FOPID":
        lam = st.slider("Lambda", 0.1, 2.0, lambda_default, 0.01)
        mu = st.slider("Mu", 0.1, 2.0, mu_default, 0.01)

elif mode == "Hysteresis":
    T_set = st.slider("Setpoint [°C]", 10.0, 20.0, 12.0, 0.1)
    dT1 = st.slider("dT1 [°C]", 0.1, 2.0, 0.5, 0.1)
    dT2 = st.slider("dT2 [°C]", 0.1, 2.0, 0.5, 0.1)

# -------------------------------
# Botón para ejecutar simulación
# -------------------------------
if st.button("Ejecutar simulación"):
    t_new = None
    Tc_sim = None
    pwm_sim = None

    # -------------------------------
    # Modo PID o FOPID
    # -------------------------------
    if mode in ["PID", "FOPID"]:
        t_new = np.linspace(0, 300, 500)
        sim = Simulator(best_params, T_start=T_start)

        if mode == "PID":
            Tc_sim, pwm_sim = sim.simulate_3nodes_FOPID(
                t_custom=t_new,
                T_set=T_set,
                Kp=Kp,
                Ki=Ki,
                Kd=Kd,
                bias=bias,
                lam=lambda_default,  # valores por defecto para PID
                mu=mu_default
            )
        else:  # FOPID
            Tc_sim, pwm_sim = sim.simulate_3nodes_FOPID(
                t_custom=t_new,
                T_set=T_set,
                Kp=Kp,
                Ki=Ki,
                Kd=Kd,
                bias=bias,
                lam=lam,
                mu=mu
            )

        # -------------------------------
        # Graficar
        # -------------------------------
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(t_new, Tc_sim, color='green', linestyle='-.', linewidth=2, label='Temperatura Simulada')
        ax.axhline(T_set, color='red', linestyle='--', linewidth=1, label=f'Setpoint {T_set}°C')
        ax.set_xlabel("Tiempo [s]")
        ax.set_ylabel("Temperatura [°C]")
        ax.set_title(f"Simulación 3-nodos + {mode}")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

    # -------------------------------
    # Modo Hysteresis
    # -------------------------------
    elif mode == "Hysteresis":
        t_new = np.linspace(0, 600, 1200)
        sim = SimulatorHysteresisReal(best_params, T_start=T_start)
        Tc, Tm, Th, pwm = sim.simulate(
            t_custom=t_new,
            T_set=T_set,
            dT1=dT1,
            dT2=dT2,
            P_max=5.0
        )

        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(t_new, Tc, linewidth=2, label='Temperatura Simulada')
        ax.axhline(T_set, linestyle='--', linewidth=1, label=f'Setpoint {T_set}°C')
        ax.set_xlabel("Tiempo [s]")
        ax.set_ylabel("Temperatura [°C]")
        ax.set_title("Simulación Histéresis")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)