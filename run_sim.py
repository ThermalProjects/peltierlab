from peltierlab.core.simulator import Simulator
import numpy as np
import matplotlib.pyplot as plt

# =================== Parámetros 3-nodos ===================
best_params = [1.9507, 2.4906, 36.4772, 0.4806, 14.0687, 2.2298, 66.5757, 11.8439]

# =================== Configuración del ensayo ===================
T_start = 19.0
T_set   = 12.0
t_new   = np.linspace(0, 300, 500)

# =================== Crear simulador ===================
sim = Simulator(best_params, T_start=T_start)

# =================== Ejecutar simulación ===================
Tc_sim, pwm_sim = sim.simulate_3nodes_FOPID(t_new, T_set=T_set)

# =================== Graficar resultados ===================
plt.figure(figsize=(10,5))
plt.plot(t_new, Tc_sim, label='Temperatura')
plt.plot(t_new, pwm_sim, label='PWM')
plt.xlabel("Tiempo [s]")
plt.ylabel("Valores")
plt.title("Simulación 3-nodos + FOPID")
plt.grid(True)
plt.legend()
plt.show()