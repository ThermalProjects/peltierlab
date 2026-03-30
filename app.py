import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Import simulators ---
from peltierlab.core.simulator import Simulator
from peltierlab.core.simulator_hysteresis_real import SimulatorHysteresisReal

# -------------------------------
# Metrics function
# -------------------------------
def compute_metrics(t, y, setpoint):
    error = y - setpoint
    overshoot = max(y) - setpoint
    overshoot_pct = (overshoot / setpoint) * 100 if setpoint != 0 else 0

    band = 0.5
    settling_time = None
    for i in range(len(y)):
        if np.all(np.abs(y[i:] - setpoint) <= band):
            settling_time = t[i]
            break

    ss_error = np.mean(error[-50:])
    rmse = np.sqrt(np.mean(error**2))
    return overshoot_pct, settling_time, ss_error, rmse

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(layout="wide")
st.title("❄️ PeltierLab Interactive Simulator")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("Control Settings")
mode = st.sidebar.selectbox("Control Mode", ["PID", "FOPID", "Hysteresis"])
T_set = st.sidebar.slider("Setpoint (°C)", 0.0, 18.0, 12.0)
T_start = st.sidebar.slider("Initial Temperature (°C)", 0.0, 30.0, 25.0)

# PID / FOPID params
Kp = st.sidebar.slider("Kp", 0.0, 200.0, 58.93)
Ki = st.sidebar.slider("Ki", 0.0, 50.0, 3.91)
Kd = st.sidebar.slider("Kd", 0.0, 50.0, 2.66)
lam = st.sidebar.slider("Lambda (λ)", 0.1, 2.0, 0.67)
mu  = st.sidebar.slider("Mu (μ)", 0.1, 2.0, 1.47)

# Hysteresis params
dT1 = st.sidebar.slider("Upper band dT1", 0.0, 1.0, 0.5)
dT2 = st.sidebar.slider("Lower band dT2", 0.0, 1.0, 0.5)

# -------------------------------
# Model parameters
# -------------------------------
best_params = [1.9507, 2.4906, 36.4772, 0.4806, 14.0687, 2.2298, 66.5757, 11.8439]

# -------------------------------
# Simulation
# -------------------------------
t_new = np.linspace(0, 300, 500)

if mode in ["PID", "FOPID"]:
    sim = Simulator(best_params, T_start=T_start)
    Tc, _ = sim.simulate_3nodes_FOPID(
        t_custom=t_new,
        T_set=T_set,
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        bias=0,
        lam=lam if mode=="FOPID" else 1.0,
        mu=mu if mode=="FOPID" else 1.0
    )

elif mode == "Hysteresis":
    sim = SimulatorHysteresisReal(best_params, T_start=T_start)
    Tc, _, _, _ = sim.simulate(
        t_custom=t_new,
        T_set=T_set,
        dT1=dT1,
        dT2=dT2,
        P_max=1.0
    )

# -------------------------------
# Plot
# -------------------------------
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(t_new, Tc, label="Cold Temperature (Tc)", linewidth=2)
ax.axhline(T_set, color="red", linestyle="--", label="Setpoint")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Temperature (°C)")
ax.set_title("Temperature Response ❄️")
ax.legend()
ax.grid()
st.pyplot(fig)

# -------------------------------
# Metrics
# -------------------------------
st.subheader("📊 Performance Metrics")
os_val, st_val, ss_val, rmse_val = compute_metrics(t_new, Tc, T_set)
st.write(f"Overshoot: {os_val:.2f} %")
st.write(f"Settling time: {st_val:.2f} s" if st_val else "Settling time: Not reached")
st.write(f"Steady-state error: {ss_val:.3f} °C")
st.write(f"RMSE: {rmse_val:.3f}")

# -------------------------------
# Interpretation
# -------------------------------
st.subheader("🧠 System Interpretation")
if os_val > 20:
    st.warning("High overshoot → controller is too aggressive (consider reducing Kp or Ki)")
elif os_val < 5:
    st.success("Low overshoot → smooth and well-damped response")
if st_val is None:
    st.warning("System does not settle within ±0.5°C band")
elif st_val > 150:
    st.info("Slow settling time → system is stable but sluggish")
else:
    st.success("Good settling time")
if abs(ss_val) > 0.5:
    st.warning("Noticeable steady-state error → consider increasing integral action (Ki)")
if rmse_val < 1:
    st.success("Excellent overall accuracy (low RMSE)")

# -------------------------------
# Model Information
# -------------------------------
with st.expander("📊 Model Information"):
    if mode in ["PID", "FOPID"]:
        st.markdown("""
### 🧊 Three-Node Thermal Model

The system is modeled using a **lumped-parameter thermal network** with three nodes:

- Cold side (Tc)
- Middle node (Tm)
- Hot side (Th)

Heat transfer is described using thermal resistances and capacitances.

---

### ⚡ Control Strategy

#### PID
u(t) = Kp·e(t) + Ki·∫e(t)dt + Kd·de(t)/dt

#### FOPID
Extends PID using fractional orders λ and μ.

---

### 🔧 Implementation

- Numerical integration
- PWM bounded [0–255]
- Ambient = 25°C
- Startup ramp applied first 5 seconds

---

### 📊 Parameters

R1, R2, Rconv, Cc, Cp, Ch, frac_cold, tau
""")
    elif mode == "Hysteresis":
        st.markdown("""
### 🔁 Hysteresis Control

ON/OFF control with deadband:

- OFF if Tc ≤ Setpoint − dT2
- ON if Tc ≥ Setpoint + dT1

Produces oscillations around setpoint.

---

### 🔧 Implementation

- Power either 0 or max
- Cold/Hot distribution fraction
- Ambient = 25°C

---

### 📊 Parameters

- `dT1` → upper hysteresis band  
- `dT2` → lower hysteresis band  
- `P_max` → maximum applied power  
""")
