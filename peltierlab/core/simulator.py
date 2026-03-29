# simulator.py
import numpy as np

class Simulator:
    def __init__(self, params, T_start=19.0):
        """
        params: lista con parámetros 3-nodos
        T_start: temperatura inicial
        """
        self.params = params
        self.T_start = T_start

    def simulate_3nodes_FOPID(self, t_custom, T_set,
                               Kp=58.93, Ki=3.91, Kd=2.66,
                               bias=0.0, lam=0.67, mu=1.47):
        """Simulación 3-nodos + FOPID con parámetros dinámicos"""
        R1,R2,Rconv,frac_cold,Cc,Cp,Ch,tau = self.params
        frac_hot = 1 - frac_cold

        T = np.zeros((len(t_custom),3))
        T[0,:] = self.T_start, self.T_start, self.T_start

        pwm_hist = np.zeros(len(t_custom))

        # FOPID histórico
        M = 30
        Ts = t_custom[1] - t_custom[0]
        e_hist = np.zeros(M+1)
        wI = np.zeros(M+1)
        wD = np.zeros(M+1)
        wI[0] = wD[0] = 1.0
        for k in range(1, M+1):
            wI[k] = wI[k-1] * ((lam - (k-1))/k)
            wD[k] = wD[k-1] * ((mu - (k-1))/k)

        MAX_PWM_SAFE = 255  # aún se puede usar como máximo físico

        for i in range(1, len(t_custom)):
            dt = t_custom[i] - t_custom[i-1]
            Tc, Tm, Th = T[i-1,:]

            # Diferenciales 3-nodos
            P_cold = frac_cold * pwm_hist[i-1] * 0.095
            P_hot  = frac_hot  * pwm_hist[i-1] * 0.095

            dTc = ((Tm - Tc)/R1 - P_cold)/Cc * dt
            dTm = ((Tc - Tm)/R1 + (Th - Tm)/R2)/Cp * dt
            dTh = ((Tm - Th)/R2 + P_hot - (Th - 25)/Rconv)/Ch * dt  # 25°C como T_amb
            T[i,:] = [Tc + dTc, Tm + dTm, Th + dTh]

            # FOPID control
            e = T[i,0] - (T_set - bias)
            e_hist[1:] = e_hist[:-1]
            e_hist[0] = e

            sumI = np.dot(wI, e_hist)
            sumD = np.dot(wD, e_hist)
            sumD_reg = sumD / (1 + abs(sumD))

            u = Kp*e + Ki*(Ts**lam)*sumI + Kd*(sumD_reg/(Ts**mu))

            # Rampa inicial 0.2
            factor_rampa = 1.0
            if t_custom[i] < 5.0:
                factor_rampa = 0.2

            pwm_new = pwm_hist[i-1] + (u - pwm_hist[i-1]) * dt / 30.0 * factor_rampa
            pwm_hist[i] = np.clip(pwm_new, 0, MAX_PWM_SAFE)

        return T[:,0], pwm_hist