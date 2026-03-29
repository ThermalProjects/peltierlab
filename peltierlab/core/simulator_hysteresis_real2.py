import numpy as np

class SimulatorHysteresisReal:

    def __init__(self, params, T_start=19.0, T_amb=25.0):
        self.R1, self.R2, self.Rconv, self.frac_cold, self.Cc, self.Cp, self.Ch, self.tau = params
        self.frac_hot = 1 - self.frac_cold

        self.T_start = T_start
        self.T_amb = T_amb

    def simulate(self, t_custom, T_set=12.0, dT1=0.5, dT2=0.5, P_max=8.0):

        N = len(t_custom)
        T = np.zeros((N,3))  # Tc, Tm, Th
        T[0,:] = self.T_start, self.T_start, self.T_start

        pwm = np.zeros(N)
        peltier_on = True

        for i in range(1, N):

            dt = t_custom[i] - t_custom[i-1]
            Tc, Tm, Th = T[i-1,:]

            # ===== HISTÉRESIS REAL =====
            if peltier_on:
                if Tc <= T_set - dT2:
                    peltier_on = False
            else:
                if Tc >= T_set + dT1:
                    peltier_on = True

            pwm[i] = 1.0 if peltier_on else 0.0

            # ===== POTENCIA =====
            P_cold = pwm[i] * self.frac_cold * P_max
            P_hot  = pwm[i] * self.frac_hot  * P_max

            # ===== DINÁMICA 3 NODOS =====
            dTc = ((Tm - Tc)/self.R1 - P_cold)/self.Cc * dt
            dTm = ((Tc - Tm)/self.R1 + (Th - Tm)/self.R2)/self.Cp * dt
            dTh = ((Tm - Th)/self.R2 + P_hot - (Th - self.T_amb)/self.Rconv)/self.Ch * dt

            T[i,0] = Tc + dTc
            T[i,1] = Tm + dTm
            T[i,2] = Th + dTh

        return T[:,0], T[:,1], T[:,2], pwm