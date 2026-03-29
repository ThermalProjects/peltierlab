import numpy as np

class SimulatorHysteresisReal:
    """
    Simulación de 3 nodos (cold, mid, hot) con control ON/OFF tipo histéresis.
    """

    def __init__(self, params, T_start=19.0, T_amb=25.0):
        # Parámetros 3-nodos reales
        self.R1, self.R2, self.Rconv, self.frac_cold, self.Cc, self.Cp, self.Ch, self.tau = params
        self.frac_hot = 1 - self.frac_cold

        # Condiciones iniciales
        self.T_start = T_start
        self.T_amb = T_amb

    def simulate(self, t_custom, T_set=12.0, dT1=0.5, dT2=0.5, P_max=1.0):
        """
        t_custom: vector de tiempo [s]
        T_set: consigna de temperatura [°C]
        dT1, dT2: histéresis [°C]
        P_max: potencia máxima relativa (0-1)
        """
        N = len(t_custom)
        T = np.zeros((N,3))  # Tc, Tm, Th
        T[0,:] = self.T_start, self.T_start, self.T_start

        pwm = np.zeros(N, dtype=float)
        peltier_on = True  # iniciar encendido

        for i in range(1,N):
            dt = t_custom[i] - t_custom[i-1]
            Tc, Tm, Th = T[i-1,:]

            # Histéresis ON/OFF
            if peltier_on:
                if Tc <= T_set - dT2:
                    peltier_on = False
            else:
                if Tc >= T_set + dT1:
                    peltier_on = True

            P_cold = pwm[i-1] * self.frac_cold * P_max
            P_hot  = pwm[i-1] * self.frac_hot * P_max
            pwm[i] = P_max if peltier_on else 0.0

            # Diferenciales 3-nodos
            dTc = ((Tm - Tc)/self.R1 - P_cold)/self.Cc * dt
            dTm = ((Tc - Tm)/self.R1 + (Th - Tm)/self.R2)/self.Cp * dt
            dTh = ((Tm - Th)/self.R2 + P_hot - (Th - self.T_amb)/self.Rconv)/self.Ch * dt

            Tc_new = Tc + dTc
            Tm_new = Tm + dTm
            Th_new = Th + dTh

            T[i,:] = [Tc_new, Tm_new, Th_new]

        Tc = T[:,0]
        Tm = T[:,1]
        Th = T[:,2]

        return Tc, Tm, Th, pwm