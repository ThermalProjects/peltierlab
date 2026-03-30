# Dentro del loop de simulación (reemplaza la sección existente de 4 FPS)
if st.session_state['running_state']:
    y_data = []
    t_data = []
    fps = 4
    interval = 1.0 / fps
    start_time = time.time()

    for i in range(len(t_full)):
        # Espera hasta que sea el momento real del segundo
        while time.time() - start_time < t_full[i]:
            # Pausa breve para permitir otras tareas
            time.sleep(0.01)

        # Añadir el punto actual
        y_data.append(Tc_full[i])
        t_data.append(t_full[i])

        # Interpolación lineal para suavizar la línea entre FPS
        if i > 0:
            t_interp = np.linspace(t_data[-2], t_data[-1], fps)
            y_interp = np.linspace(y_data[-2], y_data[-1], fps)
        else:
            t_interp = [t_data[-1]]
            y_interp = [y_data[-1]]

        # Actualiza línea punto a punto para suavidad
        for t_i, y_i in zip(t_interp, y_interp):
            line.set_data(list(t_data[:-1]) + [t_i], list(y_data[:-1]) + [y_i])
            ax.set_xlim(0, duration)
            plot_placeholder.pyplot(fig)
            time.sleep(interval)

        # Actualiza métricas y PWM
        elapsed_placeholder.markdown(f"**Time elapsed:** {int(t_full[i])} s")
        pwm_placeholder.markdown(f"**PWM:** {pwm_full[i]:.1f}")
        pwm_bar.progress(int(pwm_full[i]/255*100))

        error = np.array(y_data) - T_set
        ss_error = np.mean(error[-50:]) if len(error) > 50 else np.mean(error)
        rmse = np.sqrt(np.mean(error**2))
        settling_time = next((t_data[j] for j in range(len(y_data)) if np.all(np.abs(error[j:]) <= 0.5)), None)

        recs = []
        if abs(ss_error) > 0.5:
            recs.append("Consider tuning controller to reduce steady-state error.")
        if settling_time is None or settling_time > 150:
            recs.append("Slow response → consider increasing gains for faster settling.")
        metrics_text.markdown(
            f"**Steady-state error:** {ss_error:.3f} °C  \n"
            f"**RMSE:** {rmse:.3f}  \n"
            f"**Settling time:** {settling_time if settling_time else 'Not reached'} s  \n"
            + ("\n".join(f"- {r}" for r in recs))
        )
