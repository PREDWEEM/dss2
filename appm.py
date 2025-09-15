
# predweem_app/app.py — versión completa sin A2, basada solo en densidad efectiva x

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="PREDWEEM · Densidad efectiva (x)", layout="wide")
st.title("PREDWEEM · Supresión + Control sobre densidad efectiva (x)")
st.caption("Versión simplificada sin A2. Toda la lógica basada en densidad efectiva: x₂ (supresión), x₃ (supresión+control)")

# === Entrada simulada ===
st.sidebar.header("Parámetros")
dias = 100
fechas = pd.date_range("2025-06-01", periods=dias)
EMERREL = np.linspace(0.01, 0.05, dias)  # curva ejemplo
FC_state = np.where(np.arange(dias) < 30, 0.3, 0.6)  # fenología simplificada
Ciec = np.linspace(0.2, 0.8, dias)  # competencia del cultivo
control_factor = np.ones(dias)
control_factor[40:70] = 0.5  # control parcial entre día 40–70

MAX_PLANTS_CAP = 850
AUC = np.trapz(EMERREL, dx=1)
factor_area_to_plants = MAX_PLANTS_CAP / AUC if AUC > 0 else None

# === Cálculos densidad efectiva ===
emerrel_eff_base = EMERREL * FC_state                        # x1: sin supresión ni control
emerrel_eff_x2 = emerrel_eff_base * (1 - Ciec)               # x2: supresión
emerrel_eff_x3 = emerrel_eff_x2 * control_factor             # x3: supresión + control

plm2dia_x2 = emerrel_eff_x2 * factor_area_to_plants
plm2dia_x3 = emerrel_eff_x3 * factor_area_to_plants

X2 = np.sum(plm2dia_x2)
X3 = np.sum(plm2dia_x3)

def perdida_rinde_pct(x):
    x = np.asarray(x, dtype=float)
    return 0.375 * x / (1.0 + (0.375 * x / 76.639))

loss_x2_pct = perdida_rinde_pct(X2)
loss_x3_pct = perdida_rinde_pct(X3)

# === Visualización ===
st.subheader("Pérdida de rendimiento estimada (%) — por densidad efectiva (x)")
st.markdown(
    f'''
### x₂ — Con supresión (sin control)  
x = **{X2:,.1f}** pl·m² → pérdida estimada: **{loss_x2_pct:.2f}%**

### x₃ — Con supresión + control  
x = **{X3:,.1f}** pl·m² → pérdida estimada: **{loss_x3_pct:.2f}%**
''')

# === Gráfico de pérdida ===
x_vals = np.linspace(0, MAX_PLANTS_CAP, 500)
y_vals = perdida_rinde_pct(x_vals)

fig, ax = plt.subplots()
ax.plot(x_vals, y_vals, label="Modelo pérdida (%)")
ax.plot(X2, loss_x2_pct, "ro", label="x₂: supresión")
ax.plot(X3, loss_x3_pct, "go", label="x₃: supresión+control")
ax.set_xlabel("x (pl·m²)")
ax.set_ylabel("Pérdida de rendimiento (%)")
ax.set_title("Pérdida de rendimiento vs x")
ax.legend()
st.pyplot(fig)

# === Exportar CSV ===
df_out = pd.DataFrame({
    "fecha": fechas,
    "EMERREL": EMERREL,
    "FC_state": FC_state,
    "Ciec": Ciec,
    "control_factor": control_factor,
    "plm2dia_x2": plm2dia_x2,
    "plm2dia_x3": plm2dia_x3
})
st.download_button("Descargar resultados (CSV)", df_out.to_csv(index=False).encode("utf-8"),
                   "resultados_x2_x3.csv", "text/csv")
