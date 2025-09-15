# -*- coding: utf-8 -*-
# app.py — PREDWEEM · Supresión (EMERREL × (1−Ciec)) + Control (NR=10d) + Fenología (S1..S4)
# - Sin ICIC
# - Ciec desde canopia (FC/LAI), con curva opcional
# - Equivalencia por área: AUC[EMERREL (cruda) desde siembra] ≙ MAX_PLANTS_CAP (elegible: 850/500/250/100 pl·m²)
# - A2 = MAX_PLANTS_CAP * ( AUC[supresión] / AUC[cruda] )
# - A2_ctrl = MAX_PLANTS_CAP * ( AUC[supresión×control] / AUC[cruda] )
# - Fenología (Avena fatua) con FC fijos: S1=0.0, S2=0.3, S3=0.6, S4=1.0
#   Asignación por **edad al Periodo Crítico (PC)** con bordes: S2=7–27d; S3=28–59d; S4=≥60d; S1=resto
# - x = ∫ (pl·m²·día⁻¹_ctrl × FC_estado_en_PC) dt, desde siembra
# - Gráficos: EMERREL/Supresión/Control; A2 acumulado y x acumulado; Pérdida % vs A2/x

import io, re, json, math, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from datetime import timedelta

APP_TITLE = "PREDWEEM · Supresión (1−Ciec) + Control (AUC) + Fenología"
st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
st.title(APP_TITLE)
st.caption("AUC de EMERREL (cruda) ≙ tope A2 (850/500/250/100 pl·m²). A2, x (fenología) y pérdidas usan esa escala. Sin ICIC.")

# ========================== Constantes y helpers ==========================
NR_DAYS_DEFAULT = 10

def safe_nanmax(arr, fallback=0.0):
    try:
        val = np.nanmax(arr)
        if np.isfinite(val):
            return float(val)
        return float(fallback)
    except Exception:
        return float(fallback)

def sniff_sep_dec(text: str):
    sample = text[:8000]
    counts = {sep: sample.count(sep) for sep in [",", ";", "\t"]}
    sep_guess = max(counts, key=counts.get) if counts else ","
    dec_guess = "."
    if sample.count(",") > sample.count(".") and re.search(r",\d", sample):
        dec_guess = ","
    return sep_guess, dec_guess

@st.cache_data(show_spinner=False)
def read_raw_from_url(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=30) as r:
        return r.read()

def read_raw(up, url):
    if up is not None:
        return up.read()
    if url:
        return read_raw_from_url(url)
    raise ValueError("No hay fuente de datos.")

def parse_csv(raw, sep_opt, dec_opt, encoding="utf-8", on_bad="warn"):
    head = raw[:8000].decode("utf-8", errors="ignore")
    sep_guess, dec_guess = sniff_sep_dec(head)
    sep = sep_guess if sep_opt == "auto" else ("," if sep_opt=="," else (";" if sep_opt==";" else "\t"))
    dec = dec_guess if dec_opt == "auto" else dec_opt
    df = pd.read_csv(io.BytesIO(raw), sep=sep, decimal=dec, engine="python",
                     encoding=encoding, on_bad_lines=on_bad)
    return df, {"sep": sep, "dec": dec, "enc": encoding}

def clean_numeric_series(s: pd.Series, decimal="."):
    if s.dtype.kind in "if":
        return pd.to_numeric(s, errors="coerce")
    t = s.astype(str).str.strip().str.replace("%", "", regex=False)
    if decimal == ",":
        t = t.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    else:
        t = t.str.replace(",", "", regex=False)
    return pd.to_numeric(t, errors="coerce")

def moving_average(x: pd.Series, win=5):
    return x.rolling(win, center=True, min_periods=1).mean()

def _to_days(ts: pd.Series) -> np.ndarray:
    f = pd.to_datetime(ts)
    t = f.view("int64") / 1e9  # segundos
    t = (t - t.iloc[0]) / 86400.0  # días
    return t.to_numpy(dtype=float)

def auc_time(fecha: pd.Series, y: np.ndarray, mask=None) -> float:
    f = pd.to_datetime(fecha)
    y_arr = np.asarray(y, dtype=float)
    if mask is not None:
        f = f[mask]
        y_arr = y_arr[mask]
    if len(f) < 2:
        return 0.0
    tdays = _to_days(f)
    y_arr = np.nan_to_num(y_arr, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.trapz(y_arr, tdays))

def cumulative_auc_series(fecha: pd.Series, y: np.ndarray, mask=None) -> pd.Series:
    f = pd.to_datetime(fecha)
    y_arr = np.asarray(y, dtype=float)
    if mask is not None:
        f = f[mask]
        y_arr = y_arr[mask]
    if len(f) == 0:
        return pd.Series(dtype=float)
    t = _to_days(f)
    y_arr = np.nan_to_num(y_arr, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.zeros_like(y_arr, dtype=float)
    for i in range(1, len(y_arr)):
        dt_i = max(0.0, t[i] - t[i-1])
        out[i] = out[i-1] + 0.5 * (y_arr[i-1] + y_arr[i]) * dt_i
    return pd.Series(out, index=f)

# ======================= Canopia (sin ICIC): FC y LAI =================
# (requerida por compute_canopy -> FC/LAI y Ciec)

def compute_canopy(
    fechas: pd.Series,
    sow_date: dt.date,
    mode_canopy: str,
    t_lag: int,
    t_close: int,
    cov_max: float,
    lai_max: float,
    k_beer: float,
):
    """Devuelve (FC, LAI) estimados para el cultivo.
    - Si mode_canopy == "Cobertura dinámica (%)": crece FC con logística de t_lag a t_close (cap cov_max),
      y se deriva LAI por Beer–Lambert.
    - Caso contrario: crece LAI con logística (cap lai_max) y se proyecta FC por Beer–Lambert.
    """
    days_since_sow = np.array([(pd.Timestamp(d).date() - sow_date).days for d in fechas], dtype=float)

    def logistic_between(days, start, end, y_max):
        if end <= start:
            end = start + 1
        t_mid = 0.5 * (start + end)
        r = 4.0 / max(1.0, (end - start))
        return y_max / (1.0 + np.exp(-r * (days - t_mid)))

    if mode_canopy == "Cobertura dinámica (%)":
        fc_dyn = np.where(
            days_since_sow < t_lag, 0.0,
            logistic_between(days_since_sow, t_lag, t_close, cov_max/100.0)
        )
        fc_dyn = np.clip(fc_dyn, 0.0, 1.0)
        LAI = -np.log(np.clip(1.0 - fc_dyn, 1e-9, 1.0)) / max(1e-6, k_beer)
        LAI = np.clip(LAI, 0.0, lai_max)
    else:
        LAI = np.where(
            days_since_sow < t_lag, 0.0,
            logistic_between(days_since_sow, t_lag, t_close, lai_max)
        )
        LAI = np.clip(LAI, 0.0, lai_max)
        fc_dyn = 1 - np.exp(-k_beer * LAI)
        fc_dyn = np.clip(fc_dyn, 0.0, 1.0)
    return fc_dyn, LAI

# ========================= Sidebar: datos base =========================
with st.sidebar:
    st.header("Datos de entrada")
    up = st.file_uploader("CSV (fecha, EMERREL diaria o EMERAC)", type=["csv"])
    url = st.text_input("…o URL raw de GitHub", placeholder="https://raw.githubusercontent.com/usuario/repo/main/emer.csv")
    sep_opt = st.selectbox("Delimitador", ["auto", ",", ";", "\\t"], index=0)
    dec_opt = st.selectbox("Decimal", ["auto", ".", ","], index=0)
    dayfirst = st.checkbox("Fecha: día/mes/año (dd/mm/yyyy)", value=True)
    is_cumulative = st.checkbox("Mi CSV es acumulado (EMERAC)", value=False)
    as_percent = st.checkbox("Valores en % (no 0–1)", value=True)
    dedup = st.selectbox("Si hay fechas duplicadas…", ["sumar", "promediar", "primera"], index=0)
    fill_gaps = st.checkbox("Rellenar días faltantes con 0", value=False)

if up is None and not url:
    st.info("Subí un CSV o pegá una URL para continuar.")
    st.stop()

# ========================= Carga y parseo CSV ==========================
try:
    raw = read_raw(up, url)
    if not raw or len(raw) == 0:
        st.error("El archivo/URL está vacío."); st.stop()
    df0, meta = parse_csv(raw, sep_opt, dec_opt)
    if df0.empty:
        st.error("El CSV no tiene filas."); st.stop()
    st.success(f"CSV leído. sep='{meta['sep']}' dec='{meta['dec']}' enc='{meta['enc']}'")
except (URLError, HTTPError) as e:
    st.error(f"No se pudo acceder a la URL: {e}"); st.stop()
except Exception as e:
    st.error(f"No se pudo leer el CSV: {e}"); st.stop()

st.subheader("Vista previa (primeras 15 filas del CSV)")
st.dataframe(df0.head(15), use_container_width=True)

# ===================== Selección de columnas ===========================
cols = list(df0.columns)
with st.expander("Seleccionar columnas y depurar datos", expanded=True):
    c_fecha = st.selectbox("Columna de fecha", cols, index=0)
    c_valor = st.selectbox("Columna de valor (EMERREL diaria o EMERAC)", cols, index=1 if len(cols)>1 else 0)

    fechas = pd.to_datetime(df0[c_fecha], dayfirst=dayfirst, errors="coerce")
    sample_str = df0[c_valor].astype(str).head(200).str.cat(sep=" ")
    dec_for_col = "," if (sample_str.count(",")>sample_str.count(".") and re.search(r",\d", sample_str)) else "."
    vals = clean_numeric_series(df0[c_valor], decimal=dec_for_col)

    df = pd.DataFrame({"fecha": fechas, "valor": vals}).dropna().sort_values("fecha").reset_index(drop=True)
    if df.empty:
        st.error("Tras el parseo no quedaron filas válidas (fechas/valores NaN)."); st.stop()

    if df["fecha"].duplicated().any():
        if dedup == "sumar":
            df = df.groupby("fecha").sum(numeric_only=True).rename_axis("fecha").reset_index()
        elif dedup == "promediar":
            df = df.groupby("fecha").mean(numeric_only=True).rename_axis("fecha").reset_index()
        else:
            df = df.drop_duplicates(subset=["fecha"], keep="first")

    if fill_gaps and len(df) > 1:
        full_idx = pd.date_range(df["fecha"].min(), df["fecha"].max(), freq="D")
        df = df.set_index("fecha").reindex(full_idx).rename_axis("fecha").reset_index()
        df["valor"] = df["valor"].fillna(0.0)

    emerrel = df["valor"].astype(float)
    if as_percent:
        emerrel = emerrel / 100.0
    if is_cumulative:
        emerrel = emerrel.diff().fillna(0.0).clip(lower=0.0)
    emerrel = emerrel.clip(lower=0.0)
    df_plot = pd.DataFrame({"fecha": pd.to_datetime(df["fecha"]), "EMERREL": emerrel})

# ==================== Siembra & parámetros de canopia ==================
years = df_plot["fecha"].dt.year.dropna().astype(int)
year_ref = int(years.mode().iloc[0]) if len(years) else dt.date.today().year
sow_min = dt.date(year_ref, 5, 1); sow_max = dt.date(year_ref, 8, 1)

with st.sidebar:
    st.header("Siembra & Canopia (para Ciec)")
    st.caption(f"Ventana de siembra: **{sow_min} → {sow_max}** (1 de mayo al 1 de agosto)")
    sow_date = st.date_input("Fecha de siembra", value=sow_min, min_value=sow_min, max_value=sow_max)
    mode_canopy = st.selectbox("Canopia", ["Cobertura dinámica (%)", "LAI dinámico"], index=0)
    t_lag = st.number_input("Días a emergencia del cultivo (lag)", 0, 60, 7, 1)
    t_close = st.number_input("Días a cierre de entresurco", 10, 120, 45, 1)
    cov_max = st.number_input("Cobertura máxima (%)", 10.0, 100.0, 85.0, 1.0)
    lai_max = st.number_input("LAI máximo", 0.0, 8.0, 3.5, 0.1)
    k_beer = st.number_input("k (Beer–Lambert)", 0.1, 1.2, 0.6, 0.05)

# ========================= Sidebar: Ciec y TOPE ========================
with st.sidebar:
    st.header("Ciec (competencia del cultivo)")
    use_ciec = st.checkbox("Calcular y mostrar Ciec", value=True)
    Ca = st.number_input("Densidad real Ca (pl/m²)", 50, 700, 250, 10)
    Cs = st.number_input("Densidad estándar Cs (pl/m²)", 50, 700, 250, 10)
    LAIhc = st.number_input("LAIhc (escenario altamente competitivo)", 0.5, 10.0, 3.5, 0.1)

    st.header("Tope A2 (pl·m²)")
    cap_choice = st.selectbox("Elegí el tope para A2", [850, 500, 250, 100], index=0)
MAX_PLANTS_CAP = float(cap_choice)
DTICK_RIGHT = {850: 170, 500: 100, 250: 50, 100: 20}.get(int(MAX_PLANTS_CAP), max(10, round(MAX_PLANTS_CAP/5)))

# ========================= Periodo crítico (PC) ========================
with st.sidebar:
    st.header("Periodo crítico")
    st.caption("Ventana **11 de septiembre → 15 de noviembre** del año de referencia (solo resalta, no recorta ejes).")
    use_pc = st.checkbox("Resaltar periodo crítico", value=False)
    ref_pc = st.selectbox("Referencia de edad para clasificación", ["Punto medio", "11-Sep", "15-Nov"], index=0)

year_pc = int(sow_date.year if sow_date else (years.mode().iloc[0] if len(years) else dt.date.today().year))
PC_START = pd.to_datetime(f"{year_pc}-09-11")
PC_END   = pd.to_datetime(f"{year_pc}-11-15")
if ref_pc == "Punto medio":
    PC_REF = PC_START + (PC_END - PC_START)/2
elif ref_pc == "11-Sep":
    PC_REF = PC_START
else:
    PC_REF = PC_END

# =========== Etiquetas/visual + avanzadas =============================
with st.sidebar:
    st.header("Etiquetas y escalas")
    show_plants_axis = st.checkbox("Mostrar Plantas·m²·día⁻¹ (eje derecho)", value=True)
    show_ciec_curve = st.checkbox("Mostrar curva Ciec (0–1)", value=True)
    show_nonres_bands = st.checkbox("Marcar bandas NR (por defecto 10 días)", value=True)

if not (sow_min <= sow_date <= sow_max):
    st.error("La fecha de siembra debe estar entre el 1 de mayo y el 1 de agosto."); st.stop()

# ============================ FC/LAI + Ciec ===========================
FC, LAI = compute_canopy(
    fechas=df_plot["fecha"], sow_date=sow_date, mode_canopy=mode_canopy,
    t_lag=int(t_lag), t_close=int(t_close),
    cov_max=float(cov_max), lai_max=float(lai_max), k_beer=float(k_beer),
)

if use_ciec:
    Ca_safe = float(Ca) if float(Ca) > 0 else 1e-6
    Ciec = (LAI / max(1e-6, float(LAIhc))) * (float(Cs) / Ca_safe)
    Ciec = np.clip(Ciec, 0.0, 1.0)
else:
    Ciec = np.zeros_like(LAI, dtype=float)

df_ciec = pd.DataFrame({"fecha": df_plot["fecha"], "Ciec": Ciec})

# ===== Supresión (base agronómica) ===================================
ts = pd.to_datetime(df_plot["fecha"])  # timestamps
emerrel_supresion = (df_plot["EMERREL"].astype(float).values * (1.0 - Ciec)).clip(min=0.0)

# ================== AUC y factor de equivalencia por ÁREA =============
mask_after_sow = ts.dt.date >= sow_date
auc_cruda = auc_time(ts, df_plot["EMERREL"].to_numpy(dtype=float), mask=mask_after_sow)  # (EMERREL)·día
auc_sup   = auc_time(ts, emerrel_supresion, mask=mask_after_sow)

if auc_cruda > 0:
    factor_area_to_plants = MAX_PLANTS_CAP / auc_cruda  # [pl·m²] por (EMERREL·día)
    conv_caption = (
        f"AUC(EMERREL cruda desde siembra) = {auc_cruda:.4f} → {MAX_PLANTS_CAP:.0f} pl·m² "
        f"(factor={factor_area_to_plants:.4f} pl·m² por EMERREL·día)"
    )
else:
    factor_area_to_plants = None
    conv_caption = "No se pudo escalar por área (AUC de EMERREL cruda = 0)."

# =================== Manejo (control) y decaimientos ===================
sched_rows = []

def add_sched(nombre, fecha_ini, dias_res=None, nota=""):
    if not fecha_ini: return
    fin = (pd.to_datetime(fecha_ini) + pd.Timedelta(days=int(dias_res))).date() if dias_res else None
    sched_rows.append({"Intervención": nombre, "Inicio": str(fecha_ini), "Fin": str(fin) if fin else "—", "Nota": nota})

with st.sidebar:
    st.header("Manejo pre-siembra (manual)")
    min_date = ts.min().date(); max_date = ts.max().date()
    pre_glifo = st.checkbox("Herbicida total (glifosato)", value=False)
    pre_glifo_date = st.date_input("Fecha glifosato (pre)", value=min_date, min_value=min_date, max_value=max_date, disabled=not pre_glifo)

    pre_selNR = st.checkbox("Selectivo no residual (pre)", value=False)
    pre_selNR_date = st.date_input("Fecha selectivo no residual (pre)", value=min_date, min_value=min_date, max_value=max_date, disabled=not pre_selNR)

    pre_selR  = st.checkbox("Selectivo + residual (pre)", value=False)
    pre_res_dias = st.slider("Residualidad pre (días)", 30, 60, 45, 1, disabled=not pre_selR)
    pre_selR_date = st.date_input("Fecha selectivo + residual (pre)", value=min_date, min_value=min_date, max_value=max_date, disabled=not pre_selR)

    st.header("Manejo post-emergencia (manual)")
    post_gram = st.checkbox("Selectivo graminicida (post)", value=False)
    post_gram_date = st.date_input("Fecha graminicida (post)", value=max(min_date, sow_date), min_value=min_date, max_value=max_date, disabled=not post_gram)

    post_selR = st.checkbox("Selectivo + residual (post)", value=False)
    post_res_dias = st.slider("Residualidad post (días)", 30, 60, 45, 1, disabled=not post_selR)
    post_selR_date = st.date_input("Fecha selectivo + residual (post)", value=max(min_date, sow_date), min_value=min_date, max_value=max_date, disabled=not post_selR)

# Validaciones suaves
warnings = []

def check_pre(date_val, name):
    if date_val and date_val > sow_date:
        warnings.append(f"{name}: debería ser ≤ fecha de siembra ({sow_date}).")

def check_post(date_val, name):
    if date_val and date_val < sow_date:
        warnings.append(f"{name}: debería ser ≥ fecha de siembra ({sow_date}).")
if pre_glifo:  check_pre(pre_glifo_date, "Glifosato (pre)")
if pre_selNR:  check_pre(pre_selNR_date, "Selectivo no residual (pre)")
if pre_selR:   check_pre(pre_selR_date, "Selectivo + residual (pre)")
if post_gram:  check_post(post_gram_date, "Graminicida (post)")
if post_selR:  check_post(post_selR_date, "Selectivo + residual (post)")
for w in warnings: st.warning(w)

# Agendar (cronograma visual)
if pre_glifo: add_sched("Pre · glifosato (NSr, 1d)", pre_glifo_date, None, "Barbecho")
if pre_selNR: add_sched(f"Pre · selectivo no residual (NR)", pre_selNR_date, NR_DAYS_DEFAULT, f"NR por defecto {NR_DAYS_DEFAULT}d")
if pre_selR:  add_sched("Pre · selectivo + residual", pre_selR_date, pre_res_dias, f"Protege {pre_res_dias} días")
if post_gram:
    back_ini = (pd.to_datetime(post_gram_date) - pd.Timedelta(days=NR_DAYS_DEFAULT-1)).date()
    sched_rows.append({"Intervención": "Post · graminicida selectivo (NR, −10d)", "Inicio": str(back_ini), "Fin": str(post_gram_date), "Nota": f"Ventana hacia atrás {NR_DAYS_DEFAULT}d"})
if post_selR: add_sched("Post · selectivo + residual", post_selR_date, post_res_dias, f"Protege {post_res_dias}d")
sched = pd.DataFrame(sched_rows)

# ======================= Eficiencias y decaimientos ====================
with st.sidebar:
    st.header("Eficiencia de control (%)")
    st.caption("Reducción aplicada a EMERREL×(1−Ciec) dentro de la ventana de efecto.")
    ef_pre_glifo   = st.slider("Glifosato (pre, 1 día)", 0, 100, 90, 1) if pre_glifo else 0
    ef_pre_selNR   = st.slider(f"Selectivo no residual (pre, {NR_DAYS_DEFAULT}d)", 0, 100, 60, 1) if pre_selNR else 0
    ef_pre_selR    = st.slider("Selectivo + residual (pre, 30–60d)", 0, 100, 70, 1) if pre_selR else 0
    ef_post_gram   = st.slider(f"Graminicida selectivo (post, {NR_DAYS_DEFAULT}d)", 0, 100, 65, 1) if post_gram else 0
    ef_post_selR   = st.slider("Selectivo + residual (post, 30–60d)", 0, 100, 70, 1) if post_selR else 0

with st.sidebar:
    st.header("Decaimiento en residuales")
    decaimiento_tipo = st.selectbox("Tipo de decaimiento", ["Ninguno", "Lineal", "Exponencial"], index=0)
    if decaimiento_tipo == "Exponencial":
        half_life = st.number_input("Semivida (días) para exponencial", 1, 120, 20, 1)
        lam_exp = math.log(2) / max(1e-6, half_life)
    else:
        lam_exp = None

# =================== Factor de control diario =========================
fechas_d = ts.dt.date.values

def weights_one_day(date_val):
    if not date_val:
        return np.zeros_like(fechas_d, dtype=float)
    d0 = date_val
    return ((fechas_d >= d0) & (fechas_d < (d0 + timedelta(days=1)))).astype(float)

def weights_residual(start_date, dias):
    w = np.zeros_like(fechas_d, dtype=float)
    if (not start_date) or (not dias) or (int(dias) <= 0):
        return w
    d0 = start_date
    d1 = start_date + timedelta(days=int(dias))
    mask = (fechas_d >= d0) & (fechas_d < d1)
    if not mask.any():
        return w
    idxs = np.where(mask)[0]
    t_rel = np.arange(len(idxs), dtype=float)
    if decaimiento_tipo == "Ninguno":
        w[idxs] = 1.0
    elif decaimiento_tipo == "Lineal":
        L = max(1, len(idxs))
        w[idxs] = 1.0 - (t_rel / max(1.0, L - 1))
    else:  # Exponencial
        assert lam_exp is not None
        w[idxs] = np.exp(-lam_exp * t_rel)
    return w

def weights_backward(center_date, dias):
    """Ventana hacia atrás de longitud 'dias' que incluye el día de aplicación.
       Ej.: dias=10 → afecta [aplicación-9, …, aplicación]."""
    w = np.zeros_like(fechas_d, dtype=float)
    if (not center_date) or (not dias) or (int(dias) <= 0):
        return w
    d_center = center_date
    d0 = d_center - timedelta(days=int(dias)-1)
    d1 = d_center + timedelta(days=1)
    mask = (fechas_d >= d0) & (fechas_d < d1)
    w[mask] = 1.0
    return w

ctrl_factor = np.ones_like(fechas_d, dtype=float)

def apply_efficiency(weights, eff_pct):
    if eff_pct <= 0:
        return
    reduc = np.clip(1.0 - (eff_pct/100.0) * weights, 0.0, 1.0)
    np.multiply(ctrl_factor, reduc, out=ctrl_factor)

# Aplicar cada intervención
if pre_glifo: apply_efficiency(weights_one_day(pre_glifo_date), ef_pre_glifo)
if pre_selNR: apply_efficiency(weights_residual(pre_selNR_date, NR_DAYS_DEFAULT), ef_pre_selNR)
if pre_selR:  apply_efficiency(weights_residual(pre_selR_date, pre_res_dias), ef_pre_selR)
if post_gram: apply_efficiency(weights_backward(post_gram_date, NR_DAYS_DEFAULT), ef_post_gram)
if post_selR: apply_efficiency(weights_residual(post_selR_date, post_res_dias), ef_post_selR)

# ==================== Control sobre SUPRESIÓN ==========================
emerrel_supresion_ctrl = emerrel_supresion * ctrl_factor

# ========================= A2 por AUC (área) ===========================
auc_sup_ctrl = auc_time(ts, emerrel_supresion_ctrl, mask=mask_after_sow)

if factor_area_to_plants is not None:
    A2_sup_raw  = MAX_PLANTS_CAP * (auc_sup   / auc_cruda) if auc_cruda > 0 else float("nan")
    A2_ctrl_raw = MAX_PLANTS_CAP * (auc_sup_ctrl / auc_cruda) if auc_cruda > 0 else float("nan")
else:
    A2_sup_raw, A2_ctrl_raw = float("nan"), float("nan")

A2_sup_final  = min(MAX_PLANTS_CAP, A2_sup_raw)  if np.isfinite(A2_sup_raw)  else float("nan")
A2_ctrl_final = min(MAX_PLANTS_CAP, A2_ctrl_raw) if np.isfinite(A2_ctrl_raw) else float("nan")

# =========== Series instantáneas en plantas·m²·día⁻¹ (escala por área)
if factor_area_to_plants is not None:
    plantas_emerrel_cruda  = df_plot["EMERREL"].values * factor_area_to_plants
    plantas_supresion      = emerrel_supresion * factor_area_to_plants
    plantas_supresion_ctrl = emerrel_supresion_ctrl * factor_area_to_plants
else:
    plantas_emerrel_cruda  = np.full(len(df_plot), np.nan)
    plantas_supresion      = np.full(len(df_plot), np.nan)
    plantas_supresion_ctrl = np.full(len(df_plot), np.nan)

# ===================== Fenología S1..S4 (FC fijos) ====================
with st.sidebar:
    st.header("Fenología de la maleza (Avena fatua)")
    st.caption("FC fijos por estado: S1=0.0, S2=0.3, S3=0.6, S4=1.0. Asignación por edad al PC.")

fechas_series = ts.dt.date.values
age_at_pc = np.array([(PC_REF.date() - d).days for d in fechas_series], dtype=float)
mask_S2 = (age_at_pc >= 7)  & (age_at_pc <= 27)
mask_S3 = (age_at_pc >= 28) & (age_at_pc <= 59)
mask_S4 = (age_at_pc >= 60)
mask_S1 = ~(mask_S2 | mask_S3 | mask_S4)

FC_state = np.zeros_like(age_at_pc, dtype=float)  # S1=0.0
FC_state[mask_S2] = 0.3
FC_state[mask_S3] = 0.6
FC_state[mask_S4] = 1.0

# Densidad efectiva diaria ponderada por estado (según edad al PC)
if factor_area_to_plants is not None:
    plm2dia_ctrl_eff = plantas_supresion_ctrl * FC_state
    X_eff_pc = float(np.nansum(plm2dia_ctrl_eff[mask_after_sow]))
else:
    plm2dia_ctrl_eff = np.full(len(ts), np.nan)
    X_eff_pc = float("nan")

# =========== Curvas acumuladas (A2 y x) ==========
if factor_area_to_plants is not None:
    auc_cum_sup  = cumulative_auc_series(ts, emerrel_supresion, mask=mask_after_sow)
    auc_cum_ctrl = cumulative_auc_series(ts, emerrel_supresion_ctrl, mask=mask_after_sow)
    A2_cum_sup   = auc_cum_sup  * factor_area_to_plants
    A2_cum_ctrl  = auc_cum_ctrl * factor_area_to_plants
    A2_cum_sup_cap  = A2_cum_sup.clip(upper=MAX_PLANTS_CAP)
    A2_cum_ctrl_cap = A2_cum_ctrl.clip(upper=MAX_PLANTS_CAP)
    X_cum_eff = np.cumsum(plm2dia_ctrl_eff)
    X_cum_eff_cap = np.clip(X_cum_eff, None, MAX_PLANTS_CAP)
else:
    A2_cum_sup_cap = pd.Series(dtype=float)
    A2_cum_ctrl_cap = pd.Series(dtype=float)
    X_cum_eff_cap = np.array([])

# ============================== Gráfico 1 ==============================
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=ts, y=df_plot["EMERREL"], mode="lines",
    name="EMERREL (cruda)",
    hovertemplate="Fecha: %{x|%Y-%m-%d}<br>EMERREL: %{y:.4f}<extra></extra>"
))

fig.add_trace(go.Scatter(
    x=ts, y=emerrel_supresion, mode="lines",
    name="EMERREL × (1 − Ciec)", line=dict(dash="dashdot"),
    customdata=np.column_stack([plantas_supresion]),
    hovertemplate="Fecha: %{x|%Y-%m-%d}<br>Supresión: %{y:.4f}<br>pl·m²·día⁻¹ (sup.): %{customdata[0]:.2f}<extra></extra>"
))

fig.add_trace(go.Scatter(
    x=ts, y=emerrel_supresion_ctrl, mode="lines",
    name="EMERREL × (1 − Ciec) (control)", line=dict(dash="dot", width=2),
    customdata=np.column_stack([plantas_supresion_ctrl]),
    hovertemplate="Fecha: %{x|%Y-%m-%d}<br>Supresión (ctrl): %{y:.4f}<br>pl·m²·día⁻¹ (sup. ctrl): %{customdata[0]:.2f}<extra></extra>"
))

# Bandas de manejo (pre/post)

def _add_label(center_ts, text, bgcolor, y=0.94):
    fig.add_annotation(x=center_ts, y=y, xref="x", yref="paper",
        text=text, showarrow=False, bgcolor=bgcolor, opacity=0.9,
        bordercolor="rgba(0,0,0,0.2)", borderwidth=1, borderpad=2)

def add_residual_band(start_date, days, label):
    if start_date is None or days is None: return
    try:
        d_int = int(days)
        if d_int <= 0: return
        x0 = pd.to_datetime(start_date); x1 = x0 + pd.Timedelta(days=d_int)
        fig.add_vrect(x0=x0, x1=x1, line_width=0, fillcolor="PaleVioletRed", opacity=0.15)
        _add_label(x0 + (x1 - x0)/2, label, "rgba(219,112,147,0.8)")
    except Exception:
        return

def add_one_day_band(date_val, label):
    if date_val is None: return
    try:
        x0 = pd.to_datetime(date_val); x1 = x0 + pd.Timedelta(days=1)
        fig.add_vrect(x0=x0, x1=x1, line_width=0, fillcolor="Gold", opacity=0.25)
        _add_label(x0 + (x1 - x0)/2, label, "rgba(255,215,0,0.85)")
    except Exception:
        return

if pre_selR:  add_residual_band(pre_selR_date, pre_res_dias, f"Residual pre {pre_res_dias}d")
if post_selR: add_residual_band(post_selR_date, post_res_dias, f"Residual post {post_res_dias}d")
if show_nonres_bands:
    if pre_glifo: add_one_day_band(pre_glifo_date, "Glifo (1d)")
    if pre_selNR: add_residual_band(pre_selNR_date, NR_DAYS_DEFAULT, f"Sel. NR ({NR_DAYS_DEFAULT}d)")
    if post_gram:
        x_app = pd.to_datetime(post_gram_date)
        x0 = x_app - pd.Timedelta(days=NR_DAYS_DEFAULT-1)
        x1 = x_app + pd.Timedelta(days=1)
        fig.add_vrect(x0=x0, x1=x1, line_width=0, fillcolor="LightGreen", opacity=0.25)
        _add_label(x0 + (x1 - x0)/2, f"Graminicida (−{NR_DAYS_DEFAULT}d)", "rgba(144,238,144,0.85)")

# Resaltar Periodo Crítico
if use_pc:
    fig.add_vrect(x0=PC_START, x1=PC_END, line_width=0, fillcolor="MediumPurple", opacity=0.12)
    fig.add_annotation(x=PC_START + (PC_END-PC_START)/2, y=1.04, xref="x", yref="paper",
                       text="Periodo crítico", showarrow=False, bgcolor="rgba(147,112,219,0.85)",
                       bordercolor="rgba(0,0,0,0.2)", borderwidth=1, borderpad=2)

# Ejes y escalas
ymax = max(
    1e-6,
    1.15 * max(
        safe_nanmax(df_plot["EMERREL"].values, 0.0),
        safe_nanmax(emerrel_supresion, 0.0),
        safe_nanmax(emerrel_supresion_ctrl, 0.0),
    )
)

layout_kwargs = dict(
    margin=dict(l=10, r=10, t=40, b=10),
    title="EMERREL + Supresión (1−Ciec) + Control" + (" + Ciec" if use_ciec else ""),
    xaxis_title="Fecha",
    yaxis_title="EMERREL / Supresión",
    yaxis=dict(range=[0, ymax])
)

# Eje derecho opcional: Plantas·m²·día⁻¹ (escala por AUC)
if show_plants_axis and (factor_area_to_plants is not None) and np.isfinite(factor_area_to_plants):
    candidatos = [
        safe_nanmax(plantas_supresion, MAX_PLANTS_CAP),
        safe_nanmax(plantas_supresion_ctrl, MAX_PLANTS_CAP),
        safe_nanmax(plantas_emerrel_cruda, MAX_PLANTS_CAP),
        MAX_PLANTS_CAP
    ]
    plantas_max = float(np.nanmax(candidatos))
    if not np.isfinite(plantas_max) or plantas_max <= 0:
        plantas_max = MAX_PLANTS_CAP
    layout_kwargs["yaxis2"] = dict(
        overlaying="y",
        side="right",
        title="Plantas·m²·día⁻¹ (escala por AUC)",
        position=1.0,
        range=[0, max(plantas_max * 1.15, MAX_PLANTS_CAP * 1.15)],
        tick0=0,
        dtick=DTICK_RIGHT
    )
    fig.add_trace(go.Scatter(
        x=ts, y=plantas_emerrel_cruda, name="EMERREL→pl·m²·día⁻¹ (cruda)",
        yaxis="y2", mode="lines", line=dict(width=1),
        hovertemplate="Fecha: %{x|%Y-%m-%d}<br>pl·m²·día⁻¹ (cruda): %{y:.2f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=ts, y=plantas_supresion, name="pl·m²·día⁻¹ × (1−Ciec)", yaxis="y2", mode="lines",
        hovertemplate="Fecha: %{x|%Y-%m-%d}<br>pl·m²·día⁻¹ (sup.): %{y:.2f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=ts, y=plantas_supresion_ctrl, name="pl·m²·día⁻¹ × (1−Ciec) (ctrl)", yaxis="y2", mode="lines",
        line=dict(dash="dot"),
        hovertemplate="Fecha: %{x|%Y-%m-%d}<br>pl·m²·día⁻¹ (sup. ctrl): %{y:.2f}<extra></extra>"
    ))

# Eje auxiliar y curva Ciec (opcional)
if use_ciec and show_ciec_curve:
    layout_kwargs["yaxis3"] = dict(
        overlaying="y", side="right", title="Ciec (0–1)", position=0.97, range=[0, 1]
    )
    fig.add_trace(go.Scatter(
        x=df_ciec["fecha"], y=df_ciec["Ciec"], mode="lines", name="Ciec", yaxis="y3",
        hovertemplate="Fecha: %{x|%Y-%m-%d}<br>Ciec: %{y:.2f}<extra></extra>"
    ))

fig.update_layout(**layout_kwargs)

st.plotly_chart(fig, use_container_width=True)
st.caption(conv_caption + f" · A2 (por AUC) con tope = {MAX_PLANTS_CAP:.0f} pl·m².")

# ======================= A2 / x en UI ======================
st.subheader("Plantas·m² que escapan — Área bajo la curva (AUC) con tope")
st.markdown(
    f"""
**A2 — Solo supresión (1−Ciec):** **{A2_sup_final:,.1f}** pl·m² _(tope {MAX_PLANTS_CAP:.0f})_  
**A2 — Supresión + control post:** **{A2_ctrl_final:,.1f}** pl·m² _(tope {MAX_PLANTS_CAP:.0f})_  
**x — Densidad efectiva (edad en PC):** **{X_eff_pc:,.1f}** pl·m² _(tope {MAX_PLANTS_CAP:.0f})_
"""
)
# ======================= Pérdida de rendimiento (%) ===================

# Función: pérdida de rendimiento según densidad efectiva
def perdida_rinde_pct(x):
    x = np.asarray(x, dtype=float)
    return 0.375 * x / (1.0 + (0.375 * x / 76.639))

# Densidad efectiva diaria
emerrel_eff_base = df_plot["EMERREL"].values * FC_state               # EMERREL × FC
emerrel_eff_x2 = emerrel_eff_base * (1.0 - Ciec)                      # x₂: supresión
emerrel_eff_x3 = emerrel_eff_x2 * ctrl_factor                         # x₃: supresión + control

# Escala por AUC (plantas·m²·día⁻¹)
if factor_area_to_plants is not None:
    plm2dia_x2 = emerrel_eff_x2 * factor_area_to_plants
    plm2dia_x3 = emerrel_eff_x3 * factor_area_to_plants

    X2 = float(np.nansum(plm2dia_x2[mask_after_sow]))
    X3 = float(np.nansum(plm2dia_x3[mask_after_sow]))
else:
    plm2dia_x2 = plm2dia_x3 = np.full(len(emerrel_eff_base), np.nan)
    X2 = X3 = float("nan")

# =================== x acumulado desde siembra (x₁, x₂, x₃) ===================
if factor_area_to_plants is not None:
    plm2dia_x1 = emerrel_eff_base * factor_area_to_plants  # x₁ = EMERREL × FC
    x1_cum = np.cumsum(plm2dia_x1)
    x2_cum = np.cumsum(plm2dia_x2)
    x3_cum = np.cumsum(plm2dia_x3)
else:
    plm2dia_x1 = np.full(len(ts), np.nan)
    x1_cum = x2_cum = x3_cum = np.full(len(ts), np.nan)

# ======================= Gráfico: x acumulado (x₁, x₂, x₃) =======================
fig_xcum = go.Figure()

fig_xcum.add_trace(go.Scatter(
    x=ts, y=x1_cum, mode="lines", name="x₁: sin supresión ni control",
    line=dict(dash="dot"), hovertemplate="x₁ acumulado: %{y:.1f}<extra></extra>"
))
fig_xcum.add_trace(go.Scatter(
    x=ts, y=x2_cum, mode="lines", name="x₂: con supresión",
    line=dict(dash="dash"), hovertemplate="x₂ acumulado: %{y:.1f}<extra></extra>"
))
fig_xcum.add_trace(go.Scatter(
    x=ts, y=x3_cum, mode="lines", name="x₃: con supresión + control",
    line=dict(width=3), hovertemplate="x₃ acumulado: %{y:.1f}<extra></extra>"
))

if use_pc:
    fig_xcum.add_vrect(x0=PC_START, x1=PC_END, line_width=0,
                       fillcolor="LightSkyBlue", opacity=0.2)

fig_xcum.update_layout(
    title="Densidad efectiva acumulada (x) desde la siembra",
    xaxis_title="Fecha", yaxis_title="x acumulado (pl·m²)",
    margin=dict(l=10, r=10, t=40, b=10)
)

st.plotly_chart(fig_xcum, use_container_width=True)


# Pérdida de rendimiento para x₂ y x₃
loss_x2_pct = float(perdida_rinde_pct(X2)) if np.isfinite(X2) else float("nan")
loss_x3_pct = float(perdida_rinde_pct(X3)) if np.isfinite(X3) else float("nan")

# Mostrar resultados
st.subheader("Pérdida de rendimiento estimada (%) — por densidad efectiva (x)")

st.markdown(
    f"""
### x₂ — Con supresión (sin control)  
x = **{X2:,.1f}** pl·m² → pérdida estimada: **{loss_x2_pct:.2f}%**

### x₃ — Con supresión + control  
x = **{X3:,.1f}** pl·m² → pérdida estimada: **{loss_x3_pct:.2f}%**
"""
)

# ================= Gráfico: Pérdida (%) vs x =================

x_curve = np.linspace(0.0, MAX_PLANTS_CAP, 400)
y_curve = perdida_rinde_pct(x_curve)

fig_loss = go.Figure()
fig_loss.add_trace(go.Scatter(
    x=x_curve, y=y_curve, mode="lines", name="Modelo pérdida % vs x",
    hovertemplate="x = %{x:.1f} pl·m²<br>Pérdida: %{y:.2f}%<extra></extra>"
))

if np.isfinite(X2):
    fig_loss.add_trace(go.Scatter(
        x=[X2], y=[loss_x2_pct], mode="markers+text",
        name="x₂: con supresión",
        text=[f"x₂ = {X2:.1f}"], textposition="top center",
        marker=dict(size=10, symbol="diamond"),
        hovertemplate="x₂ = %{x:.1f} pl·m²<br>Pérdida: %{y:.2f}%<extra></extra>"
    ))

if np.isfinite(X3):
    fig_loss.add_trace(go.Scatter(
        x=[X3], y=[loss_x3_pct], mode="markers+text",
        name="x₃: con supresión + control",
        text=[f"x₃ = {X3:.1f}"], textposition="top right",
        marker=dict(size=11, symbol="star"),
        hovertemplate="x₃ = %{x:.1f} pl·m²<br>Pérdida: %{y:.2f}%<extra></extra>"
    ))

fig_loss.update_layout(
    title="Pérdida de rendimiento (%) vs. densidad efectiva (x)",
    xaxis_title=f"x (pl·m²) — área bajo la curva desde siembra",
    yaxis_title="Pérdida de rendimiento (%)",
    margin=dict(l=10, r=10, t=40, b=10)
)
st.plotly_chart(fig_loss, use_container_width=True)

# ============================== Cronograma ============================
st.subheader("Cronograma de manejo (manual)")
if len(sched):
    st.dataframe(sched, use_container_width=True)
    st.download_button("Descargar cronograma (CSV)", sched.to_csv(index=False).encode("utf-8"),
                       "cronograma_manejo_manual.csv", "text/csv", key="dl_crono")
else:
    st.info("Activa alguna intervención y define la(s) fecha(s).")

# =========================== Descargas de series ======================
out = df_plot.copy()
out.rename(columns={"EMERREL": "EMERREL_cruda"}, inplace=True)
out["EMERREL_supresion"]      = emerrel_supresion
out["EMERREL_supresion_ctrl"] = emerrel_supresion_ctrl
if factor_area_to_plants is not None:
    out["plm2dia_cruda"]             = np.round(plantas_emerrel_cruda, 4)
    out["plm2dia_supresion"]         = np.round(plantas_supresion, 4)
    out["plm2dia_supresion_ctrl"]    = np.round(plantas_supresion_ctrl, 4)
    out["plm2dia_supresion_ctrl_eff"] = np.round(plm2dia_ctrl_eff, 4)
    a2cum_df = pd.DataFrame({
        "fecha": pd.to_datetime(ts),
        "A2_acum_sup_cap": A2_cum_sup_cap.reindex(ts, method="nearest").values if len(A2_cum_sup_cap) else np.nan,
        "A2_acum_sup_ctrl_cap": A2_cum_ctrl_cap.reindex(ts, method="nearest").values if len(A2_cum_ctrl_cap) else np.nan,
        "x_acum_eff_cap": X_cum_eff_cap if len(X_cum_eff_cap) else np.nan,
    })
else:
    a2cum_df = pd.DataFrame(columns=["fecha","A2_acum_sup_cap","A2_acum_sup_ctrl_cap","x_acum_eff_cap"])

with st.expander("Descargas de series", expanded=True):
    st.caption(conv_caption + f" · Columnas en pl·m²·día⁻¹ (escala por AUC) y acumulados (cap {MAX_PLANTS_CAP:.0f}).")
    cols_show = ["fecha","EMERREL_cruda","EMERREL_supresion","EMERREL_supresion_ctrl"]
    if factor_area_to_plants is not None:
        cols_show += ["plm2dia_cruda","plm2dia_supresion","plm2dia_supresion_ctrl","plm2dia_supresion_ctrl_eff"]
    out_show = out[cols_show].copy()
    st.dataframe(out_show.tail(20), use_container_width=True)
    st.download_button("Descargar serie procesada (CSV)",
                       out_show.to_csv(index=False).encode("utf-8"),
                       "serie_procesada_auc_fenologia.csv", "text/csv", key="dl_serie")
    if len(a2cum_df):
        st.download_button("Descargar acumulados (CSV)",
                           a2cum_df.to_csv(index=False).encode("utf-8"),
                           "acumulados_a2_x.csv", "text/csv", key="dl_a2cum")

# ============================== Diagnóstico ===========================
st.subheader("Diagnóstico")
if factor_area_to_plants is not None:
    contrib_S1 = float(np.nansum(plantas_supresion_ctrl[mask_S1 & mask_after_sow]))
    contrib_S2 = float(np.nansum(plantas_supresion_ctrl[mask_S2 & mask_after_sow]))
    contrib_S3 = float(np.nansum(plantas_supresion_ctrl[mask_S3 & mask_after_sow]))
    contrib_S4 = float(np.nansum(plantas_supresion_ctrl[mask_S4 & mask_after_sow]))
else:
    contrib_S1 = contrib_S2 = contrib_S3 = contrib_S4 = float("nan")

_diag = {
    "siembra": str(sow_date),
    "tope_A2_plm2": MAX_PLANTS_CAP,
    "PC": {"start": str(PC_START.date()), "end": str(PC_END.date()), "ref": ref_pc},
    # AUCs
    "AUC_EMERREL_cruda_desde_siembra_dias": float(auc_cruda),
    "AUC_supresion_desde_siembra_dias": float(auc_sup),
    "AUC_supresion_ctrl_desde_siembra_dias": float(auc_sup_ctrl),
    # Escala por área
    "factor_pl_m2_por_EMERREL_dia": float(factor_area_to_plants) if (factor_area_to_plants is not None) else None,
    # A2/x
    "A2_sup_raw_por_AUC": float(A2_sup_raw) if np.isfinite(A2_sup_raw) else None,
    "A2_ctrl_raw_por_AUC": float(A2_ctrl_raw) if np.isfinite(A2_ctrl_raw) else None,
    "A2_sup_cap": float(A2_sup_final) if np.isfinite(A2_sup_final) else None,
    "A2_ctrl_cap": float(A2_ctrl_final) if np.isfinite(A2_ctrl_final) else None,
    "X_eff_pc": float(X_eff_pc) if np.isfinite(X_eff_pc) else None,
    # Fenología por edad en PC
    "reglas_fenologia_por_edad_PC": {"S2": "7–27", "S3": "28–59", "S4": "≥60", "S1": "resto"},
    "FC_S": {"S1": 0.0, "S2": 0.3, "S3": 0.6, "S4": 1.0},
    "contrib_plm2_por_estado": {"S1": contrib_S1, "S2": contrib_S2, "S3": contrib_S3, "S4": contrib_S4},
    # Manejo
    "decaimiento": decaimiento_tipo,
    "NR_no_residuales_dias": NR_DAYS_DEFAULT
}
st.code(json.dumps(_diag, ensure_ascii=False, indent=2))
