# -*- coding: utf-8 -*-
# app.py — PREDWEEM · Supresión (EMERREL × (1−Ciec)) + Control (NR=10d)
# - Sin ICIC
# - Ciec desde canopia (FC/LAI), con curva opcional
# - Conversión a Plantas·m² anclada al pico de EMERREL (cruda) → 250 pl·m²
# - A2 = SUMA de valores INSTANTÁNEOS MENSUALES (1 por mes calendario desde siembra), con TOPE 250
# - Pérdida de rinde calculada con A2 topeado
# - Eje derecho: incluye "EMERREL→Plantas·m² (cruda)" y escala que siempre contiene 250

import io, re, json, math, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from datetime import timedelta

APP_TITLE = "PREDWEEM · Supresión (1−Ciec) + Control"
st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
st.title(APP_TITLE)
st.caption("Plantas·m², checkpoints mensuales y pérdida de rinde se basan en EMERREL × (1 − Ciec) (y su versión con control). No se calcula ICIC.")

# ========================== Constantes clave ==========================
NR_DAYS_DEFAULT = 10
MAX_PLANTS_CAP  = 250.0  # objetivo de pico (pl·m²)

# ============================== Utils ================================
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

def percentiles_from_emerrel(emerrel: pd.Series, dates: pd.Series):
    work = pd.DataFrame({"fecha": dates, "EMERREL": emerrel}).copy()
    work["MA5"] = moving_average(work["EMERREL"], 5)
    work["EMERAC"] = work["MA5"].cumsum()
    total = float(work["EMERAC"].iloc[-1]) if len(work) else 0.0
    if total <= 0:
        work["EMERAC_N"] = 0.0
        return {}, work, 0.0
    work["EMERAC_N"] = work["EMERAC"] / total
    def date_at(p):
        idx = (work["EMERAC_N"] - p).abs().idxmin()
        return pd.to_datetime(work.loc[idx, "fecha"])
    P = {k: date_at(k/100.0) for k in [10,25,30,40,50,60,70,80]}
    return P, work, total

# ======================= Canopia (sin ICIC): FC y LAI =================
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
    """Devuelve (FC dinámica, LAI) sin ICIC."""
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
            df = df.groupby("fecha,").sum(numeric_only=True).rename_axis("fecha").reset_index()
        elif dedup == "promediar":
            df = df.groupby("fecha,").mean(numeric_only=True).rename_axis("fecha").reset_index()
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
sow_min = dt.date(year_ref, 5, 1)
sow_max = dt.date(year_ref, 7, 1)

with st.sidebar:
    st.header("Siembra & Canopia (para Ciec)")
    st.caption(f"Ventana de siembra: **{sow_min} → {sow_max}**")
    sow_date = st.date_input("Fecha de siembra", value=sow_min, min_value=sow_min, max_value=sow_max)
    mode_canopy = st.selectbox("Canopia", ["Cobertura dinámica (%)", "LAI dinámico"], index=0)
    t_lag = st.number_input("Días a emergencia del cultivo (lag)", 0, 60, 7, 1)
    t_close = st.number_input("Días a cierre de entresurco", 10, 120, 45, 1)
    cov_max = st.number_input("Cobertura máxima (%)", 10.0, 100.0, 85.0, 1.0)
    lai_max = st.number_input("LAI máximo", 0.0, 8.0, 3.5, 0.1)
    k_beer = st.number_input("k (Beer–Lambert)", 0.1, 1.2, 0.6, 0.05)

# ========================= Sidebar: Ciec ===============================
with st.sidebar:
    st.header("Ciec (competencia del cultivo)")
    use_ciec = st.checkbox("Calcular y mostrar Ciec", value=True)
    Ca = st.number_input("Densidad real Ca (pl/m²)", 50, 700, 250, 10)
    Cs = st.number_input("Densidad estándar Cs (pl/m²)", 50, 700, 250, 10)
    LAIhc = st.number_input("LAIhc (escenario altamente competitivo)", 0.5, 10.0, 3.5, 0.1)

# =========== Etiquetas/visual + avanzadas =============================
with st.sidebar:
    st.header("Etiquetas y escalas")
    show_plants_axis = st.checkbox("Mostrar Plantas·m² en eje derecho", value=True)
    show_ciec_curve = st.checkbox("Mostrar curva Ciec (0–1)", value=True)
    highlight_labels = st.checkbox("Etiquetas destacadas en bandas", value=True)

    st.header("Visualización avanzada")
    show_emac_curve_raw = st.checkbox("Mostrar EMERAC (curva) cruda", value=False)
    show_emac_points_raw = st.checkbox("Mostrar EMERAC (puntos) cruda", value=False)
    show_nonres_bands = st.checkbox("Marcar bandas NR (por defecto 10 días)", value=True)

if not (sow_min <= sow_date <= sow_max):
    st.error("La fecha de siembra debe estar entre el 1 de mayo y el 1 de julio."); st.stop()

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
    Ciec = np.full_like(LAI, np.nan, dtype=float)

df_ciec = pd.DataFrame({"fecha": df_plot["fecha"], "Ciec": Ciec})
_, work_base, tot_base = percentiles_from_emerrel(df_plot["EMERREL"], df_plot["fecha"])

# ===== Supresión (base agronómica) ===================================
emerrel_supresion = (df_plot["EMERREL"].astype(float).values * (1.0 - Ciec)).clip(min=0.0) if use_ciec else np.full(len(df_plot), np.nan)

# ===== Conversión: EMERREL (cruda) → Plantas·m² (pico = 250) =========
pico_emerrel = float(df_plot["EMERREL"].max())
if pico_emerrel > 0:
    factor_pl = MAX_PLANTS_CAP / pico_emerrel   # anclado a EMERREL cruda
    conv_caption = (
        f"Conversión: pico EMERREL(cruda)={pico_emerrel:.4f} → {MAX_PLANTS_CAP:.0f} pl·m² "
        f"⇒ factor={factor_pl:.2f} pl·m² por unidad EMERREL"
    )
else:
    factor_pl = None
    conv_caption = "No se pudo calcular Plantas·m² (pico de EMERREL cruda = 0)."

# Chequeo visual y diagnóstico del mapeo a 250
if factor_pl is not None and np.isfinite(factor_pl) and np.isfinite(pico_emerrel):
    mapped_max = pico_emerrel * factor_pl
    st.success(f"Chequeo conversión: pico EMERREL × factor = {mapped_max:.2f} pl·m² (objetivo: {MAX_PLANTS_CAP:.0f}).")
else:
    mapped_max = float("nan")
    st.warning("No fue posible verificar el mapeo del pico (factor o pico no finitos).")

# Densidades resultantes (supresión y supresión+control) en plantas·m²
plantas_supresion = (emerrel_supresion * factor_pl) if (factor_pl is not None and np.isfinite(factor_pl)) else np.full(len(df_plot), np.nan)

# =================== Manejo (control) y decaimientos ===================
sched_rows = []
def add_sched(nombre, fecha_ini, dias_res=None, nota=""):
    if not fecha_ini: return
    fin = (pd.to_datetime(fecha_ini) + pd.Timedelta(days=int(dias_res))).date() if dias_res else None
    sched_rows.append({"Intervención": nombre, "Inicio": str(fecha_ini), "Fin": str(fin) if fin else "—", "Nota": nota})

with st.sidebar:
    st.header("Manejo pre-siembra (manual)")
    min_date = df_plot["fecha"].min().date()
    max_date = df_plot["fecha"].max().date()
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

# Agendar (NR por defecto para selectivos NO residuales)
if pre_glifo: add_sched("Pre · glifosato (NSr, 1d)", pre_glifo_date, None, "Barbecho")
if pre_selNR: add_sched(f"Pre · selectivo no residual (NR)", pre_selNR_date, NR_DAYS_DEFAULT, f"NR por defecto {NR_DAYS_DEFAULT}d")
if pre_selR:  add_sched("Pre · selectivo + residual", pre_selR_date, pre_res_dias, f"Protege {pre_res_dias} días")
if post_gram: add_sched(f"Post · graminicida selectivo (NR)", post_gram_date, NR_DAYS_DEFAULT, f"NR por defecto {NR_DAYS_DEFAULT}d")
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
    if decaimiento_tipo == "Lineal":
        st.caption("Peso(t)=1 al inicio → 0 al final; e(t)=eficiencia×Peso(t)")
    if decaimiento_tipo == "Exponencial":
        half_life = st.number_input("Semivida (días) para exponencial", 1, 120, 20, 1)
        lam_exp = math.log(2) / max(1e-6, half_life)
    else:
        lam_exp = None

# =================== Factor de control diario =========================
fechas_d = df_plot["fecha"].dt.date.values

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
if post_gram: apply_efficiency(weights_residual(post_gram_date, NR_DAYS_DEFAULT), ef_post_gram)
if post_selR: apply_efficiency(weights_residual(post_selR_date, post_res_dias), ef_post_selR)

# ==================== Control sobre SUPRESIÓN ==========================
emerrel_supresion_ctrl = emerrel_supresion * ctrl_factor
if use_ciec:
    esperado = (df_plot["EMERREL"].values * (1.0 - Ciec)) * ctrl_factor
    assert np.allclose(emerrel_supresion_ctrl, esperado, atol=1e-12, rtol=1e-8), \
        "El control debe aplicarse sobre EMERREL×(1−Ciec)"

plantas_supresion = (emerrel_supresion * factor_pl) if (factor_pl is not None and np.isfinite(factor_pl)) else np.full(len(df_plot), np.nan)
plantas_supresion_ctrl = (emerrel_supresion_ctrl * factor_pl) if (factor_pl is not None and np.isfinite(factor_pl)) else np.full(len(df_plot), np.nan)

# ======= MENSUALES: SUMA de valores instantáneos (1 por mes) ==========
ts_norm = pd.to_datetime(df_plot["fecha"]).dt.normalize()
mask_after_sow = ts_norm.dt.date >= sow_date

df_m = pd.DataFrame({
    "fecha": ts_norm,
    "Plantas_m2_supresion": plantas_supresion,
    "Plantas_m2_supresion_ctrl": plantas_supresion_ctrl
})
df_m = df_m.loc[mask_after_sow].copy()
if len(df_m):
    df_m["ym"] = df_m["fecha"].dt.to_period("M")
    idx_first_of_month = df_m.groupby("ym")["fecha"].idxmin()
    df_mensual = df_m.loc[idx_first_of_month].sort_values("fecha").reset_index(drop=True)
else:
    df_mensual = df_m.copy()

A2_sup_raw  = float(np.nansum(df_mensual["Plantas_m2_supresion"])) if "Plantas_m2_supresion" in df_mensual else float("nan")
A2_ctrl_raw = float(np.nansum(df_mensual["Plantas_m2_supresion_ctrl"])) if "Plantas_m2_supresion_ctrl" in df_mensual else float("nan")

A2_sup_final  = min(MAX_PLANTS_CAP, A2_sup_raw)  if np.isfinite(A2_sup_raw)  else float("nan")
A2_ctrl_final = min(MAX_PLANTS_CAP, A2_ctrl_raw) if np.isfinite(A2_ctrl_raw) else float("nan")

# ============================== Gráfico 1 ==============================
fig = go.Figure()

# EMERREL cruda (informativa)
fig.add_trace(go.Scatter(
    x=df_plot["fecha"], y=df_plot["EMERREL"], mode="lines",
    name="EMERREL (cruda)",
    hovertemplate="Fecha: %{x|%Y-%m-%d}<br>EMERREL: %{y:.4f}<extra></extra>"
))

# Supresión base
fig.add_trace(go.Scatter(
    x=df_plot["fecha"], y=emerrel_supresion, mode="lines",
    name="EMERREL × (1 − Ciec)", line=dict(dash="dashdot"),
    customdata=np.column_stack([plantas_supresion]),
    hovertemplate="Fecha: %{x|%Y-%m-%d}<br>Supresión: %{y:.4f}<br>Plantas·m² (sup.): %{customdata[0]:.1f}<extra></extra>"
))

# Supresión + control
fig.add_trace(go.Scatter(
    x=df_plot["fecha"], y=emerrel_supresion_ctrl, mode="lines",
    name="EMERREL × (1 − Ciec) (control)", line=dict(dash="dot", width=2),
    customdata=np.column_stack([plantas_supresion_ctrl]),
    hovertemplate="Fecha: %{x|%Y-%m-%d}<br>Supresión (ctrl): %{y:.4f}<br>Plantas·m² (sup. ctrl): %{customdata[0]:.1f}<extra></extra>"
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
        x0 = pd.to_datetime(start_date)
        x1 = x0 + pd.Timedelta(days=d_int)
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
    if post_gram: add_residual_band(post_gram_date, NR_DAYS_DEFAULT, f"Graminicida ({NR_DAYS_DEFAULT}d)")

# Ejes y escalas
ymax = float(np.nanmax([
    df_plot["EMERREL"].max(),
    np.nanmax(emerrel_supresion) if np.isfinite(np.nanmax(emerrel_supresion)) else 0.0,
    np.nanmax(emerrel_supresion_ctrl) if np.isfinite(np.nanmax(emerrel_supresion_ctrl)) else 0.0
]))
ymax = max(1e-6, ymax * 1.15)

layout_kwargs = dict(
    margin=dict(l=10, r=10, t=40, b=10),
    title="EMERREL + Supresión (1−Ciec) + Control" + (" + Ciec" if use_ciec else ""),
    xaxis_title="Fecha",
    yaxis_title="EMERREL / Supresión",
    yaxis=dict(range=[0, ymax])
)

# ========= Eje derecho opcional: Plantas·m² (incluye 250 siempre) ========
if show_plants_axis and (factor_pl is not None) and np.isfinite(factor_pl):
    plantas_emerrel_cruda = df_plot["EMERREL"].values * factor_pl

    candidatos = [
        np.nanmax(plantas_supresion),
        np.nanmax(plantas_supresion_ctrl),
        np.nanmax(plantas_emerrel_cruda),
        MAX_PLANTS_CAP
    ]
    plantas_max = float(np.nanmax(candidatos))
    if not np.isfinite(plantas_max) or plantas_max <= 0:
        plantas_max = MAX_PLANTS_CAP

    layout_kwargs["yaxis2"] = dict(
        overlaying="y",
        side="right",
        title="Plantas·m²",
        position=1.0,
        range=[0, max(plantas_max * 1.15, MAX_PLANTS_CAP * 1.15)],
        tick0=0,
        dtick=50
    )

    fig.add_trace(go.Scatter(
        x=df_plot["fecha"], y=plantas_emerrel_cruda,
        name="EMERREL→Plantas·m² (cruda)",
        yaxis="y2", mode="lines", line=dict(width=1),
        hovertemplate="Fecha: %{x|%Y-%m-%d}<br>EMERREL→Plantas·m²: %{y:.1f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=df_plot["fecha"], y=plantas_supresion,
        name="Plantas·m² × (1−Ciec)", yaxis="y2", mode="lines",
        hovertemplate="Fecha: %{x|%Y-%m-%d}<br>Plantas·m² (sup.): %{y:.1f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=df_plot["fecha"], y=plantas_supresion_ctrl,
        name="Plantas·m² × (1−Ciec) (control)", yaxis="y2", mode="lines",
        line=dict(dash="dot"),
        hovertemplate="Fecha: %{x|%Y-%m-%d}<br>Plantas·m² (sup. ctrl): %{y:.1f}<extra></extra>"
    ))

# Eje auxiliar y curva Ciec (opcional)
if use_ciec:
    layout_kwargs["yaxis3"] = dict(
        overlaying="y", side="right", title="Ciec (0–1)", position=0.97, range=[0, 1]
    )
    fig.add_trace(go.Scatter(
        x=df_ciec["fecha"], y=df_ciec["Ciec"], mode="lines", name="Ciec", yaxis="y3",
        hovertemplate="Fecha: %{x|%Y-%m-%d}<br>Ciec: %{y:.2f}<extra></extra>"
    ))

fig.update_layout(**layout_kwargs)
st.plotly_chart(fig, use_container_width=True)
st.caption(f"{conv_caption} · NR por defecto = {NR_DAYS_DEFAULT} días · A2 (mensual) con tope máx = {MAX_PLANTS_CAP:.0f} pl·m² (sumatoria de checkpoints mensuales).")

# ======================= A2 en UI (SUMA MENSUAL con tope) =============
st.subheader("Plantas·m² que escapan — Sumatoria MENSUAL (con tope)")
st.markdown(
    f"""
**Sumatoria — Solo supresión (1−Ciec):** **{A2_sup_final:,.1f}** pl·m² _(tope {MAX_PLANTS_CAP:.0f})_  
**Sumatoria — Supresión + control post:** **{A2_ctrl_final:,.1f}** pl·m² _(tope {MAX_PLANTS_CAP:.0f})_
"""
)

with st.expander("Ver checkpoints mensuales (instantáneos)", expanded=False):
    if len(df_mensual):
        df_mensual_show = df_mensual[["fecha","Plantas_m2_supresion","Plantas_m2_supresion_ctrl"]].copy()
        df_mensual_show["fecha"] = pd.to_datetime(df_mensual_show["fecha"]).dt.date
        st.dataframe(df_mensual_show, use_container_width=True)
        st.download_button(
            "Descargar checkpoints mensuales (CSV)",
            df_mensual_show.to_csv(index=False).encode("utf-8"),
            "checkpoints_mensuales.csv",
            "text/csv"
        )
    else:
        st.info("No hay checkpoints mensuales (revisá el rango de fechas y la siembra).")

# ======================= Pérdida de rendimiento (%) ===================
def perdida_rinde_pct(A2):
    A2 = np.asarray(A2, dtype=float)
    return 0.375 * A2 / (1.0 + (0.375 * A2 / 76.639))

loss_sup_pct  = float(perdida_rinde_pct(A2_sup_final))  if np.isfinite(A2_sup_final)  else float("nan")
loss_ctrl_pct = float(perdida_rinde_pct(A2_ctrl_final)) if np.isfinite(A2_ctrl_final) else float("nan")

st.subheader("Pérdida de rendimiento estimada (%)")
st.markdown(
    f"""
**Sólo supresión (1−Ciec):** **{loss_sup_pct:,.2f}%**  &nbsp;|&nbsp;  A2 = {A2_sup_final:,.1f} pl·m²  
**Supresión + control post:** **{loss_ctrl_pct:,.2f}%**  &nbsp;|&nbsp;  A2 = {A2_ctrl_final:,.1f} pl·m²
"""
)

# =================== Gráfico 2: Pérdida (%) vs A2 (pl·m²) =============
x_curve = np.linspace(0.0, MAX_PLANTS_CAP, 400)
y_curve = perdida_rinde_pct(x_curve)

fig_loss = go.Figure()
fig_loss.add_trace(go.Scatter(x=x_curve, y=y_curve, mode="lines", name="Modelo pérdida %"))

if np.isfinite(A2_sup_final):
    fig_loss.add_trace(go.Scatter(
        x=[A2_sup_final], y=[loss_sup_pct], mode="markers+text",
        name="Solo supresión", text=["Supresión"], textposition="top center",
        marker=dict(size=10, symbol="circle")
    ))
if np.isfinite(A2_ctrl_final):
    fig_loss.add_trace(go.Scatter(
        x=[A2_ctrl_final], y=[loss_ctrl_pct], mode="markers+text",
        name="Supresión + control", text=["Supresión+control"], textposition="bottom center",
        marker=dict(size=10, symbol="diamond")
    ))

fig_loss.update_layout(
    title="Pérdida de rendimiento (%) vs. A2 (pl·m²)",
    xaxis_title="A2 (pl·m²) — sumatoria mensual (tope 250)",
    yaxis_title="Pérdida de rendimiento (%)",
    margin=dict(l=10, r=10, t=40, b=10)
)
st.plotly_chart(fig_loss, use_container_width=True)

# ============================== Cronograma ============================
st.subheader("Cronograma de manejo (manual)")
if len(sched):
    st.dataframe(sched, use_container_width=True)
    st.download_button("Descargar cronograma (CSV)", sched.to_csv(index=False).encode("utf-8"),
                       "cronograma_manejo_manual.csv", "text/csv")
else:
    st.info("Activa alguna intervención y define la(s) fecha(s).")

# =========================== Descargas de series ======================
out = df_plot.copy()
out["EMERREL_supresion"]           = emerrel_supresion
out["EMERREL_supresion_ctrl"]      = emerrel_supresion_ctrl
out["Plantas_m2_supresion"]        = np.round(plantas_supresion, 1)
out["Plantas_m2_supresion_ctrl"]   = np.round(plantas_supresion_ctrl, 1)

# columnas mensuales (solo las fechas checkpoint)
if len(df_mensual):
    mcols = df_mensual[["fecha","Plantas_m2_supresion","Plantas_m2_supresion_ctrl"]].copy()
    mcols.columns = ["fecha_mensual","Plantas_m2_supresion_M","Plantas_m2_supresion_ctrl_M"]
else:
    mcols = pd.DataFrame(columns=["fecha_mensual","Plantas_m2_supresion_M","Plantas_m2_supresion_ctrl_M"])

with st.expander("Descargas de series", expanded=True):
    st.caption(conv_caption + f" · A2 (mensual) con tope máx = {MAX_PLANTS_CAP:.0f} pl·m² · A2 = suma de checkpoints mensuales")
    only_plants = st.checkbox("Mostrar/descargar sólo columnas de Plantas·m² (supresión)", value=False)

    default_cols = [
        "fecha",
        "EMERREL_supresion", "EMERREL_supresion_ctrl",
        "Plantas_m2_supresion", "Plantas_m2_supresion_ctrl",
    ]
    out_show = out[default_cols].copy()
    if len(mcols):
        # Se muestran aparte los checkpoints mensuales en el CSV adicional
        pass

    st.dataframe(out_show.tail(20), use_container_width=True)
    st.download_button("Descargar serie procesada (CSV)",
                       out_show.to_csv(index=False).encode("utf-8"),
                       "serie_procesada.csv", "text/csv")
    if len(mcols):
        st.download_button("Descargar checkpoints mensuales (CSV)",
                           mcols.to_csv(index=False).encode("utf-8"),
                           "checkpoints_mensuales.csv", "text/csv")

# ============================== Diagnóstico ===========================
st.subheader("Diagnóstico")
diag = {
    "siembra": str(sow_date),
    "pico_emerrel_cruda": float(pico_emerrel),
    "factor_pl_m2_por_EMERREL_cruda": float(factor_pl) if (factor_pl is not None) else None,
    "pico_emerrel_mapeado_pl_m2": float(mapped_max) if np.isfinite(mapped_max) else None,
    "target_pl_m2": MAX_PLANTS_CAP,
    "suma_supresion_EMERRELx(1-Ciec)": float(np.nansum(emerrel_supresion)),
    "suma_supresion_ctrl_EMERRELx(1-Ciec)xcontrol": float(np.nansum(emerrel_supresion_ctrl)),
    # A2 por sumatoria de checkpoints mensuales
    "A2_sup_raw_sum_mensual": A2_sup_raw,
    "A2_sup_ctrl_raw_sum_mensual": A2_ctrl_raw,
    "A2_sup_cap": A2_sup_final,
    "A2_sup_ctrl_cap": A2_ctrl_final,
    # Ciec
    "LAIhc": float(LAIhc),
    "Ciec_min_max": (float(np.nanmin(Ciec)), float(np.nanmax(Ciec))) if use_ciec else None,
    "decaimiento": decaimiento_tipo,
    "NR_no_residuales_dias": NR_DAYS_DEFAULT
}
st.code(json.dumps(diag, ensure_ascii=False, indent=2))

