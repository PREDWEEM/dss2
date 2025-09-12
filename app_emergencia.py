# -*- coding: utf-8 -*-
# app.py — PREDWEEM · ICIC + supresión (1−Ciec) + Control (NR=10d por defecto)
# - EMERREL × (1 − Ciec) (supresión) + versiones con control post
# - Conversión EMERREL→plantas·m² con tope máx 250 pl·m²
# - Totales de escapes SOLO QUINCENALES (con tope)
# - Series exportables (solo quincenales para escapes) + diagnóstico

import io, re, json, math, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from datetime import timedelta

APP_TITLE = "PREDWEEM · ICIC + supresión (1−Ciec) + Control"
st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
st.title(APP_TITLE)
st.caption("EMERREL/ICIC/Ciec con manejo manual, eficiencias de control y decaimiento en residuales. EMERAC opcional. Usa supresión: EMERREL × (1 − Ciec).")

# ========================== Constantes clave ==========================
NR_DAYS_DEFAULT = 10          # no residuales
MAX_PLANTS_CAP  = 250.0       # límite máximo de plantas·m² (tope)

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

# ======================= ICIC (y LAI) encapsulado =====================
def compute_icic(
    fechas: pd.Series,
    sow_date: dt.date,
    mode_canopy: str,
    t_lag: int,
    t_close: int,
    cov_max: float,
    lai_max: float,
    k_beer: float,
    alpha: float,
    beta: float,
    gamma: float,
    dens: float, dens_ref: float,
    row_cm: float, row_ref: float,
):
    """
    ICIC = (alpha*FC + beta*F_d + gamma*F_s - offset) / (1 - offset),
    offset = beta*F_d + gamma*F_s ; truncado a [0,1]
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

    F_d = min(1.0, dens / max(1e-6, dens_ref))
    F_s = min(1.0, row_ref / max(1e-6, row_cm))

    icic_bruto = alpha * fc_dyn + beta * F_d + gamma * F_s
    offset = beta * F_d + gamma * F_s
    den = max(1e-6, 1.0 - offset)
    icic = (icic_bruto - offset) / den
    icic = np.clip(icic, 0.0, 1.0)
    return icic, fc_dyn, LAI

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
            df = df.groupby("fecha", as_index=False)["valor"].sum()
        elif dedup == "promediar":
            df = df.groupby("fecha", as_index=False)["valor"].mean()
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

# ==================== Siembra & parámetros ICIC ========================
years = df_plot["fecha"].dt.year.dropna().astype(int)
year_ref = int(years.mode().iloc[0]) if len(years) else dt.date.today().year
sow_min = dt.date(year_ref, 5, 1)
sow_max = dt.date(year_ref, 7, 1)

with st.sidebar:
    st.header("Siembra & ICIC")
    st.caption(f"Ventana de siembra: **{sow_min} → {sow_max}**")
    sow_date = st.date_input("Fecha de siembra", value=sow_min, min_value=sow_min, max_value=sow_max)
    mode_canopy = st.selectbox("Canopia", ["Cobertura dinámica (%)", "LAI dinámico"], index=0)
    t_lag = st.number_input("Días a emergencia del cultivo (lag)", 0, 60, 7, 1)
    t_close = st.number_input("Días a cierre de entresurco", 10, 120, 45, 1)
    cov_max = st.number_input("Cobertura máxima (%)", 10.0, 100.0, 85.0, 1.0)
    lai_max = st.number_input("LAI máximo", 0.0, 8.0, 3.5, 0.1)
    k_beer = st.number_input("k (Beer–Lambert)", 0.1, 1.2, 0.6, 0.05)
    dens = st.number_input("Densidad cultivo (pl/m²)", 20, 700, 250, 10)
    dens_ref = st.number_input("Densidad ref (pl/m²)", 20, 700, 250, 10)
    row_cm = st.number_input("Entre-hileras (cm)", 10, 60, 19, 1)
    row_ref = st.number_input("Entre-hileras ref (cm)", 10, 60, 19, 1)
    alpha = st.slider("α (peso canopia)", 0.0, 1.0, 0.60, 0.05)
    beta  = st.slider("β (densidad)",     0.0, 1.0, 0.25, 0.05)
    gamma = st.slider("γ (entrehileras)", 0.0, 1.0, 0.15, 0.05)
    theta = st.slider("Umbral ICIC (θ)", 0.2, 0.9, 0.6, 0.05)

# ========================= Sidebar: Ciec ===============================
with st.sidebar:
    st.header("Ciec (competencia del cultivo)")
    use_ciec = st.checkbox("Calcular y mostrar Ciec", value=True)
    Ca = st.number_input("Densidad real Ca (pl/m²)", 50, 700, 250, 10)
    Cs = st.number_input("Densidad estándar Cs (pl/m²)", 50, 700, 250, 10)
    LAIhc = st.number_input("LAIhc (escenario altamente competitivo)", 0.5, 10.0, 3.5, 0.1)  # default 3.5

# =========== Etiquetas/visual + avanzadas (EMERAC y barras) ===========
with st.sidebar:
    st.header("Etiquetas y escalas")
    highlight_labels = st.checkbox("Etiquetas destacadas en bandas", value=True)
    show_plants_axis = st.checkbox("Mostrar segunda escala derecha en Plantas·m²", value=True)
    st.header("Visualización avanzada")
    show_emac_curve_raw = st.checkbox("Mostrar EMERAC (curva) cruda", value=False)
    show_emac_curve_adj = st.checkbox("Mostrar EMERAC (curva) ajustada", value=False)
    show_emac_points_raw = st.checkbox("Mostrar EMERAC (puntos) cruda", value=False)
    show_emac_points_adj = st.checkbox("Mostrar EMERAC (puntos) ajustada", value=False)
    show_nonres_bands = st.checkbox("Marcar bandas NR (por defecto 10 días)", value=True)
    show_sup_density = st.checkbox("Mostrar EMERREL×(1−Ciec) en Plantas·m² (barras)", value=False)
    show_icic_density = st.checkbox("Mostrar EMERREL_eff (ICIC) en Plantas·m² (barras)", value=False)

if not (sow_min <= sow_date <= sow_max):
    st.error("La fecha de siembra debe estar entre el 1 de mayo y el 1 de julio."); st.stop()

# ============================ ICIC + Ciec =============================
ICIC, FC, LAI = compute_icic(
    fechas=df_plot["fecha"],
    sow_date=sow_date,
    mode_canopy=mode_canopy,
    t_lag=int(t_lag), t_close=int(t_close),
    cov_max=float(cov_max), lai_max=float(lai_max), k_beer=float(k_beer),
    alpha=float(alpha), beta=float(beta), gamma=float(gamma),
    dens=float(dens), dens_ref=float(dens_ref),
    row_cm=float(row_cm), row_ref=float(row_ref),
)
if len(ICIC):
    ICIC[0] = 0.0  # asegurar ICIC(siem.)=0

if use_ciec:
    Ca_safe = float(Ca) if float(Ca) > 0 else 1e-6
    Ciec = (LAI / max(1e-6, float(LAIhc))) * (float(Cs) / Ca_safe)
    Ciec = np.clip(Ciec, 0.0, 1.0)
else:
    Ciec = np.full_like(ICIC, np.nan, dtype=float)

df_ic = pd.DataFrame({"fecha": df_plot["fecha"], "ICIC": ICIC, "Ciec": Ciec})
t_star = df_ic.loc[df_ic["ICIC"] >= theta, "fecha"].min() if (df_ic["ICIC"] >= theta).any() else None

# Serie efectiva por ICIC (aplica desde t*)
df_eff = df_plot.copy()
if t_star is not None:
    mask = df_eff["fecha"] >= t_star
    df_eff["EMERREL_eff"] = df_eff["EMERREL"].where(~mask, df_eff["EMERREL"]*(1 - df_ic["ICIC"].values))
else:
    df_eff["EMERREL_eff"] = df_eff["EMERREL"]

# Percentiles y acumuladas (solo para información visual)
_, work_base, tot_base = percentiles_from_emerrel(df_plot["EMERREL"], df_plot["fecha"])
_, work_star, tot_star = percentiles_from_emerrel(df_eff["EMERREL_eff"], df_eff["fecha"])

# ===== Conversión EMERREL → Plantas·m² (pico ≙ 250) =====
pico = float(df_plot["EMERREL"].max())
if pico > 0:
    factor = MAX_PLANTS_CAP / pico
    plantas_cruda = (df_plot["EMERREL"] * factor).values
    plantas_eff   = (df_eff["EMERREL_eff"] * factor).values
    conv_caption = (f"Conversión: pico EMERREL={pico:.4f} → {MAX_PLANTS_CAP:.0f} pl·m² "
                    f"⇒ factor={factor:.2f} pl·m² por unidad EMERREL")
else:
    factor = None
    plantas_cruda = np.full(len(df_plot), np.nan)
    plantas_eff   = np.full(len(df_plot), np.nan)
    conv_caption = "No se pudo calcular Plantas·m² (pico EMERREL = 0)."
has_factor = (factor is not None) and np.isfinite(factor)

# ---- EMERREL × (1 − Ciec) → efecto de supresión ----
emerrel_supresion = (df_plot["EMERREL"].astype(float).values * (1.0 - Ciec)).clip(min=0.0) if use_ciec else np.full(len(df_plot), np.nan)
plantas_supresion = (emerrel_supresion * factor) if (has_factor and use_ciec) else np.full(len(df_plot), np.nan)

# =================== Cronograma y manejo manual =======================
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

# ======================= Eficiencias de control =======================
with st.sidebar:
    st.header("Eficiencia de control (%)")
    st.caption("Reducción relativa aplicada a EMERREL (ef-ICIC) y EMERREL×(1−Ciec) dentro de la ventana de efecto.")
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
        w[idxs] = 1.0 - (t_rel / max(1.0, L - 1))  # 1 → 0
    else:  # Exponencial
        assert lam_exp is not None
        w[idxs] = np.exp(-lam_exp * t_rel)
    return w

# Composición multiplicativa
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

# ==================== Aplicar control a series ========================
emerrel_eff_ctrl = df_eff["EMERREL_eff"].values * ctrl_factor
emerrel_supresion_ctrl = emerrel_supresion * ctrl_factor  # maneja NaN si no hay Ciec

if has_factor:
    plantas_eff_ctrl = plantas_eff * ctrl_factor
    plantas_supresion_ctrl = plantas_supresion * ctrl_factor
else:
    plantas_eff_ctrl = np.full(len(df_plot), np.nan)
    plantas_supresion_ctrl = np.full(len(df_plot), np.nan)

# ======= Sumatoria QUINCENAL (cada 15 días desde la siembra) =======
# Normalizar timestamps y restar contra sow_date para obtener días enteros
ts_norm = pd.to_datetime(df_plot["fecha"]).dt.normalize()
days_since = (ts_norm - pd.Timestamp(sow_date)).dt.days
mask_quince = (days_since >= 0) & (days_since % 15 == 0)
mask_q = mask_quince.to_numpy()

# Totales quincenales (solo entradas en días múltiplos de 15)
N_escape_sup_raw_quincenal = float(np.nansum(plantas_supresion[mask_q])) if has_factor else float("nan")
N_escape_sup_ctrl_raw_quincenal = float(np.nansum(plantas_supresion_ctrl[mask_q])) if has_factor else float("nan")

# Aplicar tope máximo posible (cap a 250)
N_escape_sup_quincenal = min(MAX_PLANTS_CAP, N_escape_sup_raw_quincenal) if np.isfinite(N_escape_sup_raw_quincenal) else float("nan")
N_escape_sup_ctrl_quincenal = min(MAX_PLANTS_CAP, N_escape_sup_ctrl_raw_quincenal) if np.isfinite(N_escape_sup_ctrl_raw_quincenal) else float("nan")

# ============================== Gráfico ===============================
fig = go.Figure()

# Customdata para tooltips (Plantas·m²)
cd_cruda = np.column_stack([plantas_cruda])
cd_eff   = np.column_stack([plantas_eff])
cd_sup   = np.column_stack([plantas_supresion])

fig.add_trace(go.Scatter(
    x=df_plot["fecha"], y=df_plot["EMERREL"], mode="lines",
    name="EMERREL (cruda)",
    customdata=cd_cruda,
    hovertemplate="Fecha: %{x|%Y-%m-%d}<br>EMERREL: %{y:.4f}<br>Plantas·m² (est): %{customdata[0]:.1f}<extra></extra>"
))
fig.add_trace(go.Scatter(
    x=df_eff["fecha"], y=df_eff["EMERREL_eff"], mode="lines",
    name="EMERREL (efectiva · ICIC)", line=dict(dash="dot"),
    customdata=cd_eff,
    hovertemplate="Fecha: %{x|%Y-%m-%d}<br>EMERREL (ef): %{y:.4f}<br>Plantas·m² (ef): %{customdata[0]:.1f}<extra></extra>"
))
fig.add_trace(go.Scatter(
    x=df_eff["fecha"], y=emerrel_eff_ctrl, mode="lines",
    name="EMERREL (ef·ICIC·control)", line=dict(dash="solid", width=2)
))

# Supresión por Ciec
if use_ciec:
    fig.add_trace(go.Scatter(
        x=df_plot["fecha"], y=emerrel_supresion, mode="lines",
        name="EMERREL × (1 − Ciec)", line=dict(dash="dashdot"),
        customdata=cd_sup,
        hovertemplate="Fecha: %{x|%Y-%m-%d}<br>EMERREL supresión: %{y:.4f}<br>Plantas·m² (sup.): %{customdata[0]:.1f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=df_plot["fecha"], y=emerrel_supresion_ctrl, mode="lines",
        name="EMERREL × (1 − Ciec) (control)", line=dict(dash="dot", width=2)
    ))

# ICIC / Ciec / EMERAC → ejes derechos
def add_icic_emac_traces(to_yaxis="y3", show_raw=False, show_adj=False, show_pr=False, show_padj=False):
    fig.add_trace(go.Scatter(x=df_ic["fecha"], y=df_ic["ICIC"], mode="lines", name="ICIC", yaxis=to_yaxis))
    if use_ciec:
        fig.add_trace(go.Scatter(x=df_ic["fecha"], y=df_ic["Ciec"], mode="lines", name="Ciec", yaxis=to_yaxis))
    if show_raw and len(work_base):
        fig.add_trace(go.Scatter(x=work_base["fecha"], y=work_base["EMERAC_N"], mode="lines",
                                 name="EMERAC (curva) cruda", yaxis=to_yaxis))
    if show_adj and len(work_star):
        fig.add_trace(go.Scatter(x=work_star["fecha"], y=work_star["EMERAC_N"], mode="lines",
                                 name="EMERAC (curva) ajustada", yaxis=to_yaxis, line=dict(dash="dash")))
    if show_pr and len(work_base):
        fig.add_trace(go.Scatter(x=work_base["fecha"], y=work_base["EMERAC_N"], mode="markers",
                                 name="EMERAC (puntos) cruda", yaxis=to_yaxis, marker=dict(size=6)))
    if show_padj and len(work_star):
        fig.add_trace(go.Scatter(x=work_star["fecha"], y=work_star["EMERAC_N"], mode="markers",
                                 name="EMERAC (puntos) ajustada", yaxis=to_yaxis, marker=dict(size=6, symbol="x")))

# Sombrear t*
if t_star is not None:
    fig.add_vrect(x0=t_star, x1=df_plot["fecha"].max(), fillcolor="LightGreen", opacity=0.15,
                  line_width=0, annotation_text="Impacto ICIC≥θ", annotation_position="top left")

# Helpers de bandas
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
        if highlight_labels:
            _add_label(x0 + (x1 - x0)/2, label, "rgba(219,112,147,0.8)")
    except Exception:
        return

def add_one_day_band(date_val, label):
    if date_val is None: return
    try:
        x0 = pd.to_datetime(date_val)
        x1 = x0 + pd.Timedelta(days=1)
        fig.add_vrect(x0=x0, x1=x1, line_width=0, fillcolor="Gold", opacity=0.25)
        if highlight_labels:
            _add_label(x0 + (x1 - x0)/2, label, "rgba(255,215,0,0.85)")
    except Exception:
        return

# Bandas residuales (pre/post R) y NR=10d para selectivos NO residuales
if pre_selR:  add_residual_band(pre_selR_date, pre_res_dias, f"Residual pre {pre_res_dias}d")
if post_selR: add_residual_band(post_selR_date, post_res_dias, f"Residual post {post_res_dias}d")
if show_nonres_bands:
    if pre_glifo: add_one_day_band(pre_glifo_date, "Glifo (1d)")
    if pre_selNR: add_residual_band(pre_selNR_date, NR_DAYS_DEFAULT, f"Sel. NR ({NR_DAYS_DEFAULT}d)")
    if post_gram: add_residual_band(post_gram_date, NR_DAYS_DEFAULT, f"Graminicida ({NR_DAYS_DEFAULT}d)")

# ======================== Ejes y escalas ==============================
ymax_base = float(df_plot["EMERREL"].max())
try:
    ymax_ctrl = float(np.nanmax(emerrel_eff_ctrl))
except Exception:
    ymax_ctrl = 0.0
if not np.isfinite(ymax_ctrl):
    ymax_ctrl = 0.0
ymax = max(1e-6, (max(ymax_base, ymax_ctrl) * 1.15))

layout_kwargs = dict(
    margin=dict(l=10, r=10, t=40, b=10),
    title=f"EMERREL + ICIC + supresión (1−Ciec) + Control (NR no residuales = {NR_DAYS_DEFAULT} días; residuales con decaimiento opcional)",
    xaxis_title="Fecha",
    yaxis_title="EMERREL",
    yaxis=dict(range=[0, ymax])
)

if show_plants_axis and has_factor:
    plantas_max = float(np.nanmax([
        np.nanmax(plantas_cruda), np.nanmax(plantas_eff),
        np.nanmax(plantas_supresion), np.nanmax(plantas_eff_ctrl),
        np.nanmax(plantas_supresion_ctrl)
    ]))
    if not np.isfinite(plantas_max) or plantas_max <= 0: plantas_max = 1.0

    layout_kwargs["yaxis2"] = dict(
        overlaying="y", side="right", title="Plantas·m² (estimadas)", position=1.0,
        range=[0, plantas_max*1.15]
    )
    fig.add_trace(go.Scatter(
        x=df_plot["fecha"], y=plantas_cruda,
        name="Plantas·m² (escala)", yaxis="y2",
        line=dict(width=0.1), opacity=0.0, showlegend=False, hoverinfo="skip"
    ))

    layout_kwargs["yaxis3"] = dict(
        overlaying="y", side="right", title="ICIC / Ciec / EMERAC (0–1)", position=0.97, range=[0, 1]
    )
    add_icic_emac_traces("y3", show_emac_curve_raw, show_emac_curve_adj,
                               show_emac_points_raw, show_emac_points_adj)

    if use_ciec and show_sup_density:
        fig.add_bar(x=df_plot["fecha"], y=plantas_supresion, name="Plantas·m² × (1−Ciec)", yaxis="y2", opacity=0.35, legendgroup="densidad")
        fig.add_bar(x=df_plot["fecha"], y=plantas_supresion_ctrl, name="Plantas·m² × (1−Ciec) (control)", yaxis="y2", opacity=0.35, legendgroup="densidad")
    if show_icic_density:
        fig.add_bar(x=df_eff["fecha"], y=plantas_eff,      name="Plantas·m² (ef · ICIC)",          yaxis="y2", opacity=0.35, legendgroup="densidad")
        fig.add_bar(x=df_eff["fecha"], y=plantas_eff_ctrl, name="Plantas·m² (ef · ICIC · control)", yaxis="y2", opacity=0.35, legendgroup="densidad")
else:
    layout_kwargs["yaxis2"] = dict(overlaying="y", side="right", title="ICIC / Ciec / EMERAC (0–1)", range=[0, 1])
    add_icic_emac_traces("y2", show_emac_curve_raw, show_emac_curve_adj,
                               show_emac_points_raw, show_emac_points_adj)

fig.update_layout(**layout_kwargs)
st.plotly_chart(fig, use_container_width=True)
st.caption(f"{conv_caption} · No residuales con NR por defecto = {NR_DAYS_DEFAULT} días. · Tope máx = {MAX_PLANTS_CAP:.0f} pl·m².")

# ======================= Totales visibles en UI (SOLO QUINCENAL) =====
st.subheader("Plantas·m² que escapan — Sumatoria QUINCENAL (con tope)")
st.markdown(
    f"""
**Quincenal — Solo supresión (1−Ciec):** **{N_escape_sup_quincenal:,.1f}** pl·m² _(tope {MAX_PLANTS_CAP:.0f})_  
**Quincenal — Supresión + control post:** **{N_escape_sup_ctrl_quincenal:,.1f}** pl·m² _(tope {MAX_PLANTS_CAP:.0f})_
"""
)

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
out["EMERREL_eff"] = df_eff["EMERREL_eff"]
out["EMERREL_supresion"] = emerrel_supresion
out["EMERREL_eff_ctrl"] = np.round(emerrel_eff_ctrl, 6)
out["EMERREL_supresion_ctrl"] = np.round(emerrel_supresion_ctrl, 6)

if has_factor:
    out["Plantas_m2_est_cruda"]            = np.round(plantas_cruda, 1)
    out["Plantas_m2_est_ef"]               = np.round(plantas_eff, 1)
    out["Plantas_m2_est_supresion"]        = np.round(plantas_supresion, 1)
    out["Plantas_m2_est_ef_ctrl"]          = np.round(plantas_eff_ctrl, 1)
    out["Plantas_m2_est_supresion_ctrl"]   = np.round(plantas_supresion_ctrl, 1)

    # Series QUINCENALES (solo en días múltiplos de 15 desde siembra; NaN resto)
    sup_q = np.where(mask_q, plantas_supresion, np.nan)
    sup_ctrl_q = np.where(mask_q, plantas_supresion_ctrl, np.nan)
    out["Plantas_m2_supresion_quincenal"] = np.round(sup_q, 1)
    out["Plantas_m2_supresion_ctrl_quincenal"] = np.round(sup_ctrl_q, 1)

    # Acumuladas QUINCENALES sin tope (para ver progresión intermedia)
    out["Plantas_m2_supresion_quincenal_acum"] = np.round(np.nancumsum(np.nan_to_num(sup_q, nan=0.0)), 1)
    out["Plantas_m2_supresion_ctrl_quincenal_acum"] = np.round(np.nancumsum(np.nan_to_num(sup_ctrl_q, nan=0.0)), 1)

    # Acumuladas QUINCENALES con tope 250
    out["Plantas_m2_supresion_quincenal_acum_cap"] = np.round(
        np.minimum(out["Plantas_m2_supresion_quincenal_acum"].values, MAX_PLANTS_CAP), 1
    )
    out["Plantas_m2_supresion_ctrl_quincenal_acum_cap"] = np.round(
        np.minimum(out["Plantas_m2_supresion_ctrl_quincenal_acum"].values, MAX_PLANTS_CAP), 1
    )
else:
    out["Plantas_m2_est_cruda"] = np.nan
    out["Plantas_m2_est_ef"]    = np.nan
    out["Plantas_m2_est_supresion"] = np.nan
    out["Plantas_m2_est_ef_ctrl"] = np.nan
    out["Plantas_m2_est_supresion_ctrl"] = np.nan
    out["Plantas_m2_supresion_quincenal"] = np.nan
    out["Plantas_m2_supresion_ctrl_quincenal"] = np.nan
    out["Plantas_m2_supresion_quincenal_acum"] = np.nan
    out["Plantas_m2_supresion_ctrl_quincenal_acum"] = np.nan
    out["Plantas_m2_supresion_quincenal_acum_cap"] = np.nan
    out["Plantas_m2_supresion_ctrl_quincenal_acum_cap"] = np.nan

with st.expander("Descargas de series", expanded=True):
    st.caption(conv_caption + f" · Tope máx = {MAX_PLANTS_CAP:.0f} pl·m² · Totales: SOLO quincenales")
    only_plants = st.checkbox("Mostrar/descargar sólo columnas de Plantas·m²", value=False)

    default_cols = [
        "fecha",
        "EMERREL", "EMERREL_eff", "EMERREL_supresion",
        "EMERREL_eff_ctrl", "EMERREL_supresion_ctrl",
        "Plantas_m2_est_cruda", "Plantas_m2_est_ef", "Plantas_m2_est_supresion",
        "Plantas_m2_est_ef_ctrl", "Plantas_m2_est_supresion_ctrl",
        # SOLO quincenales:
        "Plantas_m2_supresion_quincenal", "Plantas_m2_supresion_ctrl_quincenal",
        "Plantas_m2_supresion_quincenal_acum", "Plantas_m2_supresion_ctrl_quincenal_acum",
        "Plantas_m2_supresion_quincenal_acum_cap", "Plantas_m2_supresion_ctrl_quincenal_acum_cap"
    ]
    available_cols = [c for c in default_cols if c in out.columns] + [c for c in out.columns if c not in default_cols]
    preselect = [c for c in available_cols if c.startswith("Plantas_m2")] if only_plants else [c for c in available_cols if c in default_cols]
    selected_cols = st.multiselect("Columnas a incluir en la tabla y el CSV:", available_cols, default=preselect or default_cols)

    if not selected_cols:
        st.info("Seleccioná al menos una columna para mostrar/descargar.")
        selected_cols = preselect or available_cols

    st.dataframe(out[selected_cols].tail(20), use_container_width=True)
    st.download_button("Descargar serie procesada (CSV)",
                       out[selected_cols].to_csv(index=False).encode("utf-8"),
                       "serie_procesada.csv", "text/csv")

# ============================== Diagnóstico ===========================
st.subheader("Diagnóstico")
diag = {
    "siembra": str(sow_date),
    "t_impacto_ICIC": str(pd.to_datetime(t_star).date()) if t_star is not None else None,
    "suma_EMERREL": float(df_plot["EMERREL"].sum()),
    "suma_EMERREL_ef": float(df_eff["EMERREL_eff"].sum()),
    "suma_EMERREL_ef_ctrl": float(np.nansum(emerrel_eff_ctrl)),
    "suma_EMERREL_supresion": float(np.nansum(emerrel_supresion)),
    "suma_EMERREL_supresion_ctrl": float(np.nansum(emerrel_supresion_ctrl)),
    # Totales SOLO quincenales
    "N_escape_sup_quincenal_pl_m2_raw": N_escape_sup_raw_quincenal,
    "N_escape_sup_y_control_quincenal_pl_m2_raw": N_escape_sup_ctrl_raw_quincenal,
    "N_escape_sup_quincenal_pl_m2_cap": N_escape_sup_quincenal,
    "N_escape_sup_y_control_quincenal_pl_m2_cap": N_escape_sup_ctrl_quincenal,
    # Otros
    "cap_max_pl_m2": MAX_PLANTS_CAP,
    "tot_base": float(tot_base),
    "tot_ajustada": float(tot_star),
    "pico_EMERREL": float(df_plot["EMERREL"].max()),
    "factor_pl_m2_por_EMERREL": (MAX_PLANTS_CAP / float(df_plot["EMERREL"].max())) if float(df_plot["EMERREL"].max()) > 0 else None,
    "ICIC_sowing_day": float(ICIC[0]) if len(ICIC) else None,
    "decaimiento": decaimiento_tipo,
    "NR_no_residuales_dias": NR_DAYS_DEFAULT
}
st.code(json.dumps(diag, ensure_ascii=False, indent=2))
