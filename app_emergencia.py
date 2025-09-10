# -*- coding: utf-8 -*-
# app_emergencia.py — AVEFA (empalme estricto, JD como verdad, sin calendarizar) + Ciec (trigo) opcional
# - Histórico: BORDE2025.csv (01-ene-2025 → 03-sep-2025 inclusive, desde GitHub)
# - Futuro: meteo_history.csv (04-sep-2025 → última fecha realmente presente)
# - Lectura determinista por JD/Julian_days en ambos archivos (evita ambigüedades de formato)
# - Sin calendarizar ni interpolar días faltantes; la RN se alimenta SOLO con días existentes
# - Validador de continuidad (histórico y futuro)
# - NUEVO: Cálculo de TT → LAI_trigo (ec.16) → Ciec_t (ec.9) con **Dt reiniciado en siembra** y
#          ajuste opcional de EMERREL por supervivencia (EMERREL_ajustada = EMERREL × Ciec)

import streamlit as st
import numpy as np
import pandas as pd
from io import BytesIO, StringIO
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from pathlib import Path
import plotly.graph_objects as go
from typing import Callable, Any, List

# Defaults defensivos para evitar NameError en reruns parciales
use_ciec: bool = False
use_icic: bool = False

# =================== CONFIG UI / LOCKDOWN ===================
st.set_page_config(page_title="Predicción de Emergencia Agrícola AVEFA", layout="wide")
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header [data-testid="stToolbar"] {visibility: hidden;}
    .viewerBadge_container__1QSob {visibility: hidden;}
    .stAppDeployButton {display: none;}
    </style>
    """,
    unsafe_allow_html=True
)

# =================== HELPERS ===================

def safe_run(fn: Callable[[], Any], user_msg: str):
    try:
        return fn()
    except Exception:
        st.error(user_msg)
        return None

def _safe_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            st.warning("No pude forzar el rerun automáticamente. Volvé a ejecutar la app.")

def _to_raw_url(u: str) -> str:
    """Convierte URLs GitHub 'blob' a 'raw'. Deja intacto si ya es raw u otro host."""
    if not isinstance(u, str) or not u:
        return u
    u = u.strip()
    if "github.com" in u and "/blob/" in u:
        return u.replace("https://github.com/", "https://raw.githubusercontent.com/").replace("/blob/", "/")
    return u

def _fetch_bytes(url: str, timeout: int = 20) -> bytes:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except (HTTPError, URLError):
        raise RuntimeError(f"No se pudo descargar: {url}")
    except Exception:
        raise RuntimeError("Error descargando recurso remoto.")

# =================== MODELO / PESOS ===================
GITHUB_BASE_URL = "https://raw.githubusercontent.com/PREDWEEM/AVEFA2/main"
FNAME_IW, FNAME_BIW, FNAME_LW, FNAME_BOUT = "IW.npy", "bias_IW.npy", "LW.npy", "bias_out.npy"

@st.cache_data(ttl=1800)
def load_npy_from_fixed(filename: str) -> np.ndarray:
    raw = _fetch_bytes(f"{GITHUB_BASE_URL}/{filename}")
    return np.load(BytesIO(raw), allow_pickle=False)

class PracticalANNModel:
    def __init__(self, IW, bias_IW, LW, bias_out):
        self.IW, self.bias_IW, self.LW = IW, bias_IW, LW
        self.bias_out = float(bias_out)
        # Orden de entrada: [Julian_days, TMIN, TMAX, Prec]
        self.input_min = np.array([1, -7, 0, 0], dtype=float)
        self.input_max = np.array([300, 25.5, 41, 84], dtype=float)
        self._den = np.maximum(self.input_max - self.input_min, 1e-9)

    def _tansig(self, x): return np.tanh(x)
    def _normalize(self, X):
        Xc = np.clip(X, self.input_min, self.input_max)
        return 2 * (Xc - self.input_min) / self._den - 1
    def _denorm_out(self, y, ymin=-1, ymax=1): return (y - ymin) / (ymax - ymin)

    def predict(self, X_real, thr_bajo_medio, thr_medio_alto):
        Xn = self._normalize(X_real)
        z1 = Xn @ self.IW + self.bias_IW
        a1 = self._tansig(z1)
        LW2 = self.LW.reshape(1, -1) if self.LW.ndim == 1 else self.LW
        z2 = (a1 @ LW2.T).ravel() + self.bias_out
        y  = self._denorm_out(self._tansig(z2))  # [0,1]
        ac = np.cumsum(y) / 8.05
        diff = np.diff(ac, prepend=0)
        niveles = np.where(diff <= THR_BAJO_MEDIO, "Bajo",
                   np.where(diff <= THR_MEDIO_ALTO, "Medio", "Alto"))
        return pd.DataFrame({"EMERREL(0-1)": diff, "Nivel_Emergencia_relativa": niveles})

def cargar_modelo():
    IW = load_npy_from_fixed(FNAME_IW)
    b1 = load_npy_from_fixed(FNAME_BIW)
    LW = load_npy_from_fixed(FNAME_LW)
    bo = load_npy_from_fixed(FNAME_BOUT)
    if LW.ndim == 1: LW = LW.reshape(1, -1)
    bias_out = float(bo if np.ndim(bo)==0 else np.ravel(bo)[0])
    assert IW.shape[0] == 4 and b1.shape[0] == IW.shape[1] and LW.shape == (1, IW.shape[1])
    return PracticalANNModel(IW, b1, LW, bias_out)

# =================== PARÁMETROS ===================
THR_BAJO_MEDIO = 0.01
THR_MEDIO_ALTO = 0.05
EMEAC_MIN_DEN, EMEAC_ADJ_DEN, EMEAC_MAX_DEN = 3.0, 4.0, 5.0
HIST_START = pd.Timestamp(2025, 1, 1)
HIST_END   = pd.Timestamp(2025, 9, 3)  # inclusive
COLOR_MAP = {"Bajo": "#00A651", "Medio": "#FFC000", "Alto": "#E53935"}

# =================== SANITIZACIÓN (JD como verdad) ===================

def _sanitize_generic_prefer_jd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanea meteo para BORDE2025 y meteo_history:
    - Si hay JD/Julian_days/jd/etc, se usa para construir Fecha (verdad) desde 2025-01-01 + (JD-1).
    - Si no hay JD, se parsea Fecha y se deriva JD.
    - No calendariza ni rellena. Corrige Tmin/Tmax y clip Prec.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    def pick(*cands):
        for c in cands:
            if c in lower: return lower[c]
        return None

    c_fecha = pick("fecha","date")
    c_jd    = pick("julian_days","julianday","doy","dia_juliano","dayofyear","juliano","jd")
    c_tmax  = pick("tmax","t_max","tx")
    c_tmin  = pick("tmin","t_min","tn")
    c_prec  = pick("prec","ppt","precip","lluvia","prcp","mm")

    mapping = {}
    if c_fecha: mapping[c_fecha] = "Fecha"
    if c_jd:    mapping[c_jd]    = "Julian_days"
    if c_tmax:  mapping[c_tmax]  = "TMAX"
    if c_tmin:  mapping[c_tmin]  = "TMIN"
    if c_prec:  mapping[c_prec]  = "Prec"
    if mapping: df = df.rename(columns=mapping)

    # Tipos numéricos
    if "Julian_days" in df.columns:
        df["Julian_days"] = pd.to_numeric(df["Julian_days"], errors="coerce")

    for c in ["TMAX","TMIN","Prec"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Fecha/JD
    if "Julian_days" in df.columns and df["Julian_days"].notna().any():
        df["Fecha"] = pd.to_datetime("2025-01-01") + pd.to_timedelta(df["Julian_days"] - 1, unit="D")
    elif "Fecha" in df.columns:
        d1 = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=True)
        d2 = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=False)
        use_dayfirst = (d1.dt.year==2025).sum() >= (d2.dt.year==2025).sum()
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=use_dayfirst)
        df["Julian_days"] = df["Fecha"].dt.dayofyear
    else:
        return pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])

    # Saneos físicos
    if "Prec" in df.columns: df["Prec"] = df["Prec"].clip(lower=0)
    if {"TMAX","TMIN"}.issubset(df.columns):
        m = (df["TMAX"] < df["TMIN"])
        if m.any():
            df.loc[m, ["TMAX","TMIN"]] = df.loc[m, ["TMIN","TMAX"]].values

    # Orden final
    df = df.dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)
    return df[["Fecha","Julian_days","TMAX","TMIN","Prec"]]

# =================== CARGA BORDE2025 (GitHub) ===================
HIST_CSV_URL_SECRET = st.secrets.get("HIST_CSV_URL", "").strip()
HIST_CSV_URLS: List[str] = [
    "https://raw.githubusercontent.com/PREDWEEM/AVEFA2/main/BORDE2025.csv",
    "https://raw.githubusercontent.com/PREDWEEM/ANN/gh-pages/BORDE2025.csv",
    "https://PREDWEEM.github.io/ANN/BORDE2025.csv",
]

def _try_read_csv_semicolon_first(url: str) -> pd.DataFrame:
    raw = _fetch_bytes(url)
    for opt in [dict(sep=";", encoding="utf-8-sig"), dict(encoding="utf-8-sig")]:
        try:
            return pd.read_csv(BytesIO(raw), **opt)
        except Exception:
            continue
    raise RuntimeError("No se pudo leer CSV.")

@st.cache_data(ttl=900)
def load_borde_from_github() -> pd.DataFrame:
    urls = [HIST_CSV_URL_SECRET] if HIST_CSV_URL_SECRET else []
    urls += HIST_CSV_URLS
    last_err = None
    for url in urls:
        try:
            df = _try_read_csv_semicolon_first(url)
            if not df.empty: return df
        except Exception as e:
            last_err = e; continue
    raise RuntimeError(f"No pude leer BORDE2025.csv (último error: {last_err})")

def _load_borde_hist() -> pd.DataFrame:
    df_raw = load_borde_from_github()
    df = _sanitize_generic_prefer_jd(df_raw)
    return df[(df["Fecha"] >= HIST_START) & (df["Fecha"] <= HIST_END)].copy()

# =================== CARGA meteo_history (GitHub o local) ===================
@st.cache_data(ttl=900)
def load_meteo_history_csv(url_override: str = "") -> pd.DataFrame:
    urls: List[str] = []
    if url_override.strip(): urls.append(_to_raw_url(url_override))
    urls.append("https://raw.githubusercontent.com/PREDWEEM/AVEFA2/main/meteo_history.csv")
    urls += [
        "https://PREDWEEM.github.io/ANN/meteo_history.csv",
        "https://raw.githubusercontent.com/PREDWEEM/ANN/gh-pages/meteo_history.csv",
    ]
    last_err = None
    for url in urls:
        try:
            raw = _fetch_bytes(url)
            for opt in [dict(sep=";", decimal=",", encoding="utf-8-sig"),
                        dict(sep=";", decimal=".", encoding="utf-8-sig"),
                        dict(sep=",", decimal=".", encoding="utf-8-sig")]:
                try:
                    df0 = pd.read_csv(BytesIO(raw), engine="python", **opt)
                    df1 = _sanitize_generic_prefer_jd(df0)
                    if not df1.empty: return df1, url
                except Exception as e:
                    last_err = e; continue
        except Exception as e:
            last_err = e; continue
    # Fallback local
    try:
        p = Path("/mnt/data/meteo_history.csv")
        if p.exists():
            try:
                df0 = pd.read_csv(p, sep=";", decimal=",", engine="python", encoding="utf-8-sig")
            except Exception:
                df0 = pd.read_csv(p, engine="python", encoding="utf-8-sig")
            return _sanitize_generic_prefer_jd(df0), str(p)
    except Exception as e:
        last_err = e
    raise RuntimeError(f"No se pudo cargar meteo_history.csv (último error: {last_err})")

# =================== EMPALME ESTRICTO ===================

def construir_empalme(url_override: str = "") -> pd.DataFrame:
    # 1) Histórico: 01-ene → 03-sep (incl.)
    df_hist = _load_borde_hist()
    df_hist = df_hist[df_hist["Fecha"] <= HIST_END].copy()

    # 2) Futuro: meteo_history desde 04-sep → último día realmente presente
    try:
        df_mh, used = load_meteo_history_csv(url_override)
        st.success(f"meteo_history.csv cargado desde: {used}")
    except Exception as e:
        st.warning(str(e))
        df_mh = pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])

    df_mh = df_mh[df_mh["Fecha"] >= (HIST_END + pd.Timedelta(days=1))].copy()
    end_date = df_mh["Fecha"].max() if not df_mh.empty else HIST_END
    if pd.isna(end_date): end_date = HIST_END
    if not df_mh.empty:
        df_mh = df_mh[df_mh["Fecha"] <= end_date].copy()

    # 3) Unir y limitar a [HIST_START, end_date]
    df_emp = pd.concat([df_hist, df_mh], ignore_index=True)
    if not df_emp.empty:
        df_emp = df_emp[(df_emp["Fecha"] >= HIST_START) & (df_emp["Fecha"] <= end_date)].copy()
        df_emp = df_emp.dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)

    # 4) Diagnóstico
    if df_mh.empty:
        st.info("No hay días en meteo_history.csv posteriores al 2025-09-03; el empalme queda en 2025-09-03.")
    else:
        st.info(
            f"Empalme: {df_emp['Fecha'].min().date()} → {df_emp['Fecha'].max().date()} "
            f"({len(df_emp)} fila(s)). Último día leído en meteo_history.csv = {end_date.date()}."
        )
    return df_emp

# =================== VALIDACIÓN DE CONTINUIDAD ===================

def _validar_continuidad(df: pd.DataFrame, desde: pd.Timestamp, hasta: pd.Timestamp, etiqueta="Empalme"):
    """Chequea que haya TODOS los días calendario entre 'desde' y 'hasta' (incl.)."""
    if df.empty:
        st.error(f"{etiqueta}: DF vacío.")
        return
    cal = pd.date_range(desde, hasta, freq="D")
    present = pd.to_datetime(df["Fecha"]).dt.normalize().unique()
    missing = [d for d in cal if d.to_datetime64() not in present]
    if missing:
        st.error(f"{etiqueta}: faltan {len(missing)} día(s): " +
                 ", ".join(pd.DatetimeIndex(missing).strftime("%d-%m").tolist()))
    else:
        st.success(f"{etiqueta}: secuencia completa {desde.date()} → {hasta.date()} (sin huecos).")

# =================== (NUEVO) CIEC TRIGO ===================
# TT (°Cd) con reinicio en siembra, LAI trigo (ec. 16), Ciec_t (ec. 9)

def _acum_tt(df: pd.DataFrame, tb_crop: float = 0.0, fecha_siembra=None) -> pd.Series:
    """Acumula TT (°Cd) y reinicia a cero en la fecha de siembra si se provee."""
    if not {"TMAX","TMIN"}.issubset(df.columns):
        return pd.Series(np.nan, index=df.index)
    tmean = (pd.to_numeric(df["TMAX"], errors="coerce") + pd.to_numeric(df["TMIN"], errors="coerce")) / 2.0
    ttd = np.maximum(tmean - float(tb_crop), 0.0)
    Dt = pd.Series(ttd, index=df.index).cumsum()
    if fecha_siembra is not None:
        sow = pd.to_datetime(fecha_siembra).normalize()
        fechas = pd.to_datetime(df["Fecha"]).dt.normalize()
        if (fechas == sow).any():
            offset = float(Dt.loc[fechas == sow].iloc[0])
            Dt = Dt - offset
            Dt[Dt < 0] = 0.0
    return Dt

def _lai_trigo(Dt: np.ndarray,
               p1=0.1138, p2=3.71e-3, p3=47.98, p4=0.08012, p5=5.02e-5, p6=1.07e-8,
               G1=1116.0, G2=2260.0) -> np.ndarray:
    Dt = np.asarray(Dt, dtype=float)
    lai = np.zeros_like(Dt, dtype=float)
    idx1 = Dt < G1; idx2 = (Dt >= G1) & (Dt < G2)
    lai[idx1] = p1*Dt[idx1] + p2*(Dt[idx1]**2)
    lai[idx2] = p3 + p4*Dt[idx2] + p5*(Dt[idx2]**2) + p6*(Dt[idx2]**3)
    return np.clip(lai, 0.0, None)

def _ciec(LAI_t: np.ndarray, LAIhc=6.0, Cs=200.0, Ca=200.0) -> np.ndarray:
    ciec = (np.asarray(LAI_t, float) / float(LAIhc)) * (float(Cs) / float(Ca))
    return np.minimum(ciec, 1.0)

# =================== SIDEBAR ===================

st.sidebar.header("Opciones")
if "cache_bust" not in st.session_state: st.session_state.cache_bust = 0
col_a, col_b = st.sidebar.columns(2)
with col_a:
    if st.button("🔄 Actualizar"):
        st.cache_data.clear(); st.session_state.cache_bust += 1; _safe_rerun()
with col_b:
    if st.button("🧹 Limpiar caché"):
        st.cache_data.clear(); st.success("Caché limpiada. Volvé a correr o tocá Actualizar.")

rango_opcion = st.sidebar.radio(
    "Rango para mostrar", ["1/feb → 1/nov", "Todo el empalme"], index=0
)
st.sidebar.markdown("---")
meteo_history_url_override = st.sidebar.text_input(
    "URL de meteo_history.csv (opcional)",
    value="https://github.com/PREDWEEM/AVEFA2/blob/main/meteo_history.csv",
    help="Podés pegar blob o raw; se convierte a raw automáticamente."
)
ALPHA = st.sidebar.slider("Opacidad relleno MA5", 0.0, 1.0, 0.70, 0.05)
# Mostrar eje Y con valores reales de la MA5 (pl·m⁻² día⁻¹)
Y_MA5_REAL = st.sidebar.checkbox("Eje Y: mostrar valores reales de MA5 (pl·m⁻² d⁻¹)", value=False)

# (NUEVO) Bloque Ciec (trigo)
st.sidebar.markdown("---")
st.sidebar.header("Competencia del cultivo (trigo) — Ciec")
use_ciec = st.sidebar.checkbox("Calcular Ciec (TT→LAI→Ciec)", value=False)
apply_ciec_to_emerrel = st.sidebar.checkbox("Ajustar EMERREL por Ciec (EMERREL×Ciec)", value=False, help="Interpreta Ciec como supervivencia de plántulas en s=1")
Tb_crop = st.sidebar.number_input("T° base trigo (°C)", min_value=-5.0, max_value=10.0, value=0.0, step=0.5)
Ca = st.sidebar.number_input("Ca (densidad real trigo, pl·m⁻²)", min_value=80.0, max_value=600.0, value=200.0, step=10.0)
Cs = st.sidebar.number_input("Cs (densidad estándar trigo, pl·m⁻²)", min_value=80.0, max_value=600.0, value=200.0, step=10.0)
LAIhc = st.sidebar.number_input("LAIhc (escenario altamente competitivo)", min_value=0.5, max_value=10.0, value=6.0, step=0.1)
with st.sidebar.expander("Parámetros LAI (ec.16)"):
    p1 = st.number_input("p1", value=0.1138)
    p2 = st.number_input("p2", value=3.71e-3, format="%.6f")
    p3 = st.number_input("p3", value=47.98)
    p4 = st.number_input("p4", value=0.08012)
    p5 = st.number_input("p5", value=5.02e-5, format="%.6f")
    p6 = st.number_input("p6", value=1.07e-8, format="%.8f")
    G1 = st.number_input("G1 (°Cd a LAI máx)", value=1116.0)
    G2 = st.number_input("G2 (°Cd a madurez)", value=2260.0)
# La fecha de siembra para Ciec se elige más abajo (necesita el DF empalmado para un valor por defecto)

# =================== CARGA MODELO ===================
modelo = safe_run(cargar_modelo, "No se pudieron cargar los archivos del modelo.")
if modelo is None: st.stop()

# =================== CONSTRUIR EMPALME, VALIDAR Y DIAGNOSTICAR ===================
df_empalmado = construir_empalme(meteo_history_url_override)
if df_empalmado.empty:
    st.error("No hay datos tras el empalme (¿falta histórico o futuro?).")
    st.stop()

# Validaciones
_validar_continuidad(
    df_empalmado[df_empalmado["Fecha"] <= HIST_END],
    HIST_START, HIST_END, etiqueta="Histórico (BORDE2025)"
)
mask_fut = df_empalmado["Fecha"] > HIST_END
if mask_fut.any():
    fut_min, fut_max = df_empalmado.loc[mask_fut, "Fecha"].min(), df_empalmado.loc[mask_fut, "Fecha"].max()
    _validar_continuidad(
        df_empalmado[mask_fut],
        fut_min.normalize(), fut_max.normalize(), etiqueta="Futuro (meteo_history)"
    )

with st.expander("🔎 Diagnóstico del empalme (histórico + futuro)"):
    def _fmt(x):
        try: return pd.to_datetime(x).strftime("%Y-%m-%d") if pd.notna(x) else "—"
        except Exception: return "—"
    hist_last = df_empalmado.loc[df_empalmado["Fecha"] <= HIST_END, "Fecha"].max()
    fut_min   = df_empalmado.loc[mask_fut, "Fecha"].min() if mask_fut.any() else pd.NaT
    st.write("Último histórico:", _fmt(hist_last))
    st.write("Primer futuro:", _fmt(fut_min))
    st.write("Última fecha del empalme:", _fmt(df_empalmado["Fecha"].max()))
    st.write("Filas empalmadas:", len(df_empalmado))

# =================== PREDICCIÓN (sólo días presentes) ===================
X_all = df_empalmado[["Julian_days","TMIN","TMAX","Prec"]].to_numpy(float)
pred = modelo.predict(X_all, thr_bajo_medio=THR_BAJO_MEDIO, thr_medio_alto=THR_MEDIO_ALTO)
pred["Fecha"] = df_empalmado["Fecha"].values
pred["Julian_days"] = df_empalmado["Julian_days"].values

# ======= Ciec: elegir fecha de siembra y calcular (LAI comienza en 0 ese día) =======
fecha_siembra_ciec = None
if use_ciec:
    fecha_siembra_ciec = pd.to_datetime(df_empalmado["Fecha"].min()).date()
    fecha_siembra_ciec = st.sidebar.date_input("Fecha de siembra trigo (para Ciec)", value=fecha_siembra_ciec)
    Dt = _acum_tt(df_empalmado, tb_crop=Tb_crop, fecha_siembra=fecha_siembra_ciec)
    LAI = _lai_trigo(Dt.values, p1,p2,p3,p4,p5,p6, G1,G2)
    Ciec = _ciec(LAI, LAIhc=LAIhc, Cs=Cs, Ca=Ca)
    pred["LAI_trigo"] = LAI
    pred["Ciec_trigo"] = Ciec
else:
    pred["LAI_trigo"] = np.nan
    pred["Ciec_trigo"] = np.nan

# EMERREL (y ajuste opcional por Ciec)
base_emerrel = pred["EMERREL(0-1)"].copy()
if use_ciec and apply_ciec_to_emerrel:
    pred["EMERREL(0-1)"] = (pred["EMERREL(0-1)"] * pred["Ciec_trigo"].fillna(1.0)).clip(lower=0)
    pred["Ajuste_Ciec_aplicado"] = True
else:
    pred["Ajuste_Ciec_aplicado"] = False

pred["EMERREL acumulado"] = pred["EMERREL(0-1)"].cumsum()

# ======= Escalado "figura original": EMERREL en pl·m⁻² día⁻¹ (pico = 350) =======
max_val = pd.to_numeric(pred["EMERREL(0-1)"], errors="coerce").max()
scale_factor = 350.0 / max_val if pd.notna(max_val) and max_val > 0 else np.nan
pred["EMERREL_pl_m2"] = pred["EMERREL(0-1)"].astype(float) * (scale_factor if pd.notna(scale_factor) else np.nan)
# MA5 global (normalizada y en pl·m⁻² día⁻¹)
pred["EMERREL_MA5"] = pred["EMERREL(0-1)"].rolling(5, min_periods=1).mean()
pred["EMERREL_MA5_pl_m2"] = pred["EMERREL_MA5"] * (scale_factor if pd.notna(scale_factor) else np.nan)
# Si luego ajustamos por Ciec, el mismo factor permite comparar en la misma escala (barras)


# EMEAC (global)
pred["EMEAC (0-1) - mínimo"]    = pred["EMERREL acumulado"] / EMEAC_MIN_DEN
pred["EMEAC (0-1) - máximo"]    = pred["EMERREL acumulado"] / EMEAC_MAX_DEN
pred["EMEAC (0-1) - ajustable"] = pred["EMERREL acumulado"] / EMEAC_ADJ_DEN
for col in ["EMEAC (0-1) - mínimo","EMEAC (0-1) - máximo","EMEAC (0-1) - ajustable"]:
    pred[col.replace("(0-1)","(%)")] = (pred[col]*100).clip(0,100)

# Recalcular niveles después del posible ajuste
THR_BM, THR_MA = float(THR_BAJO_MEDIO), float(THR_MEDIO_ALTO)
pred["Nivel_Emergencia_relativa"] = np.where(
    pred["EMERREL(0-1)"] <= THR_BM, "Bajo",
    np.where(pred["EMERREL(0-1)"] <= THR_MA, "Medio", "Alto")
)

# =================== RANGO VISUAL ===================
if rango_opcion == "Todo el empalme":
    pred_vis = pred.copy()
    fi, ff = pred_vis["Fecha"].min(), pred_vis["Fecha"].max()
    y_min = pred_vis["EMEAC (%) - mínimo"]; y_max = pred_vis["EMEAC (%) - máximo"]; y_adj = pred_vis["EMEAC (%) - ajustable"]
    rango_txt = f"{fi.date()} → {ff.date()}"
else:
    years = pred["Fecha"].dt.year.unique()
    yr = int(years[0]) if len(years)==1 else int(st.sidebar.selectbox("Año (reinicio 1/feb → 1/nov)", sorted(years)))
    fi, ff = pd.Timestamp(yr,2,1), pd.Timestamp(yr,11,1)
    m = (pred["Fecha"]>=fi)&(pred["Fecha"]<=ff)
    pred_vis = pred.loc[m].copy() if m.any() else pred.copy()
    fi = pred_vis["Fecha"].min(); ff = pred_vis["Fecha"].max()
    pred_vis["EMERREL acumulado (reiniciado)"] = pred_vis["EMERREL(0-1)"].cumsum()
    pred_vis["EMEAC (0-1) - mínimo (rango)"]    = pred_vis["EMERREL acumulado (reiniciado)"]/EMEAC_MIN_DEN
    pred_vis["EMEAC (0-1) - máximo (rango)"]    = pred_vis["EMERREL acumulado (reiniciado)"]/EMEAC_MAX_DEN
    pred_vis["EMEAC (0-1) - ajustable (rango)"] = pred_vis["EMERREL acumulado (reiniciado)"]/EMEAC_ADJ_DEN
    for col in ["EMEAC (0-1) - mínimo (rango)","EMEAC (0-1) - máximo (rango)","EMEAC (0-1) - ajustable (rango)"]:
        pred_vis[col.replace("(0-1)","(%)")] = (pred_vis[col]*100).clip(0,100)
    y_min = pred_vis["EMEAC (%) - mínimo (rango)"]; y_max = pred_vis["EMEAC (%) - máximo (rango)"]; y_adj = pred_vis["EMEAC (%) - ajustable (rango)"]
    rango_txt = "1/feb → 1/nov"

# =================== GRÁFICOS ===================
st.title("Predicción de Emergencia Agrícola AVEFA")

# EMERGENCIA RELATIVA DIARIA
st.subheader("EMERGENCIA RELATIVA DIARIA")
# MA5 por vista y su versión en pl·m⁻²
pred_vis["EMERREL_MA5"] = pred_vis["EMERREL(0-1)"].rolling(5, min_periods=1).mean()
pred_vis["EMERREL_MA5_pl_m2"] = pred_vis["EMERREL_MA5"] * (scale_factor if pd.notna(scale_factor) else np.nan)
colores_vis = pred_vis["Nivel_Emergencia_relativa"].map(COLOR_MAP).fillna("#808080").to_numpy()

fig_er = go.Figure()
if Y_MA5_REAL:
    # --- Escala real (pl·m⁻² d⁻¹) ---
    # Barras en pl·m⁻² d⁻¹
    fig_er.add_bar(
        x=pred_vis["Fecha"], y=pred_vis["EMERREL_pl_m2"],
        marker=dict(color=colores_vis.tolist()),
        customdata=pred_vis["Nivel_Emergencia_relativa"].map({"Bajo":"🟢 Bajo","Medio":"🟡 Medio","Alto":"🔴 Alto"}),
        hovertemplate="Fecha: %{x|%d-%b-%Y}<br>EMERREL: %{y:.1f} pl·m⁻² d⁻¹<br>Nivel: %{customdata}<extra></extra>",
        name="EMERREL (pl·m⁻² d⁻¹)"
    )
    # Relleno tricolor interno bajo MA5 real
    x = pred_vis["Fecha"]; ma_pl = pred_vis["EMERREL_MA5_pl_m2"].clip(lower=0)
    thr_low_pl = (float(THR_BAJO_MEDIO) * scale_factor) if pd.notna(scale_factor) else np.nan
    thr_med_pl = (float(THR_MEDIO_ALTO) * scale_factor) if pd.notna(scale_factor) else np.nan
    y0 = np.zeros(len(ma_pl)); y1 = np.minimum(ma_pl, thr_low_pl); y2 = np.minimum(ma_pl, thr_med_pl); y3 = ma_pl
    GREEN_RGBA  = f"rgba(0,166,81,{ALPHA})"; YELLOW_RGBA = f"rgba(255,192,0,{ALPHA})"; RED_RGBA = f"rgba(229,57,53,{ALPHA})"
    fig_er.add_trace(go.Scatter(x=x, y=y0, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig_er.add_trace(go.Scatter(x=x, y=y1, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=GREEN_RGBA, hoverinfo="skip", showlegend=False, name="Zona baja (verde)"))
    fig_er.add_trace(go.Scatter(x=x, y=y1, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig_er.add_trace(go.Scatter(x=x, y=y2, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=YELLOW_RGBA, hoverinfo="skip", showlegend=False, name="Zona media (amarillo)"))
    fig_er.add_trace(go.Scatter(x=x, y=y2, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig_er.add_trace(go.Scatter(x=x, y=y3, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=RED_RGBA, hoverinfo="skip", showlegend=False, name="Zona alta (rojo)"))
    fig_er.add_trace(go.Scatter(x=x, y=ma_pl, mode="lines", line=dict(width=2), name="Media móvil 5 días (pl·m⁻² d⁻¹)",
                                hovertemplate="Fecha: %{x|%d-%b-%Y}<br>MA5: %{y:.1f} pl·m⁻² d⁻¹<extra></extra>"))
    # Umbrales en pl·m⁻² d⁻¹
    if pd.notna(thr_low_pl):
        fig_er.add_trace(go.Scatter(x=[x.min(), x.max()], y=[thr_low_pl, thr_low_pl], mode="lines", line=dict(color=COLOR_MAP["Bajo"], dash="dot"), name=f"Bajo (≤ {thr_low_pl:.1f})", hoverinfo="skip"))
    if pd.notna(thr_med_pl):
        fig_er.add_trace(go.Scatter(x=[x.min(), x.max()], y=[thr_med_pl, thr_med_pl], mode="lines", line=dict(color=COLOR_MAP["Medio"], dash="dot"), name=f"Medio (≤ {thr_med_pl:.1f})", hoverinfo="skip"))
    fig_er.update_layout(xaxis_title="Fecha", yaxis_title="EMERREL / MA5 (pl·m⁻² d⁻¹)", hovermode="x unified", legend_title="Referencias", height=650)
else:
    # --- Escala normalizada (0–1) ---
    fig_er.add_bar(
        x=pred_vis["Fecha"], y=pred_vis["EMERREL(0-1)"],
        marker=dict(color=colores_vis.tolist()),
        customdata=pred_vis["Nivel_Emergencia_relativa"].map({"Bajo":"🟢 Bajo","Medio":"🟡 Medio","Alto":"🔴 Alto"}),
        hovertemplate="Fecha: %{x|%d-%b-%Y}<br>EMERREL: %{y:.3f}<br>Nivel: %{customdata}<extra></extra>",
        name="EMERREL (0-1)"
    )
    x = pred_vis["Fecha"]; ma = pred_vis["EMERREL_MA5"].clip(lower=0)
    thr_low, thr_med = float(THR_BAJO_MEDIO), float(THR_MEDIO_ALTO)
    y0 = np.zeros(len(ma)); y1 = np.minimum(ma, thr_low); y2 = np.minimum(ma, thr_med); y3 = ma
    GREEN_RGBA  = f"rgba(0,166,81,{ALPHA})"; YELLOW_RGBA = f"rgba(255,192,0,{ALPHA})"; RED_RGBA = f"rgba(229,57,53,{ALPHA})"
    fig_er.add_trace(go.Scatter(x=x, y=y0, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig_er.add_trace(go.Scatter(x=x, y=y1, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=GREEN_RGBA, hoverinfo="skip", showlegend=False, name="Zona baja (verde)"))
    fig_er.add_trace(go.Scatter(x=x, y=y1, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig_er.add_trace(go.Scatter(x=x, y=y2, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=YELLOW_RGBA, hoverinfo="skip", showlegend=False, name="Zona media (amarillo)"))
    fig_er.add_trace(go.Scatter(x=x, y=y2, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig_er.add_trace(go.Scatter(x=x, y=y3, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=RED_RGBA, hoverinfo="skip", showlegend=False, name="Zona alta (rojo)"))
    fig_er.add_trace(go.Scatter(x=x, y=ma, mode="lines", line=dict(width=2), name="Media móvil 5 días",
                                hovertemplate="Fecha: %{x|%d-%b-%Y}<br>MA5: %{y:.3f}<extra></extra>"))
    fig_er.add_trace(go.Scatter(x=[x.min(), x.max()], y=[thr_low, thr_low], mode="lines", line=dict(color=COLOR_MAP["Bajo"], dash="dot"), name=f"Bajo (≤ {thr_low:.3f})", hoverinfo="skip"))
    fig_er.add_trace(go.Scatter(x=[x.min(), x.max()], y=[thr_med, thr_med], mode="lines", line=dict(color=COLOR_MAP["Medio"], dash="dot"), name=f"Medio (≤ {thr_med:.3f})", hoverinfo="skip"))
    fig_er.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=COLOR_MAP["Alto"], dash="dot"), name=f"Alto (> {thr_med:.3f})", hoverinfo="skip"))
    fig_er.update_layout(xaxis_title="Fecha", yaxis_title="EMERREL (0-1)", hovermode="x unified", legend_title="Referencias", height=650)

fi, ff = pred_vis["Fecha"].min(), pred_vis["Fecha"].max()
fig_er.update_xaxes(range=[fi, ff], dtick="D1" if (ff-fi).days <= 31 else "M1", tickformat="%d-%b" if (ff-fi).days <= 31 else "%b")
fig_er.update_yaxes(rangemode="tozero")
st.plotly_chart(fig_er, use_container_width=True, theme="streamlit")

# ======= Gráfico combinado solicitado =======
st.subheader("EMERREL normalizada y en pl·m⁻², con Ciec/ICIC opcionales (eje secundario)")
try:
    fig_all = go.Figure()

    # --- BARRAS en eje primario (densidad pl·m⁻² día⁻¹) ---
    fig_all.add_bar(x=pred["Fecha"], y=pred["EMERREL_pl_m2"], name="EMERREL (pl·m⁻² día⁻¹)", opacity=0.35, yaxis="y1")
    if use_ciec:
        emerrel_x_ciec = (base_emerrel * pred["Ciec_trigo"].fillna(1.0)).clip(lower=0)
        emerrel_x_ciec_plm2 = emerrel_x_ciec.astype(float) * (scale_factor if pd.notna(scale_factor) else np.nan)
        fig_all.add_bar(x=pred["Fecha"], y=emerrel_x_ciec_plm2, name="EMERREL×Ciec (pl·m⁻² día⁻¹)", opacity=0.65, yaxis="y1")

    # --- LÍNEAS en eje secundario (0–1) ---
    # EMERREL normalizada (0-1) como línea para comparar forma
    fig_all.add_trace(go.Scatter(x=pred["Fecha"], y=base_emerrel, mode="lines", name="EMERREL (0–1)", yaxis="y2"))
    if use_ciec:
        fig_all.add_trace(go.Scatter(x=pred["Fecha"], y=pred["Ciec_trigo"], mode="lines", name="Ciec_t (0–1)", yaxis="y2"))
    if use_icic:
        fig_all.add_trace(go.Scatter(x=pred["Fecha"], y=pred["ICIC_M_t"], mode="lines", name="ICIC: M_t (0–1)", yaxis="y2"))

    # Layout de doble eje
    # Añadimos MA5 real en eje izquierdo
fig_all.add_trace(go.Scatter(x=pred["Fecha"], y=pred["EMERREL_MA5_pl_m2"], mode="lines", name="MA5 (pl·m⁻² d⁻¹)", yaxis="y1"))
fig_all.update_layout(
    hovermode="x unified",
    height=560,
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
    xaxis=dict(title="Fecha"),
    yaxis=dict(title="Densidad diaria (pl·m⁻² día⁻¹)", rangemode="tozero"),
    yaxis2=dict(title="Valores normalizados (0–1)", overlaying='y', side='right', range=[0,1])
)
st.plotly_chart(fig_all, use_container_width=True, theme="streamlit")
except Exception as e:
    st.warning(f"No se pudo renderizar el gráfico combinado (versiones + ejes): {e}")
except Exception as e:
    st.warning(f"No se pudo renderizar el gráfico combinado (EMERREL×Ciec/ICIC). Detalle: {e}")
except Exception as e:
    st.warning(f"No se pudo renderizar el gráfico combinado (error: {e})")

# (NUEVO) Panel combinado: EMERREL, EMERREL ajustada, Ciec, ICIC
if use_ciec:
    st.subheader("Competencia y emergencia (Ciec + ICIC + EMERREL)")
    figc = go.Figure()
    # Curvas EMERREL
    figc.add_trace(go.Scatter(x=pred["Fecha"], y=base_emerrel, name="EMERREL original", mode="lines", line=dict(color="#1f77b4")))
    if apply_ciec_to_emerrel:
        figc.add_trace(go.Scatter(x=pred["Fecha"], y=pred["EMERREL(0-1)"], name="EMERREL × Ciec", mode="lines", line=dict(color="#ff7f0e")))
    # Curva Ciec
    figc.add_trace(go.Scatter(x=pred["Fecha"], y=pred["Ciec_trigo"], name="Ciec_t (0-1)", mode="lines", line=dict(dash="dot", color="#2ca02c")))
    # ICIC si existe en pred
    if "M_t" in pred.columns:
        figc.add_trace(go.Scatter(x=pred["Fecha"], y=pred["M_t"], name="ICIC M_t (0-1)", mode="lines", line=dict(dash="dash", color="#d62728")))
    # Línea vertical de siembra
    try:
        figc.add_vline(x=pd.to_datetime(fecha_siembra_ciec), line_dash="dot", line_color="#555", annotation_text="Siembra", annotation_position="top left")
    except Exception:
        pass
    figc.update_layout(height=600, margin=dict(l=10,r=10,t=30,b=10), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0))
    st.plotly_chart(figc, use_container_width=True, theme="streamlit")

# EMERGENCIA ACUMULADA
st.subheader("EMERGENCIA ACUMULADA DIARIA")
if rango_opcion == "Todo el empalme":
    y_min = pred_vis["EMEAC (%) - mínimo"]; y_max = pred_vis["EMEAC (%) - máximo"]; y_adj = pred_vis["EMEAC (%) - ajustable"]
else:
    y_min = pred_vis["EMEAC (%) - mínimo (rango)"]; y_max = pred_vis["EMEAC (%) - máximo (rango)"]; y_adj = pred_vis["EMEAC (%) - ajustable (rango)"]

fig = go.Figure()
fig.add_trace(go.Scatter(x=pred_vis["Fecha"], y=y_max, mode="lines", line=dict(width=0), name="Máximo", hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Máximo: %{y:.1f}%<extra></extra>"))
fig.add_trace(go.Scatter(x=pred_vis["Fecha"], y=y_min, mode="lines", line=dict(width=0), fill="tonexty", name="Mínimo", hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Mínimo: %{y:.1f}%<extra></extra>"))
fig.add_trace(go.Scatter(x=pred_vis["Fecha"], y=y_adj, mode="lines", line=dict(width=2.5), name=f"Umbral ajustable (/{EMEAC_ADJ_DEN:.2f})", hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Ajustable: %{y:.1f}%<extra></extra>"))
for nivel in [25, 50, 75, 90]:
    try:
        fig.add_hline(y=nivel, line_dash="dash", opacity=0.6, annotation_text=f"{nivel}%")
    except Exception:
        fig.add_trace(go.Scatter(x=[pred_vis["Fecha"].min(), pred_vis["Fecha"].max()], y=[nivel, nivel], mode="lines", line=dict(dash="dash"), showlegend=False))
fig.update_layout(xaxis_title="Fecha", yaxis_title="EMEAC (%)", yaxis=dict(range=[0, 100]), hovermode="x unified", legend_title="Referencias", height=600)
fig.update_xaxes(range=[fi, ff], dtick="D1" if (ff-fi).days <= 31 else "M1", tickformat="%d-%b" if (ff-fi).days <= 31 else "%b")
st.plotly_chart(fig, use_container_width=True, theme="streamlit")

# =================== TABLA & DESCARGA ===================
st.subheader(f"Resultados ({'1/feb → 1/nov' if rango_opcion!='Todo el empalme' else f'{fi.date()} → {ff.date()}'}) - Histórico GitHub + meteo_history.csv")
col_emeac = "EMEAC (%) - ajustable" if rango_opcion == "Todo el empalme" else "EMEAC (%) - ajustable (rango)"
nivel_icono = {"Bajo":"🟢 Bajo","Medio":"🟡 Medio","Alto":"🔴 Alto"}
cols_base = ["Fecha","Julian_days","Nivel_Emergencia_relativa",col_emeac]
tabla = pred_vis[cols_base].copy()
tabla["Nivel de EMERREL"] = tabla["Nivel_Emergencia_relativa"].map(nivel_icono).fillna("🟢 Bajo")
tabla = tabla.rename(columns={col_emeac:"EMEAC (%)"})
tabla["EMEAC (%)"] = pd.to_numeric(tabla["EMEAC (%)"], errors="coerce").fillna(0).clip(0,100)

# Añadir columnas Ciec si se activó
if use_ciec:
    add_cols = ["Fecha","Ciec_trigo","LAI_trigo","Ajuste_Ciec_aplicado"]
    tabla = tabla.merge(pred[add_cols], on="Fecha", how="left")

st.dataframe(tabla.sort_values("Fecha").reset_index(drop=True), use_container_width=True)

buf = StringIO(); tabla.to_csv(buf, index=False)
st.download_button(
    f"Descargar resultados ({'todo' if rango_opcion=='Todo el empalme' else 'rango'})",
    data=buf.getvalue(), file_name=f"AVEFA_resultados_{'todo' if rango_opcion=='Todo el empalme' else 'rango'}.csv",
    mime="text/csv"
)

