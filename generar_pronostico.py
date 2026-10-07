import os
import time
import datetime
import requests
import s3fs
import xarray as xr
import cartopy.crs as ccrs
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Zona horaria local de Chile
ZONA_CHILE = "America/Santiago"

# ==============================================================================
# 1. PUNTOS DE LA CUENCA Y PARÁMETROS HIPSOMÉTRICOS (1.109,29 km²)
# ==============================================================================
PUNTOS = {
    "Termas del Flaco":   {"lat": -34.957319, "lon": -70.435943, "alt": 1746},
    "BT Tinguiririca":    {"lat": -34.916224, "lon": -70.505040, "alt": 1450},
    "BT Portillo":        {"lat": -34.770882, "lon": -70.449070, "alt": 1470},
    "HLC":                {"lat": -34.833021, "lon": -70.555089, "alt": 1100},
    "HLH":                {"lat": -34.751769, "lon": -70.713893, "alt": 730},
    "La Rufina":          {"lat": -34.742984, "lon": -70.753104, "alt": 730},
    "San Fernando":       {"lat": -34.585194, "lon": -70.987161, "alt": 341},
    "Mi Casa":            {"lat": -34.679066, "lon": -70.999811, "alt": 357}
}

# Curva hipsométrica oficial de la cuenca (extraída de Cuenca_Hipsometria_Analisis.xlsx)
COTAS_HIPSO = [700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000]
AREAS_HIPSO = [0.0, 1.6, 6.45, 13.89, 24.24, 35.16, 47.82, 63.88, 85.23, 108.94, 135.72, 168.47, 199.91, 230.92, 264.47, 299.47, 338.39, 382.33, 430.04, 479.66, 529.92, 579.79, 632.17, 685.14, 735.32, 783.59, 832.26, 877.91, 921.37, 958.59, 989.78, 1015.67, 1036.97, 1054.1, 1070.82, 1083.4, 1092.16, 1099.17, 1103.65, 1106.38, 1107.79, 1108.59, 1109.17, 1109.29]
PCTS_HIPSO  = [0.0, 0.14, 0.58, 1.25, 2.18, 3.17, 4.31, 5.76, 7.68, 9.82, 12.23, 15.19, 18.02, 20.82, 23.84, 27.0, 30.5, 34.47, 38.77, 43.24, 47.77, 52.27, 56.99, 61.76, 66.29, 70.64, 75.03, 79.14, 83.06, 86.42, 89.23, 91.56, 93.48, 95.02, 96.53, 97.67, 98.46, 99.09, 99.49, 99.74, 99.86, 99.94, 99.99, 100.0]

def interpolar_area_pluvial(cota):
    if cota <= 700:
        return 0.0, 0.0
    if cota >= 5000:
        return 1109.29, 100.0
    area = float(np.interp(cota, COTAS_HIPSO, AREAS_HIPSO))
    pct = float(np.interp(cota, COTAS_HIPSO, PCTS_HIPSO))
    return round(area, 1), round(pct, 1)

AMBIENT_API_KEY = os.environ.get("AMBIENT_API_KEY", "").strip()
AMBIENT_APP_KEY = os.environ.get("AMBIENT_APPLICATION_KEY", "").strip()

def formatear_mac(mac_raw):
    limpia = mac_raw.replace(":", "").replace("-", "").strip().lower()
    if len(limpia) == 12:
        return ":".join(limpia[i:i+2] for i in range(0, 12, 2))
    return mac_raw.strip().lower()

ESTACIONES_AMBIENT = {
    "BT Portillo": formatear_mac(os.environ.get("MAC_PORTILLO", "")),
    "BT Tinguiririca": formatear_mac(os.environ.get("MAC_TINGUIRIRICA", ""))
}

HISTORICO_DIR = "historico"
os.makedirs(HISTORICO_DIR, exist_ok=True)
FILE_HIST_OBS = os.path.join(HISTORICO_DIR, "observaciones_ambient.csv")
FILE_HIST_FCST = os.path.join(HISTORICO_DIR, "pronosticos_wrf.csv")

# ==============================================================================
# 2. CONSULTA DE AMBIENT WEATHER (ESTACIONES EN SUPERFICIE)
# ==============================================================================
def obtener_datos_estacion(mac, nombre, limit=288):
    if not AMBIENT_API_KEY or not AMBIENT_APP_KEY or not mac:
        return pd.DataFrame()
    url = f"https://rt.ambientweather.net/v1/devices/{mac}"
    params = {"apiKey": AMBIENT_API_KEY, "applicationKey": AMBIENT_APP_KEY, "limit": limit}
    try:
        time.sleep(2)
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            return pd.DataFrame()
        data = resp.json()
        if not data or not isinstance(data, list):
            return pd.DataFrame()

        registros = []
        for r in data:
            fecha_chile = pd.to_datetime(r.get("dateutc"), unit='ms', utc=True).tz_convert(ZONA_CHILE).tz_localize(None)
            tf = r.get("tempf")
            tc = round((tf - 32) * 5/9, 2) if tf is not None else None
            daily_in = r.get("dailyrainin", 0.0)
            daily_mm = round(daily_in * 25.4, 2) if daily_in is not None else 0.0
            registros.append({"Punto": nombre, "Fecha_Local": fecha_chile, "T2_Obs": tc, "Daily_PP_mm": daily_mm})

        df_raw = pd.DataFrame(registros).sort_values("Fecha_Local").reset_index(drop=True)
        df_raw['PP_diff'] = df_raw['Daily_PP_mm'].diff().fillna(0)
        df_raw.loc[df_raw['PP_diff'] < 0, 'PP_diff'] = df_raw['Daily_PP_mm']

        df_obs = df_raw.resample('1h', on='Fecha_Local').agg({
            'Punto': 'first', 'T2_Obs': 'mean', 'PP_diff': 'sum'
        }).reset_index().rename(columns={'PP_diff': 'PP_Obs'})
        df_obs['PP_Obs'] = df_obs['PP_Obs'].round(2)
        df_obs['PP_Obs_acum'] = df_obs['PP_Obs'].cumsum().round(2)
        return df_obs
    except Exception as e:
        print(f"[Error Ambient] {nombre}: {e}")
        return pd.DataFrame()

datos_observados = {}
obs_acumuladas = []
for nom_est, mac_est in ESTACIONES_AMBIENT.items():
    if mac_est:
        df_est = obtener_datos_estacion(mac_est, nom_est)
        if not df_est.empty:
            datos_observados[nom_est] = df_est
            obs_acumuladas.append(df_est[['Punto', 'Fecha_Local', 'T2_Obs', 'PP_Obs']])

if obs_acumuladas:
    df_nuevas_obs = pd.concat(obs_acumuladas, ignore_index=True)
    if os.path.exists(FILE_HIST_OBS):
        df_prev_obs = pd.read_csv(FILE_HIST_OBS)
        if "Fecha_Validez" in df_prev_obs.columns and "Fecha_Local" not in df_prev_obs.columns:
            df_prev_obs.rename(columns={"Fecha_Validez": "Fecha_Local"}, inplace=True)
        df_prev_obs['Fecha_Local'] = pd.to_datetime(df_prev_obs['Fecha_Local'])
        df_total_obs = pd.concat([df_prev_obs, df_nuevas_obs]).drop_duplicates(subset=['Punto', 'Fecha_Local'], keep='last')
    else:
        df_total_obs = df_nuevas_obs
    df_total_obs.to_csv(FILE_HIST_OBS, index=False)

# ==============================================================================
# 3. CONSULTA MULTI-MODELO (ECMWF IFS + GFS VÍA OPEN-METEO API)
# ==============================================================================
def obtener_multi_modelo(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": ["temperature_2m", "precipitation"],
        "models": ["ecmwf_ifs025", "gfs_seamless"],
        "timezone": ZONA_CHILE,
        "forecast_days": 4
    }
    try:
        r = requests.get(url, params=params, timeout=12)
        if r.status_code == 200:
            h = r.json().get("hourly", {})
            df_m = pd.DataFrame({
                "Fecha_Local": pd.to_datetime(h.get("time")),
                "T2_ECMWF": h.get("temperature_2m_ecmwf_ifs025"),
                "PP_ECMWF": h.get("precipitation_ecmwf_ifs025"),
                "T2_GFS": h.get("temperature_2m_gfs_seamless"),
                "PP_GFS": h.get("precipitation_gfs_seamless")
            })
            df_m["PP_ECMWF_acum"] = df_m["PP_ECMWF"].cumsum()
            df_m["PP_GFS_acum"] = df_m["PP_GFS"].cumsum()
            return df_m
    except Exception as e:
        print(f"[Aviso Multi-modelo] No se pudo consultar Open-Meteo: {e}")
    return pd.DataFrame()

# Descargar ECMWF y GFS para los puntos clave
datos_multi_modelo = {}
for p_nom in ["BT Portillo", "BT Tinguiririca", "Termas del Flaco"]:
    p_meta = PUNTOS[p_nom]
    df_mm = obtener_multi_modelo(p_meta["lat"], p_meta["lon"])
    if not df_mm.empty:
        datos_multi_modelo[p_nom] = df_mm

# ==============================================================================
# 4. EXTRACCIÓN DE PRONÓSTICO DE AWS S3 (WRF-SMN 72 HORAS EN HORA LOCAL)
# ==============================================================================
ahora_utc = datetime.datetime.now(datetime.timezone.utc)
fecha_corrida = datetime.datetime(ahora_utc.year, ahora_utc.month, ahora_utc.day, 0)
if ahora_utc.hour < 7:
    fecha_corrida = fecha_corrida - datetime.timedelta(hours=12)

fs = s3fs.S3FileSystem(anon=True)
registros = []
coords_proy = {}
data_crs = None

print(f"Consultando corrida {fecha_corrida:%Y-%m-%d %H:00} UTC...")

for lead_time in range(1, 73):
    s3_path = f"smn-ar-wrf/DATA/WRF/DET/{fecha_corrida:%Y/%m/%d/%H}/WRFDETAR_01H_{fecha_corrida:%Y%m%d_%H}_{lead_time:03d}.nc"
    if not fs.exists(s3_path):
        continue
    try:
        with fs.open(s3_path) as f:
            with xr.open_dataset(f, decode_coords="all", engine="h5netcdf") as ds:
                if data_crs is None:
                    lc = ds["Lambert_Conformal"].attrs
                    data_crs = ccrs.LambertConformal(
                        central_longitude=lc["longitude_of_central_meridian"],
                        central_latitude=lc["latitude_of_projection_origin"],
                        standard_parallels=lc["standard_parallel"]
                    )
                    for nom, c in PUNTOS.items():
                        coords_proy[nom] = data_crs.transform_point(c["lon"], c["lat"], src_crs=ccrs.PlateCarree())

                f_validez_utc = pd.to_datetime(fecha_corrida + datetime.timedelta(hours=lead_time)).tz_localize('UTC')
                f_validez_chile = f_validez_utc.tz_convert(ZONA_CHILE).tz_localize(None)

                # Extraer temperatura simultánea en valle (La Rufina / San Fernando) y alta montaña (Termas)
                nodo_valle = ds.sel(x=coords_proy["La Rufina"][0], y=coords_proy["La Rufina"][1], method="nearest")
                nodo_alta = ds.sel(x=coords_proy["Termas del Flaco"][0], y=coords_proy["Termas del Flaco"][1], method="nearest")
                t_valle = float(nodo_valle["T2"].values)
                t_alta = float(nodo_alta["T2"].values)
                delta_z = PUNTOS["Termas del Flaco"]["alt"] - PUNTOS["La Rufina"]["alt"]  # 1746 - 730 = 1016 m
                
                # Gradiente real dinámico del modelo (°C / 1000m)
                gamma_dinamico = round(((t_valle - t_alta) / delta_z) * 1000, 2)
                # Restringir a valores físicos válidos en atmósfera (entre 4.5 y 9.8 °C/km)
                if gamma_dinamico < 4.5 or gamma_dinamico > 9.8:
                    gamma_dinamico = 6.5

                for nom, (xp, yp) in coords_proy.items():
                    nodo = ds.sel(x=xp, y=yp, method="nearest")
                    t2_val = round(float(nodo["T2"].values), 2)
                    pp_val = round(float(nodo["PP"].values), 2)
                    alt_real = PUNTOS[nom]["alt"]

                    # Cálculo con gradiente real dinámico
                    iso_dinamica = round(alt_real + (t2_val / gamma_dinamico) * 1000, 1)
                    iso_alta = round(alt_real + (t2_val / 6.5) * 1000, 1)
                    iso_baja = round(alt_real + (t2_val / 9.0) * 1000, 1)

                    # Interpolar área activa líquida en base a la curva hipsométrica
                    area_liq, pct_liq = interpolar_area_pluvial(iso_dinamica)

                    registros.append({
                        "Punto": nom,
                        "Fecha_Corrida_UTC": fecha_corrida,
                        "Fecha_Local": f_validez_chile,
                        "Lead_Time": lead_time,
                        "Altitud_msnm": alt_real,
                        "T2": t2_val,
                        "PP": pp_val,
                        "Gamma_Dinamico": gamma_dinamico,
                        "Iso0_Dinamica": iso_dinamica,
                        "Iso0_Alta": iso_alta,
                        "Iso0_Baja": iso_baja,
                        "Area_Pluvial_km2": area_liq,
                        "Pct_Pluvial": pct_liq
                    })
    except Exception as e:
        print(f"Error en paso +{lead_time}: {e}")

df = pd.DataFrame(registros)
if df.empty:
    raise RuntimeError("No se descargaron datos de los archivos NetCDF.")

# Actualizar CSV acumulado de pronósticos
df_nuevo_fcst = df[['Punto', 'Fecha_Corrida_UTC', 'Fecha_Local', 'Lead_Time', 'T2', 'PP']]
if os.path.exists(FILE_HIST_FCST):
    df_prev_fcst = pd.read_csv(FILE_HIST_FCST)
    if "Fecha_Validez" in df_prev_fcst.columns and "Fecha_Local" not in df_prev_fcst.columns:
        df_prev_fcst.rename(columns={"Fecha_Validez": "Fecha_Local"}, inplace=True)
    df_prev_fcst['Fecha_Local'] = pd.to_datetime(df_prev_fcst['Fecha_Local'])
    df_total_fcst = pd.concat([df_prev_fcst, df_nuevo_fcst]).drop_duplicates(subset=['Punto', 'Fecha_Corrida_UTC', 'Lead_Time'], keep='last')
else:
    df_total_fcst = df_nuevo_fcst
df_total_fcst.to_csv(FILE_HIST_FCST, index=False)

# ==============================================================================
# 5. MÓDULO DE VALIDACIÓN HISTÓRICA (HORA LOCAL CHILE)
# ==============================================================================
html_validacion = ""
metricas_filas = []

if os.path.exists(FILE_HIST_OBS) and os.path.exists(FILE_HIST_FCST):
    df_h_obs = pd.read_csv(FILE_HIST_OBS)
    df_h_fcst = pd.read_csv(FILE_HIST_FCST)
    if "Fecha_Validez" in df_h_obs.columns and "Fecha_Local" not in df_h_obs.columns:
        df_h_obs.rename(columns={"Fecha_Validez": "Fecha_Local"}, inplace=True)
    if "Fecha_Validez" in df_h_fcst.columns and "Fecha_Local" not in df_h_fcst.columns:
        df_h_fcst.rename(columns={"Fecha_Validez": "Fecha_Local"}, inplace=True)

    df_h_obs['Fecha_Local'] = pd.to_datetime(df_h_obs['Fecha_Local'])
    df_h_fcst['Fecha_Local'] = pd.to_datetime(df_h_fcst['Fecha_Local'])

    df_fcst_24 = df_h_fcst[(df_h_fcst['Lead_Time'] >= 1) & (df_h_fcst['Lead_Time'] <= 24)]
    df_fcst_unicos = df_fcst_24.sort_values('Lead_Time').drop_duplicates(subset=['Punto', 'Fecha_Local'], keep='first')
    df_cruce = pd.merge(df_fcst_unicos, df_h_obs, on=['Punto', 'Fecha_Local'], how='inner')

    for pto in ["BT Portillo", "BT Tinguiririca"]:
        sub = df_cruce[df_cruce['Punto'] == pto].copy().dropna(subset=['T2', 'T2_Obs'])
        if len(sub) >= 1:
            error_t = sub['T2'] - sub['T2_Obs']
            mae_t = round(float(np.mean(np.abs(error_t))), 2)
            bias_t = round(float(np.mean(error_t)), 2)

            sub_pp = sub.dropna(subset=['PP', 'PP_Obs'])
            if len(sub_pp) >= 1:
                error_pp = sub_pp['PP'] - sub_pp['PP_Obs']
                mae_pp = round(float(np.mean(np.abs(error_pp))), 2)
                bias_pp = round(float(np.mean(error_pp)), 2)
                total_pp_fcst = round(float(sub_pp['PP'].sum()), 1)
                total_pp_obs = round(float(sub_pp['PP_Obs'].sum()), 1)
            else:
                mae_pp, bias_pp, total_pp_fcst, total_pp_obs = "-", "-", "-", "-"

            metricas_filas.append({
                "Punto": pto, "Horas_Evaluadas": len(sub),
                "MAE_Temp": f"{mae_t} °C", "Sesgo_Temp": f"{bias_t:+0.2f} °C",
                "MAE_PP": f"{mae_pp} mm/h" if mae_pp != "-" else "-",
                "Sesgo_PP": f"{bias_pp:+0.2f} mm/h" if bias_pp != "-" else "-",
                "PP_Acum_WRF": f"{total_pp_fcst} mm" if total_pp_fcst != "-" else "-",
                "PP_Acum_Real": f"{total_pp_obs} mm" if total_pp_obs != "-" else "-"
            })

    if metricas_filas:
        df_metricas = pd.DataFrame(metricas_filas)
        tabla_html = "<table style='width:100%; border-collapse:collapse; margin-top:15px; font-size:14px; text-align:center;'>"
        tabla_html += "<tr style='background-color:#2b6cb0; color:white;'>"
        for col in ["Estación", "Horas Muestreadas", "MAE Temp", "Sesgo Temp", "MAE Precipitación", "Sesgo Precipitación", "Lluvia Acum. WRF", "Lluvia Acum. Real"]:
            tabla_html += f"<th style='padding:10px; border:1px solid #cbd5e0;'>{col}</th>"
        tabla_html += "</tr>"
        for _, row in df_metricas.iterrows():
            tabla_html += "<tr style='background-color:#ffffff;'>"
            tabla_html += f"<td style='padding:8px; border:1px solid #e2e8f0; font-weight:bold;'>{row['Punto']}</td>"
            tabla_html += f"<td style='padding:8px; border:1px solid #e2e8f0;'>{row['Horas_Evaluadas']}</td>"
            tabla_html += f"<td style='padding:8px; border:1px solid #e2e8f0;'>{row['MAE_Temp']}</td>"
            tabla_html += f"<td style='padding:8px; border:1px solid #e2e8f0; color:{'#c53030' if '+' in str(row['Sesgo_Temp']) else '#2b6cb0'}; font-weight:bold;'>{row['Sesgo_Temp']}</td>"
            tabla_html += f"<td style='padding:8px; border:1px solid #e2e8f0;'>{row['MAE_PP']}</td>"
            tabla_html += f"<td style='padding:8px; border:1px solid #e2e8f0;'>{row['Sesgo_PP']}</td>"
            tabla_html += f"<td style='padding:8px; border:1px solid #e2e8f0;'>{row['PP_Acum_WRF']}</td>"
            tabla_html += f"<td style='padding:8px; border:1px solid #e2e8f0;'>{row['PP_Acum_Real']}</td>"
            tabla_html += "</tr>"
        tabla_html += "</table>"
        html_validacion = f"""
        <div style='background:#f7fafc; padding:20px; border-radius:8px; border:1px solid #e2e8f0; margin-bottom:40px;'>
            <p style='margin:0; font-size:14px; color:#4a5568;'>Métricas de desempeño para pronósticos de corto plazo (1 a 24 horas) contrastados con datos de estaciones en superficie:</p>
            {tabla_html}
        </div>
        """

# ==============================================================================
# 6. GENERACIÓN DE GRÁFICOS PLOTLY (HIPSOMETRÍA NIEVE/AGUA + EMPALME WRF)
# ==============================================================================

# Cargar histórico de pronósticos WRF para empalmar horas pasadas en los gráficos
df_h_wrf_prev = pd.DataFrame()
if os.path.exists(FILE_HIST_FCST):
    try:
        df_h_wrf_prev = pd.read_csv(FILE_HIST_FCST)
        if "Fecha_Validez" in df_h_wrf_prev.columns and "Fecha_Local" not in df_h_wrf_prev.columns:
            df_h_wrf_prev.rename(columns={"Fecha_Validez": "Fecha_Local"}, inplace=True)
        df_h_wrf_prev['Fecha_Local'] = pd.to_datetime(df_h_wrf_prev['Fecha_Local'])
    except Exception as e:
        print(f"[Aviso] No se pudo leer histórico previo para empalme: {e}")

# A. Gráfico Hipsométrico e Isoterma Dinámica (Lluvia Líquida vs. Nieve)
df_flaco = df[df["Punto"] == "Termas del Flaco"].copy()
fig_iso = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.18,
    subplot_titles=(
        "<b>Cota de Isoterma 0 °C Proyectada (m s. n. m.) con Gradiente Real Dinámico</b>",
        "<b>Distribución Hipsométrica de la Cuenca: Lluvia Líquida vs. Nieve (1.109,3 km²)</b>"
    ),
    specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
)

# Panel 1: Isoterma y Banda
fig_iso.add_trace(go.Scatter(
    x=df_flaco["Fecha_Local"], y=df_flaco["Iso0_Alta"],
    mode='lines', line=dict(width=0), showlegend=False, name='Límite Superior'
), row=1, col=1)
fig_iso.add_trace(go.Scatter(
    x=df_flaco["Fecha_Local"], y=df_flaco["Iso0_Baja"],
    mode='lines', line=dict(width=0), fill='tonexty',
    fillcolor='rgba(178, 235, 242, 0.40)', name='Banda Teórica (6.5 - 9.0 °C/km)'
), row=1, col=1)
fig_iso.add_trace(go.Scatter(
    x=df_flaco["Fecha_Local"], y=df_flaco["Iso0_Dinamica"],
    mode='lines+markers', line=dict(color='#0288d1', width=3),
    name='Isoterma 0 °C (Gradiente Dinámico Real)'
), row=1, col=1)

colores_ref = ['#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
for i, (nombre_p, meta) in enumerate(list(PUNTOS.items())[:5]):
    fig_iso.add_hline(
        y=meta["alt"], line_dash="dot", line_color=colores_ref[i % len(colores_ref)],
        annotation_text=f"{nombre_p} ({meta['alt']} m)", annotation_position="bottom right",
        row=1, col=1
    )

# Panel 2: Distribución Nieve vs. Lluvia Líquida
# Capa 1: Lluvia líquida (desde cota mínima hasta isoterma)
fig_iso.add_trace(go.Scatter(
    x=df_flaco["Fecha_Local"], y=df_flaco["Area_Pluvial_km2"],
    mode='lines', line=dict(color='#00acc1', width=2),
    fill='tozeroy', fillcolor='rgba(77, 208, 225, 0.45)',
    name='Área con Lluvia Líquida (Escorrentía)'
), row=2, col=1, secondary_y=False)

# Capa 2: Nieve acumulada en cordillera (desde la isoterma hasta los 1.109,3 km²)
area_total_vec = [1109.29] * len(df_flaco)
fig_iso.add_trace(go.Scatter(
    x=df_flaco["Fecha_Local"], y=area_total_vec,
    mode='lines', line=dict(color='#90a4ae', width=1, dash='dot'),
    fill='tonexty', fillcolor='rgba(236, 239, 241, 0.75)',
    name='Área con Nieve Sólida (Retención Nival)'
), row=2, col=1, secondary_y=False)

# Línea del porcentaje activo sobre eje secundario
fig_iso.add_trace(go.Scatter(
    x=df_flaco["Fecha_Local"], y=df_flaco["Pct_Pluvial"],
    mode='lines', line=dict(color='#004d40', width=2.5, dash='dash'),
    name='% Cuenca Activa (Líquida)'
), row=2, col=1, secondary_y=True)

fig_iso.update_layout(
    height=760, hovermode="x unified", template="plotly_white",
    margin=dict(l=65, r=50, t=50, b=40)
)
fig_iso.update_yaxes(title_text="Altitud (m)", row=1, col=1)
fig_iso.update_yaxes(title_text="Superficie de Cuenca (km²)", range=[0, 1150], row=2, col=1, secondary_y=False)
fig_iso.update_yaxes(
    title_text="% Cuenca Líquida",
    range=[0, 100],
    tickmode='linear',
    tick0=0,
    dtick=20,
    ticksuffix="%",
    row=2, col=1, secondary_y=True
)
fig_iso.update_xaxes(title_text="Fecha y Hora (Hora Local de Chile)", row=2, col=1)

html_isoterma = f"<div style='margin-bottom: 45px;'>{fig_iso.to_html(full_html=False, include_plotlyjs='cdn')}</div>"

# B. Gráficos Detallados por Estación (con Empalme Histórico WRF)
html_puntos = ""
for punto in PUNTOS.keys():
    df_p = df[df["Punto"] == punto].copy()
    if df_p.empty:
        continue

    # Empalmar con corridas anteriores del WRF si existen en el histórico
    if not df_h_wrf_prev.empty:
        sub_h = df_h_wrf_prev[df_h_wrf_prev['Punto'] == punto].copy()
        # Tomar registros anteriores a la corrida actual
        sub_h = sub_h[sub_h['Fecha_Local'] < df_p['Fecha_Local'].min()]
        if not sub_h.empty:
            sub_h = sub_h.sort_values('Lead_Time').drop_duplicates(subset=['Fecha_Local'], keep='first')
            cols_unir = [c for c in ['Fecha_Local', 'T2', 'PP'] if c in sub_h.columns]
            df_p = pd.concat([sub_h[cols_unir], df_p], ignore_index=True).sort_values('Fecha_Local').reset_index(drop=True)

    # Calcular acumulado completo continuo
    df_p["PP_acum"] = df_p["PP"].cumsum()

    subtitulos = (
        f"<b>Temperatura (°C) - {punto} ({PUNTOS[punto]['alt']} m s. n. m.)</b>",
        f"<b>Precipitación Horaria y Acumulada - {punto}</b>"
    )

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
        subplot_titles=subtitulos,
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
    )

    # 1. TEMPERATURA
    fig.add_trace(
        go.Scatter(x=df_p["Fecha_Local"], y=df_p["T2"], name="Temp WRF-SMN",
                   line=dict(color="#d9381e", width=2.5)), row=1, col=1
    )

    if punto in datos_multi_modelo:
        df_mm = datos_multi_modelo[punto]
        fig.add_trace(
            go.Scatter(x=df_mm["Fecha_Local"], y=df_mm["T2_ECMWF"], name="Temp ECMWF IFS",
                       line=dict(color="#ff7f0e", width=2, dash="dash")), row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df_mm["Fecha_Local"], y=df_mm["T2_GFS"], name="Temp GFS (NOAA)",
                       line=dict(color="#9467bd", width=1.8, dash="dot")), row=1, col=1
        )

    if punto in datos_observados and not datos_observados[punto].empty:
        df_o = datos_observados[punto]
        fig.add_trace(
            go.Scatter(x=df_o["Fecha_Local"], y=df_o["T2_Obs"], name="Temp Real (Estación)",
                       line=dict(color="#2ca02c", width=2.5)), row=1, col=1
        )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="0°C", row=1, col=1)

    # 2. PRECIPITACIÓN
    fig.add_trace(
        go.Bar(x=df_p["Fecha_Local"], y=df_p["PP"], name="PP WRF (mm/h)",
               marker_color="#29b6f6", opacity=0.75), row=2, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=df_p["Fecha_Local"], y=df_p["PP_acum"], name="Acum. WRF (mm)",
                   line=dict(color="#01579b", width=2.5)), row=2, col=1, secondary_y=True
    )

    if punto in datos_multi_modelo:
        df_mm = datos_multi_modelo[punto]
        fig.add_trace(
            go.Scatter(x=df_mm["Fecha_Local"], y=df_mm["PP_ECMWF_acum"], name="Acum. ECMWF (mm)",
                       line=dict(color="#ff7f0e", width=2.2, dash="dash")), row=2, col=1, secondary_y=True
        )
        fig.add_trace(
            go.Scatter(x=df_mm["Fecha_Local"], y=df_mm["PP_GFS_acum"], name="Acum. GFS (mm)",
                       line=dict(color="#9467bd", width=2, dash="dot")), row=2, col=1, secondary_y=True
        )

    if punto in datos_observados and not datos_observados[punto].empty:
        df_o = datos_observados[punto]
        fig.add_trace(
            go.Bar(x=df_o["Fecha_Local"], y=df_o["PP_Obs"], name="PP Real Estación (mm/h)",
                   marker_color="#2ca02c", opacity=0.65), row=2, col=1, secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=df_o["Fecha_Local"], y=df_o["PP_Obs_acum"], name="Acum. Real Estación (mm)",
                       line=dict(color="#006400", width=2.5)), row=2, col=1, secondary_y=True
        )

    layout_update = dict(
        height=680, hovermode="x unified", template="plotly_white",
        bargap=0.15,
        margin=dict(l=55, r=50, t=50, b=40)
    )

    if punto in ["BT Portillo", "BT Tinguiririca"]:
        fig.update_xaxes(
            rangeslider=dict(visible=True, thickness=0.06),
            rangeselector=dict(
                buttons=list([
                    dict(count=24, label="24h", step="hour", stepmode="backward"),
                    dict(count=48, label="48h", step="hour", stepmode="backward"),
                    dict(step="all", label="Ver Todo")
                ])
            ),
            row=2, col=1
        )

    fig.update_layout(**layout_update)
    fig.update_xaxes(title_text="Fecha y Hora (Hora Local de Chile)", row=2, col=1)
    fig.update_yaxes(title_text="Temp (°C)", row=1, col=1)
    fig.update_yaxes(title_text="PP (mm/h)", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Acumulada (mm)", row=2, col=1, secondary_y=True)

    html_puntos += f"<div style='margin-bottom: 50px;'>{fig.to_html(full_html=False, include_plotlyjs=False)}</div>"

# ==============================================================================
# 7. ENSAMBLAJE HTML COMPLETO
# ==============================================================================
ahora_chile = datetime.datetime.now(datetime.timezone.utc).astimezone(pd.Timestamp.now(tz=ZONA_CHILE).tzinfo)

html_final = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Monitoreo Meteorológico, Isoterma e Hipsometría - Cuenca Tinguiririca</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #f0f2f5; margin: 0; padding: 20px; color: #2d3748; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }}
        h1 {{ color: #1a365d; margin-top: 0; font-size: 26px; }}
        .meta {{ background: #edf2f7; padding: 12px 18px; border-radius: 6px; margin-bottom: 25px; font-size: 14px; line-height: 1.6; }}
        .section-title {{ font-size: 20px; color: #2c5282; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; margin-top: 35px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Sistema de Pronóstico, Monitoreo y Validación - Cuenca del Tinguiririca</h1>
        <div class="meta">
            <strong>Modelos Atmosféricos:</strong> Ensamble WRF-SMN + ECMWF IFS + GFS (NOAA)<br>
            <strong>Superficie Cuenca:</strong> 1.109,29 km² (Cota media: 2.713 m s. n. m.) | 
            <strong>Actualizado:</strong> {ahora_chile:%Y-%m-%d %H:%M} (Hora de Chile)<br>
            <strong>Estaciones en Superficie:</strong> BT Portillo y BT Tinguiririca (Ambient Weather)
        </div>

        <div class="section-title">1. Proyección de Isoterma Cero y Respuesta Hipsométrica de la Cuenca</div>
        <p style="font-size: 14px; color: #4a5568;">
            El panel superior proyecta la altitud de la Isoterma 0 °C calculada mediante el <strong>gradiente térmico real dinámico</strong> entre el valle y la alta cordillera. 
            El panel inferior traduce dicha cota en tiempo real al <strong>área activa drenante bajo lluvia líquida (km² y %)</strong> según la curva hipsométrica oficial de la cuenca.
        </p>
        {html_isoterma}

        <div class="section-title">2. Validación Histórica: Pronóstico WRF vs. Estaciones Reales</div>
        {html_validacion}

        <div class="section-title">3. Pronósticos Detallados y Comparativa Multi-Modelo (72 Horas)</div>
        <p style="font-size: 14px; color: #4a5568;">
            Contraste entre los modelos <strong>WRF-SMN (azul)</strong>, <strong>ECMWF IFS (naranja)</strong>, <strong>GFS (púrpura)</strong> y las observaciones registradas por las estaciones en superficie (verde).
        </p>
        {html_puntos}
    </div>
</body>
</html>"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_final)

print("Reporte con hipsometría y ensamble multi-modelo generado exitosamente.")
