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
# 1. PUNTOS DE LA CUENCA (COORDENADAS Y COTAS OFICIALES MSNM)
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

AMBIENT_API_KEY = os.environ.get("AMBIENT_API_KEY", "").strip()
AMBIENT_APP_KEY = os.environ.get("AMBIENT_APPLICATION_KEY", "").strip()

def formatear_mac(mac_raw):
    limpia = mac_raw.replace(":", "").replace("-", "").strip().lower()
    if len(limpia) == 12:
        return ":".join(limpia[i:i+2] for i in range(0, 12, 2))
    return mac_raw.strip().lower()

MAC_PORTILLO = formatear_mac(os.environ.get("MAC_PORTILLO", ""))
MAC_TINGUIRIRICA = formatear_mac(os.environ.get("MAC_TINGUIRIRICA", ""))

ESTACIONES_AMBIENT = {
    "BT Portillo": MAC_PORTILLO,
    "BT Tinguiririca": MAC_TINGUIRIRICA
}

HISTORICO_DIR = "historico"
os.makedirs(HISTORICO_DIR, exist_ok=True)
FILE_HIST_OBS = os.path.join(HISTORICO_DIR, "observaciones_ambient.csv")
FILE_HIST_FCST = os.path.join(HISTORICO_DIR, "pronosticos_wrf.csv")

# ==============================================================================
# 2. CONSULTA Y PERSISTENCIA DE AMBIENT WEATHER (HORA LOCAL CHILE)
# ==============================================================================
def obtener_datos_estacion(mac, nombre, limit=288):
    if not AMBIENT_API_KEY or not AMBIENT_APP_KEY or not mac:
        print(f"[Aviso] Faltan credenciales o MAC para {nombre}.")
        return pd.DataFrame()

    url = f"https://rt.ambientweather.net/v1/devices/{mac}"
    params = {"apiKey": AMBIENT_API_KEY, "applicationKey": AMBIENT_APP_KEY, "limit": limit}

    try:
        time.sleep(2)
        resp = requests.get(url, params=params, timeout=15)
        print(f"[Ambient] Consulta a {nombre} ({mac}) -> Código {resp.status_code}")

        if resp.status_code != 200:
            print(f"[Aviso] No fue posible obtener datos para {nombre}: {resp.status_code} - {resp.text}")
            return pd.DataFrame()

        data = resp.json()
        if not data or not isinstance(data, list):
            print(f"[Aviso] Respuesta vacía de Ambient Weather para {nombre}.")
            return pd.DataFrame()

        registros = []
        for r in data:
            # Timestamp UTC de la estación convertido a Hora Local de Chile
            fecha_chile = pd.to_datetime(r.get("dateutc"), unit='ms', utc=True).tz_convert(ZONA_CHILE).tz_localize(None)
            tf = r.get("tempf")
            tc = round((tf - 32) * 5/9, 2) if tf is not None else None
            hourly_in = r.get("hourlyrainin", 0.0)
            pp_h = round(hourly_in * 25.4, 2) if hourly_in is not None else 0.0

            registros.append({
                "Punto": nombre,
                "Fecha_Local": fecha_chile,
                "T2_Obs": tc,
                "PP_Obs": pp_h
            })

        df_obs = pd.DataFrame(registros).sort_values("Fecha_Local").reset_index(drop=True)
        # Resampleo a nivel horario
        df_obs = df_obs.resample('1h', on='Fecha_Local').agg({'Punto': 'first', 'T2_Obs': 'mean', 'PP_Obs': 'max'}).reset_index()
        df_obs['PP_Obs_acum'] = df_obs['PP_Obs'].cumsum()
        print(f"[Ambient] Correcto: {len(df_obs)} registros para {nombre}.")
        return df_obs
    except Exception as e:
        print(f"[Error] Excepción descargando {nombre}: {e}")
        return pd.DataFrame()

datos_observados = {}
obs_acumuladas = []
for nom_est, mac_est in ESTACIONES_AMBIENT.items():
    if mac_est:
        print(f"Descargando observaciones de {nom_est}...")
        df_est = obtener_datos_estacion(mac_est, nom_est)
        if not df_est.empty:
            datos_observados[nom_est] = df_est
            obs_acumuladas.append(df_est[['Punto', 'Fecha_Local', 'T2_Obs', 'PP_Obs']])

# Actualizar CSV acumulado de observaciones
if obs_acumuladas:
    df_nuevas_obs = pd.concat(obs_acumuladas, ignore_index=True)
    if os.path.exists(FILE_HIST_OBS):
        df_prev_obs = pd.read_csv(FILE_HIST_OBS)
        df_prev_obs['Fecha_Local'] = pd.to_datetime(df_prev_obs['Fecha_Local'])
        df_total_obs = pd.concat([df_prev_obs, df_nuevas_obs]).drop_duplicates(subset=['Punto', 'Fecha_Local'], keep='last')
    else:
        df_total_obs = df_nuevas_obs
    df_total_obs.to_csv(FILE_HIST_OBS, index=False)

# ==============================================================================
# 3. EXTRACCIÓN DE PRONÓSTICO DE AWS S3 (WRF-SMN 72 HORAS EN HORA LOCAL)
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

                # Fecha de validez original en UTC
                f_validez_utc = pd.to_datetime(fecha_corrida + datetime.timedelta(hours=lead_time)).tz_localize('UTC')
                # Conversión a Hora Local de Chile
                f_validez_chile = f_validez_utc.tz_convert(ZONA_CHILE).tz_localize(None)

                for nom, (xp, yp) in coords_proy.items():
                    nodo = ds.sel(x=xp, y=yp, method="nearest")
                    t2_val = round(float(nodo["T2"].values), 2)
                    pp_val = round(float(nodo["PP"].values), 2)
                    alt_real = PUNTOS[nom]["alt"]

                    iso_alta = round(alt_real + (t2_val / 6.5) * 1000, 1)
                    iso_baja = round(alt_real + (t2_val / 9.0) * 1000, 1)
                    iso_media = round((iso_alta + iso_baja) / 2, 1)

                    registros.append({
                        "Punto": nom,
                        "Fecha_Corrida_UTC": fecha_corrida,
                        "Fecha_Local": f_validez_chile,
                        "Lead_Time": lead_time,
                        "Altitud_msnm": alt_real,
                        "T2": t2_val,
                        "PP": pp_val,
                        "Iso0_Media": iso_media,
                        "Iso0_Alta": iso_alta,
                        "Iso0_Baja": iso_baja
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
    df_prev_fcst['Fecha_Local'] = pd.to_datetime(df_prev_fcst['Fecha_Local'])
    df_total_fcst = pd.concat([df_prev_fcst, df_nuevo_fcst]).drop_duplicates(subset=['Punto', 'Fecha_Corrida_UTC', 'Lead_Time'], keep='last')
else:
    df_total_fcst = df_nuevo_fcst
df_total_fcst.to_csv(FILE_HIST_FCST, index=False)

# ==============================================================================
# 4. MÓDULO DE VALIDACIÓN HISTÓRICA (HORA LOCAL CHILE)
# ==============================================================================
html_validacion = ""
metricas_filas = []

if os.path.exists(FILE_HIST_OBS) and os.path.exists(FILE_HIST_FCST):
    df_h_obs = pd.read_csv(FILE_HIST_OBS)
    df_h_fcst = pd.read_csv(FILE_HIST_FCST)
    df_h_obs['Fecha_Local'] = pd.to_datetime(df_h_obs['Fecha_Local'])
    df_h_fcst['Fecha_Local'] = pd.to_datetime(df_h_fcst['Fecha_Local'])

    # Cruzar pronósticos de corto plazo (1 a 24h) con observaciones por Fecha_Local
    df_fcst_24 = df_h_fcst[(df_h_fcst['Lead_Time'] >= 1) & (df_h_fcst['Lead_Time'] <= 24)]
    df_cruce = pd.merge(df_fcst_24, df_h_obs, on=['Punto', 'Fecha_Local'], how='inner')

    for pto in ["BT Portillo", "BT Tinguiririca"]:
        sub = df_cruce[df_cruce['Punto'] == pto].copy()
        sub = sub.dropna(subset=['T2', 'T2_Obs'])
        if len(sub) >= 1:
            error_t = sub['T2'] - sub['T2_Obs']
            mae_t = round(float(np.mean(np.abs(error_t))), 2)
            bias_t = round(float(np.mean(error_t)), 2)

            # Precipitación
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
                "Punto": pto,
                "Horas_Evaluadas": len(sub),
                "MAE_Temp": f"{mae_t} °C",
                "Sesgo_Temp": f"{bias_t:+0.2f} °C",
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
            <p style='margin:0; font-size:14px; color:#4a5568;'>
                Métricas de desempeño para pronósticos de corto plazo (1 a 24 horas) contrastados con datos de estaciones en superficie:
            </p>
            {tabla_html}
            <p style='font-size:12px; color:#718096; margin-top:8px;'>
                * <strong>Sesgo (Bias):</strong> Positivo indica sobreestimación del modelo; negativo indica subestimación.<br>
                * <strong>MAE:</strong> Magnitud promedio del error hora a hora.
            </p>
        </div>
        """

if not html_validacion:
    html_validacion = """
    <div style='background:#fffaf0; padding:15px; border-left:4px solid #dd6b20; border-radius:4px; margin-bottom:30px; font-size:14px; color:#7b341e;'>
        <strong>Acumulando histórico:</strong> El sistema está registrando las corridas horarias. Las métricas aparecerán tras las primeras coincidencias.
    </div>
    """

# ==============================================================================
# 5. GENERACIÓN DE GRÁFICOS PLOTLY (EN HORA LOCAL DE CHILE)
# ==============================================================================

# Gráfico de Isoterma Cero
df_flaco = df[df["Punto"] == "Termas del Flaco"].copy()
fig_iso = go.Figure()
fig_iso.add_trace(go.Scatter(
    x=df_flaco["Fecha_Local"], y=df_flaco["Iso0_Alta"],
    mode='lines', line=dict(width=0), showlegend=False, name='Límite Superior'
))
fig_iso.add_trace(go.Scatter(
    x=df_flaco["Fecha_Local"], y=df_flaco["Iso0_Baja"],
    mode='lines', line=dict(width=0), fill='tonexty',
    fillcolor='rgba(0, 128, 255, 0.22)', name='Banda de Seguridad (6.5 - 9.0 °C/km)'
))
fig_iso.add_trace(go.Scatter(
    x=df_flaco["Fecha_Local"], y=df_flaco["Iso0_Media"],
    mode='lines+markers', line=dict(color='#0052cc', width=2.5), name='Isoterma 0 °C Estimada'
))

colores_ref = ['#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
for i, (nombre_p, meta) in enumerate(list(PUNTOS.items())[:5]):
    fig_iso.add_hline(
        y=meta["alt"], line_dash="dot", line_color=colores_ref[i % len(colores_ref)],
        annotation_text=f"{nombre_p} ({meta['alt']} m)", annotation_position="bottom right"
    )

fig_iso.update_layout(
    title="<b>Proyección de Isoterma 0 °C y Banda de Seguridad en la Cuenca Alta</b>",
    xaxis_title="Fecha y Hora (Hora Local de Chile)",
    yaxis_title="Altitud (m s. n. m.)", hovermode="x unified",
    template="plotly_white", height=480, margin=dict(l=40, r=40, t=60, b=40)
)
html_isoterma = f"<div style='margin-bottom: 45px;'>{fig_iso.to_html(full_html=False, include_plotlyjs='cdn')}</div>"

# Gráficos por Punto
html_puntos = ""
for punto in PUNTOS.keys():
    df_p = df[df["Punto"] == punto].copy()
    if df_p.empty:
        continue
    df_p["PP_acum"] = df_p["PP"].cumsum()

    subtitulos = (
        f"Temperatura (°C) - {punto} ({PUNTOS[punto]['alt']} m s. n. m.)",
        f"Precipitación Horaria y Acumulada - {punto}"
    )
    if punto in datos_observados and not datos_observados[punto].empty:
        subtitulos = (
            f"Temperatura (°C) - {punto} [Pronóstico vs. Estación Real]",
            f"Precipitación - {punto} [Pronóstico vs. Estación Real]"
        )

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=subtitulos,
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
    )

    # 1. Curva de Temperatura Pronosticada
    fig.add_trace(
        go.Scatter(x=df_p["Fecha_Local"], y=df_p["T2"], name="Temp. Pronosticada (WRF)",
                   line=dict(color="#d9381e", width=2.5)),
        row=1, col=1
    )

    # 1b. Curva de Temperatura Observada
    if punto in datos_observados and not datos_observados[punto].empty:
        df_o = datos_observados[punto]
        fig.add_trace(
            go.Scatter(x=df_o["Fecha_Local"], y=df_o["T2_Obs"], name="Temp. Observada (Estación)",
                       line=dict(color="#2ca02c", width=2, dash="dot")),
            row=1, col=1
        )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="0°C", row=1, col=1)

    # 2. Precipitación Pronosticada
    fig.add_trace(
        go.Bar(x=df_p["Fecha_Local"], y=df_p["PP"], name="PP Pronosticada (mm/h)",
               marker_color="#3182bd", opacity=0.7),
        row=2, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=df_p["Fecha_Local"], y=df_p["PP_acum"], name="PP Acumulada Pronóstico (mm)",
                   line=dict(color="#08519c", width=2.5)),
        row=2, col=1, secondary_y=True
    )

    # 2b. Precipitación Observada
    if punto in datos_observados and not datos_observados[punto].empty:
        df_o = datos_observados[punto]
        fig.add_trace(
            go.Bar(x=df_o["Fecha_Local"], y=df_o["PP_Obs"], name="PP Observada Estación (mm/h)",
                   marker_color="#2ca02c", opacity=0.6),
            row=2, col=1, secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=df_o["Fecha_Local"], y=df_o["PP_Obs_acum"], name="PP Acumulada Estación (mm)",
                       line=dict(color="#006400", width=2, dash="dash")),
            row=2, col=1, secondary_y=True
        )

    fig.update_layout(
        height=620, hovermode="x unified", template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    fig.update_xaxes(title_text="Fecha y Hora (Hora Local de Chile)", row=2, col=1)
    fig.update_yaxes(title_text="Temp (°C)", row=1, col=1)
    fig.update_yaxes(title_text="PP (mm/h)", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Acumulada (mm)", row=2, col=1, secondary_y=True)

    html_puntos += f"<div style='margin-bottom: 50px;'>{fig.to_html(full_html=False, include_plotlyjs=False)}</div>"

# ==============================================================================
# 6. ENSAMBLAJE HTML COMPLETO
# ==============================================================================
ahora_chile = datetime.datetime.now(datetime.timezone.utc).astimezone(pd.Timestamp.now(tz=ZONA_CHILE).tzinfo)

html_final = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Monitoreo Meteorológico e Isoterma - Cuenca Tinguiririca</title>
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
            <strong>Modelo Atmosférico:</strong> WRF-SMN (Resolución operativa en AWS Open Data)<br>
            <strong>Corrida:</strong> {fecha_corrida:%Y-%m-%d %H:00} UTC | 
            <strong>Actualizado:</strong> {ahora_chile:%Y-%m-%d %H:%M} (Hora de Chile)<br>
            <strong>Estaciones en Superficie:</strong> BT Portillo y BT Tinguiririca (Ambient Weather)<br>
            <strong>Zona Horaria de Visualización:</strong> Hora Oficial de Chile (CLT/CLST)
        </div>

        <div class="section-title">1. Proyección de Isoterma Cero y Análisis de Altitud</div>
        <p style="font-size: 14px; color: #4a5568;">Banda calculada con gradientes térmicos verticales de 6.5 a 9.0 °C/1.000 m sobre la alta cuenca.</p>
        {html_isoterma}

        <div class="section-title">2. Validación Histórica: Pronóstico WRF vs. Estaciones Reales</div>
        {html_validacion}

        <div class="section-title">3. Pronósticos Detallados por Estación (72 Horas)</div>
        {html_puntos}
    </div>
</body>
</html>"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_final)

print("Reporte con conversión a Hora Local y validación histórica generado exitosamente.")
