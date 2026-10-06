import datetime
import s3fs
import xarray as xr
import cartopy.crs as ccrs
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==============================================================================
# 1. PUNTOS ACTUALIZADOS DE LA CUENCA (CON ALTITUD REAL MSNM)
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

# ==============================================================================
# 2. DEFINICIÓN DE LA CORRIDA DISPONIBLE
# ==============================================================================
ahora = datetime.datetime.now(datetime.timezone.utc)
fecha_corrida = datetime.datetime(ahora.year, ahora.month, ahora.day, 0)
if ahora.hour < 7:
    fecha_corrida = fecha_corrida - datetime.timedelta(hours=12)

fs = s3fs.S3FileSystem(anon=True)

# ==============================================================================
# 3. EXTRACCIÓN DE DATOS DE AWS S3 (72 HORAS)
# ==============================================================================
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

                f_validez = fecha_corrida + datetime.timedelta(hours=lead_time)

                for nom, (xp, yp) in coords_proy.items():
                    nodo = ds.sel(x=xp, y=yp, method="nearest")
                    t2_val = round(float(nodo["T2"].values), 2)
                    pp_val = round(float(nodo["PP"].values), 2)
                    alt_real = PUNTOS[nom]["alt"]

                    # Cálculo de la cota de la Isoterma 0 °C:
                    # z_0 = z_estacion + (T2 / Gradiente) * 1000
                    # Banda: 6.5 °C/1000m (conservador/húmedo) a 9.0 °C/1000m (seco/convectivo)
                    iso_alta = round(alt_real + (t2_val / 6.5) * 1000, 1)
                    iso_baja = round(alt_real + (t2_val / 9.0) * 1000, 1)
                    iso_media = round((iso_alta + iso_baja) / 2, 1)

                    registros.append({
                        "Punto": nom,
                        "Fecha_Validez": f_validez,
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
    raise RuntimeError("No se descargaron datos de los archivos NetCDF. Verifica la corrida o la conexión a S3.")

# Guardar copia local de respaldo
df.to_csv("serie_wrf_tinguiririca_ultima.csv", index=False)

# ==============================================================================
# 4. GENERACIÓN DE GRÁFICOS INTERACTIVOS (PLOTLY)
# ==============================================================================

# A. Gráfico General de Isoterma Cero en la Alta Cuenca (Termas del Flaco)
df_flaco = df[df["Punto"] == "Termas del Flaco"].copy()
fig_iso = go.Figure()

# Banda de seguridad (6.5 a 9.0 °C / 1000 m)
fig_iso.add_trace(go.Scatter(
    x=df_flaco["Fecha_Validez"], y=df_flaco["Iso0_Alta"],
    mode='lines', line=dict(width=0), showlegend=False,
    name='Límite Superior (6.5°C/km)'
))
fig_iso.add_trace(go.Scatter(
    x=df_flaco["Fecha_Validez"], y=df_flaco["Iso0_Baja"],
    mode='lines', line=dict(width=0), fill='tonexty',
    fillcolor='rgba(0, 128, 255, 0.22)',
    name='Banda de Seguridad (6.5 - 9.0 °C/km)'
))
fig_iso.add_trace(go.Scatter(
    x=df_flaco["Fecha_Validez"], y=df_flaco["Iso0_Media"],
    mode='lines+markers', line=dict(color='#0052cc', width=2.5),
    name='Isoterma 0 °C Estimada (Media)'
))

# Líneas de referencia con las cotas de tus instalaciones
colores_ref = ['#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
for i, (nombre_p, meta) in enumerate(list(PUNTOS.items())[:5]):
    fig_iso.add_hline(
        y=meta["alt"], line_dash="dot", line_color=colores_ref[i % len(colores_ref)],
        annotation_text=f"{nombre_p} ({meta['alt']} m)",
        annotation_position="bottom right"
    )

fig_iso.update_layout(
    title="<b>Proyección de Isoterma 0 °C y Banda de Seguridad en la Cuenca Alta</b>",
    yaxis_title="Altitud (m s. n. m.)",
    hovermode="x unified",
    template="plotly_white",
    height=480,
    margin=dict(l=40, r=40, t=60, b=40)
)
html_isoterma = f"<div style='margin-bottom: 45px;'>{fig_iso.to_html(full_html=False, include_plotlyjs='cdn')}</div>"

# B. Gráficos de Temperatura y Precipitación para cada punto
html_puntos = ""
for punto in PUNTOS.keys():
    df_p = df[df["Punto"] == punto].copy()
    if df_p.empty:
        continue
    df_p["PP_acum"] = df_p["PP"].cumsum()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            f"Temperatura (°C) - {punto} ({PUNTOS[punto]['alt']} m s. n. m.)",
            f"Precipitación Horaria y Acumulada - {punto}"
        ),
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
    )

    # Curva de Temperatura
    fig.add_trace(
        go.Scatter(x=df_p["Fecha_Validez"], y=df_p["T2"], name="Temperatura (2m)",
                   line=dict(color="#d9381e", width=2.5)),
        row=1, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="0°C", row=1, col=1)

    # Precipitación horaria
    fig.add_trace(
        go.Bar(x=df_p["Fecha_Validez"], y=df_p["PP"], name="Precipitación horaria (mm)",
               marker_color="#3182bd", opacity=0.75),
        row=2, col=1, secondary_y=False
    )
    # Precipitación acumulada
    fig.add_trace(
        go.Scatter(x=df_p["Fecha_Validez"], y=df_p["PP_acum"], name="Acumulada (mm)",
                   line=dict(color="#08519c", width=2.5)),
        row=2, col=1, secondary_y=True
    )

    fig.update_layout(
        height=620,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    fig.update_yaxes(title_text="Temp (°C)", row=1, col=1)
    fig.update_yaxes(title_text="PP (mm/h)", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Acumulada (mm)", row=2, col=1, secondary_y=True)

    html_puntos += f"<div style='margin-bottom: 50px;'>{fig.to_html(full_html=False, include_plotlyjs=False)}</div>"

# ==============================================================================
# 5. ENSAMBLAJE DEL REPORTE HTML COMPLETO
# ==============================================================================
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
        .section-title {{ font-size: 20px; color: #2c5282; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; margin-top: 30px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Sistema de Pronóstico Cuenca del Tinguiririca</h1>
        <div class="meta">
            <strong>Modelo:</strong> WRF-SMN (Argentina en AWS) | 
            <strong>Corrida:</strong> {fecha_corrida:%Y-%m-%d %H:00} UTC | 
            <strong>Actualizado:</strong> {ahora:%Y-%m-%d %H:%M} UTC<br>
            <strong>Puntos Monitoreados:</strong> {", ".join(PUNTOS.keys())}
        </div>
        
        <div class="section-title">1. Proyección de Isoterma Cero y Análisis de Altitud</div>
        <p style="font-size: 14px; color: #4a5568;">La banda sombreada representa el rango de incertidumbre atmosférico en cordillera, delimitado entre un gradiente húmedo/saturado (6.5 °C/1.000 m) y un gradiente seco (9.0 °C/1.000 m). Las líneas punteadas indican las cotas de tus instalaciones clave.</p>
        {html_isoterma}

        <div class="section-title">2. Pronósticos Detallados por Estación (72 Horas)</div>
        {html_puntos}
    </div>
</body>
</html>"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_final)

print("Reporte generado exitosamente en 'index.html'.")
