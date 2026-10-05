{\rtf1\ansi\ansicpg1252\cocoartf2907
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import datetime\
import s3fs\
import xarray as xr\
import cartopy.crs as ccrs\
import pandas as pd\
import plotly.graph_objects as go\
from plotly.subplots import make_subplots\
\
# 1. Puntos de la cuenca\
PUNTOS = \{\
    "Termas del Flaco":       \{"lat": -34.9570, "lon": -70.4350\},\
    "La Rufina":              \{"lat": -34.7330, "lon": -70.7600\},\
    "Puente Negro":           \{"lat": -34.6730, "lon": -70.8750\},\
    "San Fernando (Valle)":   \{"lat": -34.5840, "lon": -70.9890\},\
    "Sector Cordillera":      \{"lat": -34.9800, "lon": -70.3600\}\
\}\
\
# 2. Determinar la corrida m\'e1s reciente disponible (00 UTC de hoy o 12 UTC de ayer)\
ahora = datetime.datetime.now(datetime.timezone.utc)\
# Intentar corrida de 00 UTC del d\'eda de hoy\
fecha_corrida = datetime.datetime(ahora.year, ahora.month, ahora.day, 0)\
# Si a\'fan es muy temprano (antes de las ~07:00 UTC), retroceder a la de 12 UTC de ayer\
if ahora.hour < 7:\
    fecha_corrida = fecha_corrida - datetime.timedelta(hours=12)\
\
fs = s3fs.S3FileSystem(anon=True)\
\
# 3. Procesar las 72 horas\
registros = []\
coords_proy = \{\}\
data_crs = None\
\
print(f"Consultando corrida \{fecha_corrida:%Y-%m-%d %H:00\} UTC...")\
\
for lead_time in range(1, 73):\
    s3_path = f"smn-ar-wrf/DATA/WRF/DET/\{fecha_corrida:%Y/%m/%d/%H\}/WRFDETAR_01H_\{fecha_corrida:%Y%m%d_%H\}_\{lead_time:03d\}.nc"\
    if not fs.exists(s3_path):\
        continue\
    try:\
        with fs.open(s3_path) as f:\
            with xr.open_dataset(f, decode_coords="all", engine="h5netcdf") as ds:\
                if data_crs is None:\
                    lc = ds["Lambert_Conformal"].attrs\
                    data_crs = ccrs.LambertConformal(\
                        central_longitude=lc["longitude_of_central_meridian"],\
                        central_latitude=lc["latitude_of_projection_origin"],\
                        standard_parallels=lc["standard_parallel"]\
                    )\
                    for nom, c in PUNTOS.items():\
                        coords_proy[nom] = data_crs.transform_point(c["lon"], c["lat"], src_crs=ccrs.PlateCarree())\
                \
                f_validez = fecha_corrida + datetime.timedelta(hours=lead_time)\
                for nom, (xp, yp) in coords_proy.items():\
                    nodo = ds.sel(x=xp, y=yp, method="nearest")\
                    registros.append(\{\
                        "Punto": nom,\
                        "Fecha_Validez": f_validez,\
                        "Lead_Time": lead_time,\
                        "T2": round(float(nodo["T2"].values), 2),\
                        "PP": round(float(nodo["PP"].values), 2)\
                    \})\
    except Exception as e:\
        print(f"Error en paso +\{lead_time\}: \{e\}")\
\
df = pd.DataFrame(registros)\
\
# 4. Crear el reporte HTML interactivo con Plotly\
html_plots = ""\
\
for punto in PUNTOS.keys():\
    df_p = df[df["Punto"] == punto].copy()\
    if df_p.empty:\
        continue\
    df_p["PP_acum"] = df_p["PP"].cumsum()\
    \
    fig = make_subplots(\
        rows=2, cols=1, shared_xaxes=True,\
        vertical_spacing=0.08,\
        subplot_titles=(f"Temperatura (\'b0C) - \{punto\}", f"Precipitaci\'f3n Horaria y Acumulada - \{punto\}"),\
        specs=[[\{"secondary_y": False\}], [\{"secondary_y": True\}]]\
    )\
    \
    # Temperatura\
    fig.add_trace(\
        go.Scatter(x=df_p["Fecha_Validez"], y=df_p["T2"], name="Temperatura (2m)",\
                   line=dict(color="#d9381e", width=2.5)),\
        row=1, col=1\
    )\
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="0\'b0C", row=1, col=1)\
    \
    # Precipitaci\'f3n horaria\
    fig.add_trace(\
        go.Bar(x=df_p["Fecha_Validez"], y=df_p["PP"], name="Precipitaci\'f3n horaria (mm)",\
               marker_color="#3182bd", opacity=0.75),\
        row=2, col=1, secondary_y=False\
    )\
    # Precipitaci\'f3n acumulada\
    fig.add_trace(\
        go.Scatter(x=df_p["Fecha_Validez"], y=df_p["PP_acum"], name="Acumulada (mm)",\
                   line=dict(color="#08519c", width=2.5)),\
        row=2, col=1, secondary_y=True\
    )\
    \
    fig.update_layout(\
        height=650,\
        hovermode="x unified",\
        template="plotly_white",\
        margin=dict(l=40, r=40, t=50, b=40)\
    )\
    fig.update_yaxes(title_text="Temp (\'b0C)", row=1, col=1)\
    fig.update_yaxes(title_text="PP (mm/h)", row=2, col=1, secondary_y=False)\
    fig.update_yaxes(title_text="Acumulada (mm)", row=2, col=1, secondary_y=True)\
    \
    html_plots += f"<div style='margin-bottom: 50px;'>\{fig.to_html(full_html=False, include_plotlyjs='cdn')\}</div>"\
\
# Ensamblado del archivo HTML completo\
html_final = f"""<!DOCTYPE html>\
<html lang="es">\
<head>\
    <meta charset="UTF-8">\
    <meta name="viewport" content="width=device-width, initial-scale=1.0">\
    <title>Pron\'f3stico Cuenca Tinguiririca - WRF SMN</title>\
    <style>\
        body \{\{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #f7f9fa; margin: 0; padding: 20px; color: #333; \}\}\
        .container \{\{ max-width: 1100px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); \}\}\
        h1 \{\{ color: #1a365d; margin-top: 0; \}\}\
        .meta \{\{ color: #718096; margin-bottom: 25px; \}\}\
    </style>\
</head>\
<body>\
    <div class="container">\
        <h1>Monitoreo Meteorol\'f3gico Cuenca del Tinguiririca</h1>\
        <div class="meta">\
            <strong>Modelo:</strong> WRF-SMN (Argentina en AWS) | \
            <strong>Corrida:</strong> \{fecha_corrida:%Y-%m-%d %H:00\} UTC | \
            <strong>Actualizado:</strong> \{ahora:%Y-%m-%d %H:%M\} UTC\
        </div>\
        \{html_plots\}\
    </div>\
</body>\
</html>"""\
\
with open("index.html", "w", encoding="utf-8") as f:\
    f.write(html_final)\
\
print("Reporte generado exitosamente en 'index.html'.")\
}