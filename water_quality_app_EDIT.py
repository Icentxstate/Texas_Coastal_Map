import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import os
from folium.plugins import FloatImage

# عنوان اپلیکیشن
st.title("Texas Coastal Hydrologic Monitoring Project")

# مسیر فایل shapefile
shapefile_path = st.text_input("Enter Shapefile Path:", "C:/Users/qrb31/Downloads/New folder/CZB.shp")

# بررسی وجود فایل shapefile
if os.path.exists(shapefile_path):
    gdf = gpd.read_file(shapefile_path).to_crs(epsg=4326)

    # مرکز نقشه
    center = gdf.geometry.centroid.iloc[0]
    m = folium.Map(location=[center.y, center.x], zoom_start=7, tiles="CartoDB positron")

    # Popup با استایل سایت‌پسند
    popup_html = '''
    <div style="font-family: 'Segoe UI', sans-serif; font-size: 14px; line-height: 1.6;">
      <h4 style="margin-bottom: 5px;">Texas Coastal Hydrologic Monitoring Project</h4>
      <p><strong style="color:#0b5394;">Why this project?</strong><br>
      Texas lacks long-term, consistent hydrologic data across its coast. This project addresses that gap.</p>
    </div>
    '''

    # افزودن منطقه پروژه با Popup
    folium.GeoJson(
        gdf,
        style_function=lambda x: {
            "fillColor": "#0b5394",
            "color": "#0b5394",
            "weight": 2,
            "fillOpacity": 0.4,
        },
        popup=folium.Popup(popup_html, max_width=450)
    ).add_to(m)

    # نمایش نقشه
    st_folium(m, width=800, height=500)

else:
    st.error("Shapefile path is incorrect. Please enter a valid path.")
