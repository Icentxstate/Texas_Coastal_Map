import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import os
from folium.plugins import FloatImage

# عنوان اپلیکیشن
st.set_page_config(page_title='Texas Coastal Hydrologic Monitoring Project', layout='wide')
st.title("Texas Coastal Hydrologic Monitoring Project")

# مسیر فایل shapefile
shapefile_path = st.text_input("Enter Shapefile Path:", "CZB.shp")

# بررسی وجود فایل shapefile
if os.path.exists(shapefile_path):
    gdf = gpd.read_file(shapefile_path).to_crs(epsg=4326)

    # مرکز نقشه
    center = gdf.geometry.centroid.iloc[0]
    m = folium.Map(location=[center.y, center.x], zoom_start=7, tiles="CartoDB positron")

    # افزودن منطقه پروژه با Popup
    for _, row in gdf.iterrows():
        folium.GeoJson(
            row.geometry,
            style_function=lambda x: {
                "fillColor": "#0b5394",
                "color": "#0b5394",
                "weight": 2,
                "fillOpacity": 0.4,
            },
            popup=folium.Popup(f"""
                <div style='font-family: Segoe UI, sans-serif; font-size: 14px; line-height: 1.6;'>
                  <h4>Texas Coastal Hydrologic Monitoring Project</h4>
                  <p><strong>Purpose:</strong> Develop a stakeholder-driven, long-term coastal hydrologic monitoring plan (LTCHMP).</p>
                  <p><strong>Goal:</strong> Create sustainable, data-informed tools for decision-making, planning, and resilience.</p>
                </div>
            """, max_width=450)
        ).add_to(m)

    # افزودن لوگوی محلی Meadows Center
    logo_path = "meadows-logo.png"
    if os.path.exists(logo_path):
        FloatImage(logo_path, bottom=5, left=5).add_to(m)

    # نمایش نقشه به صورت تمام صفحه
    st.markdown("<style>div.st_folium {height: 95vh !important; width: 100vw !important;}</style>", unsafe_allow_html=True)
    st_folium(m, width=1500, height=900)
else:
    st.error("⚠️ فایل Shapefile یافت نشد. لطفاً مسیر صحیح را وارد کنید.")
