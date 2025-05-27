import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import os
from folium.plugins import FloatImage
import base64

# تنظیمات اولیه Streamlit
st.set_page_config(layout="wide")  # حالت عریض

# حذف حاشیه‌ها و فضای اضافی
st.markdown("""
    <style>
        .block-container {
            padding: 0 !important;
        }
        .main {
            padding: 0 !important;
        }
    </style>
""", unsafe_allow_html=True)

# مسیر فایل shapefile
shapefile_path = st.text_input("Enter Shapefile Path:", "CZB.shp")

# بررسی وجود فایل shapefile
if os.path.exists(shapefile_path):
    gdf = gpd.read_file(shapefile_path).to_crs(epsg=4326)

    # مرکز نقشه
    center = gdf.geometry.centroid.iloc[0]
    m = folium.Map(location=[center.y, center.x], zoom_start=7, tiles="CartoDB positron")

    # Popup
    popup_html = '''
    <div style="font-family: 'Segoe UI', sans-serif; font-size: 14px; line-height: 1.6;">
      <h4 style="margin-bottom: 5px;">Texas Coastal Hydrologic Monitoring Project</h4>
      <p><strong style="color:#0b5394;">Why this project?</strong><br>
      Texas lacks long-term, consistent hydrologic data across its coast. This project addresses that gap through collaboration and innovation.</p>
      <p><strong>Purpose:</strong> Develop a stakeholder-driven, long-term coastal hydrologic monitoring plan (LTCHMP).</p>
      <p><strong>Goal:</strong> Create sustainable, data-informed tools for decision-making, planning, and resilience.</p>
    </div>
    '''

    # افزودن منطقه پروژه
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

    # افزودن لوگو
    logo_path = "meadows-logo.png"
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            logo_base64 = base64.b64encode(f.read()).decode("utf-8")
        logo_html = f'''
        <img src="data:image/png;base64,{logo_base64}" 
             style="position: fixed; bottom: 10px; left: 10px; width: 120px; z-index: 1000;">
        '''
        m.get_root().html.add_child(folium.Element(logo_html))

    # نقشه تمام‌صفحه
    st_folium(m, use_container_width=True, height=900)

else:
    st.error("⚠️ فایل Shapefile یافت نشد. لطفاً مسیر صحیح را وارد کنید.")
