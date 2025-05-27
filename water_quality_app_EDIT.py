import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import os
import base64
from folium.plugins import FloatImage

# پیکربندی صفحه به حالت عریض
st.set_page_config(layout="wide")

# حذف padding اضافی
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

# عنوان اپلیکیشن
st.title("Texas Coastal Hydrologic Monitoring Project")

# وارد کردن مسیر فایل Shapefile
shapefile_path = st.text_input("Enter Shapefile Path:", "CZB.shp")

# بررسی وجود فایل
if os.path.exists(shapefile_path):
    gdf = gpd.read_file(shapefile_path).to_crs(epsg=4326)

    # مرکز نقشه بر اساس هندسه منطقه
    center = gdf.geometry.centroid.iloc[0]
    m = folium.Map(location=[center.y, center.x], zoom_start=7, tiles="CartoDB positron")

    # محتوای Popup
    popup_html = '''
    <div style="font-family: 'Segoe UI', sans-serif; font-size: 14px; line-height: 1.6;">
      <h4 style="margin-bottom: 5px;">Texas Coastal Hydrologic Monitoring Project</h4>
      <p><strong style="color:#0b5394;">Why this project?</strong><br>
      Texas lacks long-term, consistent hydrologic data across its coast. This project addresses that gap through collaboration and innovation.</p>
      <p><strong>Purpose:</strong> Develop a stakeholder-driven, long-term coastal hydrologic monitoring plan (LTCHMP).</p>
      <p><strong>Goal:</strong> Create sustainable, data-informed tools for decision-making, planning, and resilience.</p>
    </div>
    '''

    # اضافه کردن ناحیه shapefile
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

    # اضافه کردن لوگوی Meadows Center
    logo_path = "meadows-logo.png"  # مطمئن شو این فایل کنار app هست
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as image_file:
            encoded_logo = base64.b64encode(image_file.read()).decode("utf-8")

        logo_html = f'''
        <div style="position: fixed; bottom: 10px; left: 10px; z-index: 1000;">
            <img src="data:image/png;base64,{encoded_logo}" style="width: 140px; opacity: 0.9;">
        </div>
        '''
        m.get_root().html.add_child(folium.Element(logo_html))

    # نمایش نقشه
    st_folium(m, use_container_width=True, height=900)

else:
    st.error("⚠️ فایل Shapefile یافت نشد. لطفاً مسیر صحیح را وارد کنید.")
