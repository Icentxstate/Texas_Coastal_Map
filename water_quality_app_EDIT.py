import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import os

# عنوان صفحه و توضیحات پروژه
st.set_page_config(page_title='Texas Coastal Hydrologic Monitoring Project')
st.title('Texas Coastal Hydrologic Monitoring Project')
st.markdown("""
### Why this project?
Texas lacks long-term, consistent hydrologic data across its coast. This project addresses that gap through collaboration and innovation.

### Purpose
Develop a stakeholder-driven, long-term coastal hydrologic monitoring plan (LTCHMP).

### Goal
Create sustainable, data-informed tools for decision-making, planning, and resilience.
""")

# بارگذاری فایل shapefile
shapefile_path = st.text_input("Enter the path to your Shapefile:", value="C:\\Users\\qrb31\\Downloads\\New folder\\CZB.shp")

if os.path.exists(shapefile_path):
    gdf = gpd.read_file(shapefile_path).to_crs(epsg=4326)

    # ایجاد نقشه
    center = gdf.geometry.centroid.iloc[0]
    m = folium.Map(location=[center.y, center.x], zoom_start=7, tiles="CartoDB positron")

    # افزودن منطقه پروژه با Popup
    popup_html = """
    <div style='font-family: "Segoe UI", sans-serif; font-size: 14px; line-height: 1.6;'>
      <h4 style='margin-bottom: 5px;'>Texas Coastal Hydrologic Monitoring Project</h4>
      <p><strong style='color:#0b5394;'>Why this project?</strong><br>
      Texas lacks long-term, consistent hydrologic data across its coast.</p>
      <p><strong>Purpose:</strong> Develop a stakeholder-driven, long-term coastal hydrologic monitoring plan (LTCHMP).</p>
      <p><strong>Goal:</strong> Create sustainable, data-informed tools for decision-making, planning, and resilience.</p>
    </div>
    """

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
    st.error("⚠️ فایل Shapefile یافت نشد. لطفاً مسیر صحیح را وارد کنید.")
