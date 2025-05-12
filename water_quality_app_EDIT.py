import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import os

# عنوان صفحه و توضیحات پروژه
st.set_page_config(page_title='Texas Coastal Hydrologic Monitoring Project', layout='wide')
st.title('Texas Coastal Hydrologic Monitoring Project')

# بارگذاری فایل shapefile
shapefile_path = st.text_input("Enter the path to your Shapefile:", value="C:\\Users\\qrb31\\Downloads\\New folder\\CZB.shp")

if os.path.exists(shapefile_path):
    gdf = gpd.read_file(shapefile_path).to_crs(epsg=4326)

    # ایجاد نقشه
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
            popup=folium.Popup(f"<h4>Texas Coastal Hydrologic Monitoring Project</h4><p><strong>Purpose:</strong> Develop a stakeholder-driven, long-term coastal hydrologic monitoring plan (LTCHMP).</p><p><strong>Goal:</strong> Create sustainable, data-informed tools for decision-making, planning, and resilience.</p>", max_width=450)
        ).add_to(m)

    # نمایش نقشه به صورت تمام صفحه
    st_folium(m, width=1200, height=800)
else:
    st.error("⚠️ فایل Shapefile یافت نشد. لطفاً مسیر صحیح را وارد کنید.")
