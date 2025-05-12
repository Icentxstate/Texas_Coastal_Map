import geopandas as gpd
import folium
import os
from folium.plugins import FloatImage

# Path to the shapefile
shapefile_path = r"CZB.shp"
gdf = gpd.read_file(shapefile_path).to_crs(epsg=4326)

# Output directory and file
output_dir = os.path.dirname(shapefile_path)
output_file = os.path.join(output_dir, "Texas_Coastal_Interactive_Map.html")

# Map center based on the shapefile's centroid
center = gdf.geometry.centroid.iloc[0]
m = folium.Map(location=[center.y, center.x], zoom_start=7, tiles="CartoDB positron")

#Styled Popup with Project Information
popup_html = """
<div style="font-family: 'Segoe UI', sans-serif; font-size: 14px; line-height: 1.6;">
  <h4 style="margin-bottom: 5px;">Texas Coastal Hydrologic Monitoring Project</h4>
  <p><strong style="color:#0b5394;">Project Overview:</strong><br>
  Texas lacks long-term, consistent hydrologic data across its coast. This project aims to address that gap through collaboration and innovation.</p>
  <p><strong>Purpose:</strong> To develop a stakeholder-driven, long-term coastal hydrologic monitoring plan (LTCHMP).</p>
  <p><strong>Goal:</strong> To create sustainable, data-informed tools for decision-making, planning, and resilience.</p>
</div>
"""

# Adding the project area with a Popup
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

# Adding the logo (must be in the same directory as the output file)
local_logo_path = "meadows-vertical-txstate-blue-gold.png"
logo_full_path = os.path.join(output_dir, local_logo_path)
if os.path.exists(logo_full_path):
    FloatImage(local_logo_path, bottom=5, left=5).add_to(m)
else:
    print("⚠️ Logo file not found at the expected path:", logo_full_path)

# Saving the map
m.save(output_file)
print(" Map with logo saved at:")
print(output_file)
