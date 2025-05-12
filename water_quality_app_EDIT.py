import geopandas as gpd
import folium
import os
from folium.plugins import FloatImage

# Path to the shapefile
shapefile_path = r"CZB.shp"
gdf = gpd.read_file(shapefile_path)

# Reproject to EPSG:4326 if needed
if gdf.crs.to_string() != 'EPSG:4326':
    gdf = gdf.to_crs(epsg=4326)

# Output directory and file
output_dir = os.path.dirname(shapefile_path)
output_file = os.path.join(output_dir, "Texas_Coastal_Interactive_Map.html")

# Map center based on the shapefile's centroid
center = gdf.geometry.centroid.iloc[0]
m = folium.Map(location=[center.y, center.x], zoom_start=7, tiles="CartoDB positron")

# Styled Popup with Project Information
popup_html = """
<div style='font-family: Arial, sans-serif; font-size: 14px;'>
  <h4>Texas Coastal Hydrologic Monitoring Project</h4>
  <p><strong>Project Overview:</strong> Texas lacks long-term, consistent hydrologic data across its coast.</p>
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
logo_path = os.path.join(output_dir, "meadows-vertical-txstate-blue-gold.png")
if os.path.exists(logo_path):
    FloatImage(logo_path, bottom=5, left=5).add_to(m)
else:
    print("⚠️ Logo file not found at the expected path:", logo_path)
