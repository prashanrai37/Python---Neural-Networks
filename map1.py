import folium
map = folium.Map(location=[53.799665237832066, -1.553104940964487], zoom_start=6, tiles = "Stamen Terrain")

map.add_child(folium.Marker(location=[57.070815515729436, -3.669454422744673], popup = "Ben Macdui", icon=folium.Icon(color="green")))

map.save("Map1.html")

