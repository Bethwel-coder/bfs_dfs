# Australia map coloring using graph coloring (max 3 colors)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Define 5 regions and their neighbors (adjacency list)
regions = {
    "Western Australia":  ["Northern Territory", "South Australia"],
    "Northern Territory": ["Western Australia", "Queensland", "South Australia"],
    "Queensland":         ["Northern Territory", "South Australia", "NSW & Victoria"],
    "South Australia":    ["Western Australia", "Northern Territory", "Queensland", "NSW & Victoria"],
    "NSW & Victoria":     ["Queensland", "South Australia"]
}

colors = ["Blue", "Red", "Green"]
color_map = {
    "Blue":  "#0000FF", 
    "Red":   "#FF0000",
    "Green": "#00FF00"
}

assigned = {}

def can_assign(region, color):
    for neighbor in regions[region]:
        if assigned.get(neighbor) == color:
            return False
    return True

def color_regions():
    for region in regions:
        for color in colors:
            if can_assign(region, color):
                assigned[region] = color
                break

color_regions()

# Print result
print("Region Coloring Result:")
print("-" * 40)
for region, color in assigned.items():
    print(f"  {region:<25} -> {color}")
print(f"\nTotal colors used: {len(set(assigned.values()))}")
print(f"Valid coloring:    {len(set(assigned.values())) <= 3}")

# --- Matplotlib Visualization ---
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title("Australia Map Coloring (Max 3 Colors)", fontsize=14, fontweight='bold')

# Define region shapes as polygons [x, y coordinates]
shapes = {
    "Western Australia":  [[1, 1], [1, 7], [4, 7], [4, 4], [3.5, 1]],
    "Northern Territory": [[4, 4], [4, 7], [6, 7], [6, 4]],
    "Queensland":         [[6, 4], [6, 7], [9, 7], [9, 3], [7.5, 3]],
    "South Australia":    [[3.5, 1], [4, 4], [6, 4], [7.5, 3], [7, 1]],
    "NSW & Victoria":     [[7, 1], [7.5, 3], [9, 3], [9, 1]]
}

# Draw each region
for region, coords in shapes.items():
    color_name = assigned[region]
    hex_color = color_map[color_name]
    polygon = plt.Polygon(coords, closed=True, facecolor=hex_color,
                          edgecolor='white', linewidth=2, alpha=0.85)
    ax.add_patch(polygon)

    # Add region label at centroid
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    ax.text(cx, cy, region.replace(" & ", "\n& "), ha='center',
            va='center', fontsize=8, fontweight='bold', color='white')

# Add legend
legend_patches = [
    mpatches.Patch(color=color_map["Blue"],  label="Color 1 - Blue"),
    mpatches.Patch(color=color_map["Red"],   label="Color 2 - Red"),
    mpatches.Patch(color=color_map["Green"], label="Color 3 - Green"),
]
ax.legend(handles=legend_patches, loc='lower left', fontsize=9)

plt.tight_layout()
plt.savefig("australia_map.png", dpi=150)
plt.show()
