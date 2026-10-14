import os
from PIL import Image, ImageOps

base_dir = r"d:\DL_PROJECT"
store_dir = os.path.join(base_dir, "store_assets")
os.makedirs(store_dir, exist_ok=True)

icon_path = os.path.join(base_dir, "app_icon.png")
if not os.path.exists(icon_path):
    print("app_icon.png not found!")
    exit(1)

icon = Image.open(icon_path).convert("RGBA")

# Required Microsoft Store Square and Tile Assets
sizes = {
    "Square44x44Logo.png": (44, 44),
    "Square50x50Logo.png": (50, 50),
    "Square71x71Logo.png": (71, 71),
    "Square150x150Logo.png": (150, 150),
    "Square300x300Logo.png": (300, 300),
    "Square310x310Logo.png": (310, 310),
    "StoreLogo_50x50.png": (50, 50),
    "StoreLogo_300x300.png": (300, 300),
}

for name, size in sizes.items():
    resized = icon.resize(size, Image.Resampling.LANCZOS)
    out_path = os.path.join(store_dir, name)
    resized.save(out_path, "PNG")
    print(f"Generated: {name} ({size[0]}x{size[1]})")

# Wide Tile (310x150)
wide = Image.new("RGBA", (310, 150), (20, 20, 25, 255))
icon_120 = icon.resize((120, 120), Image.Resampling.LANCZOS)
wide.paste(icon_120, (15, 15), icon_120)
wide_path = os.path.join(store_dir, "Wide310x150Logo.png")
wide.save(wide_path, "PNG")
print("Generated: Wide310x150Logo.png")

# Promotional Screenshot (1920x1080)
preview_path = os.path.join(base_dir, "screenshots", "easy_gesture_hud_preview.png")
if os.path.exists(preview_path):
    preview = Image.open(preview_path).convert("RGB")
    screenshot_1080 = preview.resize((1920, 1080), Image.Resampling.LANCZOS)
    screenshot_1080.save(os.path.join(store_dir, "Screenshot_HUD_1920x1080.png"), "PNG")
    print("Generated: Screenshot_HUD_1920x1080.png")

print("All Microsoft Store assets generated successfully in store_assets/!")
