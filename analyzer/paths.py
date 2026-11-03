"""Vị trí mô hình AI (nằm ngoài Dropbox để không bị đồng bộ hàng GB dữ liệu)."""
from pathlib import Path

BASE = Path.home() / "court-vision-ai"
MODELS_DIR = BASE / "models"
TRACKNET_DIR = MODELS_DIR / "tracknet"   # chỉ chứa trọng số TrackNetV3 + LICENSE
FONT_REGULAR = Path("C:/Windows/Fonts/segoeui.ttf")
FONT_BOLD = Path("C:/Windows/Fonts/segoeuib.ttf")
