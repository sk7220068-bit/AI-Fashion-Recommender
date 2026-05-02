from typing import Dict, Any, List, Tuple

from PIL import Image, ImageDraw, ImageFont


def _safe_font(size: int = 18):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _apply_tint(base: Image.Image, bbox: List[int], rgb: Tuple[int, int, int], alpha: int = 95) -> None:
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    draw.rounded_rectangle([x1, y1, x2, y2], radius=12, fill=(rgb[0], rgb[1], rgb[2], alpha))
    base.alpha_composite(overlay)


def _render_variant(image: Image.Image, upgrade_plan: Dict[str, Any], title: str, tint_rgb: Tuple[int, int, int]) -> Image.Image:
    canvas = image.copy().convert("RGBA")
    detected_items: List[Dict[str, Any]] = upgrade_plan.get("detectedItems", []) or []
    to_replace = {str(x).lower() for x in (upgrade_plan.get("itemsToReplace", []) or [])}
    to_add = upgrade_plan.get("itemsToAdd", []) or []
    occasion = upgrade_plan.get("occasion", "casual")

    title_font = _safe_font(22)
    text_font = _safe_font(16)

    # Simulated "upgrade": tint replace regions + highlight retained regions.
    for item in detected_items:
        category = str(item.get("category", "")).lower()
        bbox = item.get("boundingBox") or item.get("bounding_box")
        if not bbox or len(bbox) != 4:
            continue

        if category in to_replace:
            _apply_tint(canvas, bbox, tint_rgb)
        else:
            _apply_tint(canvas, bbox, (16, 185, 129), alpha=45)

    flat = canvas.convert("RGB")

    panel_width = min(360, max(240, int(flat.width * 0.36)))
    out = Image.new("RGB", (flat.width + panel_width, flat.height), (15, 23, 42))
    out.paste(flat, (0, 0))
    panel = ImageDraw.Draw(out)
    x0 = flat.width + 16
    y = 18

    panel.text((x0, y), f"{title} ({occasion})", fill=(255, 255, 255), font=title_font)
    y += 40

    panel.text((x0, y), "Replace:", fill=(148, 163, 184), font=text_font)
    y += 24
    replace_items = list(to_replace)[:4]
    if replace_items:
        for idx, item in enumerate(replace_items, start=1):
            panel.text((x0, y), f"{idx}. {item}", fill=(248, 113, 113), font=text_font)
            y += 22
    else:
        panel.text((x0, y), "No replacements", fill=(148, 163, 184), font=text_font)
        y += 22

