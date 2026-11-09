"""Image helpers: decode uploads (JPG/PNG/PDF), crop and normalise signatures.

PDF rendering uses pypdfium2 (Apache-2.0 / BSD-3). PyMuPDF is NOT used because it is AGPL.
"""
import io

from PIL import Image, ImageOps

from ..config import settings


def load_pages(data: bytes):
    """Return a list of RGB PIL pages."""
    if data[:4] == b"%PDF":
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(data)
        pages = []
        for i in range(min(len(pdf), settings.max_pages)):
            pages.append(pdf[i].render(scale=150 / 72).to_pil().convert("RGB"))
        return pages
    img = Image.open(io.BytesIO(data))
    return [ImageOps.exif_transpose(img).convert("RGB")]


def crop(page: Image.Image, box, pad=0.04):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1, y1 = max(0, x1 - w * pad), max(0, y1 - h * pad)
    x2, y2 = min(page.width, x2 + w * pad), min(page.height, y2 + h * pad)
    return page.crop((int(x1), int(y1), int(x2), int(y2)))


def _otsu(arr):
    import numpy as np
    hist = np.bincount(arr.ravel(), minlength=256).astype(float)
    total, cum, cum_mean = arr.size, np.cumsum(hist), np.cumsum(hist * np.arange(256))
    w0, w1 = cum, total - cum
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (cum_mean[-1] * w0 - cum_mean * total) ** 2 / (w0 * w1)
    return int(np.nanargmax(between))


def crop_to_ink(g: Image.Image, margin=0.06) -> Image.Image:
    """Cut away empty paper around the strokes so two crops of the same signature line up,
    whatever box size the detector (or the user) used."""
    import numpy as np
    arr = np.asarray(g)
    ys, xs = np.where(arr < _otsu(arr))
    if len(xs) < 20:
        return g
    x1, x2 = np.percentile(xs, [0.5, 99.5]).astype(int)   # ignore isolated specks
    y1, y2 = np.percentile(ys, [0.5, 99.5]).astype(int)
    mx, my = int((x2 - x1) * margin) + 2, int((y2 - y1) * margin) + 2
    return g.crop((max(0, x1 - mx), max(0, y1 - my), min(g.width, x2 + mx + 1), min(g.height, y2 + my + 1)))


def normalise_signature(img: Image.Image, size=224) -> Image.Image:
    """Grayscale, autocontrast, crop to the ink, pad to a white square: removes colour,
    scale and framing differences between scans before embedding."""
    g = crop_to_ink(ImageOps.autocontrast(img.convert("L"), cutoff=1))
    side = max(g.size)
    canvas = Image.new("L", (side, side), 255)
    canvas.paste(g, ((side - g.width) // 2, (side - g.height) // 2))
    return canvas.resize((size, size), Image.BICUBIC).convert("RGB")
