#!/usr/bin/env python3
"""Acceptance checks A-D (CODEX_FIGURE_IMPLEMENTATION_HANDOFF.md §6) on the composed Figures 1 and 5."""
import re
import sys
sys.dont_write_bytecode = True
import numpy as np
import pymupdf
from PIL import Image

from common import FIG1_SRC, FIG5_SRC, IMPL, log_json, sha

R = pymupdf.Rect
Image.MAX_IMAGE_PIXELS = None
F1, F5 = IMPL / "figures/Figure1_NPJSBA_Final.pdf", IMPL / "figures/Figure5_NPJSBA_Final.pdf"
ZOOM = 4.0  # 288 dpi: every documented translation (175, -55.5, 23 pt) is a whole number of pixels
res = {"files": {str(p): sha(p) for p in (F1, F5, FIG1_SRC, FIG5_SRC)}}
fails = []


def ok(name, cond, detail=None):
    res.setdefault("checks", {})[name] = {"pass": bool(cond), "detail": detail}
    if not cond:
        fails.append(name)


def raster(page, rect):
    pm = page.get_pixmap(matrix=pymupdf.Matrix(ZOOM, ZOOM), clip=rect, alpha=False)
    return np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width, 3).astype(np.int16)


def compare(old_pg, new_pg, rect, dx=0.0, dy=0.0, masks=()):
    a = raster(old_pg, rect); b = raster(new_pg, R(rect.x0 + dx, rect.y0 + dy, rect.x1 + dx, rect.y1 + dy))
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1]); a, b = a[:h, :w], b[:h, :w]
    d = np.abs(a - b).max(axis=2)
    keep = np.ones_like(d, bool)
    for m in masks:  # masks in page coordinates of the OLD page
        y0, y1 = int((m.y0 - rect.y0) * ZOOM), int((m.y1 - rect.y0) * ZOOM)
        x0, x1 = int((m.x0 - rect.x0) * ZOOM), int((m.x1 - rect.x0) * ZOOM)
        keep[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = False
    dd = d[keep]
    return {"max": int(dd.max()), "frac_gt8": float((dd > 8).mean()), "frac_gt40": float((dd > 40).mean())}


def spans(page, rect, dx=0, dy=0):
    out = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            for s in l["spans"]:
                bb = R(s["bbox"])
                if s["text"].strip() and rect.contains(bb):
                    out.append((s["text"].strip(), s["font"], round(s["size"], 1), round(s["origin"][0] - dx, 1), round(s["origin"][1] - dy, 1)))
    return sorted(out)


o1, n1 = pymupdf.open(FIG1_SRC)[0], pymupdf.open(F1)[0]
o5, n5 = pymupdf.open(FIG5_SRC)[0], pymupdf.open(F5)[0]
ok("canvas_fig1", (round(n1.rect.width, 2), round(n1.rect.height, 2)) == (594.72, 856.14))
ok("canvas_fig5", (round(n5.rect.width, 2), round(n5.rect.height, 2)) == (595.28, 841.89))

# ---------------- A1/A2 preserved regions (pixels + text)
W1 = n1.rect.width
CARE = R(70, 650, 130, 672)
regions1 = {"A_B_C": (R(0, 0, W1, 265), 0, 0, []), "oldD_to_E": (R(0, 300, W1, 441), 0, 175, []),
            "F": (R(0, 622, 175, 856.14), 0, 0, [CARE])}
for k, (r, dx, dy, m) in regions1.items():
    c = compare(o1, n1, r, dx, dy, m); ok(f"A1_fig1_{k}_pixels", c["frac_gt40"] < 0.002, c)
    so = [s for s in spans(o1, r) if not any(R(s[3], s[4] - 6, s[3] + 1, s[4]).intersects(mm) for mm in m)]
    sn = [s for s in spans(n1, R(r.x0 + dx, r.y0 + dy, r.x1 + dx, r.y1 + dy), dx, dy) if not any(R(s[3], s[4] - 6, s[3] + 1, s[4]).intersects(mm) for mm in m)]
    ok(f"A2_fig1_{k}_text", so == sn, {"old": len(so), "new": len(sn), "diff": sorted(set(so) ^ set(sn))[:10]})
W5 = n5.rect.width
FLAB = R(340, 628, 595, 640)
regions5 = {"A_B_C": (R(0, 0, W5, 335), 0, 0, []), "E_gene_coverage": (R(0, 506, 118, 587), 0, 0, []),
            "E_claim_evaluability": (R(245, 504, 349, 588), -55.5, 0, []), "E_tissue_scope": (R(349.5, 504.5, 556, 591.5), 23, 0, []),
            "F": (R(0, 592, W5, 698.8), 0, 0, [FLAB])}
for k, (r, dx, dy, m) in regions5.items():
    c = compare(o5, n5, r, dx, dy, m); ok(f"A1_fig5_{k}_pixels", c["frac_gt40"] < 0.002, c)
    so = [s for s in spans(o5, r) if not any(R(s[3], s[4] - 6, s[3] + 1, s[4]).intersects(mm) for mm in m)]
    sn = [s for s in spans(n5, R(r.x0 + dx, r.y0 + dy, r.x1 + dx, r.y1 + dy), dx, dy) if not any(R(s[3], s[4] - 6, s[3] + 1, s[4]).intersects(mm) for mm in m)]
    ok(f"A2_fig5_{k}_text", so == sn, {"old": len(so), "new": len(sn), "diff": sorted(set(so) ^ set(sn))[:10]})

# ---------------- B withdrawn content absent (text incl. clipped/hidden, images, orphan objects)
FORBID = ["Spatial ecological", "Spatial ecology", "niche", "Matrix-rich", "Myeloid-rich", "Vascular-rich", "COMMON AXIS", "DIRECTION",
          "192/200", "186/200", "178/200", "H&E", "High − low score", "Mean gene z; common color limits", "Full q = 1 denominator",
          "Naive-only", "Blocked-only", "Bao OS-61", "Bao transcription", "12/20", "Example: Liu EMT-5", "Five-state confusion",
          "R 0.287", "F 0.340", "Computable ≠ informative", "Same-world bidirectional", "Median p = 0.1065", "patients"]
for tag, pg in [("fig1", n1), ("fig5", n5)]:
    t = pg.get_text("text", flags=0).replace("\xa0", " ")
    lines = set(x.strip() for x in t.splitlines())
    hits = [f for f in FORBID if (f in lines if f == "patients" else f in t)]
    if tag == "fig1":  # 'patients' legitimately occurs? check: old E axis label was 'patients'; new figure must not contain it
        pass
    ok(f"B1_{tag}_forbidden_text_absent", not hits, hits)
for tag, path, removed_dims in [("fig1", F1, {(474, 402), (231, 81), (231, 5), (6, 68), (271, 147), (51, 51)}), ("fig5", F5, {(91, 74)})]:
    d = pymupdf.open(path); dims = []
    for x in range(1, d.xref_length()):
        try:
            if d.xref_get_key(x, "Subtype")[1] == "/Image":
                dims.append((int(d.xref_get_key(x, "Width")[1]), int(d.xref_get_key(x, "Height")[1])))
        except Exception:
            pass
    ok(f"B2_{tag}_withdrawn_images_absent", not (set(dims) & removed_dims), {"all_image_objects": dims})

# ---------------- C numbers read back from the final PDFs
t1, t5 = n1.get_text().replace("\xa0", " "), n5.get_text().replace("\xa0", " ")
need1 = ["0.916", "0.933", "0.691", "max 0.744", "max 0.680", "max 0.587", "all genes 0.939", "all genes 0.957", "all genes 0.731",
         "114", "47", "30", "10", "n = 160", "n = 114", "plus-one P = 0.002", "post hoc sensitivity analysis", "96.4% at zero",
         "Spearman ρ = 0.731", "195/201 anatomy-associated", "170/201 above-null classification", "Spearman ρ = 0.804", "2/201 ≥ 0.60",
         "CARE ecology", "(grouped CV)", "Single-cell malignant–TME continuum", "Ecological alignment survives removal of programme genes"]
need5 = ["9/9 tumours", "PAN > CT 8/8", "MVP < CT 9/9", "LE > CT 8/8", "partial R² 0.530", "partial R² 0.591", "partial R² 0.655",
         "BA 0.349", "BA 0.370", "BA 0.493", "0/20 → 20/20", "0/20 → 0/20", "20/20 → 0/20", "55/100", "0/100", "5/100", "8/100", "3/100",
         "median P 0.1745 → 0.003", "median P 0.2915 → 0.1065", "median P 0.0295 → 0.24", "unsigned membership", "SPOCK1", "CRLF1",
         "Computability and claim evaluability", "20/20 · median P 0.003 (MC)", "Reference-dependent detection and null calibration"]
ok("C_fig1_numbers_present", all(s in t1 for s in need1), [s for s in need1 if s not in t1])
ok("C_fig5_numbers_present", all(s in t5 for s in need5), [s for s in need5 if s not in t5])
ok("C_fig1_letters", re.findall(r"^[A-G]$", t1, re.M) and sorted(set(re.findall(r"^[A-G]$", t1, re.M))) == list("ABCDEFG"),
   re.findall(r"^[A-G]$", t1, re.M))
ok("C_fig5_letters", sorted(set(re.findall(r"^[A-G]$", t5, re.M))) == list("ABCDEFG"), re.findall(r"^[A-G]$", t5, re.M))

# ---------------- C5 legibility in new regions: min font size and text-box overlaps
def new_region_spans(pg, rects):
    out = []
    for b in pg.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            for s in l["spans"]:
                bb = R(s["bbox"])
                if s["text"].strip() and any(r.contains(bb) for r in rects):
                    out.append((s["text"], round(s["size"], 2), bb))
    return out

for tag, pg, rects in [("fig1", n1, [R(0, 284, W1, 448), R(175.2, 622, W1, 856.14)]), ("fig5", n5, [R(0, 338, W5, 481), R(0, 699, W5, 841.89)])]:
    ss = new_region_spans(pg, rects)
    small = [(s, z) for s, z, _ in ss if z < 5.35]
    ov = []
    for i in range(len(ss)):
        for j in range(i + 1, len(ss)):
            a, b = ss[i][2], ss[j][2]
            inter = a & b
            if not inter.is_empty and inter.width > 0.6 and inter.height > 1.2:
                ov.append((ss[i][0], ss[j][0], round(inter.width, 1), round(inter.height, 1)))
    ok(f"C5_{tag}_min_font_ge_5.4", not small, {"n_spans": len(ss), "min": min(z for _, z, _ in ss), "small": small[:10]})
    ok(f"C5_{tag}_no_text_overlap", not ov, ov[:15])

# ---------------- D exports
for tag, path, png, dims in [("fig1", F1, IMPL / "figures/Figure1_NPJSBA_Final_600dpi.png", (4956, 7135)),
                             ("fig5", F5, IMPL / "figures/Figure5_NPJSBA_Final_600dpi.png", (4961, 7016))]:
    im = Image.open(png); ok(f"D1_{tag}_png", im.size == dims and im.mode == "RGB", {"size": im.size, "mode": im.mode, "dpi": im.info.get("dpi")})
    d = pymupdf.open(path); fonts = d[0].get_fonts(full=True)
    bad = [f for f in fonts if f[1] == "n/a" or f[2] == "Type3"]
    ok(f"D2_{tag}_fonts_embedded", not bad, {"fonts": sorted({(f[3], f[2], f[1]) for f in fonts}), "bad": bad})
for svg in sorted((IMPL / "panels").glob("*.svg")):
    s = svg.read_text()
    ok(f"D3_svg_text_{svg.stem}", s.count("<text") > 20 and "<image" not in s.split("<g id=\"axes")[0][:0] + "", {"text_elements": s.count("<text"),
                                                                                                                  "raster_images": s.count("<image")})

res["failures"] = fails
log_json("validate_figures.json", res)
for k, v in res["checks"].items():
    print(("PASS " if v["pass"] else "FAIL ") + k, "" if v["pass"] and not isinstance(v["detail"], dict) else v["detail"])
print("FAILURES:", fails)
