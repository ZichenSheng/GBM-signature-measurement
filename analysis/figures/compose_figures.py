#!/usr/bin/env python3
"""Compose the final Figures 1 and 5 from (i) redacted copies of the authoritative Illustrator pages, (ii) isolated
translated pieces of those pages and (iii) the newly built vector panels. The original PDFs are opened read-only;
all writes go to 06_FIGURE_IMPLEMENTATION. Usage: compose_figures.py [1] [5]"""
import sys
sys.dont_write_bytecode = True
import pymupdf

from pathlib import Path
from common import (ARIAL, ARIAL_B, FIG1_SRC, FIG5_SRC, IMPL, log_json, sha)

R = pymupdf.Rect
WORK, FIGS, PAN = IMPL / "work", IMPL / "figures", IMPL / "panels"
G_NONE, G_COVERED, G_TOUCHED = (pymupdf.PDF_REDACT_LINE_ART_NONE, pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED,
                                 pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED)
I_NONE, I_REMOVE = pymupdf.PDF_REDACT_IMAGE_NONE, pymupdf.PDF_REDACT_IMAGE_REMOVE


def hexrgb(h):
    h = int(h.lstrip("#"), 16); return ((h >> 16) / 255, ((h >> 8) & 255) / 255, (h & 255) / 255)


def inventory(page, rect):
    """Objects intersecting rect: (contained, crossing) for text spans, drawings, images."""
    out = {"text_in": [], "text_cross": [], "draw_in": 0, "draw_cross": [], "img_in": [], "img_cross": []}
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            for s in l["spans"]:
                bb = R(s["bbox"])
                if s["text"].strip() and bb.intersects(rect):
                    (out["text_in"] if rect.contains(bb) else out["text_cross"]).append((s["text"], [round(v, 1) for v in bb]))
    for d in page.get_drawings():
        r = d["rect"]
        if r.intersects(rect) or (r.width == 0 or r.height == 0) and rect.contains(r.tl):
            if rect.contains(r): out["draw_in"] += 1
            else: out["draw_cross"].append([round(v, 1) for v in r])
    for im in page.get_image_info(xrefs=True):
        bb = R(im["bbox"])
        if bb.intersects(rect):
            (out["img_in"] if rect.contains(bb) else out["img_cross"]).append([round(v, 1) for v in bb])
    return out


def redact(page, rects, graphics=G_COVERED, images=I_REMOVE):
    for r in rects:
        page.add_redact_annot(r, fill=False)
    page.apply_redactions(images=images, graphics=graphics, text=pymupdf.PDF_REDACT_TEXT_REMOVE)


def isolate(src, keep, name, extra=()):
    """Copy of src page keeping only content inside `keep` (complement redaction)."""
    d = pymupdf.open(src); p = d[0]; W, H = p.rect.width, p.rect.height
    comp = [r for r in [R(0, 0, W, keep.y0), R(0, keep.y1, W, H), R(0, keep.y0, keep.x0, keep.y1), R(keep.x1, keep.y0, W, keep.y1)]
            if r.width > 0.01 and r.height > 0.01]
    redact(p, comp + list(extra))
    p = d.reload_page(p)
    drop_xobjects(d, p, [keep], inside=False)
    if extra:
        drop_xobjects(d, p, list(extra), inside=True)
    out = WORK / f"{name}.pdf"; d.save(out, garbage=4, deflate=True, clean=True); d.close()
    return out


def drop_xobjects(doc, page, rects, inside=True):
    """Delete Form/Image XObjects from the page resources whose bbox lies wholly inside (inside=True) or wholly
    outside (inside=False) the given top-left-origin rects, so that garbage collection removes their data."""
    H = page.rect.height
    res = doc.xref_get_key(page.xref, "Resources")
    rx, pre = (int(res[1].split()[0]), "") if res[0] == "xref" else (page.xref, "Resources/")
    dropped = []
    for xref, name, invoker, bb in page.get_xobjects():
        if invoker != 0:
            continue
        r = R(bb[0], H - bb[3], bb[2], H - bb[1])
        hit = any(q.contains(r) for q in rects)
        if (hit if inside else not any(q.intersects(r) for q in rects)):
            # keep the resource name valid but make the form empty and detach its resources (images become garbage)
            doc.update_stream(xref, b" ")
            doc.xref_set_key(xref, "Resources", "<<>>")
            dropped.append((name, [round(v, 1) for v in r]))
    for im in page.get_images(full=True):
        if im[9] == 0:
            bbs = [b for b in page.get_image_rects(im[0])]
            if bbs and all((any(q.contains(b) for q in rects) if inside else not any(q.intersects(b) for q in rects)) for b in bbs):
                dropped.append((im[7], "direct image still invoked - left in place"))
            elif not bbs:
                doc.xref_set_key(rx, f"{pre}XObject/{im[7]}", "null"); dropped.append((im[7], "uninvoked direct image removed"))
    return dropped


def put_text(page, x, y, s, size, bold=False, color="#000000", align="left"):
    ff = ARIAL_B if bold else ARIAL
    w = pymupdf.Font(fontfile=ff).text_length(s, fontsize=size)
    x = x if align == "left" else (x - w / 2 if align == "center" else x - w)
    page.insert_text((x, y), s, fontsize=size, fontfile=ff, fontname="ArialBoldX" if bold else "ArialX", color=hexrgb(color))
    return R(x, y - size, x + w, y + 0.25 * size)


def export(doc_path, stem):
    d = pymupdf.open(doc_path); p = d[0]
    p.get_pixmap(dpi=600, alpha=False).save(FIGS / f"{stem}_600dpi.png")
    d.close()


# ======================================================================= Figure 1
def figure1():
    log = {"source": str(FIG1_SRC), "source_sha256": sha(FIG1_SRC)}
    src = pymupdf.open(FIG1_SRC); W, H = src[0].rect.width, src[0].rect.height
    assert (round(W, 2), round(H, 2)) == (594.72, 856.14)
    OLD_D = R(0, 283, W, 441)                    # single-cell panel (moves to E)
    rm = {"old_D_E": R(0, 282.5, W, 622),        # old D (re-inserted translated) + withdrawn spatial E
          "C_footer": R(236, 265.5, W, 282.5),   # R/F dumbbells (overlap content moves to new D)
          "old_G": R(175.2, 622, W, H)}          # rebuilt natively (Liu EMT-5 inset removed)
    log["pre_redaction_inventory"] = {k: inventory(src[0], r) for k, r in rm.items()}
    piece = isolate(FIG1_SRC, OLD_D, "fig1_piece_oldD_singlecell", extra=[R(236, 265, W, 298)])  # C footer residue (e.g. "F 0.731" at y 283.0)
    log["piece_oldD_inventory"] = inventory(pymupdf.open(piece)[0], R(0, 0, W, H))
    page = src[0]
    # CARE header text (text only)
    care = [R(92.0, 655.5, 109.2, 664.0), R(89.3, 661.6, 112.0, 670.2)]
    redact(page, care, graphics=G_NONE, images=I_NONE)
    redact(page, [rm["old_D_E"]])                                   # B log-histogram bars cross this box: covered-only
    redact(page, [rm["C_footer"], rm["old_G"]], graphics=G_TOUCHED)  # no crossing paths
    page = src.reload_page(page)                                       # avoid PyMuPDF image-info cache
    log["dropped_xobjects"] = drop_xobjects(src, page, list(rm.values()), inside=True)
    page = src.reload_page(page)
    log["post_redaction_residue"] = {k: inventory(page, r) for k, r in rm.items()}
    # place pieces
    dy = 175.0
    page.show_pdf_page(R(OLD_D.x0, OLD_D.y0 + dy, OLD_D.x1, OLD_D.y1 + dy), pymupdf.open(piece), 0, clip=OLD_D)
    page.show_pdf_page(R(0, 284, W, 448), pymupdf.open(PAN / "Fig1D_overlap_controlled_alignment.pdf"), 0)
    page.show_pdf_page(R(175.2, 622, W, H), pymupdf.open(PAN / "Fig1G_state_resolution_rebuilt.pdf"), 0)
    put_text(page, 10.99, 288.8 + dy, "E", 12, True, "#111111")
    cx = (89.67 + 111.6) / 2
    put_text(page, cx, 661.63, "CARE ecology", 5.8, True, "#152033", "center")
    put_text(page, cx, 667.76, "(grouped CV)", 5.8, True, "#152033", "center")
    out = FIGS / "Figure1_NPJSBA_Final.pdf"
    src.save(out, garbage=4, deflate=True, clean=True)
    log["placements"] = {"old_D->E": f"translate +{dy} pt; clip {list(OLD_D)}", "new_D": [0, 284, W, 448], "new_G": [175.2, 622, W, H]}
    log["output"] = {str(out): sha(out)}
    export(out, "Figure1_NPJSBA_Final")
    log_json("compose_figure1.json", log)
    print("Figure 1 composed:", out)


# ======================================================================= Figure 5
def figure5():
    log = {"source": str(FIG5_SRC), "source_sha256": sha(FIG5_SRC)}
    src = pymupdf.open(FIG5_SRC); W, H = src[0].rect.width, src[0].rect.height
    assert (round(W, 2), round(H, 2)) == (595.28, 841.89)
    rm = {"old_D": R(0, 337.8, W, 481.2), "old_D_letter": R(12.5, 336.0, 22.0, 353.0),
          "E_ivy_partialR2": R(122, 506, 242, 587), "old_G": R(0, 699.0, W, H), "old_G_letter": R(9.5, 693.0, 19.6, 711.0)}
    log["pre_redaction_inventory"] = {k: inventory(src[0], r) for k, r in rm.items()}
    page = src[0]
    # text-only replacements: E title, F median labels
    txt = {"E_title": R(26.3, 491.6, 206.2, 506.0), "F1": R(354.2, 630.5, 410.8, 638.0), "F2": R(442.0, 630.5, 498.6, 638.0),
           "F3": R(532.7, 630.5, 583.6, 638.0)}
    log["pre_text_inventory"] = {k: inventory(page, r) for k, r in txt.items()}
    redact(page, list(txt.values()), graphics=G_NONE, images=I_NONE)
    assert all(not v["draw_cross"] for v in log["pre_redaction_inventory"].values())
    redact(page, list(rm.values()), graphics=G_TOUCHED)
    page = src.reload_page(page)
    log["dropped_xobjects"] = drop_xobjects(src, page, list(rm.values()), inside=True)
    page = src.reload_page(page)
    log["post_redaction_residue"] = {k: inventory(page, r) for k, r in rm.items()}
    E_MOVE = __import__("json").loads(Path(__file__).with_name("fig5_E_layout.json").read_text()) if Path(__file__).with_name("fig5_E_layout.json").exists() else None
    if E_MOVE:  # optional translation of E sub-panels (decided after visual inspection)
        for name, (x0, y0, x1, y1, dx) in E_MOVE.items():
            keep = R(x0, y0, x1, y1)
            piece = isolate(FIG5_SRC, keep, f"fig5_piece_{name}")
            redact(page, [keep])
            page.show_pdf_page(R(x0 + dx, y0, x1 + dx, y1), pymupdf.open(piece), 0, clip=keep)
        log["E_moves"] = E_MOVE
    page.show_pdf_page(R(0, 338, W, 481), pymupdf.open(PAN / "Fig5D_ivy_anatomical_profiles.pdf"), 0)
    page.show_pdf_page(R(0, 699, W, H), pymupdf.open(PAN / "Fig5G_reference_dependent_detection.pdf"), 0)
    put_text(page, 26.53, 502.01, "Computability and claim evaluability", 10, True, "#231815")
    centres = [354.42 + (410.5 - 354.42) / 2, 442.22 + (498.3 - 442.22) / 2, 532.95 + (583.3 - 532.95) / 2]
    for cxx, s in zip(centres, ["20/20 · median P 0.003 (MC)", "0/20 · median P 0.107 (MC)", "0/20 · median P 0.24 (MC)"]):
        put_text(page, cxx, 636.01, s, 5.19, False, "#231815", "center")
    out = FIGS / "Figure5_NPJSBA_Final.pdf"
    src.save(out, garbage=4, deflate=True, clean=True)
    log["output"] = {str(out): sha(out)}
    export(out, "Figure5_NPJSBA_Final")
    log_json("compose_figure5.json", log)
    print("Figure 5 composed:", out)


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "5"]
    if "1" in which: figure1()
    if "5" in which: figure5()
