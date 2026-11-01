#!/usr/bin/env python3
"""Generate an intuitive, attractive PowerPoint training course for
Scorecards4Extremes and its diagnostic tools.

Run:  .venv/bin/python diagnostics/panels/make_training_pptx.py
Output: presentations/scorecards4extremes_training.pptx

The deck embeds three real example figures produced by the toolkit:
  * a scorecard heatmap panel   (the flagship output)
  * the IFS-vs-AIFS Q-Q panel   (day-3 extremes diagnostic)
  * an ensemble reliability diagnostic
"""

from __future__ import annotations

import os

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.expanduser("~")
OUT = os.path.join(OUT_DIR, "scorecards4extremes_training.pptx")

IMG_SCORECARD = os.path.join(
    ROOT,
    "plots/ENS_ifs_hybrid/tp24/tp24_ens_datasetclim_p99obs_ifs4hybrid/"
    "heatmap_smooth_panel_tp24_ifs_ens4hybrid_vs_hybrid_ens_all_conditions.png",
)
IMG_QQ = os.path.join(ROOT, "plots/qq_day3_ifsvsaifs/panel_qq_ifs_vs_aifs_day3.png")
IMG_DIAG = os.path.join(
    ROOT, "plots/warm_extremes_diagnostics/warm_extremes_diagnostics_day3.png"
)

# ---------------------------------------------------------------------------
# Colour palette (Okabe-Ito, colour-blind safe, matches the toolkit's plots)
# ---------------------------------------------------------------------------
NAVY = RGBColor(0x0E, 0x22, 0x33)   # deep background for title / dividers
BLUE = RGBColor(0x00, 0x72, 0xB2)   # IFS / primary accent
ORANGE = RGBColor(0xD5, 0x5E, 0x00) # AIFS / secondary accent
GREEN = RGBColor(0x00, 0x9E, 0x73)  # observations / positive
SKY = RGBColor(0x56, 0xB4, 0xE9)    # light accent
YELLOW = RGBColor(0xF0, 0xE4, 0x42) # highlight
GREY = RGBColor(0x3A, 0x3A, 0x3A)   # body text
LIGHT = RGBColor(0xF3, 0xF6, 0xFA)  # light panel background
MIDGREY = RGBColor(0x77, 0x77, 0x77)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
CODEBG = RGBColor(0x1B, 0x2B, 0x34)
CODEFG = RGBColor(0xE8, 0xED, 0xF0)

FONT = "Calibri"
MONO = "Consolas"

# ---------------------------------------------------------------------------
# Presentation set-up (16:9 widescreen)
# ---------------------------------------------------------------------------
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]

_slide_no = 0


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def _no_line(shape):
    shape.line.fill.background()


def _solid(shape, color):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    _no_line(shape)
    shape.shadow.inherit = False


def add_rect(slide, x, y, w, h, color, shape=MSO_SHAPE.RECTANGLE):
    sp = slide.shapes.add_shape(shape, x, y, w, h)
    _solid(sp, color)
    return sp


def add_text(
    slide, x, y, w, h, text, size=18, color=GREY, bold=False, italic=False,
    align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, font=FONT, line_spacing=1.0,
):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = align
    if line_spacing:
        p.line_spacing = line_spacing
    r = p.add_run()
    r.text = text
    f = r.font
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.name = font
    f.color.rgb = color
    return tb


def _set_bg(slide, color):
    bg = add_rect(slide, 0, 0, SW, SH, color)
    slide.shapes._spTree.remove(bg._element)
    slide.shapes._spTree.insert(2, bg._element)
    return bg


def _footer(slide, dark=False):
    """Slide number + branding footer for content slides."""
    global _slide_no
    _slide_no += 1
    col = RGBColor(0xB8, 0xC4, 0xCE) if dark else MIDGREY
    add_text(slide, Inches(0.5), Inches(7.02), Inches(6), Inches(0.35),
             "Scorecards4Extremes  ·  Training course", size=10, color=col)
    add_text(slide, Inches(11.3), Inches(7.02), Inches(1.53), Inches(0.35),
             str(_slide_no), size=10, color=col, align=PP_ALIGN.RIGHT)


def content_header(slide, title, kicker=None, accent=BLUE):
    """Standard content-slide header: accent bar + title (+ optional kicker)."""
    _set_bg(slide, WHITE)
    add_rect(slide, 0, 0, Inches(0.28), SH, accent)              # left spine
    add_rect(slide, Inches(0.6), Inches(0.55), Inches(0.55), Inches(0.09), accent)
    if kicker:
        add_text(slide, Inches(0.6), Inches(0.66), Inches(11), Inches(0.35),
                 kicker.upper(), size=12.5, color=accent, bold=True)
        ty = Inches(1.0)
    else:
        ty = Inches(0.78)
    add_text(slide, Inches(0.6), ty, Inches(12.1), Inches(0.9),
             title, size=30, color=NAVY, bold=True)
    add_rect(slide, Inches(0.62), Inches(1.72), Inches(12.1), Pt(1.4), RGBColor(0xE1, 0xE7, 0xEC))


def add_bullets(slide, bullets, x=Inches(0.7), y=Inches(2.0),
                w=Inches(11.9), h=Inches(4.6), size=17, gap=8):
    """bullets: list of dicts {level, lead, text, color, bold}."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, b in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        lvl = b.get("level", 0)
        p.level = lvl
        p.space_after = Pt(b.get("gap", gap))
        p.line_spacing = b.get("line_spacing", 1.05)
        # bullet glyph via a coloured lead marker
        marker = b.get("marker", "●" if lvl == 0 else "–")
        mcol = b.get("accent", BLUE if lvl == 0 else SKY)
        rm = p.add_run()
        rm.text = f"{marker}  "
        rm.font.size = Pt(size if lvl == 0 else size - 2)
        rm.font.name = FONT
        rm.font.color.rgb = mcol
        rm.font.bold = True
        if b.get("lead"):
            rl = p.add_run()
            rl.text = b["lead"]
            rl.font.size = Pt(size if lvl == 0 else size - 2)
            rl.font.bold = True
            rl.font.name = FONT
            rl.font.color.rgb = b.get("lead_color", NAVY)
        rt = p.add_run()
        rt.text = b.get("text", "")
        rt.font.size = Pt(size if lvl == 0 else size - 2)
        rt.font.name = FONT
        rt.font.color.rgb = b.get("color", GREY)
    return tb


def add_chip(slide, x, y, w, h, text, fill, fg=WHITE, size=13, bold=True):
    sp = add_rect(slide, x, y, w, h, fill, MSO_SHAPE.ROUNDED_RECTANGLE)
    try:
        sp.adjustments[0] = 0.18
    except Exception:
        pass
    tf = sp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_top = Pt(2)
    tf.margin_bottom = Pt(2)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.name = FONT
    r.font.color.rgb = fg
    return sp


def add_image_fit(slide, path, x, y, w, h, frame=True):
    """Add an image scaled to fit inside the (w,h) box, centred, keeping aspect."""
    from PIL import Image
    try:
        iw, ih = Image.open(path).size
    except Exception:
        iw, ih = 1600, 1000
    box_ratio = w / h
    img_ratio = iw / ih
    if img_ratio > box_ratio:
        nw = w
        nh = int(w / img_ratio)
    else:
        nh = h
        nw = int(h * img_ratio)
    nx = x + (w - nw) // 2
    ny = y + (h - nh) // 2
    if frame:
        pad = Emu(int(Inches(0.06)))
        add_rect(slide, nx - pad, ny - pad, nw + 2 * pad, nh + 2 * pad,
                 RGBColor(0xDD, 0xE3, 0xE8))
    slide.shapes.add_picture(path, nx, ny, nw, nh)
    return nx, ny, nw, nh


def new_slide():
    return prs.slides.add_slide(BLANK)


# ---------------------------------------------------------------------------
# Slide builders
# ---------------------------------------------------------------------------
def title_slide():
    s = new_slide()
    _set_bg(s, NAVY)
    # decorative accent bands
    add_rect(s, 0, Inches(5.55), SW, Inches(0.10), BLUE)
    add_rect(s, 0, Inches(5.68), SW, Inches(0.06), GREEN)
    add_rect(s, 0, Inches(5.78), SW, Inches(0.04), ORANGE)
    # colour dots motif (like q-q markers)
    for i, c in enumerate([BLUE, SKY, GREEN, YELLOW, ORANGE]):
        add_rect(s, Inches(0.9 + i * 0.42), Inches(0.9), Inches(0.26), Inches(0.26),
                 c, MSO_SHAPE.OVAL)
    add_text(s, Inches(0.9), Inches(1.9), Inches(11.6), Inches(1.7),
             "Scorecards4Extremes", size=54, color=WHITE, bold=True)
    add_text(s, Inches(0.92), Inches(3.15), Inches(11.5), Inches(0.9),
             "Verifying weather forecasts where it matters most — the extremes",
             size=23, color=SKY, bold=False)
    add_text(s, Inches(0.95), Inches(4.15), Inches(11), Inches(0.6),
             "A hands-on training course  ·  the tool & its diagnostic suite",
             size=17, color=RGBColor(0xC7, 0xD2, 0xDC))
    add_text(s, Inches(0.95), Inches(6.15), Inches(11), Inches(0.6),
             "IFS  ·  AIFS  ·  research experiments      vs      SYNOP observations",
             size=14, color=RGBColor(0x9A, 0xAA, 0xB6), bold=True)


def divider_slide(num, title, subtitle=""):
    s = new_slide()
    _set_bg(s, NAVY)
    add_rect(s, 0, Inches(2.55), Inches(0.9), Inches(2.0), BLUE)
    add_text(s, Inches(1.15), Inches(2.35), Inches(2.2), Inches(2.2),
             f"{num:02d}", size=96, color=RGBColor(0x2B, 0x4C, 0x63), bold=True)
    add_text(s, Inches(3.4), Inches(2.9), Inches(9.2), Inches(1.4),
             title, size=40, color=WHITE, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    if subtitle:
        add_text(s, Inches(3.45), Inches(4.25), Inches(9.0), Inches(0.8),
                 subtitle, size=18, color=SKY)
    add_rect(s, Inches(3.45), Inches(4.15), Inches(4.5), Pt(2), GREEN)


def pipeline_slide():
    s = new_slide()
    content_header(s, "The pipeline: one config → nine automatic steps",
                   kicker="How it works", accent=BLUE)
    steps = [
        ("1", "READ DATA", "GRIB forecasts + obs", BLUE),
        ("2", "PRE-PROCESS", "units, lapse-rate, accum.", BLUE),
        ("3", "EXTRACT", "interpolate to stations", GREEN),
        ("4", "FILTER", "season · terrain · QC", GREEN),
        ("5", "THRESHOLD", "define the extreme event", ORANGE),
        ("6", "SCORES", "ETS · PSS · twMAE · …", ORANGE),
        ("7", "BOOTSTRAP", "95% confidence", SKY),
        ("8", "SAVE", "results to CSV", SKY),
        ("9", "PLOT", "heatmap scorecards", GREEN),
    ]
    # 3 columns x 3 rows grid of step cards
    x0, y0 = Inches(0.75), Inches(2.05)
    cw, ch = Inches(3.85), Inches(1.28)
    gx, gy = Inches(0.28), Inches(0.24)
    for i, (n, name, desc, col) in enumerate(steps):
        r, c = divmod(i, 3)
        x = x0 + c * (cw + gx)
        y = y0 + r * (ch + gy)
        card = add_rect(s, x, y, cw, ch, LIGHT, MSO_SHAPE.ROUNDED_RECTANGLE)
        try:
            card.adjustments[0] = 0.08
        except Exception:
            pass
        add_rect(s, x, y, Inches(0.14), ch, col)
        num = add_rect(s, x + Inches(0.32), y + Inches(0.30), Inches(0.66),
                       Inches(0.66), col, MSO_SHAPE.OVAL)
        tf = num.text_frame
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        rr = p.add_run(); rr.text = n
        rr.font.size = Pt(20); rr.font.bold = True
        rr.font.name = FONT; rr.font.color.rgb = WHITE
        add_text(s, x + Inches(1.15), y + Inches(0.20), cw - Inches(1.25),
                 Inches(0.5), name, size=15.5, color=NAVY, bold=True)
        add_text(s, x + Inches(1.15), y + Inches(0.70), cw - Inches(1.25),
                 Inches(0.5), desc, size=11.5, color=MIDGREY)
    add_text(s, Inches(0.75), Inches(6.55), Inches(11.8), Inches(0.4),
             "Run everything with a single command:   python run.py my_config.yaml",
             size=14, color=BLUE, bold=True)
    _footer(s)


def code_block(slide, x, y, w, h, lines, size=13):
    box = add_rect(slide, x, y, w, h, CODEBG, MSO_SHAPE.ROUNDED_RECTANGLE)
    try:
        box.adjustments[0] = 0.03
    except Exception:
        pass
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Pt(14)
    tf.margin_right = Pt(10)
    tf.margin_top = Pt(10)
    tf.margin_bottom = Pt(10)
    for i, (txt, col) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = 1.12
        r = p.add_run()
        r.text = txt
        r.font.size = Pt(size)
        r.font.name = MONO
        r.font.color.rgb = col
    return box


# ===========================================================================
# BUILD THE DECK
# ===========================================================================

# 1. Title
title_slide()

# 2. Agenda
s = new_slide()
content_header(s, "What we will cover today", kicker="Agenda", accent=GREEN)
add_bullets(s, [
    {"lead": "Why extremes? ", "text": "the problem standard scores can't see", "accent": ORANGE},
    {"lead": "The big idea — ", "text": "two-model comparison + threshold-weighted metrics", "accent": ORANGE},
    {"lead": "The pipeline — ", "text": "nine steps, one YAML config file", "accent": BLUE},
    {"lead": "Configuring a run — ", "text": "variables, data retrieval, thresholds, filters", "accent": BLUE},
    {"lead": "The scores — ", "text": "ETS, PSS, twMAE/twRMSE, twCRPS, BSS", "accent": BLUE},
    {"lead": "Running it — ", "text": "deterministic or ensemble — interactively or on the HPC (SLURM)", "accent": GREEN},
    {"lead": "Reading the outputs — ", "text": "the scorecard heatmap", "accent": GREEN},
    {"lead": "The diagnostic toolbox — ", "text": "Q-Q plots, 22-plot diagnostics, case studies", "accent": ORANGE},
    {"lead": "Hands-on — ", "text": "your first run + tips and common pitfalls", "accent": SKY},
], size=18, gap=9)
_footer(s)

# 3. Section 1 divider
divider_slide(1, "Why verify the extremes?",
              "Heat waves · cold spells · heavy rain · damaging winds")

# 4. The problem
s = new_slide()
content_header(s, "Averages hide the events that matter", kicker="Motivation", accent=ORANGE)
add_bullets(s, [
    {"lead": "The problem: ", "text": "MAE, RMSE and CRPS are dominated by the huge population of ordinary, non-extreme days.", "accent": ORANGE},
    {"lead": "", "text": "A model can look excellent on average yet badly miss the heat wave, the storm or the flood.", "level": 1, "accent": ORANGE},
    {"lead": "Who needs the tails? ", "text": "Warnings, energy, aviation, hydrology, insurance — all live in the extremes.", "accent": BLUE},
    {"lead": "New models, new questions: ", "text": "AI forecasts (AIFS) are sharp and skilful on average — but do they capture extremes as well as the physics-based IFS?", "accent": GREEN},
    {"lead": "Our answer: ", "text": "score only where it counts — the extreme tail — and always compare two models head-to-head.", "accent": NAVY, "lead_color": GREEN},
], size=19, gap=14)
# highlight strip
add_rect(s, Inches(0.7), Inches(6.35), Inches(11.9), Inches(0.7), LIGHT, MSO_SHAPE.ROUNDED_RECTANGLE)
add_text(s, Inches(0.95), Inches(6.46), Inches(11.5), Inches(0.5),
         "Key idea:  concentrate the whole evaluation on cases that exceed a climatological extreme threshold.",
         size=15, color=NAVY, bold=True)
_footer(s)

# 5. The big idea - two model comparison
s = new_slide()
content_header(s, "The big idea: relative skill, model 1 vs model 2", kicker="Core concept", accent=BLUE)
# formula card
add_rect(s, Inches(0.7), Inches(2.0), Inches(6.0), Inches(2.0), LIGHT, MSO_SHAPE.ROUNDED_RECTANGLE)
add_text(s, Inches(0.95), Inches(2.2), Inches(5.6), Inches(0.5),
         "Every cell shows relative % improvement", size=15, color=NAVY, bold=True)
add_text(s, Inches(0.95), Inches(2.75), Inches(5.6), Inches(0.9),
         "Skill  =  (S₂ − S₁) / S₂  × 100 %", size=24, color=BLUE, bold=True)
add_text(s, Inches(0.95), Inches(3.55), Inches(5.6), Inches(0.4),
         "S₁, S₂ = score of model 1, model 2", size=13, color=MIDGREY, italic=True)
# green/red meaning
add_chip(s, Inches(7.1), Inches(2.1), Inches(5.3), Inches(0.7),
         "GREEN  →  model 1 is better", GREEN)
add_chip(s, Inches(7.1), Inches(2.95), Inches(5.3), Inches(0.7),
         "RED  →  model 2 is better", ORANGE)
add_chip(s, Inches(7.1), Inches(3.8), Inches(5.3), Inches(0.7),
         "DOTTED cell  →  significant at 95 %", SKY, fg=NAVY)
add_bullets(s, [
    {"lead": "Stratified everywhere: ", "text": "results broken down by lead time × season (DJF/MAM/JJA/SON) × terrain (flat / hilly / complex).", "accent": BLUE},
    {"lead": "Bootstrap confidence: ", "text": "1000 resamples give 95% intervals, so you know which differences are real.", "accent": GREEN},
], y=Inches(4.9), size=17, gap=10)
_footer(s)

# 6. Section 2 divider
divider_slide(2, "Configuring a run", "One YAML file describes the whole experiment")

# 7. Retrieval methods
s = new_slide()
content_header(s, "Getting the data in: retrieval methods", kicker="Step 1 · read_data", accent=BLUE)
add_text(s, Inches(0.7), Inches(1.9), Inches(6.2), Inches(0.35),
         "Forecasts  —  forecast_model{1,2}:  source:", size=14.5, color=BLUE, bold=True)
fc_methods = [
    ("local_grib", "GRIB already on disk (2t_YYYYMMDD.grib, 10u/10v, tp…)", GREEN),
    ("mars", "Fetch from MARS; folder derived from class/stream/expver", BLUE),
    ("quaver", "In-memory ECMWF backend — no GRIB staged to disk", SKY),
]
ry = Inches(2.32)
for tag, desc, col in fc_methods:
    add_rect(s, Inches(0.7), ry, Inches(5.9), Inches(0.55), LIGHT, MSO_SHAPE.ROUNDED_RECTANGLE)
    add_chip(s, Inches(0.82), ry + Inches(0.07), Inches(1.35), Inches(0.41), tag, col, size=11.5)
    add_text(s, Inches(2.35), ry + Inches(0.04), Inches(4.15), Inches(0.47), desc,
             size=11.5, color=GREY, anchor=MSO_ANCHOR.MIDDLE)
    ry += Inches(0.63)
add_text(s, Inches(0.7), Inches(4.3), Inches(6.2), Inches(0.35),
         "Observations  —  observation_source:", size=14.5, color=GREEN, bold=True)
obs_methods = [
    ("local_gpt", "Geopoints .gpt/.geo: lat, lon, height, date, value", GREEN),
    ("stvl", "Pull SYNOP per valid time → *_obs_*.geo", BLUE),
    ("cams_netcdf", "CAMS species: fc + obs straight from NetCDF", ORANGE),
]
ry = Inches(4.72)
for tag, desc, col in obs_methods:
    add_rect(s, Inches(0.7), ry, Inches(5.9), Inches(0.55), LIGHT, MSO_SHAPE.ROUNDED_RECTANGLE)
    add_chip(s, Inches(0.82), ry + Inches(0.07), Inches(1.35), Inches(0.41), tag, col, size=11.5)
    add_text(s, Inches(2.35), ry + Inches(0.04), Inches(4.15), Inches(0.47), desc,
             size=11.5, color=GREY, anchor=MSO_ANCHOR.MIDDLE)
    ry += Inches(0.63)
add_text(s, Inches(7.0), Inches(1.9), Inches(5.6), Inches(0.35),
         "Switch to direct retrieval (ECMWF)", size=14.5, color=NAVY, bold=True)
code_block(s, Inches(7.0), Inches(2.32), Inches(5.65), Inches(3.5), [
    ("forecast_model1:", SKY),
    ('  source: "mars"', CODEFG),
    ("  mars:", SKY),
    ("    class: rd      # od | ai | rd", CODEFG),
    ("    stream: oper   # enfo if ensemble", CODEFG),
    ('    expver: "j5vo"', CODEFG),
    ('    grid: "0.25/0.25"', CODEFG),
    ("    base_path: /ec/vol/.../raw", CODEFG),
    ("", CODEFG),
    ('observation_source: "stvl"', SKY),
    ("stvl:", SKY),
    ('  sources: ["synop"]', CODEFG),
    ("  base_path: /ec/vol/.../obs", CODEFG),
], size=11.5)
add_rect(s, Inches(7.0), Inches(6.0), Inches(5.65), Inches(0.85), LIGHT, MSO_SHAPE.ROUNDED_RECTANGLE)
add_text(s, Inches(7.2), Inches(6.11), Inches(5.3), Inches(0.65),
         "ECMWF-only: mars · quaver · stvl.  External users: local_grib + local_gpt (or dataset_climatology).",
         size=11.5, color=NAVY, bold=True)
_footer(s)

# 7b. Choosing your variable
s = new_slide()
content_header(s, "Choosing your variable", kicker="One tool, many parameters", accent=ORANGE)
var_cards = [
    ("2t", "2 m temperature  (°C)",
     "Warm p99 AND cold p1 tails · lapse-rate correction to station height", ORANGE),
    ("10ff", "10 m wind speed  (m/s)",
     "Computed from u/v components · strong-wind p99", BLUE),
    ("tp24", "24 h precipitation  (mm)",
     "Accumulated to 24 h · heavy-rain p99 or a fixed mm warning level", GREEN),
    ("CAMS", "aod500 · pm2p5/pm10 · go3, no2, so2, co, no",
     "Air-quality species read from per-site NetCDF · deterministic mode only", SKY),
]
cy = Inches(2.05)
for tag, name, note, col in var_cards:
    card = add_rect(s, Inches(0.7), cy, Inches(11.9), Inches(1.05), LIGHT, MSO_SHAPE.ROUNDED_RECTANGLE)
    card.adjustments[0] = 0.10
    add_rect(s, Inches(0.7), cy, Inches(0.14), Inches(1.05), col)
    add_chip(s, Inches(1.0), cy + Inches(0.24), Inches(1.55), Inches(0.57), tag, col, size=15)
    add_text(s, Inches(2.85), cy + Inches(0.14), Inches(9.4), Inches(0.45), name, size=16, color=NAVY, bold=True)
    add_text(s, Inches(2.85), cy + Inches(0.58), Inches(9.4), Inches(0.42), note, size=12.5, color=GREY)
    cy += Inches(1.18)
_footer(s)

# 8. Config walkthrough (code)
s = new_slide()
content_header(s, "The config file: edit five fields, run", kicker="config.yaml", accent=GREEN)
code_block(s, Inches(0.7), Inches(2.0), Inches(7.5), Inches(4.75), [
    ('variable: "tp24"          # 2t | 10ff | tp24 | CAMS', SKY),
    ('start_date: "2024-01-01"   # \u2190 CHANGE', CODEFG),
    ('end_date:   "2024-03-31"   # \u2190 CHANGE', CODEFG),
    ('mode: "deterministic"      # or ensemble', CODEFG),
    ('', CODEFG),
    ('read_data:', SKY),
    ('  forecast_model1:', CODEFG),
    ('    name: "IFS"            # \u2190 CHANGE', CODEFG),
    ('    source: "local_grib"', CODEFG),
    ('    local_grib: {path: /data/ifs}   # \u2190 CHANGE', CODEFG),
    ('  forecast_model2:', CODEFG),
    ('    name: "AIFS"           # \u2190 CHANGE', CODEFG),
    ('    local_grib: {path: /data/aifs}  # \u2190 CHANGE', CODEFG),
    ('  observation_source: "local_gpt"', CODEFG),
    ('  local_gpt: {path: /data/obs}', CODEFG),
], size=13)
add_bullets(s, [
    {"lead": "One file = one experiment. ", "text": "It is your reproducible record.", "accent": GREEN},
    {"lead": "Start from a template: ", "text": "cp config_example.yaml my_run.yaml", "accent": BLUE},
    {"lead": "Skip flags ", "text": "let you re-run only the plots without re-extracting or re-scoring.", "accent": ORANGE},
    {"lead": "Ready-made configs ", "text": "live in configs/deterministic, /ensemble, /cams.", "accent": SKY},
], x=Inches(8.4), y=Inches(2.05), w=Inches(4.3), h=Inches(4.7), size=15, gap=13)
_footer(s)

# 8b. Deterministic vs ensemble methodology
s = new_slide()
content_header(s, "Two methodologies, one tool: deterministic & ensemble", kicker="mode:", accent=BLUE)
# left panel — deterministic
lp = add_rect(s, Inches(0.6), Inches(1.95), Inches(6.0), Inches(4.9), LIGHT, MSO_SHAPE.ROUNDED_RECTANGLE)
lp.adjustments[0] = 0.04
add_text(s, Inches(0.6), Inches(2.12), Inches(6.0), Inches(0.5),
         "DETERMINISTIC", size=19, color=BLUE, bold=True, align=PP_ALIGN.CENTER)
add_rect(s, Inches(2.6), Inches(2.68), Inches(2.0), Pt(3), BLUE)
code_block(s, Inches(0.9), Inches(2.95), Inches(5.4), Inches(1.35), [
    ('mode: "deterministic"', SKY),
    ("forecast_days: [1,2,3,4,5]", CODEFG),
    ("lead_time_frequency: 24", CODEFG),
], size=12)
add_bullets(s, [
    {"lead": "", "text": "Single-valued forecast vs obs", "accent": BLUE},
    {"lead": "Scores: ", "text": "ETS · PSS · POD · FAR · twMAE · twRMSE", "accent": BLUE},
    {"lead": "Extractor: ", "text": "extract_points.py", "accent": BLUE},
    {"lead": "", "text": "Lightest to run — ideal first pass", "accent": GREEN},
], x=Inches(0.95), y=Inches(4.5), w=Inches(5.35), h=Inches(2.3), size=13.5, gap=9)
# right panel — ensemble
rp = add_rect(s, Inches(6.85), Inches(1.95), Inches(6.0), Inches(4.9), LIGHT, MSO_SHAPE.ROUNDED_RECTANGLE)
rp.adjustments[0] = 0.04
add_text(s, Inches(6.85), Inches(2.12), Inches(6.0), Inches(0.5),
         "ENSEMBLE", size=19, color=ORANGE, bold=True, align=PP_ALIGN.CENTER)
add_rect(s, Inches(8.85), Inches(2.68), Inches(2.0), Pt(3), ORANGE)
code_block(s, Inches(7.15), Inches(2.95), Inches(5.4), Inches(1.35), [
    ('mode: "ensemble"', SKY),
    ("steps: [24, 48, ... , 240]", CODEFG),
    ("ensemble: {n_members: 50,", CODEFG),
    ("          include_control: true}", CODEFG),
], size=12)
add_bullets(s, [
    {"lead": "", "text": "Full probabilistic forecast vs obs", "accent": ORANGE},
    {"lead": "Scores: ", "text": "twCRPS · Brier · BSS · twQS (tails)", "accent": ORANGE},
    {"lead": "Extractor: ", "text": "extract_points_ensemble.py  (\u2265 64 GB)", "accent": ORANGE},
    {"lead": "50 vs 51 members OK · ", "text": "control_source: own|omit|external", "accent": GREEN},
], x=Inches(7.2), y=Inches(4.5), w=Inches(5.35), h=Inches(2.3), size=13.5, gap=9)
_footer(s)

# 9. Thresholds
s = new_slide()
content_header(s, "Defining the extreme event: thresholds", kicker="Step 5", accent=ORANGE)
add_bullets(s, [
    {"lead": "local_obs_climatology  ", "text": "(recommended) — a per-station percentile from 15–20 yrs of independent obs; removes local climate differences.", "accent": GREEN, "lead_color": GREEN},
    {"lead": "fixed  ", "text": "— a physical warning level, e.g. 30 mm / 24 h or 35 °C.", "accent": BLUE},
    {"lead": "dataset_climatology  ", "text": "— Nth percentile of the obs already in your run; great for a first exploratory test (no extra files).", "accent": BLUE},
    {"lead": "station_climatology  ", "text": "— per-station percentile from STVL archive.", "accent": SKY},
    {"lead": "event_type: ", "text": "above (warm / wet / windy)  or  below (cold).", "accent": ORANGE},
], size=17, gap=13, h=Inches(3.4))
add_rect(s, Inches(0.7), Inches(5.7), Inches(11.9), Inches(1.15), LIGHT, MSO_SHAPE.ROUNDED_RECTANGLE)
add_text(s, Inches(0.95), Inches(5.85), Inches(11.4), Inches(0.5),
         "Why per-station?", size=15, color=NAVY, bold=True)
add_text(s, Inches(0.95), Inches(6.25), Inches(11.4), Inches(0.5),
         "The 99th percentile of temperature in Seville is very different from Reykjavík. A single fixed value would compare apples to oranges.",
         size=13.5, color=GREY)
_footer(s)

# 10. Filters & stratification
s = new_slide()
content_header(s, "Slice the sample: filters & stratification", kicker="Step 4", accent=BLUE)
add_bullets(s, [
    {"lead": "Season — ", "text": "DJF · MAM · JJA · SON (or custom windows like ASO).", "accent": BLUE},
    {"lead": "Terrain from sdfor — ", "text": "flat (<40 m) · hilly (40–120 m) · complex (≥120 m) sub-grid orography.", "accent": GREEN},
    {"lead": "Coastal removal — ", "text": "drop stations where land-sea mask < 0.9 (mixed land/sea grid boxes).", "accent": ORANGE},
    {"lead": "Quality control — ", "text": "physical bounds on 2t and a precipitation ceiling reject bad reports.", "accent": SKY},
    {"lead": "Lead time — ", "text": "verify all forecast days, or select e.g. [24, 48, 72] h.", "accent": BLUE},
], size=18, gap=15)
add_rect(s, Inches(0.7), Inches(6.2), Inches(11.9), Inches(0.75), LIGHT, MSO_SHAPE.ROUNDED_RECTANGLE)
add_text(s, Inches(0.95), Inches(6.33), Inches(11.4), Inches(0.5),
         "Stratifying by terrain often reveals the biggest model differences — extremes in complex terrain are the hardest to forecast.",
         size=14, color=NAVY, bold=True)
_footer(s)

# 11. Section 3 divider
divider_slide(3, "The scores", "Categorical skill + threshold-weighted errors")

# 12. Threshold-weighted idea
s = new_slide()
content_header(s, "Threshold-weighted scores: the key innovation", kicker="Tail-focused metrics", accent=ORANGE)
add_rect(s, Inches(0.7), Inches(2.0), Inches(11.9), Inches(1.5), LIGHT, MSO_SHAPE.ROUNDED_RECTANGLE)
add_text(s, Inches(0.95), Inches(2.15), Inches(11.4), Inches(0.5),
         "twMAE  =  (1/N) Σ  w(y, ŷ) · | y − ŷ |", size=24, color=BLUE, bold=True)
add_text(s, Inches(0.95), Inches(2.85), Inches(11.4), Inches(0.55),
         "weight w = 1 when the obs OR the forecast reaches the extreme threshold T, else 0",
         size=15, color=GREY, italic=True)
add_bullets(s, [
    {"lead": "The error is only counted for extreme cases ", "text": "— ordinary days are switched off.", "accent": ORANGE},
    {"lead": "Family (Taggart, 2022): ", "text": "twMAE, twRMSE, twCRPS (ensemble), twQS (quantile score).", "accent": BLUE},
    {"lead": "Decomposable: ", "text": "the diagnostics split twMAE into hit / miss / false-alarm contributions.", "accent": GREEN},
    {"lead": "Complemented by categorical scores ", "text": "on the same threshold (next slide).", "accent": SKY},
], y=Inches(3.75), size=17, gap=13)
_footer(s)

# 13. Deterministic scores table
s = new_slide()
content_header(s, "Deterministic scores at a glance", kicker="Step 6 · deterministic", accent=BLUE)
rows = [
    ("POD", "Probability of detection — fraction of events caught", "1", GREEN),
    ("FAR", "False-alarm ratio — fraction of warnings that were wrong", "0", ORANGE),
    ("ETS", "Equitable Threat Score — hits corrected for chance", "1", BLUE),
    ("PSS", "Peirce Skill Score — detection minus false-detection", "1", BLUE),
    ("twMAE", "Threshold-weighted mean absolute error", "0", ORANGE),
    ("twRMSE", "Threshold-weighted root-mean-square error", "0", ORANGE),
    ("bias / MAE / RMSE", "Classic errors (all cases) for context", "0", MIDGREY),
]
# header row
hx, hy = Inches(0.7), Inches(2.05)
add_rect(s, hx, hy, Inches(2.6), Inches(0.5), NAVY)
add_rect(s, hx + Inches(2.6), hy, Inches(7.6), Inches(0.5), NAVY)
add_rect(s, hx + Inches(10.2), hy, Inches(2.4), Inches(0.5), NAVY)
add_text(s, hx + Inches(0.15), hy + Inches(0.06), Inches(2.4), Inches(0.4), "Score", size=13, color=WHITE, bold=True)
add_text(s, hx + Inches(2.75), hy + Inches(0.06), Inches(7.3), Inches(0.4), "What it tells you", size=13, color=WHITE, bold=True)
add_text(s, hx + Inches(10.35), hy + Inches(0.06), Inches(2.1), Inches(0.4), "Perfect", size=13, color=WHITE, bold=True)
ry = hy + Inches(0.5)
for i, (name, desc, best, col) in enumerate(rows):
    bg = WHITE if i % 2 == 0 else LIGHT
    add_rect(s, hx, ry, Inches(12.6 - 2.4 + 2.4), Inches(0.6), bg)  # full width
    add_rect(s, hx, ry, Inches(0.10), Inches(0.6), col)
    add_text(s, hx + Inches(0.25), ry + Inches(0.10), Inches(2.4), Inches(0.45), name, size=14, color=col, bold=True)
    add_text(s, hx + Inches(2.75), ry + Inches(0.10), Inches(7.3), Inches(0.45), desc, size=13, color=GREY)
    add_text(s, hx + Inches(10.35), ry + Inches(0.10), Inches(2.1), Inches(0.45), best, size=14, color=NAVY, bold=True)
    ry += Inches(0.6)
_footer(s)

# 14. Ensemble scores
s = new_slide()
content_header(s, "Ensemble mode: probabilistic scores", kicker="Step 6 · ensemble", accent=GREEN)
add_bullets(s, [
    {"lead": "twCRPS — ", "text": "threshold-weighted CRPS: the headline probabilistic tail score.", "accent": GREEN},
    {"lead": "Brier & BSS — ", "text": "accuracy and skill of the event probability vs climatology.", "accent": BLUE},
    {"lead": "twQS — ", "text": "threshold-weighted quantile score, incl. extreme q0.01 / q0.05 tails.", "accent": ORANGE},
    {"lead": "extreme_spread_skill_ratio — ", "text": "is the ensemble spread reliable at the tail?", "accent": SKY},
    {"lead": "Reliability & rank histograms ", "text": "diagnose over/under-dispersion (see diagnostics).", "accent": BLUE},
], size=18, gap=13, h=Inches(3.1))
add_rect(s, Inches(0.7), Inches(5.55), Inches(11.9), Inches(1.3), LIGHT, MSO_SHAPE.ROUNDED_RECTANGLE)
add_text(s, Inches(0.95), Inches(5.7), Inches(11.4), Inches(0.45),
         "Switch modes in one line:", size=15, color=NAVY, bold=True)
add_text(s, Inches(0.95), Inches(6.15), Inches(11.4), Inches(0.55),
         'mode: "ensemble"     steps: [24, 48, 72, … 240]     scores.ensemble: [twCRPS, BSS, tw_quantile_score, …]',
         size=13.5, color=BLUE, bold=True, font=MONO)
_footer(s)

# 15. Section 4 divider
divider_slide(4, "Running the tool", "Interactively, then at scale on the HPC")

# 16. Running
s = new_slide()
content_header(s, "From a quick test to the full period", kicker="Execution", accent=BLUE)
add_text(s, Inches(0.7), Inches(1.95), Inches(6), Inches(0.4), "1 · Interactive sanity check", size=16, color=GREEN, bold=True)
code_block(s, Inches(0.7), Inches(2.4), Inches(5.9), Inches(1.55), [
    ("source .venv/bin/activate", CODEFG),
    ("python run.py my_run.yaml", SKY),
    ("# runs all 9 steps in the terminal", MIDGREY),
], size=13)
add_text(s, Inches(0.7), Inches(4.15), Inches(6), Inches(0.4), "2 · Full run on SLURM", size=16, color=ORANGE, bold=True)
code_block(s, Inches(0.7), Inches(4.6), Inches(5.9), Inches(1.85), [
    ("sbatch scripts/submit_job.sh my_run.yaml", SKY),
    ("squeue -u $USER", CODEFG),
    ("tail -f scorecards_<JOBID>.out", CODEFG),
], size=13)
add_bullets(s, [
    {"lead": "Small ranges ", "text": "run fine interactively.", "accent": BLUE},
    {"lead": "Large 2t / 10ff sets & ensembles ", "text": "need ≥ 64 GB — always submit to the HPC.", "accent": ORANGE},
    {"lead": "Green cells = model 1 better; ", "text": "dotted = significant.", "accent": GREEN},
    {"lead": "Re-run plots only ", "text": "by setting both skip_*_if_exists flags to true.", "accent": SKY},
    {"lead": "Everything is cached ", "text": "as parquet in extracted_points/.", "accent": BLUE},
], x=Inches(7.0), y=Inches(2.05), w=Inches(5.7), h=Inches(4.6), size=15, gap=14)
_footer(s)

# 17. Reading the scorecard (image)
s = new_slide()
content_header(s, "Reading a scorecard heatmap", kicker="The flagship output", accent=GREEN)
add_image_fit(s, IMG_SCORECARD, Inches(0.55), Inches(1.95), Inches(6.6), Inches(4.9))
add_bullets(s, [
    {"lead": "Rows = lead time ", "text": "(day 1 → day 10).", "accent": BLUE},
    {"lead": "Columns = terrain ", "text": "(LOW / MID / HIGH).", "accent": GREEN},
    {"lead": "Each panel = one score ", "text": "(Brier, twCRPS, QS, spread ratio…).", "accent": ORANGE},
    {"lead": "Colour = % skill ", "text": "of model 1 vs model 2.", "accent": SKY},
    {"lead": "Small number ", "text": "= the raw score value.", "accent": MIDGREY},
    {"lead": "One figure per season ", "text": "with terrain as column groups.", "accent": BLUE},
], x=Inches(7.5), y=Inches(2.1), w=Inches(5.3), h=Inches(4.7), size=15, gap=13)
_footer(s)

# 18. Section 5 divider
divider_slide(5, "The diagnostic toolbox",
              "Go beyond the scorecard — understand WHY")

# 19. Diagnostics overview
s = new_slide()
content_header(s, "A toolbox to explain the numbers", kicker="diagnostics/", accent=ORANGE)
tools = [
    ("plot_qq.py / plot_qq_extremes.py", "Q-Q plots — where in the distribution does each model drift?", BLUE),
    ("diagnose_extremes.py", "22-plot deep dive for one (day, season, terrain, threshold)", ORANGE),
    ("diagnose_det/twcrps_simple.py", "Fast 3-panel visual check (det. & ensemble)", GREEN),
    ("plot_station_diagnostics.py", "Per-station maps of bias / error / event class", SKY),
    ("case_studies/", "Rank & visualise the dates with the biggest model gap", ORANGE),
    ("analysis/", "Publication panels (e.g. the IFS-vs-AIFS Q-Q figure)", BLUE),
]
ty = Inches(2.05)
for name, desc, col in tools:
    add_rect(s, Inches(0.7), ty, Inches(11.9), Inches(0.72), LIGHT, MSO_SHAPE.ROUNDED_RECTANGLE)
    add_rect(s, Inches(0.7), ty, Inches(0.13), Inches(0.72), col)
    add_text(s, Inches(1.0), ty + Inches(0.10), Inches(5.0), Inches(0.55), name, size=14.5, color=NAVY, bold=True, font=MONO, anchor=MSO_ANCHOR.MIDDLE)
    add_text(s, Inches(6.1), ty + Inches(0.10), Inches(6.3), Inches(0.55), desc, size=13.5, color=GREY, anchor=MSO_ANCHOR.MIDDLE)
    ty += Inches(0.8)
_footer(s)

# 20. Q-Q plot (image)
s = new_slide()
content_header(s, "Q-Q plots: is the tail under-forecast?", kicker="IFS vs AIFS · day 3", accent=BLUE)
add_image_fit(s, IMG_QQ, Inches(0.55), Inches(1.95), Inches(7.4), Inches(4.9))
add_bullets(s, [
    {"lead": "Axes: ", "text": "observed vs forecast quantiles.", "accent": BLUE},
    {"lead": "On the 1:1 line ", "text": "= perfect distribution match.", "accent": GREEN},
    {"lead": "Below the line ", "text": "= the model under-forecasts the extreme.", "accent": ORANGE},
    {"lead": "○ every 1 %, ✕ ", "text": "on the 99.x % tail (and cold tail for 2t).", "accent": SKY},
    {"lead": "Split by terrain ", "text": "(flat vs complex) reveals where it breaks.", "accent": BLUE},
    {"lead": "Read here: ", "text": "both under-shoot extremes; AIFS more so for wind & rain.", "accent": NAVY, "lead_color": ORANGE},
], x=Inches(8.2), y=Inches(2.1), w=Inches(4.6), h=Inches(4.7), size=14.5, gap=11)
_footer(s)

# 21. Case studies (1) — find
s = new_slide()
content_header(s, "Case studies 1 — find the dates that matter", kicker="case_studies/find_case_studies.py", accent=GREEN)
add_bullets(s, [
    {"lead": "What it does: ", "text": "loops every date × forecast-day, classifies each station hit / miss / false-alarm for both models, and ranks dates by a composite performance gap.", "accent": GREEN},
    {"lead": "Consistent with the scorecard: ", "text": "uses the SAME threshold method (incl. local_obs_climatology) as the main run.", "accent": BLUE},
], y=Inches(1.95), h=Inches(1.45), size=15, gap=10)
code_block(s, Inches(0.7), Inches(3.45), Inches(11.9), Inches(1.15), [
    ("python case_studies/find_case_studies.py --config <cfg> \\", SKY),
    ("       --days 3 5 7 --season JJA --orog flat --top-n 20", CODEFG),
], size=12.5)
add_text(s, Inches(0.7), Inches(4.8), Inches(11.8), Inches(0.35),
         "Outputs", size=14.5, color=ORANGE, bold=True)
add_bullets(s, [
    {"lead": "case_study_ranking_*.csv — ", "text": "one row per date: FA/miss counts, POD, FAR, ETS, twMAE, composite rank.", "accent": ORANGE},
    {"lead": "case_study_summary_*.txt — ", "text": "readable top-N worst cases per day, tagged by case type (e.g. M1_FALSE_ALARM).", "accent": BLUE},
    {"lead": "Refine with: ", "text": "--min-stations · --min-concentration · --season · --orog.", "accent": SKY},
], y=Inches(5.2), h=Inches(1.7), size=14, gap=8)
_footer(s)

# 22. Case studies (2) — visualise
s = new_slide()
content_header(s, "Case studies 2 — visualise the event", kicker="case_studies/plot_case_study.py", accent=ORANGE)
code_block(s, Inches(0.7), Inches(2.0), Inches(11.9), Inches(1.1), [
    ("python case_studies/plot_case_study.py --config <cfg> \\", SKY),
    ("       --date 20250718 --day 5", CODEFG),
], size=12.5)
add_text(s, Inches(0.7), Inches(3.3), Inches(11.8), Inches(0.35),
         "Three figures for the chosen date", size=14.5, color=BLUE, bold=True)
figs = [
    ("Fig 1", "4-panel hit / miss / false-alarm station maps for both models + score table", GREEN),
    ("Fig 2", "Observed vs model-1 vs model-2 value maps on a shared colour scale", BLUE),
    ("Fig 3", "Raw 0.25° GRIB field with observation dots overlaid (one panel per model)", ORANGE),
]
fy = Inches(3.75)
for tag, desc, col in figs:
    add_rect(s, Inches(0.7), fy, Inches(11.9), Inches(0.7), LIGHT, MSO_SHAPE.ROUNDED_RECTANGLE)
    add_rect(s, Inches(0.7), fy, Inches(0.13), Inches(0.7), col)
    add_chip(s, Inches(0.95), fy + Inches(0.12), Inches(1.15), Inches(0.46), tag, col, size=12.5)
    add_text(s, Inches(2.35), fy + Inches(0.08), Inches(10.0), Inches(0.55), desc,
             size=13.5, color=GREY, anchor=MSO_ANCHOR.MIDDLE)
    fy += Inches(0.8)
add_rect(s, Inches(0.7), Inches(6.25), Inches(11.9), Inches(0.62), LIGHT, MSO_SHAPE.ROUNDED_RECTANGLE)
add_text(s, Inches(0.95), Inches(6.34), Inches(11.4), Inches(0.45),
         "Turns an abstract score gap into a real weather map for stakeholders. (Fig 3 needs cartopy + raw GRIB.)",
         size=13, color=NAVY, bold=True)
_footer(s)

# 23. Hands-on exercise
s = new_slide()
content_header(s, "Your turn: a 6-step first run", kicker="Hands-on", accent=BLUE)
add_bullets(s, [
    {"lead": "1.  Copy a template — ", "text": "cp config_example.yaml my_run.yaml", "accent": BLUE},
    {"lead": "2.  Edit the 5 ← CHANGE fields ", "text": "(dates, model paths, obs path).", "accent": BLUE},
    {"lead": "3.  Keep it short first: ", "text": "one month, dataset_climatology threshold.", "accent": GREEN},
    {"lead": "4.  Run interactively: ", "text": "python run.py my_run.yaml", "accent": GREEN},
    {"lead": "5.  Open the heatmap ", "text": "in results/ — who's greener, and where?", "accent": ORANGE},
    {"lead": "6.  Dig deeper: ", "text": "plot_qq_extremes.py --day 3 --season DJF --orog complex", "accent": ORANGE},
], size=18, gap=14, h=Inches(3.8))
add_rect(s, Inches(0.7), Inches(6.1), Inches(11.9), Inches(0.8), LIGHT, MSO_SHAPE.ROUNDED_RECTANGLE)
add_text(s, Inches(0.95), Inches(6.24), Inches(11.4), Inches(0.55),
         "Challenge:  flip event_type to \"below\" and compare the models on cold extremes instead of warm ones.",
         size=15, color=NAVY, bold=True)
_footer(s)

# 24. Tips & pitfalls
s = new_slide()
content_header(s, "Tips & common pitfalls", kicker="Survival guide", accent=ORANGE)
add_bullets(s, [
    {"lead": "Memory: ", "text": "big 2t/10ff & ensemble runs get OOM-killed interactively — submit with 64 GB.", "accent": ORANGE},
    {"lead": "All-grey heatmap? ", "text": "too few events — lower the percentile or use dataset_climatology.", "accent": BLUE},
    {"lead": "\u2018No stations found\u2019? ", "text": "your season/terrain filter is too strict — start with null.", "accent": BLUE},
    {"lead": "Never launch two configs ", "text": "that write the same shared forecast file at once — serialise with --dependency.", "accent": ORANGE, "lead_color": ORANGE},
    {"lead": "Re-use cached data: ", "text": "skip_extraction_if_exists / skip_scoring_if_exists save hours.", "accent": GREEN},
    {"lead": "Set TMPDIR ", "text": "to scratch before running on the HPC.", "accent": SKY},
], size=17, gap=13)
_footer(s)

# 25. Summary / closing
s = new_slide()
_set_bg(s, NAVY)
add_rect(s, 0, Inches(1.35), Inches(0.9), Inches(1.0), BLUE)
add_text(s, Inches(1.15), Inches(1.35), Inches(11), Inches(1.0),
         "Key takeaways", size=36, color=WHITE, bold=True, anchor=MSO_ANCHOR.MIDDLE)
add_rect(s, Inches(1.18), Inches(2.35), Inches(4.2), Pt(2), GREEN)
tb = s.shapes.add_textbox(Inches(1.15), Inches(2.7), Inches(11.2), Inches(3.4))
tf = tb.text_frame; tf.word_wrap = True
pts = [
    ("Score where it matters — ", "threshold-weighted metrics on the extreme tail.", GREEN),
    ("Always compare two models — ", "relative % skill, stratified by lead / season / terrain.", SKY),
    ("One YAML, nine steps — ", "reproducible from raw GRIB to publication figure.", BLUE),
    ("Trust it — ", "bootstrap 95% significance on every difference.", ORANGE),
    ("Explain it — ", "Q-Q, 22-plot diagnostics and case studies reveal the WHY.", YELLOW),
]
for i, (lead, rest, col) in enumerate(pts):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.space_after = Pt(12)
    rm = p.add_run(); rm.text = "●  "; rm.font.color.rgb = col; rm.font.size = Pt(18); rm.font.bold = True; rm.font.name = FONT
    rl = p.add_run(); rl.text = lead; rl.font.color.rgb = WHITE; rl.font.bold = True; rl.font.size = Pt(18); rl.font.name = FONT
    rt = p.add_run(); rt.text = rest; rt.font.color.rgb = RGBColor(0xC7, 0xD2, 0xDC); rt.font.size = Pt(18); rt.font.name = FONT
add_rect(s, 0, Inches(6.35), SW, Inches(0.06), BLUE)
add_text(s, Inches(1.15), Inches(6.55), Inches(11), Inches(0.5),
         "Docs:  README.md · docs/QUICKSTART.md · docs/USER_GUIDE.md · docs/SCIENCE.md · docs/CASE_STUDIES.md",
         size=13, color=SKY, bold=True)

# ---------------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
prs.save(OUT)
print(f"Saved {len(prs.slides._sldIdLst)} slides to {OUT}")
