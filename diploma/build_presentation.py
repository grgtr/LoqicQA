"""
Generates diploma/vkr_presentation.pptx
4-5 minute talk: improvements to LogicQA pipeline + results.
Uses evaluation_framework_logicqa.pptx as style template.
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from lxml import etree
import copy

TEMPLATE = "/home/chikibriki/LoqicQA/diploma/evaluation_framework_logicqa.pptx"
OUTPUT   = "/home/chikibriki/LoqicQA/diploma/vkr_presentation.pptx"

# ---------------------------------------------------------------------------
# Colors (matching original Google Slides export)
# ---------------------------------------------------------------------------
C_BG_DARK   = RGBColor(0x1C, 0x1C, 0x2E)   # dark navy background
C_BG_LIGHT  = RGBColor(0xF5, 0xF5, 0xF7)   # light section background
C_ACCENT    = RGBColor(0x4A, 0x86, 0xC8)   # MIPT blue accent
C_ACCENT2   = RGBColor(0xE8, 0xA0, 0x20)   # amber accent
C_WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
C_TEXT_DARK = RGBColor(0x1A, 0x1A, 0x2E)
C_GREEN     = RGBColor(0x27, 0xAE, 0x60)
C_RED       = RGBColor(0xC0, 0x39, 0x2B)
C_GRAY      = RGBColor(0x95, 0xA5, 0xA6)

W = Inches(10.0)
H = Inches(5.625)

# ---------------------------------------------------------------------------
# Fresh presentation — no template (we use explicit RGB colors everywhere)
# ---------------------------------------------------------------------------
prs = Presentation()
prs.slide_width  = W
prs.slide_height = H

# Use blank layout from default master
layout_map = {l.name: l for l in prs.slide_masters[0].slide_layouts}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_bg(slide, color: RGBColor):
    """Fill slide background with solid color."""
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_textbox(slide, text, left, top, width, height,
                font_name="Times New Roman", font_size=18,
                bold=False, color=C_WHITE, align=PP_ALIGN.LEFT,
                wrap=True, italic=False):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.name = font_name
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return txBox


def add_bullet_textbox(slide, title, bullets, left, top, width, height,
                       title_size=22, bullet_size=17, title_color=C_ACCENT,
                       bullet_color=C_WHITE, bg_color=None, indent_level=0):
    """Textbox with bold title line + bullet paragraphs."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True

    # Optionally fill background
    if bg_color:
        fill = txBox.fill
        fill.solid()
        fill.fore_color.rgb = bg_color

    # Title paragraph
    p0 = tf.paragraphs[0]
    p0.alignment = PP_ALIGN.LEFT
    run0 = p0.add_run()
    run0.text = title
    run0.font.name = "Times New Roman"
    run0.font.size = Pt(title_size)
    run0.font.bold = True
    run0.font.color.rgb = title_color

    for bullet in bullets:
        p = tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        # Space before
        pPr = p._pPr
        if pPr is None:
            pPr = p._p.get_or_add_pPr()
        spcBef = etree.SubElement(pPr, qn('a:spcBef'))
        spcPts = etree.SubElement(spcBef, qn('a:spcPts'))
        spcPts.set('val', '120')

        run = p.add_run()
        run.text = f"▸  {bullet}"
        run.font.name = "Times New Roman"
        run.font.size = Pt(bullet_size)
        run.font.color.rgb = bullet_color
    return txBox


def add_divider(slide, top, color=C_ACCENT):
    """Horizontal line as a thin rectangle."""
    line = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        Inches(0.45), top, Inches(9.1), Emu(40000)
    )
    line.fill.solid()
    line.fill.fore_color.rgb = color
    line.line.fill.background()
    return line


def add_table(slide, headers, rows, left, top, width, height,
              hdr_color=C_ACCENT, hdr_text_color=C_WHITE,
              row_colors=(RGBColor(0x25,0x25,0x3D), RGBColor(0x2E,0x2E,0x4A)),
              text_color=C_WHITE, font_size=13):
    """Add a styled table."""
    cols = len(headers)
    table = slide.shapes.add_table(1 + len(rows), cols, left, top, width, height).table
    col_w = width // cols
    for c in range(cols):
        table.columns[c].width = col_w

    # Header row
    for c, h in enumerate(headers):
        cell = table.cell(0, c)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = hdr_color
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.runs[0] if p.runs else p.add_run()
        run.font.name = "Times New Roman"
        run.font.size = Pt(font_size)
        run.font.bold = True
        run.font.color.rgb = hdr_text_color

    # Data rows
    for r, row in enumerate(rows):
        bg = row_colors[r % len(row_colors)]
        for c, val in enumerate(row):
            cell = table.cell(r + 1, c)
            cell.text = str(val)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            run = p.runs[0] if p.runs else p.add_run()
            run.font.name = "Times New Roman"
            run.font.size = Pt(font_size)
            run.font.color.rgb = text_color
    return table


def add_badge(slide, text, left, top, width, height, bg=C_GREEN, fg=C_WHITE, font_size=14):
    """Colored badge box with centered text."""
    box = slide.shapes.add_shape(1, left, top, width, height)
    box.fill.solid()
    box.fill.fore_color.rgb = bg
    box.line.fill.background()
    tf = box.text_frame
    tf.word_wrap = False
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = text
    run.font.name = "Times New Roman"
    run.font.size = Pt(font_size)
    run.font.bold = True
    run.font.color.rgb = fg
    return box


# ---------------------------------------------------------------------------
# SLIDE 1: Титул
# ---------------------------------------------------------------------------
blank_layout = prs.slide_masters[0].slide_layouts[6]  # Blank layout
slide = prs.slides.add_slide(blank_layout)
set_bg(slide, C_BG_DARK)

# Top accent bar
bar = slide.shapes.add_shape(1, 0, 0, W, Emu(180000))
bar.fill.solid(); bar.fill.fore_color.rgb = C_ACCENT; bar.line.fill.background()

# MIPT label
add_textbox(slide, "МОСКОВСКИЙ ФИЗИКО-ТЕХНИЧЕСКИЙ ИНСТИТУТ",
            Inches(0.5), Emu(200000), Inches(9), Inches(0.4),
            font_size=11, color=C_GRAY, align=PP_ALIGN.CENTER)

# Main title
add_textbox(slide,
    "Улучшения фреймворка обнаружения\nлогических аномалий и результаты",
    Inches(0.8), Inches(1.5), Inches(8.4), Inches(1.8),
    font_size=34, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)

# Subtitle
add_textbox(slide,
    "Training-free VLM-пайплайн на основе LogicQA · InternVL2.5-8B · MVTec LOCO AD",
    Inches(0.8), Inches(3.0), Inches(8.4), Inches(0.5),
    font_size=16, color=C_GRAY, align=PP_ALIGN.CENTER)

# Divider
add_divider(slide, Inches(3.7))

# Author
add_textbox(slide, "Д. А. Сахаров",
            Inches(0.5), Inches(3.9), Inches(9), Inches(0.35),
            font_size=17, bold=True, color=C_ACCENT2, align=PP_ALIGN.CENTER)

add_textbox(slide, "Научный руководитель: Копылов И.С.",
            Inches(0.5), Inches(4.3), Inches(9), Inches(0.3),
            font_size=14, color=C_GRAY, align=PP_ALIGN.CENTER)

add_textbox(slide, "2026",
            Inches(0.5), Inches(4.85), Inches(9), Inches(0.3),
            font_size=13, color=C_GRAY, align=PP_ALIGN.CENTER)

# Bottom accent bar
bar2 = slide.shapes.add_shape(1, 0, H - Emu(120000), W, Emu(120000))
bar2.fill.solid(); bar2.fill.fore_color.rgb = C_ACCENT; bar2.line.fill.background()


# ---------------------------------------------------------------------------
# SLIDE 2: Проблема — логическая аномалия
# ---------------------------------------------------------------------------
slide = prs.slides.add_slide(blank_layout)
set_bg(slide, C_BG_DARK)

add_textbox(slide, "Проблема: логические аномалии",
            Inches(0.45), Inches(0.2), Inches(9.1), Inches(0.6),
            font_size=28, bold=True, color=C_WHITE)
add_divider(slide, Inches(0.85))

# Left column
add_bullet_textbox(slide,
    "Структурная аномалия",
    ["Царапина, трещина, загрязнение",
     "Локальный дефект пикселей",
     "PatchCore, DRAEM → AUROC > 0.95"],
    Inches(0.45), Inches(1.05), Inches(4.3), Inches(2.5),
    title_size=18, bullet_size=15, title_color=C_ACCENT)

# Right column
add_bullet_textbox(slide,
    "Логическая аномалия  ← наша задача",
    ["Нарушение правил комплектации",
     "Каждый пиксель нормален — дефект только в смысле",
     "«2 мандарина» вместо «1» → все методы слепы"],
    Inches(5.25), Inches(1.05), Inches(4.3), Inches(2.5),
    title_size=18, bullet_size=15, title_color=C_ACCENT2)

# VS divider
add_textbox(slide, "VS", Inches(4.55), Inches(1.8), Inches(0.7), Inches(0.6),
            font_size=20, bold=True, color=C_GRAY, align=PP_ALIGN.CENTER)

# Key insight box
key_box = slide.shapes.add_shape(1, Inches(0.45), Inches(3.75), Inches(9.1), Inches(1.5))
key_box.fill.solid(); key_box.fill.fore_color.rgb = RGBColor(0x1A, 0x3A, 0x5C)
key_box.line.color.rgb = C_ACCENT; key_box.line.width = Emu(25000)
tf = key_box.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
run = p.add_run()
run.text = "Единственный подход, способный работать с логическими аномалиями:"
run.font.name = "Times New Roman"; run.font.size = Pt(14); run.font.color.rgb = C_GRAY

p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.LEFT
run2 = p2.add_run()
run2.text = "визуально-языковая модель как рассуждающий агент — формирует и проверяет семантические ограничения нормальности"
run2.font.name = "Times New Roman"; run2.font.size = Pt(17)
run2.font.bold = True; run2.font.color.rgb = C_WHITE


# ---------------------------------------------------------------------------
# SLIDE 3: Исходная точка — baseline
# ---------------------------------------------------------------------------
slide = prs.slides.add_slide(blank_layout)
set_bg(slide, C_BG_DARK)

add_textbox(slide, "Исходная точка: воспроизведение LogicQA",
            Inches(0.45), Inches(0.2), Inches(9.1), Inches(0.6),
            font_size=28, bold=True, color=C_WHITE)
add_divider(slide, Inches(0.85))

# Baseline table
add_textbox(slide, "InternVL2.5-8B · n_shots=3 · монолитный Stage 1 · без CoT",
            Inches(0.45), Inches(1.0), Inches(9.1), Inches(0.35),
            font_size=13, color=C_GRAY, italic=True)

add_table(slide,
    ["Класс", "AUROC", "F1-max", "TP", "FP", "FN"],
    [
        ["breakfast_box",       "0.589", "0.619", "90",  "42",  "83"],
        ["juice_bottle",        "0.537", "0.751", "127", "44",  "109"],
        ["pushpins",            "0.543", "0.569", "51",  "29",  "121"],
        ["screw_bag",           "0.484", "0.692", "73",  "44",  "146"],
        ["splicing_connectors", "0.481", "0.645", "160", "106", "33"],
        ["AVG",                 "0.527", "0.655", "—",   "—",   "—"],
    ],
    Inches(0.45), Inches(1.4), Inches(5.8), Inches(2.9),
    font_size=12,
    row_colors=(RGBColor(0x25,0x25,0x3D), RGBColor(0x2E,0x2E,0x4A))
)

# GPT-4o comparison badge
add_textbox(slide, "GPT-4o (оригинал):", Inches(6.6), Inches(1.5), Inches(3.0), Inches(0.35),
            font_size=13, color=C_GRAY)
add_badge(slide, "AVG AUROC ≈ 0.876",
          Inches(6.6), Inches(1.9), Inches(3.0), Inches(0.55),
          bg=RGBColor(0x27,0x6E,0x2E), fg=C_WHITE, font_size=15)

add_textbox(slide, "Наш baseline:", Inches(6.6), Inches(2.65), Inches(3.0), Inches(0.35),
            font_size=13, color=C_GRAY)
add_badge(slide, "AVG AUROC = 0.527",
          Inches(6.6), Inches(3.05), Inches(3.0), Inches(0.55),
          bg=C_RED, fg=C_WHITE, font_size=15)

# Two root problems
add_bullet_textbox(slide,
    "Два корневых дефекта:",
    ["Монолитное описание галлюцинирует и пропускает компоненты (CCR=63%)",
     "Модель отвечает «Yes» без реального анализа — yes-bias (FP=42)"],
    Inches(0.45), Inches(4.45), Inches(9.1), Inches(1.0),
    title_size=15, bullet_size=14, title_color=C_ACCENT2,
    bullet_color=RGBColor(0xFF, 0xCC, 0x80))


# ---------------------------------------------------------------------------
# SLIDE 4: Улучшение A — Per-component Stage 1
# ---------------------------------------------------------------------------
slide = prs.slides.add_slide(blank_layout)
set_bg(slide, C_BG_DARK)

add_textbox(slide, "Улучшение A: декомпозированный Stage 1",
            Inches(0.45), Inches(0.2), Inches(9.1), Inches(0.6),
            font_size=28, bold=True, color=C_WHITE)
add_divider(slide, Inches(0.85))

# Left: problem → solution
add_bullet_textbox(slide,
    "Проблема",
    ["Один запрос на всё изображение",
     "VLM «галлюцинирует» объекты",
     "Упускает редкие компоненты",
     "CCR baseline = 63.3%"],
    Inches(0.45), Inches(1.05), Inches(3.3), Inches(2.4),
    title_size=17, bullet_size=14, title_color=C_RED)

arrow_box = slide.shapes.add_shape(1, Inches(3.85), Inches(1.8), Inches(0.5), Inches(0.7))
arrow_box.fill.background(); arrow_box.line.fill.background()
add_textbox(slide, "→", Inches(3.85), Inches(1.9), Inches(0.5), Inches(0.5),
            font_size=28, bold=True, color=C_ACCENT, align=PP_ALIGN.CENTER)

add_bullet_textbox(slide,
    "Решение",
    ["6 отдельных запросов по компонентам",
     "tangerines / nectarine / muesli /",
     "banana chips / almonds / yogurt",
     "5 изображений × 6 = 30 VLM-вызовов"],
    Inches(4.45), Inches(1.05), Inches(3.3), Inches(2.4),
    title_size=17, bullet_size=14, title_color=C_GREEN)

# Result badges
add_textbox(slide, "Результат:", Inches(7.85), Inches(1.15), Inches(1.9), Inches(0.35),
            font_size=13, color=C_GRAY, align=PP_ALIGN.CENTER)
add_badge(slide, "CCR: 63% → 100%",
          Inches(7.85), Inches(1.55), Inches(1.95), Inches(0.55),
          bg=C_GREEN, fg=C_WHITE, font_size=13)
add_badge(slide, "MACE: 0.44 → 0.24",
          Inches(7.85), Inches(2.2), Inches(1.95), Inches(0.55),
          bg=RGBColor(0x27,0x6E,0x2E), fg=C_WHITE, font_size=13)

# Mechanism explanation box
mbox = slide.shapes.add_shape(1, Inches(0.45), Inches(3.65), Inches(9.1), Inches(1.7))
mbox.fill.solid(); mbox.fill.fore_color.rgb = RGBColor(0x1A, 0x3A, 0x5C)
mbox.line.color.rgb = C_ACCENT; mbox.line.width = Emu(20000)
tf = mbox.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]
run = p.add_run()
run.text = "Почему работает: "
run.font.name = "Times New Roman"; run.font.size = Pt(15); run.font.bold = True
run.font.color.rgb = C_ACCENT2
run2 = p.add_run()
run2.text = ("при фокусированном запросе «опиши только отсек с мандаринами» "
             "VLM не может галлюцинировать другие объекты — внимание жёстко ограничено компонентом. "
             "Все нормативные ограничения попадают в итоговое описание.")
run2.font.name = "Times New Roman"; run2.font.size = Pt(15); run2.font.color.rgb = C_WHITE


# ---------------------------------------------------------------------------
# SLIDE 5: Улучшения B C D E F
# ---------------------------------------------------------------------------
slide = prs.slides.add_slide(blank_layout)
set_bg(slide, C_BG_DARK)

add_textbox(slide, "Улучшения пайплайна B – F",
            Inches(0.45), Inches(0.2), Inches(9.1), Inches(0.6),
            font_size=28, bold=True, color=C_WHITE)
add_divider(slide, Inches(0.85))

# 5 cards in two rows: 3 on top, 2 on bottom (centered)
cards = [
    ("B", "Hedge-filtering",     "Stage 2",
     "Исключаем утверждения с «sometimes», «may», «appears to» из нормативного определения",
     "Устойчивость нормативного определения к случайным наблюдениям"),

    ("C", "Count-bypass",        "Stage 3b",
     "Счётные вопросы («exactly N») не проходят 80%-порог фильтрации на val-изображениях",
     "VLM систематически ошибается в счёте → без bypass теряются ключевые вопросы"),

    ("D", "Stem-matching",       "Stage 3b",
     "Лемматизация при парсинге ответа: «almonds»→«almond», «tangerines»→«tangerine»",
     "Устраняет грамматические ложные отклонения при поиске Yes/No-токена"),

    ("E", "Sub-Q аугментация ×4","Stage 3c",
     "Каждый вопрос перефразируется в 4 варианта; финальный ответ — majority vote",
     "+0.03–0.05 AUROC; устойчивость к поверхностным лингвистическим вариациям"),

    ("F", "min_failures = 2",    "Stage 4",
     "Аномалия: ≥ 2 основных вопроса получили «No» (вместо порога = 1)",
     "FP: 22 → 12 — отсекаем одиночные галлюцинации при несогласии вопросов"),
]

card_w = Inches(3.0)
card_h = Inches(1.9)
card_gap = Inches(0.12)
row1_tops = [Inches(1.0)] * 3
row2_tops = [Inches(3.05)] * 2

positions = [
    (Inches(0.45),              row1_tops[0]),
    (Inches(0.45) + card_w + card_gap, row1_tops[1]),
    (Inches(0.45) + 2*(card_w + card_gap), row1_tops[2]),
    (Inches(0.45) + card_w/2 + card_gap/2 - Inches(0.05), row2_tops[0]),
    (Inches(0.45) + card_w/2 + card_gap/2 + card_w + card_gap - Inches(0.05), row2_tops[1]),
]

accent_colors = [
    RGBColor(0x16, 0x7A, 0x8A),  # teal
    RGBColor(0x8E, 0x44, 0xAD),  # purple
    RGBColor(0xD3, 0x54, 0x00),  # orange
    RGBColor(0x27, 0x6E, 0x2E),  # green
    RGBColor(0x1A, 0x6E, 0x9A),  # blue
]

for (letter, name, stage, desc, effect), (left, top), acolor in zip(cards, positions, accent_colors):
    # Card background
    card = slide.shapes.add_shape(1, left, top, card_w, card_h)
    card.fill.solid(); card.fill.fore_color.rgb = RGBColor(0x22, 0x22, 0x3E)
    card.line.color.rgb = acolor; card.line.width = Emu(25000)

    # Top accent bar
    cbar = slide.shapes.add_shape(1, left, top, card_w, Emu(90000))
    cbar.fill.solid(); cbar.fill.fore_color.rgb = acolor; cbar.line.fill.background()

    # Letter badge
    add_textbox(slide, letter,
                left + Emu(80000), top + Emu(15000), Emu(220000), Emu(70000),
                font_size=16, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)

    # Name + stage
    add_textbox(slide, f"{name}  ·  {stage}",
                left + Emu(310000), top + Emu(20000), card_w - Emu(360000), Emu(70000),
                font_size=12, bold=True, color=C_WHITE)

    # Description
    add_textbox(slide, desc,
                left + Emu(60000), top + Emu(130000), card_w - Emu(120000), Emu(620000),
                font_size=11, color=RGBColor(0xCC, 0xCC, 0xCC), wrap=True)

    # Effect (bottom, amber)
    add_textbox(slide, f"→ {effect}",
                left + Emu(60000), top + Emu(770000), card_w - Emu(120000), Emu(400000),
                font_size=11, color=C_ACCENT2, wrap=True, italic=True)


# ---------------------------------------------------------------------------
# SLIDE 6: Обоснование выбора метрик
# ---------------------------------------------------------------------------
slide = prs.slides.add_slide(blank_layout)
set_bg(slide, C_BG_DARK)

add_textbox(slide, "Обоснование выбора метрик",
            Inches(0.45), Inches(0.2), Inches(9.1), Inches(0.6),
            font_size=28, bold=True, color=C_WHITE)
add_divider(slide, Inches(0.85))

add_textbox(slide, "Каждая метрика мотивирована внешней работой — не выбрана произвольно",
            Inches(0.45), Inches(0.92), Inches(9.1), Inches(0.3),
            font_size=13, color=C_GRAY, italic=True)

metric_rows = [
    # (metric, level, paper_ref, paper_short, justification, color)
    ("CLIPScore",       "L1",   "[Hessel et al., EMNLP 2021]",
     "CLIP ViT-B/32",
     "Reference-free метрика семантического сходства текст–изображение; широко принята как стандарт оценки качества описаний",
     C_ACCENT),

    ("CCR  (LLM-as-Judge)", "L1", "[Zheng et al., NeurIPS 2023]  [Lee et al., ACL 2024]",
     "MT-Bench · Prometheus-Vision",
     "VLM-as-Judge коррелирует с оценками людей лучше, чем text-match метрики; Prometheus-Vision — наивысшая корреляция Пирсона среди open-source",
     RGBColor(0x16, 0x7A, 0x8A)),

    ("MACE · SRA",      "L2",   "[Kamath et al., EMNLP 2023]  [Dou & Peng, ACL 2024]",
     "What's 'up' · VALOR-EVAL",
     "VLM достигают 56% точности на пространственных задачах vs 99% людей; VALOR-EVAL показал: объектных метрик недостаточно, нужны атрибуты и отношения",
     RGBColor(0x8E, 0x44, 0xAD)),

    ("FaithScore / CCR-atomic","L1","[Jing et al., arXiv 2023]",
     "FaithScore",
     "Атомарная верификация фактов без эталона коррелирует с оценками людей — обосновывает подход «одно ограничение = один вопрос судье»",
     RGBColor(0xD3, 0x54, 0x00)),

    ("Sub-Q Consistency","L3",  "[Kostumov et al., arXiv 2024]  [Li et al., EMNLP 2023]",
     "Uncertainty VLMs · POPE",
     "Точность VLM не согласована с неопределённостью; POPE: VLM склонны отвечать «Yes» — majority vote по перефразировкам выявляет нестабильные ответы",
     RGBColor(0x27, 0x6E, 0x2E)),
]

row_h = Emu(730000)
for i, (metric, level, ref, paper_short, just, acolor) in enumerate(metric_rows):
    top = Inches(1.3) + i * (row_h + Emu(30000))

    # Left stripe
    stripe = slide.shapes.add_shape(1, Inches(0.45), top, Emu(60000), row_h)
    stripe.fill.solid(); stripe.fill.fore_color.rgb = acolor; stripe.line.fill.background()

    # Level badge
    lvl_box = slide.shapes.add_shape(1, Inches(0.6), top + Emu(150000), Emu(420000), Emu(350000))
    lvl_box.fill.solid(); lvl_box.fill.fore_color.rgb = acolor; lvl_box.line.fill.background()
    tf = lvl_box.text_frame; p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    run = p.add_run(); run.text = level
    run.font.name = "Times New Roman"; run.font.size = Pt(11)
    run.font.bold = True; run.font.color.rgb = C_WHITE

    # Metric name
    add_textbox(slide, metric, Inches(1.25), top + Emu(40000), Inches(2.2), Emu(380000),
                font_size=14, bold=True, color=C_WHITE)

    # Paper short name
    add_textbox(slide, paper_short, Inches(1.25), top + Emu(430000), Inches(2.2), Emu(280000),
                font_size=11, color=acolor, italic=True)

    # Reference
    add_textbox(slide, ref, Inches(3.6), top + Emu(40000), Inches(3.5), Emu(380000),
                font_size=11, color=C_ACCENT2, italic=False)

    # Justification
    add_textbox(slide, just, Inches(3.6), top + Emu(430000), Inches(6.0), Emu(320000),
                font_size=11, color=RGBColor(0xCC, 0xCC, 0xCC), wrap=True)

    # Separator
    if i < 4:
        sep = slide.shapes.add_shape(1, Inches(0.45), top + row_h + Emu(5000),
                                     Inches(9.1), Emu(18000))
        sep.fill.solid(); sep.fill.fore_color.rgb = RGBColor(0x3A,0x3A,0x5A)
        sep.line.fill.background()


# ---------------------------------------------------------------------------
# SLIDE 7: Улучшение G — RC4-A CoT
# ---------------------------------------------------------------------------
slide = prs.slides.add_slide(blank_layout)
set_bg(slide, C_BG_DARK)

add_textbox(slide, "Улучшение G: inline Chain-of-Thought (RC4-A)",
            Inches(0.45), Inches(0.2), Inches(9.1), Inches(0.6),
            font_size=28, bold=True, color=C_WHITE)
add_divider(slide, Inches(0.85))

# Without CoT
add_textbox(slide, "Без CoT (r23):",
            Inches(0.45), Inches(1.05), Inches(4.2), Inches(0.35),
            font_size=15, bold=True, color=C_RED)
add_textbox(slide,
    'Is there exactly 2 tangerines?\nAnswer: Yes / No',
    Inches(0.45), Inches(1.45), Inches(4.2), Inches(0.75),
    font_name="Courier New", font_size=13, color=RGBColor(0xCC,0xCC,0xCC))
add_textbox(slide, "→ модель угадывает на основе статистических ожиданий",
            Inches(0.45), Inches(2.25), Inches(4.2), Inches(0.4),
            font_size=13, color=C_RED, italic=True)

# Arrow
add_textbox(slide, "RC4-A", Inches(4.55), Inches(1.7), Inches(0.9), Inches(0.35),
            font_size=11, color=C_ACCENT2, bold=True, align=PP_ALIGN.CENTER)
add_textbox(slide, "→", Inches(4.55), Inches(2.0), Inches(0.9), Inches(0.5),
            font_size=30, bold=True, color=C_ACCENT, align=PP_ALIGN.CENTER)

# With CoT
add_textbox(slide, "С CoT (r25):",
            Inches(5.6), Inches(1.05), Inches(4.2), Inches(0.35),
            font_size=15, bold=True, color=C_GREEN)
add_textbox(slide,
    'Step 1 - Observe:\n  Describe what you see about tangerines.\nStep 2 - Conclude:\n  Is there exactly 2? Result: Yes / No',
    Inches(5.6), Inches(1.45), Inches(4.2), Inches(1.25),
    font_name="Courier New", font_size=12, color=RGBColor(0xCC,0xCC,0xCC))

# Majority vote note
add_textbox(slide, "+ majority vote по 4 перефразировкам вопроса",
            Inches(5.6), Inches(2.8), Inches(4.2), Inches(0.4),
            font_size=13, color=C_GRAY, italic=True)

# Result badges
add_badge(slide, "FP: 12 → 0",
          Inches(0.45), Inches(3.05), Inches(2.8), Inches(0.65),
          bg=C_GREEN, fg=C_WHITE, font_size=16)
add_badge(slide, "AUROC: 0.782 → 0.852",
          Inches(3.45), Inches(3.05), Inches(3.1), Inches(0.65),
          bg=RGBColor(0x1A,0x6E,0x9A), fg=C_WHITE, font_size=16)
add_badge(slide, "F1-max: 0.746 → 0.818",
          Inches(6.75), Inches(3.05), Inches(3.0), Inches(0.65),
          bg=RGBColor(0x1A,0x6E,0x9A), fg=C_WHITE, font_size=16)

# Explanation
ebox = slide.shapes.add_shape(1, Inches(0.45), Inches(3.9), Inches(9.1), Inches(1.5))
ebox.fill.solid(); ebox.fill.fore_color.rgb = RGBColor(0x1A,0x3A,0x5C)
ebox.line.color.rgb = C_ACCENT; ebox.line.width = Emu(20000)
tf = ebox.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]
r1 = p.add_run(); r1.text = "Механизм: "
r1.font.name = "Times New Roman"; r1.font.size = Pt(15); r1.font.bold = True
r1.font.color.rgb = C_ACCENT2
r2 = p.add_run()
r2.text = ("ложноположительный ответ потребовал бы сформулировать несуществующее нарушение. "
           "Step 1 Observe создаёт верифицируемую посылку — модель не может соврать, "
           "не описав реального нарушения. Yes-bias (POPE, 2023) полностью устранён.")
r2.font.name = "Times New Roman"; r2.font.size = Pt(15); r2.font.color.rgb = C_WHITE


# ---------------------------------------------------------------------------
# SLIDE 8: Прогрессия результатов
# ---------------------------------------------------------------------------
slide = prs.slides.add_slide(blank_layout)
set_bg(slide, C_BG_DARK)

add_textbox(slide, "Прогрессия результатов: baseline → r25",
            Inches(0.45), Inches(0.2), Inches(9.1), Inches(0.6),
            font_size=28, bold=True, color=C_WHITE)
add_divider(slide, Inches(0.85))

add_textbox(slide, "breakfast_box · 50 тестовых изображений (25 good + 25 anomaly)",
            Inches(0.45), Inches(1.0), Inches(9.1), Inches(0.3),
            font_size=13, color=C_GRAY, italic=True)

# Custom row colors: highlight r25
row_normal   = RGBColor(0x25, 0x25, 0x3D)
row_alt      = RGBColor(0x2E, 0x2E, 0x4A)
row_best     = RGBColor(0x1A, 0x5C, 0x2A)  # green for r25
row_key      = RGBColor(0x1A, 0x3A, 0x5C)  # blue for r23

table_data = [
    ["Baseline (avg 5 cls)", "0.527", "0.655", "—",   "—",   "Монолитный, без CoT"],
    ["r18 (decomposed start)","0.521","0.667", "1",   "23",  "Декомпозиция (нестабильна)"],
    ["r21 (NORMALITY fix)",  "0.500", "0.667", "0",   "25",  "Регрессия промптов"],
    ["r23 (fixes 3b/3c/4)",  "0.782", "0.746", "12",  "3",   "Исправление Stage 3b/4"],
    ["r25 ★ RC4-A CoT",      "0.852", "0.818", "0",   "10",  "inline CoT · FP=0"],
]

table = slide.shapes.add_table(
    6, 6,
    Inches(0.45), Inches(1.35), Inches(9.1), Inches(2.85)
).table

headers = ["Запуск", "AUROC", "F1-max", "FP", "FN", "Ключевое"]
col_widths = [Inches(2.0), Inches(0.85), Inches(0.85), Inches(0.65), Inches(0.65), Inches(4.1)]
for c, w in enumerate(col_widths):
    table.columns[c].width = w

for c, h in enumerate(headers):
    cell = table.cell(0, c)
    cell.text = h
    cell.fill.solid(); cell.fill.fore_color.rgb = C_ACCENT
    p = cell.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    run = p.runs[0] if p.runs else p.add_run()
    run.font.name = "Times New Roman"; run.font.size = Pt(13)
    run.font.bold = True; run.font.color.rgb = C_WHITE

row_bgs = [row_normal, row_alt, row_alt, row_key, row_best]
for r, (row, bg) in enumerate(zip(table_data, row_bgs)):
    for c, val in enumerate(row):
        cell = table.cell(r + 1, c)
        cell.text = val
        cell.fill.solid(); cell.fill.fore_color.rgb = bg
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER if c != 5 else PP_ALIGN.LEFT
        run = p.runs[0] if p.runs else p.add_run()
        run.font.name = "Times New Roman"; run.font.size = Pt(12)
        is_best = (r == 4)
        run.font.bold = is_best
        run.font.color.rgb = C_WHITE

# Best result badges
add_badge(slide, "AUROC = 0.852",
          Inches(0.45), Inches(4.35), Inches(2.5), Inches(0.7),
          bg=C_GREEN, fg=C_WHITE, font_size=15)
add_badge(slide, "FP = 0",
          Inches(3.15), Inches(4.35), Inches(1.5), Inches(0.7),
          bg=C_GREEN, fg=C_WHITE, font_size=15)
add_badge(slide, "CCR = 100%",
          Inches(4.85), Inches(4.35), Inches(2.0), Inches(0.7),
          bg=C_GREEN, fg=C_WHITE, font_size=15)
add_badge(slide, "≈ GPT-4o · 0 руб./запрос",
          Inches(7.05), Inches(4.35), Inches(2.5), Inches(0.7),
          bg=RGBColor(0x1A,0x6E,0x9A), fg=C_WHITE, font_size=14)


# ---------------------------------------------------------------------------
# SLIDE 9: Иерархическая система оценки L1–L4
# ---------------------------------------------------------------------------
slide = prs.slides.add_slide(blank_layout)
set_bg(slide, C_BG_DARK)

add_textbox(slide, "Иерархическая система оценки L1–L4",
            Inches(0.45), Inches(0.2), Inches(9.1), Inches(0.6),
            font_size=28, bold=True, color=C_WHITE)
add_divider(slide, Inches(0.85))

add_textbox(slide, "Зачем: AUROC не показывает, на каком этапе пайплайна возникает ошибка",
            Inches(0.45), Inches(0.95), Inches(9.1), Inches(0.3),
            font_size=13, color=C_GRAY, italic=True)

levels = [
    ("L1  Восприятие",    C_ACCENT,
     "CLIPScore = 69.7%   ·   CCR = 100%",
     "Описание семантически верное и полное"),
    ("L2  Атрибуты",      RGBColor(0x8E,0x44,0xAD),
     "MACE = 0.24   ·   SRA = 64.3%",
     "Счёт точен · пространственные отношения — системная слабость 8B"),
    ("L2.5  Фильтрация",  RGBColor(0x16,0x7A,0x8A),
     "Precision = 21%   ·   Recall = 63%",
     "Ограничение Qwen2.5-3B-судьи: не различает перефразировки"),
    ("L3  Рассуждение",   RGBColor(0xD3,0x54,0x00),
     "Sub-Q Consistency = 70.5%",
     "Умеренная стабильность — VLM неуверена (Kostumov et al. 2024)"),
    ("L4  Задача",        C_GREEN,
     "AUROC = 0.852   ·   F1-max = 0.818   ·   FP = 0",
     "Лучший результат достигнут"),
]

for i, (title, color, metric, note) in enumerate(levels):
    top = Inches(1.35) + i * Inches(0.82)
    # Color stripe
    stripe = slide.shapes.add_shape(1, Inches(0.45), top, Inches(0.12), Inches(0.65))
    stripe.fill.solid(); stripe.fill.fore_color.rgb = color; stripe.line.fill.background()
    # Level label
    add_textbox(slide, title, Inches(0.65), top, Inches(2.0), Inches(0.35),
                font_size=14, bold=True, color=color)
    # Metric values
    add_textbox(slide, metric, Inches(2.75), top, Inches(3.5), Inches(0.35),
                font_size=14, bold=True, color=C_WHITE)
    # Note
    add_textbox(slide, note, Inches(6.35), top, Inches(3.2), Inches(0.6),
                font_size=12, color=C_GRAY, italic=True)
    # Separator line
    if i < 4:
        sep = slide.shapes.add_shape(1, Inches(0.45), top + Inches(0.72), Inches(9.1), Emu(15000))
        sep.fill.solid(); sep.fill.fore_color.rgb = RGBColor(0x3A,0x3A,0x5A); sep.line.fill.background()


# ---------------------------------------------------------------------------
# SLIDE 10: Заключение
# ---------------------------------------------------------------------------
slide = prs.slides.add_slide(blank_layout)
set_bg(slide, C_BG_DARK)

# Accent bar top
bar = slide.shapes.add_shape(1, 0, 0, W, Emu(180000))
bar.fill.solid(); bar.fill.fore_color.rgb = C_ACCENT; bar.line.fill.background()

add_textbox(slide, "Заключение",
            Inches(0.45), Inches(0.45), Inches(9.1), Inches(0.6),
            font_size=30, bold=True, color=C_WHITE)
add_divider(slide, Inches(1.1))

results = [
    ("AUROC 0.527 → 0.852",    "open-source модель на одном GPU · без API · воспроизводимо"),
    ("FP = 0  ·  CCR = 100%",  "RC4-A CoT + per-component Stage 1 решают разные проблемы"),
    ("FN = 10 → пространство", "системная слабость VLM-8B (56% vs 99% люди, Kamath 2023)"),
]

for i, (result, desc) in enumerate(results):
    top = Inches(1.3) + i * Inches(1.1)
    num_box = slide.shapes.add_shape(1, Inches(0.45), top, Inches(0.5), Inches(0.7))
    num_box.fill.solid(); num_box.fill.fore_color.rgb = C_ACCENT; num_box.line.fill.background()
    tf = num_box.text_frame
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    run = p.add_run(); run.text = str(i+1)
    run.font.name = "Times New Roman"; run.font.size = Pt(18)
    run.font.bold = True; run.font.color.rgb = C_WHITE

    add_textbox(slide, result, Inches(1.1), top, Inches(8.5), Inches(0.35),
                font_size=18, bold=True, color=C_ACCENT2)
    add_textbox(slide, desc, Inches(1.1), top + Inches(0.38), Inches(8.5), Inches(0.35),
                font_size=14, color=C_GRAY)

# Next step box
nbox = slide.shapes.add_shape(1, Inches(0.45), Inches(4.5), Inches(9.1), Inches(0.8))
nbox.fill.solid(); nbox.fill.fore_color.rgb = RGBColor(0x1A,0x3A,0x5C)
nbox.line.color.rgb = C_ACCENT2; nbox.line.width = Emu(30000)
tf = nbox.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]
r1 = p.add_run(); r1.text = "Следующий шаг: "
r1.font.name = "Times New Roman"; r1.font.size = Pt(15); r1.font.bold = True
r1.font.color.rgb = C_ACCENT2
r2 = p.add_run()
r2.text = "InternVL2.5-38B-AWQ для пространственных аномалий · все 5 классов MVTec LOCO AD"
r2.font.name = "Times New Roman"; r2.font.size = Pt(15); r2.font.color.rgb = C_WHITE

# Bottom accent bar
bar2 = slide.shapes.add_shape(1, 0, H - Emu(120000), W, Emu(120000))
bar2.fill.solid(); bar2.fill.fore_color.rgb = C_ACCENT; bar2.line.fill.background()

# ---------------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------------
prs.save(OUTPUT)
print(f"Saved: {OUTPUT}")
print(f"Slides: {len(prs.slides)}")
