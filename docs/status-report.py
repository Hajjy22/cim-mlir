"""Generate the cim-mlir project status report as a PDF.

The PDF itself is committed for convenience, but this script is the
reviewable source for it -- a checked-in binary with no text original
is exactly the "opaque in review" artifact this project avoids
elsewhere (see test/python/onnx_fixtures.py's own module docstring on
why models are built in-process rather than checked in as blobs).

Every figure in the report is a point-in-time snapshot of the
repository, not a live query: regenerate after material changes.

Usage:  pip install reportlab && python3 docs/status-report.py
"""

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, KeepTogether,
                                NextPageTemplate, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)

import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "cim-mlir-status-report.pdf")

# ---------------------------------------------------------------- palette
INK = colors.HexColor("#12161C")       # near-black body text
SLATE = colors.HexColor("#47535F")     # secondary text
NAVY = colors.HexColor("#12324D")      # headings / rules
ACCENT = colors.HexColor("#1F6F8B")    # the one accent
RULE = colors.HexColor("#C9D2DA")
PANEL = colors.HexColor("#F1F4F7")
PANEL_EDGE = colors.HexColor("#DCE3EA")

OK_BG = colors.HexColor("#E3F0E6")
OK_TX = colors.HexColor("#1E5B2E")
WARN_BG = colors.HexColor("#FBF0DC")
WARN_TX = colors.HexColor("#7A5410")
OPEN_BG = colors.HexColor("#E8ECF1")
OPEN_TX = colors.HexColor("#3B4855")
STOP_BG = colors.HexColor("#F7E2E2")
STOP_TX = colors.HexColor("#7E2020")

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm

# ---------------------------------------------------------------- styles
ss = getSampleStyleSheet()


def mk(name, **kw):
    base = dict(name=name, fontName="Helvetica", fontSize=9.2, leading=13.4,
                textColor=INK, alignment=TA_LEFT)
    base.update(kw)
    return ParagraphStyle(**base)


Body = mk("Body", spaceAfter=6)
Lead = mk("Lead", fontSize=10.6, leading=16, textColor=SLATE, spaceAfter=9)
H1 = mk("H1", fontName="Helvetica-Bold", fontSize=16.5, leading=20,
        textColor=NAVY, spaceBefore=2, spaceAfter=2)
H2 = mk("H2", fontName="Helvetica-Bold", fontSize=11.4, leading=15,
        textColor=NAVY, spaceBefore=11, spaceAfter=4)
H3 = mk("H3", fontName="Helvetica-Bold", fontSize=9.6, leading=13,
        textColor=ACCENT, spaceBefore=8, spaceAfter=3)
Small = mk("Small", fontSize=8.1, leading=11.4, textColor=SLATE)
TCell = mk("TCell", fontSize=8.2, leading=11.2)
TCellS = mk("TCellS", fontSize=7.8, leading=10.6, textColor=SLATE)
THead = mk("THead", fontName="Helvetica-Bold", fontSize=8.0, leading=10.8,
           textColor=colors.white)
Chip = mk("Chip", fontName="Helvetica-Bold", fontSize=7.2, leading=9.6,
          alignment=1)
Mono = mk("Mono", fontName="Courier", fontSize=8.0, leading=11.2)
CoverTitle = mk("CoverTitle", fontName="Helvetica-Bold", fontSize=31,
                leading=35, textColor=NAVY)
CoverSub = mk("CoverSub", fontSize=12.4, leading=17.5, textColor=SLATE)
KpiNum = mk("KpiNum", fontName="Helvetica-Bold", fontSize=17, leading=20,
            textColor=NAVY, alignment=1)
KpiLbl = mk("KpiLbl", fontSize=7.2, leading=9.4, textColor=SLATE, alignment=1)


def chip(text, kind):
    bg, tx = {"ok": (OK_BG, OK_TX), "warn": (WARN_BG, WARN_TX),
              "open": (OPEN_BG, OPEN_TX), "stop": (STOP_BG, STOP_TX)}[kind]
    p = Paragraph(f'<font color="#{tx.hexval()[2:]}">{text}</font>', Chip)
    t = Table([[p]], colWidths=[19 * mm], rowHeights=[6.4 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 1),
        ("RIGHTPADDING", (0, 0), (-1, -1), 1),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
    ]))
    return t


CONTENT_W = PAGE_W - 2 * MARGIN


def table(rows, widths, head=True, zebra=True, valign="TOP"):
    t = Table(rows, colWidths=widths, repeatRows=1 if head else 0)
    style = [
        ("VALIGN", (0, 0), (-1, -1), valign),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
    ]
    if head:
        style += [("BACKGROUND", (0, 0), (-1, 0), NAVY),
                  ("LINEBELOW", (0, 0), (-1, 0), 0, colors.white)]
    if zebra:
        start = 1 if head else 0
        for i in range(start, len(rows)):
            if (i - start) % 2 == 1:
                style.append(("BACKGROUND", (0, i), (-1, i), PANEL))
    t.setStyle(TableStyle(style))
    return t


def panel(flowables, bg=PANEL, edge=PANEL_EDGE):
    t = Table([[flowables]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("BOX", (0, 0), (-1, -1), 0.6, edge),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    return t


def rule(space_before=3, space_after=7, color=RULE, w=0.9):
    t = Table([[""]], colWidths=[CONTENT_W], rowHeights=[0.1])
    t.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), w, color)]))
    return [Spacer(1, space_before), t, Spacer(1, space_after)]


def bullets(items, style=Body):
    out = []
    for it in items:
        out.append(Paragraph(f"&bull;&nbsp;&nbsp;{it}", style))
    return out


# ---------------------------------------------------------------- canvas
def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.4)
    canvas.setFillColor(SLATE)
    canvas.drawString(MARGIN, 11 * mm,
                      "cim-mlir  |  Project Status Report  |  17 August 2026")
    canvas.drawRightString(PAGE_W - MARGIN, 11 * mm, "Page %d" % doc.page)
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, 14.5 * mm, PAGE_W - MARGIN, 14.5 * mm)
    canvas.restoreState()


def on_cover(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_H - 13 * mm, PAGE_W, 13 * mm, stroke=0, fill=1)
    canvas.setFillColor(ACCENT)
    canvas.rect(0, PAGE_H - 13 * mm, PAGE_W * 0.34, 13 * mm, stroke=0, fill=1)
    canvas.setFont("Helvetica", 7.4)
    canvas.setFillColor(SLATE)
    canvas.drawString(MARGIN, 11 * mm,
                      "Generated from repository state at commit fe3aced")
    canvas.restoreState()


doc = BaseDocTemplate(OUT, pagesize=A4,
                      leftMargin=MARGIN, rightMargin=MARGIN,
                      topMargin=20 * mm, bottomMargin=20 * mm,
                      title="cim-mlir - Project Status Report",
                      author="cim-mlir")
frame = Frame(MARGIN, 20 * mm, CONTENT_W, PAGE_H - 40 * mm, id="f")
doc.addPageTemplates([
    PageTemplate(id="cover", frames=[frame], onPage=on_cover),
    PageTemplate(id="main", frames=[frame], onPage=on_page),
])

S = []

# ================================================================ COVER
S.append(Spacer(1, 22 * mm))
S.append(Paragraph("cim-mlir", CoverTitle))
S.append(Spacer(1, 3 * mm))
S.append(Paragraph(
    "An open, retargetable compiler and runtime stack for "
    "compute-in-memory and processing-in-memory hardware.", CoverSub))
S.append(Spacer(1, 9 * mm))
S.extend(rule(0, 6, NAVY, 1.4))
S.append(Paragraph("PROJECT STATUS REPORT", mk(
    "x", fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=ACCENT)))
S.append(Paragraph(
    "Architecture, delivered scope, and launch readiness &mdash; "
    "17 August 2026", Small))
S.append(Spacer(1, 12 * mm))

kpis = [
    ("31,845", "lines of code"),
    ("8", "compiler passes"),
    ("10", "dialect ops"),
    ("465", "automated tests"),
    ("12", "target files"),
]
kt = Table([[Paragraph(n, KpiNum) for n, _ in kpis],
            [Paragraph(l, KpiLbl) for _, l in kpis]],
           colWidths=[CONTENT_W / 5.0] * 5)
kt.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), PANEL),
    ("BOX", (0, 0), (-1, -1), 0.6, PANEL_EDGE),
    ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.white),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, 0), 9),
    ("BOTTOMPADDING", (0, 1), (-1, 1), 9),
]))
S.append(kt)
S.append(Spacer(1, 10 * mm))

S.append(panel([
    Paragraph("Headline", H3),
    Paragraph(
        "A real quantized INT8 convolutional network &mdash; chained "
        "convolution layers, interleaved max-pooling, and a "
        "fully-connected head &mdash; now compiles from a standard "
        "<b>.onnx</b> file and executes, with every numerical claim "
        "checked against independent reference implementations rather "
        "than against this project's own code.", Body),
    Paragraph(
        "Milestones M0&ndash;M3 are complete and M4 is substantially "
        "complete. The remaining distance to a public v0.1 launch is "
        "dominated by one infrastructure blocker outside the codebase "
        "(exhausted CI minutes, resets 1 September 2026) and by the "
        "absence of measured hardware numbers &mdash; not by missing "
        "compiler capability.", Body),
]))

S.append(Spacer(1, 8 * mm))
meta = [
    [Paragraph("<b>Repository</b>", TCell), Paragraph("Hajjy22/cim-mlir", TCell),
     Paragraph("<b>Branch</b>", TCell),
     Paragraph("claude/cim-mlir-cost-report-zhj2ce", TCell)],
    [Paragraph("<b>HEAD</b>", TCell), Paragraph("fe3aced", TCell),
     Paragraph("<b>Base</b>", TCell), Paragraph("main @ 8325ccb", TCell)],
    [Paragraph("<b>Open PR</b>", TCell), Paragraph("#22 (6 commits)", TCell),
     Paragraph("<b>Toolchain</b>", TCell), Paragraph("MLIR / LLVM 18", TCell)],
]
mt = table(meta, [26 * mm, 52 * mm, 24 * mm, CONTENT_W - 102 * mm],
           head=False, zebra=True)
S.append(mt)

S.append(NextPageTemplate("main"))
S.append(PageBreak())

# ================================================================ 1. SUMMARY
S.append(Paragraph("1&nbsp;&nbsp;Executive summary", H1))
S.extend(rule())

S.append(Paragraph(
    "CIM/PIM hardware exists and some of it ships, but there is no open "
    "way to take a normal neural network and run it on that hardware "
    "without hand-writing assembly or using a vendor's closed, "
    "single-chip compiler. cim-mlir is a three-part open-source stack "
    "that closes that gap: an MLIR dialect and lowering pipeline, a thin "
    "C runtime ABI, and a benchmark suite with a cost model.", Lead))

S.append(Paragraph("Where the project stands", H2))
S.append(Paragraph(
    "The compiler is real and composes end to end. All eight lowering "
    "passes are implemented, and a single module has been driven through "
    "literally all eight in spec order, down to real runtime calls with "
    "no dialect ops left. The placement engine &mdash; the scheduling "
    "problem the project exists to solve &mdash; is implemented and "
    "proven optimal by exhaustive search on the shape the compiler "
    "actually emits.", Body))
S.append(Paragraph(
    "The front end has moved from a single matrix multiply to a genuine "
    "convolutional network. Over the current development cycle it gained "
    "convolution-to-convolution chaining, a real MLIR-level im2col, a "
    "conv stem feeding a fully-connected head, and most recently the "
    "first executable non-matmul operator in the dialect &mdash; a "
    "windowed maximum &mdash; together with max-pooling interleaved into "
    "a convolution chain.", Body))

S.append(Paragraph("What is not yet true", H2))
S.append(Paragraph(
    "No number in this project has been measured on real hardware. Every "
    "shipped target description is marked <i>estimated</i>, the "
    "hardware backend stubs every entry point, and one cost figure "
    "inherited from the specification is internally inconsistent by a "
    "factor of 1000 and must be resolved against a real device before it "
    "appears in a paper. These are disclosed in the repository rather "
    "than implied away, and they are the substance of what remains "
    "before launch.", Body))

S.append(Spacer(1, 4))
S.append(panel([
    Paragraph("Single critical blocker", H3),
    Paragraph(
        "GitHub Actions minutes for the account are exhausted for the "
        "billing cycle and reset on <b>1 September 2026</b>. All nine CI "
        "jobs fail instantly with empty output. There is no code-side "
        "fix. Development has continued against a full local "
        "verification gate that reproduces every CI job, and the work is "
        "staged on an open pull request ready to land the moment CI "
        "recovers.", Body),
], bg=colors.HexColor("#FBF0DC"), edge=colors.HexColor("#E8D5A8")))

# ================================================================ 2. ARCH
S.append(PageBreak())
S.append(Paragraph("2&nbsp;&nbsp;Architecture", H1))
S.extend(rule())

S.append(Paragraph(
    "The stack is deliberately three separable pieces. The hardware is "
    "described by a portable YAML target file rather than hardcoded, so "
    "a vendor should be able to support cim-mlir by writing one file.",
    Lead))

S.append(Paragraph("2.1&nbsp;&nbsp;The three parts", H2))
rows = [[Paragraph(h, THead) for h in
         ("Component", "What it is", "Key artifacts")]]
rows += [
    [Paragraph("<b>Front end</b><br/><font size=7>cim_frontend</font>", TCell),
     Paragraph("Reads a standard ONNX model and emits the MLIR the "
               "pipeline consumes. Pure Python; no ONNX dependency below "
               "this layer. Refuses rather than guesses &mdash; dropping "
               "an unrecognised operator would emit a module that runs "
               "and computes a different function.", TCell),
     Paragraph("onnx_import.py, emit.py,<br/>im2col.py, analyze.py", TCellS)],
    [Paragraph("<b>Compiler</b><br/><font size=7>cim dialect</font>", TCell),
     Paragraph("An MLIR dialect of 10 operations plus 8 lowering passes "
               "that take standard ML graphs down to CIM primitives, "
               "then to real C ABI calls that go through MLIR's own "
               "LLVM pipeline into a linkable binary.", TCell),
     Paragraph("lib/Transforms/*,<br/>lib/Dialect/, lib/Placement/", TCellS)],
    [Paragraph("<b>Runtime</b><br/><font size=7>cimrt</font>", TCell),
     Paragraph("A thin C API of 20 entry points that executes the "
               "compiled artifact. Ships with a functional INT8 "
               "simulator; the real hardware backend is stubbed.", TCell),
     Paragraph("runtime/include/cimrt.h,<br/>runtime/src/simulator/", TCellS)],
    [Paragraph("<b>Benchmarks</b><br/><font size=7>cim-bench</font>", TCell),
     Paragraph("Reproducible workloads and an analytical cost model, so "
               "\"is CIM actually better here?\" is measurable rather "
               "than a marketing claim.", TCell),
     Paragraph("tools/cim-bench,<br/>bench/workloads/", TCellS)],
]
S.append(table(rows, [34 * mm, CONTENT_W - 34 * mm - 42 * mm, 42 * mm]))

S.append(Paragraph("2.2&nbsp;&nbsp;The lowering pipeline", H2))
S.append(Paragraph(
    "Eight passes, run in specification order. Passes 1&ndash;3 are the "
    "core of the project; pass 3 is the one it exists for.", Body))
rows = [[Paragraph(h, THead) for h in ("#", "Pass", "Responsibility", "State")]]
pipeline = [
    ("1", "cim-detect",
     "Annotates INT8 matmuls that have exactly one constant weight operand; "
     "reports why it declined every other candidate.", "ok", "Complete"),
    ("2", "cim-partition",
     "Lowers a candidate into per-tile program/mvm pairs with partial-sum "
     "reduction and explicit memory-space transfers.", "ok", "Complete"),
    ("3", "cim-placement",
     "Rewrites that IR from a Belady schedule: eliminates redundant weight "
     "programming, assigns real tile ids, hoists loop-invariant programming "
     "out of loops when provably safe.", "ok", "Complete"),
    ("4", "cim-schedule",
     "Inserts barriers conservatively over placed IR. Does not reorder or "
     "overlap in v0.1.", "ok", "Scoped"),
    ("5", "cim-insert-transfers",
     "Inserts host-to-near copies where an activation is not already in "
     "near space, hoisting above a loop when the source is invariant.",
     "ok", "Complete"),
    ("6", "cim-legalize-precision",
     "Inserts a requantize after every terminal accumulator and warns when "
     "the target clamps below 8 bits.", "warn", "Partial"),
    ("7", "cim-lower-to-target",
     "Converts dialect ops into real C ABI calls so the result can become a "
     "linkable binary. Covers a deliberately scoped straight-line slice.",
     "warn", "Scoped"),
    ("8", "cim-cost-report",
     "Walks the final placed IR and emits the project's publishable "
     "energy and latency numbers.", "ok", "Complete"),
]
for num, name, desc, kind, label in pipeline:
    rows.append([
        Paragraph(f"<b>{num}</b>", TCell),
        Paragraph(f"<font face='Courier' size=7.2>{name}</font>", TCell),
        Paragraph(desc, TCell),
        chip(label, kind),
    ])
S.append(table(rows, [7 * mm, 40 * mm, CONTENT_W - 7 * mm - 40 * mm - 21 * mm,
                      21 * mm]))

S.append(PageBreak())
S.append(Paragraph("2.3&nbsp;&nbsp;The dialect", H2))
S.append(Paragraph(
    "Ten operations. Nine were present from the first milestone; "
    "<font face='Courier' size=8>cim.reduce_max</font> is new this cycle "
    "and is the first operation in the dialect that is not part of a "
    "matrix multiply.", Body))

ops = [
    ("cim.device_open", "Open a target device", "both"),
    ("cim.tile_alloc", "Reserve a physical tile", "both"),
    ("cim.tile_free", "Release a tile", "both"),
    ("cim.program", "Make a weight block resident (the expensive, "
                    "asymmetric operation)", "both"),
    ("cim.mvm", "Matrix multiply against a resident tile", "both"),
    ("cim.reduce_partial", "Elementwise integer sum of N buffers "
                           "(partial sums, and bias add)", "both"),
    ("cim.reduce_max", "Elementwise SIGNED integer maximum of N buffers "
                       "(a pooling window)", "interp"),
    ("cim.requantize", "Round, offset and clamp an accumulator", "both"),
    ("cim.copy", "Explicit memory-space transfer", "both"),
    ("cim.barrier", "Ordering", "both"),
]
rows = [[Paragraph(h, THead) for h in
         ("Operation", "Purpose", "Interpreter", "Compiled")]]
for name, desc, where in ops:
    rows.append([
        Paragraph(f"<font face='Courier' size=8>{name}</font>", TCell),
        Paragraph(desc, TCell),
        chip("Yes", "ok"),
        chip("Yes", "ok") if where == "both" else chip("Missing", "stop"),
    ])
S.append(table(rows, [38 * mm, CONTENT_W - 38 * mm - 44 * mm,
                      22 * mm, 22 * mm]))
S.append(Paragraph(
    "The single red cell is the most concrete open gap in the compiler "
    "and is addressed in section 5.", Small))

S.append(Paragraph("2.4&nbsp;&nbsp;Runtime ABI and target description", H2))
S.append(Paragraph(
    "The runtime is 20 C entry points. Every operation the runtime can "
    "execute is charged against the target's cost table &mdash; a "
    "project invariant, enforced by a differential test asserting that "
    "the statically predicted cost equals the cost actually measured at "
    "runtime. A new runtime call with no static counterpart is treated "
    "as a silent regression, not a missing feature.", Body))
S.append(Paragraph(
    "<font face='Courier' size=8>cimrt_open, cimrt_close, cimrt_query, "
    "cimrt_alloc, cimrt_free, cimrt_map, cimrt_write, cimrt_read, "
    "cimrt_copy, cimrt_copy_range, cimrt_program, cimrt_mvm, "
    "cimrt_requantize, cimrt_reduce_add, cimrt_reduce_add_inplace, "
    "cimrt_reduce_max, cimrt_barrier, cimrt_profile_start, "
    "cimrt_profile_stop, cimrt_status_string</font>", TCellS))
S.append(Spacer(1, 5))
S.append(Paragraph(
    "Twelve target description files exist &mdash; three shipped "
    "(an Erbium-8T analog part, a generic digital CIM part, and a "
    "UPMEM-like near-memory part) and nine test fixtures that vary one "
    "capability each. The reader is hand-rolled, its supported YAML "
    "subset is written down rather than inherited, and it is checked "
    "differentially against PyYAML plus an independently written schema.",
    Body))

# ================================================================ 3. BUILT
S.append(PageBreak())
S.append(Paragraph("3&nbsp;&nbsp;What has been built", H1))
S.extend(rule())

S.append(Paragraph("3.1&nbsp;&nbsp;Milestone status", H2))
rows = [[Paragraph(h, THead) for h in ("Milestone", "Scope", "State")]]
ms = [
    ("M0", "Environment and orientation", "ok", "Complete"),
    ("M1", "Dialect skeleton", "ok", "Complete"),
    ("M2", "Functional correctness &mdash; a .onnx file compiles and runs, "
           "checked for exact integer equality against ONNX's own "
           "reference implementation", "ok", "Complete"),
    ("M3", "The placement pass &mdash; Belady eviction, loop-invariant "
           "hoisting, and a steady-state pin-and-stream schedule proved "
           "optimal by exhaustive search", "ok", "Complete"),
    ("M4", "Second target and generalization &mdash; most items closed, "
           "including full cost-model integrity, batching, convolution, "
           "and real-model analysis; calibration remains", "warn", "Mostly"),
    ("M5", "Community and real hardware &mdash; hardware bring-up, "
           "upstream contribution, conference talk", "open", "Not started"),
    ("M6", "Decision point", "open", "Not started"),
]
for name, scope, kind, label in ms:
    rows.append([Paragraph(f"<b>{name}</b>", TCell), Paragraph(scope, TCell),
                 chip(label, kind)])
S.append(table(rows, [17 * mm, CONTENT_W - 17 * mm - 23 * mm, 23 * mm]))

S.append(Paragraph("3.2&nbsp;&nbsp;Delivered this development cycle", H2))
S.append(Paragraph(
    "Six commits on the open pull request, taking the front end from a "
    "single convolution to a genuine convolutional network.", Body))
rows = [[Paragraph(h, THead) for h in ("Commit", "Capability", "Why it was hard")]]
work = [
    ("Conv &rarr; matmul", "A convolution stem feeding a "
     "fully-connected head.",
     "ONNX's own matmul kernel does N-D broadcast rather than requiring "
     "2-D operands, so wiring a convolution's raw output straight in "
     "silently contracts the wrong axes instead of erroring. Fixed by "
     "requiring an explicit reshape bridge in the graph."),
    ("Interpreter reshape", "Execute shape expansion and collapse.",
     "Neither operation was executable at all. Needed before any "
     "MLIR-level im2col could reinterpret a buffer between matmul and "
     "gather shapes."),
    ("Conv &rarr; conv", "Real convolution-to-convolution chaining, with "
     "a genuine MLIR-level im2col.",
     "The interpreter cannot evaluate integer arithmetic or conditionals, "
     "ruling out both obvious im2col designs. Solved by unrolling over "
     "kernel taps as static strided views. The channel-last weight "
     "flatten for interior layers is a silent-wrong-answer hazard, "
     "mutation-tested accordingly."),
    ("Conv stem &rarr; head", "The realistic full CNN shape.",
     "Built by composing the two mechanisms above rather than "
     "reimplementing either."),
    ("reduce_max", "The first executable non-matmul primitive.",
     "Max on INT8 compares SIGNED, but the sibling addition is "
     "deliberately sign-agnostic &mdash; a copy-paste would compare raw "
     "bytes, compile, run, and quietly return the wrong answer. Landed "
     "with full cost accounting, which a first draft of the plan missed "
     "entirely."),
    ("MaxPool chaining", "Pooling interleaved into a convolution chain.",
     "The reference oracle cannot evaluate integer pooling at unit "
     "stride at all, so that case is refused on verifiability grounds "
     "and an independent hand-written oracle was added alongside."),
]
for name, cap, why in work:
    rows.append([Paragraph(f"<b>{name}</b>", TCell), Paragraph(cap, TCell),
                 Paragraph(why, TCellS)])
S.append(table(rows, [26 * mm, 36 * mm, CONTENT_W - 62 * mm]))

_cov = [Paragraph("3.3&nbsp;&nbsp;Model coverage today", H2),
        Paragraph(
            "Seven distinct graph shapes are imported and compiled. "
            "Anything outside them is refused with a stated reason rather "
            "than silently approximated.", Body)]
_cov.extend(bullets([
    "A single INT8 matrix multiply, and chains of them.",
    "A single 2-D convolution, including per-channel scale, a real bias, "
    "asymmetric zero points, dilation, stride and padding.",
    "A convolution feeding one or more matmul layers.",
    "Chained convolutions, connected directly.",
    "A convolution stem feeding a fully-connected head.",
    "Chained convolutions with max-pooling interleaved between layers.",
    "Any graph at all, for placement and cost <i>analysis</i> without "
    "execution &mdash; a permissive walker classifies every node and "
    "never aborts, and discloses what it skipped.",
]))
S.append(KeepTogether(_cov))

# ================================================================ 4. VERIF
S.append(Spacer(1, 6))
S.append(Paragraph("4&nbsp;&nbsp;Verification posture", H1))
S.extend(rule())
S.append(Paragraph(
    "The verification discipline is the project's main defence against "
    "its central failure mode: not a crash, but structurally valid code "
    "that computes a confidently wrong number.", Lead))

rows = [[Paragraph(h, THead) for h in ("Suite", "Count", "What it covers")]]
suites = [
    ("ctest", "20", "End-to-end real compiled binaries: each numerically "
     "correct case paired with a deliberately wrong one that must trap."),
    ("cim-unit-tests", "134", "Runtime ABI, target parser, placement engine, "
     "cost model, workload reader."),
    ("cim-mlir-tests", "43", "In-process pipeline, precision legalization, "
     "and static-versus-runtime cost differentials."),
    ("pytest", "268", "The ONNX front end, driven against the real shipped "
     "binaries, plus differential fuzzing of the YAML reader against PyYAML."),
    ("lit / FileCheck", "40 files", "Dialect round-trip, verifier rejection "
     "cases, per-pass structure, and full-pipeline composition."),
]
for name, count, cov in suites:
    rows.append([Paragraph(f"<b>{name}</b>", TCell),
                 Paragraph(count, TCell), Paragraph(cov, TCell)])
S.append(table(rows, [32 * mm, 18 * mm, CONTENT_W - 50 * mm]))

S.append(Paragraph("Practices applied throughout", H2))
S.extend(bullets([
    "<b>Probe before code.</b> Every new IR shape is hand-verified "
    "through the real compiler and runtime binaries before any emission "
    "code is written to produce it.",
    "<b>Independent oracles.</b> Correctness is checked against "
    "implementations written by other people for other reasons &mdash; "
    "ONNX's reference implementation, onnxruntime, NumPy, PyYAML &mdash; "
    "not against this project's own second opinion.",
    "<b>Mutation testing.</b> Every new guard is deliberately broken, "
    "confirmed to fail for the right reason, then reverted and "
    "confirmed green. A test that has never been seen to fail is not "
    "treated as evidence.",
    "<b>Refuse rather than approximate.</b> Unsupported input is "
    "rejected with a stated reason. Several restrictions exist because "
    "the <i>oracle</i> cannot evaluate a case, not because the compiler "
    "cannot compile it &mdash; and say so explicitly.",
    "<b>Cost integrity as an invariant.</b> Statically predicted cost "
    "and runtime-measured cost are asserted equal by test.",
]))

S.append(Spacer(1, 4))
S.append(panel([
    Paragraph("A defect found in this project's own primary oracle", H3),
    Paragraph(
        "ONNX's reference implementation cannot evaluate an integer "
        "max-pool at unit stride at all: it pads with a floating-point "
        "sentinel that cannot be placed into an integer array, and "
        "raises rather than returning a wrong number. This was "
        "discovered by probing, recorded in the repository so it is not "
        "rediscovered, and is the stated reason that one configuration "
        "is refused &mdash; a gap in what can be verified, not in what "
        "can be compiled.", Body),
]))

# ================================================================ 5. OPEN
S.append(PageBreak())
S.append(Paragraph("5&nbsp;&nbsp;What is still open", H1))
S.extend(rule())

S.append(Paragraph("5.1&nbsp;&nbsp;Blockers", H2))
rows = [[Paragraph(h, THead) for h in ("Item", "Detail", "Resolution")]]
rows += [[
    Paragraph("<b>CI unavailable</b>", TCell),
    Paragraph("All nine CI jobs fail instantly with empty output. GitHub "
              "Actions minutes are exhausted for the billing cycle. No "
              "code-side fix exists. Mitigated by a full local gate that "
              "reproduces every job, including sanitiser builds.", TCell),
    Paragraph("Resets automatically on 1 September 2026.", TCellS)]]
S.append(table(rows, [30 * mm, CONTENT_W - 30 * mm - 38 * mm, 38 * mm]))

S.append(Paragraph("5.2&nbsp;&nbsp;Capability gaps", H2))
rows = [[Paragraph(h, THead) for h in
         ("Gap", "Why it matters", "Size")]]
gaps = [
    ("No compiled lowering for the pooling primitive",
     "Every other dialect operation works on both the interpreter and the "
     "compiled path. A pooling network therefore runs under the "
     "interpreter but cannot be built into a real-target binary. A plan "
     "is written and ready to execute.", "Small"),
    ("No calibration",
     "Requantization scales are fixed rather than derived per layer. The "
     "arithmetic is already general; what is missing is a calibration "
     "step to choose real values from data.", "Medium"),
    ("Activation and shape operators",
     "Relu needs an asymmetric clamp the current requantize cannot "
     "express. Average-pooling needs a division primitive that exists "
     "nowhere in the stack. Concatenation and residual addition need a "
     "loader for branching, multi-producer graphs; every loader to date "
     "assumes a single linear chain.", "Large"),
    ("Control flow in the compiled path",
     "The compiled lowering covers straight-line code and simple loops. "
     "Conditionals and loop-carried values are still refused.", "Medium"),
    ("Higher-rank strided slices in the compiled path",
     "The compiled path moves one contiguous byte range; a genuinely "
     "strided gather has no runtime equivalent yet.", "Medium"),
]
for gap, why, size in gaps:
    rows.append([Paragraph(f"<b>{gap}</b>", TCell), Paragraph(why, TCell),
                 chip(size, "warn" if size != "Large" else "open")])
S.append(table(rows, [40 * mm, CONTENT_W - 40 * mm - 20 * mm, 20 * mm]))

S.append(Paragraph("5.3&nbsp;&nbsp;Credibility gaps &mdash; the ones that "
                   "matter most for launch", H2))
rows = [[Paragraph(h, THead) for h in ("Item", "Detail")]]
cred = [
    ("No measured hardware numbers",
     "Every shipped target file is marked <i>estimated</i>. The runtime's "
     "hardware backend stubs every entry point and returns a "
     "no-device error. All published energy and latency figures are "
     "therefore modelled, not observed &mdash; which the repository "
     "states plainly, but which limits what can be claimed externally."),
    ("A 1000x inconsistency inherited from the specification",
     "The specification's own worked example is internally inconsistent "
     "by three orders of magnitude on weight-programming energy. Which "
     "reading is correct decides whether installation cost is a rounding "
     "error or a material fraction of the per-inference budget &mdash; "
     "which is the crux of the project's central argument. It must be "
     "settled against real hardware before it appears in a paper."),
    ("Placeholder cost entries",
     "The newest cost-table entry has no hardware measurement behind it "
     "on any modelled target. Each file marks it as a placeholder and "
     "gives it a deliberately distinct value so a miswiring shows up as "
     "a wrong number rather than a plausible one."),
]
for item, detail in cred:
    rows.append([Paragraph(f"<b>{item}</b>", TCell), Paragraph(detail, TCell)])
S.append(table(rows, [46 * mm, CONTENT_W - 46 * mm]))

# ================================================================ 6. LAUNCH
S.append(PageBreak())
S.append(Paragraph("6&nbsp;&nbsp;Launch readiness", H1))
S.extend(rule())
S.append(Paragraph(
    "Assessed against a public v0.1 release: the project is usable by "
    "someone who is not its author, its claims survive scrutiny, and "
    "there is a reason for a stranger to care.", Lead))

rows = [[Paragraph(h, THead) for h in
         ("Dimension", "Assessment", "State")]]
dims = [
    ("Core capability",
     "The compiler does what it claims. A real quantized CNN compiles and "
     "runs, and the placement engine &mdash; the actual intellectual "
     "contribution &mdash; is implemented and proved optimal on the shape "
     "it emits.", "ok", "Ready"),
    ("Correctness evidence",
     "465 automated tests, independent oracles, mutation testing, and a "
     "static-versus-runtime cost invariant. Stronger than typical for a "
     "project at this stage.", "ok", "Ready"),
    ("Documentation",
     "A hardware abstraction model, a dialect reference, a target-format "
     "specification, a front-end guide, and a single-page website. The "
     "roadmap records not just what was done but what was tried and "
     "rejected, and why.", "ok", "Ready"),
    ("Packaging",
     "The front end installs as a Python package; the compiler builds "
     "with CMake against MLIR 18. A dedicated CI job guards packaging.",
     "ok", "Ready"),
    ("Retargetability",
     "Three shipped target descriptions across three hardware classes, "
     "a documented YAML subset, and a differential test against an "
     "independent parser. A vendor can add a target by writing one file.",
     "ok", "Ready"),
    ("Continuous integration",
     "Nine jobs covering sanitisers, valgrind, coverage, static analysis "
     "and packaging &mdash; all currently unavailable for reasons outside "
     "the codebase.", "stop", "Blocked"),
    ("Measured results",
     "No number has been observed on real hardware, and one inherited "
     "figure is inconsistent by 1000x. This is the weakest dimension and "
     "the one most likely to be challenged.", "stop", "Not ready"),
    ("Operator coverage",
     "Sufficient for a small classification network. Activations, "
     "average-pooling and branching topologies are absent, so most "
     "real-world models will still be refused.", "warn", "Partial"),
    ("External engagement",
     "No upstream contribution, no talk submitted, no external users.",
     "open", "Not started"),
]
for dim, assess, kind, label in dims:
    rows.append([Paragraph(f"<b>{dim}</b>", TCell), Paragraph(assess, TCell),
                 chip(label, kind)])
S.append(table(rows, [34 * mm, CONTENT_W - 34 * mm - 23 * mm, 23 * mm]))

S.append(Spacer(1, 5))
S.append(panel([
    Paragraph("Assessment", H3),
    Paragraph(
        "The engineering is in better shape than the evidence base. The "
        "compiler, the runtime, the target abstraction and the "
        "verification discipline are all genuinely ready for other "
        "people to look at. What is not ready is the empirical claim: "
        "no figure has been observed on hardware, and the project's "
        "central argument about weight-programming amortisation rests on "
        "a specification figure that is internally inconsistent.", Body),
    Paragraph(
        "The most efficient next effort is therefore not more compiler "
        "features. It is landing the outstanding work, closing the one "
        "remaining dialect asymmetry, and then turning to measurement "
        "&mdash; because a smaller, honestly measured claim will survive "
        "scrutiny that a broader modelled one will not.", Body),
]))

S.append(PageBreak())
S.append(Paragraph("6.1&nbsp;&nbsp;Critical path to launch", H2))
rows = [[Paragraph(h, THead) for h in ("Step", "Action", "Depends on")]]
path = [
    ("1", "Land the open pull request once CI recovers, confirming all "
          "nine jobs green on the four-commit convolution series and the "
          "two pooling commits.", "CI reset, 1 Sep 2026"),
    ("2", "Close the compiled-path gap for the pooling primitive, so "
          "every dialect operation works on both execution paths. Plan "
          "written and approved.", "Nothing"),
    ("3", "Resolve the 1000x energy inconsistency, or state a defensible "
          "reading and mark it clearly everywhere it is used.",
     "Hardware access or a specification decision"),
    ("4", "Obtain at least one measured data point on real hardware, or "
          "reframe every published figure explicitly as modelled.",
     "Hardware access"),
    ("5", "Decide the launch scope for operator coverage: ship as a "
          "convolutional-network compiler with stated limits, or first "
          "add activations and branching topologies.", "A scope decision"),
    ("6", "External engagement &mdash; upstream contribution and a talk "
          "submission.", "Steps 1 to 4"),
]
for num, action, dep in path:
    rows.append([Paragraph(f"<b>{num}</b>", TCell), Paragraph(action, TCell),
                 Paragraph(dep, TCellS)])
S.append(table(rows, [12 * mm, CONTENT_W - 12 * mm - 44 * mm, 44 * mm]))

S.append(Spacer(1, 7))
S.append(Paragraph(
    "Steps 1 and 2 are entirely within the project's own control and "
    "account for the whole of the remaining engineering work. Steps 3 "
    "and 4 are the ones that decide how strong a claim the launch can "
    "make, and both depend on access to real hardware rather than on "
    "further development.", Body))

doc.build(S)
print("wrote", OUT)
