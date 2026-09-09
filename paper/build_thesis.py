#!/usr/bin/env python
"""인제대학교 대학원 학위논문 생성기 → paper/thesis.docx

규정: CLAUDE.md '학위논문 작성 규정' 절. 19x26cm, 신명조 10pt, 줄간격 200%,
전문 면번호 로마 소문자 / 본문 아라비아, 항목번호 예시B(I. 1. A. 1)).
미확정 정보는 PLACE 딕셔너리의 ○○○ 유지.

본문 내용은 paper/thesis_text.py(초록·Ⅰ~Ⅵ·부록·참고문헌 dict)에서 가져온다.
마크업：{cite:키} / {cite:키1,키2} → 어깨번호(최초 인용 순서), {Fig:이름} / {Table:이름} → 번호,
"[Fig. 제목 — figures/이름]" / "[Table. 제목 — tables/이름.csv]" → 1개/1면 삽입,
"[Verbatim — 경로]" → 파일 원문(고정폭), "[Note. 내용]" → 직전 표/그림/원문 아래의 작은 설명(같은 면).
목차·표목차·그림목차의 면번호는 2회 빌드로 얻는다
(1차 docx → LibreOffice PDF → pdftotext로 제목의 면을 찾아 2차 빌드의 PAGEREF 캐시에 기록).

실행:  uv run python paper/build_thesis.py
"""

import csv
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

PLACE = {
    "degree_kr": "공학석사",
    # 제목은 2축(알고리즘 비교 + LLM 보상 코칭)을 반영한 제안본 — 지도교수 확정 전
    "title_kr": "사족보행 로봇의 험지 보행을 위한\n강화학습 알고리즘 비교 및\n대형 언어 모델 기반 보상 코칭에 관한 연구",
    "title_en": "A Comparative Study of Reinforcement Learning\nAlgorithms and Large-Language-Model-Based\nReward Coaching for Rough-Terrain\nQuadruped Locomotion",
    "school_kr": "인제대학교 대학원",
    "school_en": "Graduate School, Inje University",
    "dept_kr": "○○학과 ○○학 전공",
    "dept_en": "Department of ○○○",
    "author_kr": "○　　○　　○",
    "author_en": "○○○ ○○○",
    "advisor_kr": "○　　○　　○",
    "advisor_en": "Prof. ○ ○ ○",
    "date_kr": "○○○○년 12월(또는 6월)",
    "date_en": "Dec.(또는 Jun.) 20XX",
}

BODY_FONT = "신명조"
MONO_FONT = "Courier New"
PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
sys.path.insert(0, str(PAPER))
import thesis_text as T  # noqa: E402

FIG_WIDTH_CM = 13.0  # 19 cm 페이지 − 좌 3.5 − 우 2.5
# 표/그림 뒤의 글을 같은 면에 이어 쓸지(True) 항상 새 면에서 시작할지(False).
# 어느 쪽이든 표/그림은 자기 면에서 시작하고 한 면에 표/그림이 둘 이상 놓이지 않는다.
TEXT_AFTER_EMBED = True

# 본문 자리표시자 "[Fig. 제목 — figures/이름]" / "[Table. 제목 — tables/이름.csv]".
# 번호는 등장 순서대로 build()가 매긴다(아라비아 일련번호, 표·그림 별도).
EMBED_RE = re.compile(r"^\[(Fig|Table)\. (.+?) — (figures|tables)/(\S+?)\]$")
# 본문 안의 참조 "{Fig:이름}" / "{Table:이름}" → "Fig. N" / "Table N" (번호는 위 자리표시자의 등장 순서)
REF_RE = re.compile(r"\{(Fig|Table):([^}]+)\}")
# 부록의 파일 원문 "[Verbatim — configs/…/system.md]"
VERB_RE = re.compile(r"^\[Verbatim — (\S+)\]$")
# 표/그림/원문 바로 아래의 설명 "[Note. 내용]" — 새 면을 만들지 않고 8.5pt로 붙인다
NOTE_RE = re.compile(r"^\[Note\. (.+)\]$", re.S)
BODY_SIZE = 10.0
# 어깨번호 인용 "{cite:키}" / "{cite:키1,키2}" → 최초 인용 순서의 번호(위첨자)
CITE_RE = re.compile(r"\{cite:([^}]+)\}")
CITE_NUMBERS: dict[str, int] = {}

# 표 셀의 내부 실행 명칭 → 논문 표기
CELL_RENAME = {
    "LLM-PPO (v7p, evolve4)": "LLM-PPO (v3)",
    "LLM (v7p)": "LLM-PPO (v3)",
    "Rough (medium), naive reward": "Naive (rough medium)",
}
CELL_EXACT = {
    "flat": "Flat",
    "stairs": "Stairs",
    "rough": "Rough",
    "ppo": "PPO",
    "trpo": "TRPO",
    "a3c": "A3C",
    "sac": "SAC",
    "td3": "TD3",
    "ddpg": "DDPG",
    "": "—",
    "nan": "—",
}

# 표 하단 약자 풀이(규정：약자는 표 하단에 풀이)
TABLE_NOTES = {
    "table1_final_performance": "SD, standard deviation of velocity over seeds; Attitude RMS, root-mean-square of roll and pitch; "
    "Falls, falls per minute; Path Efficiency, straight-line distance / travelled distance. n = 3 seeds per cell.",
    "table2_statistics": "F and p from one-way ANOVA over algorithms (n = 3 seeds each); Eta Squared, effect size η²; "
    "Tukey HSD, Tukey honestly significant difference post-hoc test (α = 0.05).",
    "table3_learning_efficiency": "Steps to 0.8 m/s, environment steps (millions) until the evaluated velocity first reached 0.8 m/s; "
    "Censored, seeds that never reached the threshold / seeds; Normalized AUC, area under the "
    "velocity learning curve divided by the budget.",
    "table4_difficulty_scaling": "SD, standard deviation over seeds; Falls, falls per minute. Easy / Medium / Hard follow the "
    "terrain difficulty levels of the terrain configuration files.",
    "table5_anymal_transfer": "SD, standard deviation over seeds; Falls, falls per minute; Cost of Transport, E / (m·g·d).",
    "coach_versions_thesis": "v1–v3, the three coach versions reported in this thesis; Internal Revisions, the numbered revisions (#, Ⅲ.4.F) collapsed into each version; "
    "n Pairs, (setting, seed) pairs against the no-coach PPO control of the same batch; ΔJ, paired "
    "difference LLM-PPO − PPO of the objective J = success rate + 0.5 × forward velocity; CI, t-based "
    "95 % confidence interval; d, paired Cohen's d; Naive Escape, runs that left the success-rate floor "
    "of the naive reward / runs. The last column summarises the six meta-LLM candidates, each judged "
    "against its incumbent (one-sided paired t).",
    "table6_coach_per_setting_evolve4": "Values are mean ± SD over n seeds of the final fixed evaluation (256 environments); Paired Δ, "
    "LLM-PPO − PPO per seed; p, two-sided paired t-test; d, paired Cohen's d; CoT, cost of transport "
    "E / (m·g·d); Steps to Success 0.5, environment steps (millions) until the evaluated success rate "
    "first reached 0.5 (n counts seeds that reached it under both conditions).",
    "table7_coach_pooled_evolve4": "Nine (setting, seed) pairs pooled over the three settings; 95 % CI, t-based confidence interval "
    "of the paired difference; p (Wilcoxon), Wilcoxon signed-rank test; d, paired Cohen's d.",
    "coach_intervention_stats": "Int., settled coach proposals (kept or rolled back) per run; Kept, share of settled proposals "
    "not rolled back; RB, rollbacks; RS, best-snapshot restores; Clip, proposals whose applied values "
    "differ from the proposed values because of range or step limits; Lever ↓ / ↑, decreases / "
    "increases of the commanded speed per run; Final Cmd, commanded speed at the end of training "
    "(mean ± SD); Steps to 0.5, median environment steps (millions) until the evaluated success rate "
    "first reached 0.5 (runs that reached it / runs), LLM-PPO / PPO of the same batch.",
    "coach_naive_recipe": "First-Report Recipe, whether the coach's first proposal (after 2 × 10⁶ steps) lowered the "
    "commanded speed target_ms and weakened the energy penalty; Escaped Floor, runs whose evaluated "
    "success rate left 0 before the end of training; Fisher p, one-sided Fisher's exact test of the "
    "escape rate between the two recipe groups (not defined when a group is empty).",
    "table8_ablation_per_setting": "Final objective J (mean ± SD over n seeds) of the no-coach PPO and the three coaches sharing the "
    "same decision layer; Random, random coach; Hill-climb, (1+1) hill-climbing coach; LLM-PPO (v3), "
    "the final LLM coach.",
    "table8_ablation_contrasts": "ΔJ, paired difference of the final objective J between the two conditions of each contrast; SD, "
    "standard deviation of the paired differences; 95 % CI, t-based confidence interval; p (Wilcoxon) "
    "is not defined for n = 3; d, paired Cohen's d.",
    "table8_ablation_escapes": "Escaped, runs whose evaluated success rate left 0 under the naive reward; Final J, objective J of "
    "each seed at the end of training; p (Fisher), one-sided Fisher's exact test against the LLM coach.",
    "appA_coach_tunables": "Scale, linear or log; Per-Intervention Limit, maximum change of one proposal (±30 % of the current "
    "value for linear parameters, ×/÷3 for log parameters); Curriculum Lever, parameter under the "
    "release invariant of the decision layer. Bounds never straddle zero (sign lock).",
    "appB_hyperparameters": "Values as configured in configs/algorithm/*.yaml; —, not applicable. Updates / step of the "
    "off-policy algorithms was overridden to 16 in every training run. A3C was not trained "
    "(asynchronous CPU-worker design incompatible with the GPU-vectorised backend).",
    "appC_coach_runs": "Final fixed evaluation (256 environments) at the end of training; CoT, cost of transport "
    "E / (m·g·d); Steps to Success 0.5, environment steps (millions) until the evaluated success rate "
    "first reached 0.5 (—, never); Interventions, settled coach proposals (kept or rolled back); "
    "Escaped, evaluated success rate exceeded 0.1 at least once during training.",
    "appD_coach_history": "Revision, internal revision number (#, Ⅲ.4.F; p, playbook variant) — only #5 and #7p became thesis versions v2 and v3; Gen., evolution generation and proposer (LLM, meta-LLM; hand, hand-written); Change, revision "
    "relative to the parent version named first, which is also the paired comparison target unless "
    "the Note says otherwise (none, the no-coach PPO control); n, (setting, seed) pairs; ΔJ, paired difference of the final "
    "objective J; p, one-sided paired t-test (W, Wilcoxon signed-rank); d, paired Cohen's d. "
    "Revisions #1–#4 (GPT-5.4 batches) are summarised as v1 in the main text.",
    "run_history": "Noise, exploration noise of DDPG (ou, Ornstein–Uhlenbeck; parameter_space, parameter-space "
    "noise); Budget, environment steps; Wall Time, minutes on one RTX 4080; Attitude RMS, "
    "root-mean-square of roll and pitch (rad); Falls, falls per minute. Metrics from the original "
    "Part 1 evaluation protocol (20 episodes).",
}


def set_font(run, size=BODY_SIZE, bold=False, font=BODY_FONT):
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    r = run._element.get_or_add_rPr()
    rf = r.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        r.append(rf)
    rf.set(qn("w:eastAsia"), font)


def para(
    doc,
    text="",
    size=BODY_SIZE,
    bold=False,
    align=WD_ALIGN_PARAGRAPH.JUSTIFY,
    spacing=2.0,
    before=0,
    after=0,
    indent=None,
    font=BODY_FONT,
    new_page=False,
):
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.page_break_before = new_page
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = spacing
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    if indent:
        pf.first_line_indent = Cm(indent)
    for i, chunk in enumerate(text.split("\n")):
        if i:
            p.add_run().add_break()
        for j, piece in enumerate(CITE_RE.split(chunk)):
            if j % 2:  # 인용 키 목록 → 위첨자 "3)" / "3,5)" / "3-6)"
                r = p.add_run(cite_label(piece))
                set_font(r, size=size, bold=bold, font=font)
                r.font.superscript = True
            elif piece:
                set_font(p.add_run(piece), size=size, bold=bold, font=font)
    return p


def cite_label(keys: str) -> str:
    nums = sorted({CITE_NUMBERS[k.strip()] for k in keys.split(",")})  # 미등록 키는 KeyError
    parts, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        parts.append(f"{nums[i]}-{nums[j]}" if j - i >= 2 else ",".join(map(str, nums[i : j + 1])))
        i = j + 1
    return ",".join(parts) + ")"


def number_cites(sections) -> dict[str, int]:
    """본문 최초 인용 순서로 참고문헌 번호를 매긴다(초록 제외)."""
    order: dict[str, int] = {}

    def scan(x):
        if isinstance(x, str):
            for m in CITE_RE.finditer(x):
                for k in m.group(1).split(","):
                    order.setdefault(k.strip(), len(order) + 1)
        elif isinstance(x, (list, tuple)):
            for y in x:
                scan(y)

    scan(sections)
    missing = [k for k in order if k not in T.REFERENCES]
    if missing:
        raise KeyError(f"REFERENCES에 없는 인용 키: {missing}")
    unused = [k for k in T.REFERENCES if k not in order]
    if unused:
        print(f"warning: 인용되지 않아 생략된 참고문헌 {len(unused)}건: {unused}")
    return order


def bookmark(p, name, bid):
    s = OxmlElement("w:bookmarkStart")
    s.set(qn("w:id"), str(bid))
    s.set(qn("w:name"), name)
    e = OxmlElement("w:bookmarkEnd")
    e.set(qn("w:id"), str(bid))
    p._p.get_or_add_pPr().addnext(s)
    p._p.append(e)


def field(p, instr, cached, size=11):
    """복합 필드(PAGEREF 등)：워드에서 갱신 가능, 캐시값은 2차 빌드에서 채운다."""
    for kind in ("begin", None, "separate", cached, "end"):
        r = p.add_run()
        if kind is None:
            it = OxmlElement("w:instrText")
            it.set(qn("xml:space"), "preserve")
            it.text = f" {instr} "
            r._r.append(it)
        elif kind in ("begin", "separate", "end"):
            fc = OxmlElement("w:fldChar")
            fc.set(qn("w:fldCharType"), kind)
            r._r.append(fc)
        else:
            r.text = kind
            set_font(r, size=size)


def center(doc, text, size, bold=False, before=0, after=0, spacing=1.5):
    return para(
        doc,
        text,
        size=size,
        bold=bold,
        align=WD_ALIGN_PARAGRAPH.CENTER,
        spacing=spacing,
        before=before,
        after=after,
    )


def setup_section(sec, numbering=None, start=None):
    """19x26cm, 여백 상3/좌3.5/우2.5/하2.5, 면번호 형식 지정."""
    sec.page_width, sec.page_height = Cm(19), Cm(26)
    sec.top_margin, sec.bottom_margin = Cm(3), Cm(2.5)
    sec.left_margin, sec.right_margin = Cm(3.5), Cm(2.5)
    sec.footer_distance = Cm(1.5)
    if numbering:
        sp = sec._sectPr
        el = sp.find(qn("w:pgNumType"))
        if el is None:
            el = OxmlElement("w:pgNumType")
            sp.append(el)
        el.set(qn("w:fmt"), numbering)
        if start is not None:
            el.set(qn("w:start"), str(start))


def footer_page_number(sec, show=True):
    sec.footer.is_linked_to_previous = False
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in list(fp.runs):
        r._element.getparent().remove(r._element)
    if not show:
        return
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    fp._p.append(fld)


def heading(doc, text, level=1, new_page=False):
    sizes = {1: 14, 2: 12, 3: 11}
    p = para(
        doc,
        text,
        size=sizes.get(level, 11),
        bold=True,
        align=WD_ALIGN_PARAGRAPH.LEFT if level > 1 else WD_ALIGN_PARAGRAPH.CENTER,
        before=18 if level == 1 else 12,
        after=8,
        new_page=new_page,
    )
    p.paragraph_format.keep_with_next = True
    return p


def body(doc, text, new_page=False):
    return para(doc, text, indent=0.75, new_page=new_page)


def caption(doc, text, before=0, after=0, new_page=False):
    return para(
        doc,
        text,
        size=10,
        align=WD_ALIGN_PARAGRAPH.CENTER,
        spacing=1.5,
        before=before,
        after=after,
        new_page=new_page,
    )


def embed_figure(doc, num, title, name, new_page=True):
    """그림 1개/1면：그림(폭 13 cm) 위, 영문 제목 아래."""
    png = PAPER / "figures" / f"{name}.png"
    p = para(doc, align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.0, new_page=new_page)
    p.paragraph_format.keep_with_next = True  # 그림과 제목을 같은 면에
    p.add_run().add_picture(str(png), width=Cm(FIG_WIDTH_CM))
    return caption(doc, f"Fig. {num}. {title}", before=6)


def _cell_text(v):
    v = v.strip()
    for a, b in CELL_RENAME.items():
        v = v.replace(a, b)
    return CELL_EXACT.get(v.lower(), v)


def _col_widths(rows, ncol, size):
    """열 폭(cm)：줄바꿈이 안 되는 최장 토큰(머리글은 8자 이하면 통째)에 셀 여백을 더한 폭을
    보장하고, 남는 폭은 열별 최장 셀 길이(28자 절단)에 비례하여 배분한다. 폭이 모자라면 비례 축소."""
    char_cm = size * 0.55 * 0.0353  # 평균 글자 폭 ≈ 0.55 em
    need, want = [], []
    for j in range(ncol):
        cells = [r[j] if j < len(r) else "" for r in rows]
        head = cells[0] if len(cells[0]) <= 8 else max(cells[0].split(), key=len)
        # "평균 ± 표준편차", "A − B" 대비 이름은 한 줄에 유지
        tokens = [head] + [
            w
            for c in cells[1:]
            for w in ([c] if re.search(r" [±−] ", c) and len(c) <= 24 else c.split())
        ]
        need.append(max(len(t) for t in tokens) * char_cm + 0.25)  # 0.25 cm：좌우 셀 여백
        want.append(min(60, max(len(c) for c in cells)) * char_cm)
    total = sum(need)
    if total >= FIG_WIDTH_CM:
        return [Cm(FIG_WIDTH_CM * n / total) for n in need]
    slack = [max(w - n, 0.0) for n, w in zip(need, want, strict=True)]
    extra = FIG_WIDTH_CM - total
    share = sum(slack) or 1.0
    return [Cm(n + extra * sl / share) for n, sl in zip(need, slack, strict=True)]


def embed_table(doc, num, title, name, new_page=True):
    """표 1개/1면：영문 제목 위, 표, 하단 약자 풀이."""
    with open(PAPER / "tables" / name, newline="", encoding="utf-8") as f:
        rows = [[_cell_text(c) for c in r] for r in csv.reader(f)]
    ncol = max(len(r) for r in rows)
    size = 8 if ncol <= 6 else 7 if ncol <= 9 else 6.5
    if len(rows) > 28:  # 긴 표(부록)는 1면에 들어가도록 축소
        size = min(size, 6.5)
    cap = caption(doc, f"Table {num}. {title}", after=4, new_page=new_page)
    tbl = doc.add_table(rows=len(rows), cols=ncol)
    tbl.style = "Table Grid"
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    mar = OxmlElement("w:tblCellMar")  # 좌우 셀 여백 0.1 cm
    for side in ("left", "right"):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:w"), "57")
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tbl._tbl.tblPr.append(mar)
    widths = _col_widths(rows, ncol, size)
    for j, col in enumerate(tbl.columns):
        col.width = widths[j]
    for i, r in enumerate(rows):
        for j in range(ncol):
            cell = tbl.cell(i, j)
            cell.width = widths[j]
            cp = cell.paragraphs[0]
            cp.alignment = WD_ALIGN_PARAGRAPH.CENTER if j else WD_ALIGN_PARAGRAPH.LEFT
            pf = cp.paragraph_format
            pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
            pf.line_spacing = 1.0
            pf.space_before = Pt(0)
            pf.space_after = Pt(0)
            set_font(
                cp.add_run(r[j] if j < len(r) else ""),
                size=size,
                bold=(i == 0),
                font="Times New Roman",
            )
    note = TABLE_NOTES.get(name.rsplit(".", 1)[0])
    if note:
        para(doc, note, size=8, spacing=1.2, before=4, font="Times New Roman")
    return cap


def embed_verbatim(doc, path):
    """부록：파일 원문을 고정폭 글꼴로(줄 단위 문단, 빈 줄 유지)."""
    for line in (ROOT / path).read_text(encoding="utf-8").splitlines():
        para(
            doc,
            line.replace("{cite:", "{cite :") or " ",
            size=7.5,
            spacing=1.0,
            align=WD_ALIGN_PARAGRAPH.LEFT,
            font=MONO_FONT,
        )


def number_embeds(chapters, prefix="", numbers=None):
    """모든 장의 자리표시자를 등장 순서대로 번호 매김：{("Fig", 이름): "N", ("Table", 이름): "N"}.
    부록은 prefix="A"로 별도 일련번호(A1, A2, …)."""
    numbers = {} if numbers is None else numbers
    counters = {"Fig": 0, "Table": 0}
    for chapter in chapters:
        for _, paras in chapter:
            for t in paras:
                m = EMBED_RE.match(t)
                if m:
                    kind, name = m.group(1), m.group(4).rsplit(".", 1)[0]
                    counters[kind] += 1
                    numbers[(kind, name)] = (f"{prefix}{counters[kind]}", m.group(2))
    return numbers


def resolve_refs(text, numbers):
    def sub(m):
        kind, name = m.groups()
        n = numbers[(kind, name)][0]  # 없는 이름은 KeyError로 즉시 드러난다
        return f"Fig. {n}" if kind == "Fig" else f"Table {n}"

    return REF_RE.sub(sub, text)


def is_subheading(t):
    return len(t) < 40 and t[1:3] == ". "


class Toc:
    """목차 항목의 등록(문서 앞의 목차 작성 시)과 소비(본문 렌더링 시 책갈피 부여)를 같은 순서로."""

    def __init__(self):
        self.entries = []  # dict(kind, level, text, bm, page)
        self._i = 0

    def add(self, kind, level, text):
        self.entries.append(
            {
                "kind": kind,
                "level": level,
                "text": text,
                "bm": f"_toc_{len(self.entries)}",
                "page": "○",
            }
        )

    def mark(self, p, text):
        e = self.entries[self._i]
        assert e["text"] == text, (e["text"], text)
        bookmark(p, e["bm"], self._i + 1)
        self._i += 1


def starts_with_embed(paras):
    return bool(paras) and bool(EMBED_RE.match(paras[0]))


def chapter_items(doc, paras, numbers, toc, heading_on_new_page=False):
    """소제목("A. …")·본문·표/그림 자리표시자를 순서대로 배치.

    표/그림은 1개당 1면：자기 면에서 시작하고, 뒤따르는 글은 TEXT_AFTER_EMBED에 따라 같은 면에
    잇거나 새 면에서 시작한다(빈 면이 생기지 않도록 문단의 page_break_before 속성으로 처리).
    heading_on_new_page：직전의 절 제목이 새 면에서 시작하였으면 첫 표/그림을 그 면에 둔다."""
    after_embed = False
    heading_owns_page = heading_on_new_page
    for i, t in enumerate(paras):
        m = EMBED_RE.match(t)
        v = VERB_RE.match(t)
        if m:
            kind, title, _, name = m.groups()
            num = numbers[(kind, name.rsplit(".", 1)[0])][0]
            cap = (embed_figure if kind == "Fig" else embed_table)(
                doc, num, title, name, new_page=not heading_owns_page
            )
            toc.mark(cap, f"{'Fig. ' if kind == 'Fig' else 'Table '}{num}. {title}")
            after_embed, heading_owns_page = not TEXT_AFTER_EMBED, False
            next_embed = i + 1 < len(paras) and bool(EMBED_RE.match(paras[i + 1]))
            if next_embed:  # 표/그림이 연이어 오면 다음 것은 반드시 새 면
                after_embed = True
        elif v:
            embed_verbatim(doc, v.group(1))
            after_embed = False
        elif NOTE_RE.match(t):
            para(
                doc,
                resolve_refs(NOTE_RE.match(t).group(1), numbers),
                size=8.5,
                spacing=1.3,
                before=4,
                after=2,
            )
            # 설명은 표/그림과 같은 면에 두되, 뒤따르는 글은 여전히 새 면에서 시작한다
        elif is_subheading(t):
            # 소제목 바로 뒤가 표/그림이면 소제목을 표/그림의 면 머리에 둔다
            # (면 끝에 제목만 남거나, 제목만 있는 면이 생기지 않도록)
            next_embed = i + 1 < len(paras) and bool(EMBED_RE.match(paras[i + 1]))
            toc.mark(heading(doc, t, 3, new_page=after_embed or next_embed), t)
            heading_owns_page = next_embed
            after_embed = False
        else:
            body(doc, resolve_refs(t, numbers), new_page=after_embed)
            after_embed = False
    return after_embed


# ===================================================================== 목차·면번호
CHAPTERS = [  # (제목, 내용, 부록 여부)
    ("Ⅲ. 연구재료 및 방법", T.MATERIALS),
    ("Ⅳ. 연구성적", T.RESULTS),
    ("Ⅴ. 고　　찰", T.DISCUSSION),
]


def collect_toc(numbers):
    """본문 렌더링 순서와 같은 순서로 목차 항목을 등록한다."""
    toc = Toc()
    toc.add("front", 1, "국문초록")
    toc.add("front", 1, "ABSTRACT")
    toc.add("body", 1, "Ⅰ. 서　　론")
    for sub, paras in T.INTRO:
        toc.add("body", 2, sub)
        for t in paras:
            if is_subheading(t):
                toc.add("body", 3, t)
    toc.add("body", 1, "Ⅱ. 연구목적")
    for sub, _ in T.PURPOSE:
        toc.add("body", 2, sub)
    for title, chapter in CHAPTERS:
        toc.add("body", 1, title)
        _collect_chapter(toc, chapter, numbers)
    toc.add("body", 1, "Ⅵ. 결　　론")
    toc.add("body", 1, "참고문헌")
    toc.add("body", 1, "부　　록")
    _collect_chapter(toc, T.APPENDIX, numbers)
    return toc


def _collect_chapter(toc, chapter, numbers):
    for sub, paras in chapter:
        toc.add("body", 2, sub)
        for t in paras:
            m = EMBED_RE.match(t)
            if m:
                kind, title, _, name = m.groups()
                num = numbers[(kind, name.rsplit(".", 1)[0])][0]
                toc.add(kind, 0, f"{'Fig. ' if kind == 'Fig' else 'Table '}{num}. {title}")
            elif is_subheading(t):
                toc.add("body", 3, t)


def toc_line(doc, text, page, bm, indent_cm, size=11):
    p = para(doc, "", spacing=1.25, align=WD_ALIGN_PARAGRAPH.LEFT)
    hang = (
        1.2 if len(text) > 10 else 0.0
    )  # 짧은 항목은 내어쓰기 위치가 암묵적 탭이 되는 것을 피한다
    p.paragraph_format.left_indent = Cm(indent_cm + hang)
    p.paragraph_format.first_line_indent = Cm(-hang)
    set_font(p.add_run(text), size=size)
    tabs = OxmlElement("w:tabs")
    tab = OxmlElement("w:tab")
    tab.set(qn("w:val"), "right")
    tab.set(qn("w:leader"), "dot")
    tab.set(qn("w:pos"), str(int(13 * 567)))  # 본문 폭 19-3.5-2.5 cm
    tabs.append(tab)
    p._p.get_or_add_pPr().append(tabs)
    p.add_run("\t")
    if str(page).isdigit() or page == "○":
        field(p, f"PAGEREF {bm} \\h", str(page), size=size)
    else:  # 로마 소문자 면번호는 뷰어마다 PAGEREF 서식이 달라 고정 문자열로 둔다
        set_font(p.add_run(str(page)), size=size)


def write_toc(doc, toc):
    center(doc, "목　　차", 14, bold=True, before=6, after=14)
    indent = {1: 0.0, 2: 0.6, 3: 1.2}
    for e in toc.entries:
        if e["kind"] in ("front", "body"):
            toc_line(
                doc,
                e["text"],
                e["page"],
                e["bm"],
                indent[e["level"]],
                size=11 if e["level"] < 3 else 10.5,
            )
    for kind, title in (("Table", "표 목 차"), ("Fig", "그 림 목 차")):
        doc.add_page_break()
        center(doc, title, 14, bold=True, before=6, after=14)
        for e in toc.entries:
            if e["kind"] == kind:
                toc_line(doc, e["text"], e["page"], e["bm"], 0.0, size=10)


def _norm(s):
    return re.sub(r"[\s　]+", "", s)


def _roman(n):
    out, table = "", [(10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
    for v, sym in table:
        while n >= v:
            out, n = out + sym, n - v
    return out


def page_map(docx_path, toc):
    """1차 docx → PDF → 면별 텍스트에서 목차 항목의 면을 찾아 {책갈피: 면번호}를 만든다."""
    if not (shutil.which("soffice") and shutil.which("pdftotext")):
        print("warning: soffice/pdftotext 없음 — 목차 면번호를 ○로 둔다")
        return {}
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["soffice", "--headless", "--convert-to", "pdf", "--outdir", tmp, str(docx_path)],
            check=True,
            capture_output=True,
            timeout=600,
        )
        pdf = Path(tmp) / (Path(docx_path).stem + ".pdf")
        txt = subprocess.run(
            ["pdftotext", "-layout", str(pdf), "-"], check=True, capture_output=True, text=True
        ).stdout
    pages = [[_norm(ln) for ln in pg.splitlines() if ln.strip()] for pg in txt.split("\f")]
    front0 = next(i for i, pg in enumerate(pages) if "목차" in pg)
    body0 = next(i for i, pg in enumerate(pages) if _norm("Ⅰ. 서　　론") in pg)
    found, cursor = {}, front0
    for e in toc.entries:
        key = _norm(e["text"])
        if e["kind"] in ("Fig", "Table"):
            key = re.match(r"(Table|Fig\.)[A-Z]?\d+\.", key).group(0)  # "Table3." / "Fig.3."
            hit = lambda ln, k=key: ln.startswith(k)  # noqa: E731
        else:
            hit = lambda ln, k=key: ln == k  # noqa: E731
        start = max(cursor, body0 if e["kind"] != "front" else front0)
        for i in range(start, len(pages)):
            if any(hit(ln) for ln in pages[i]):
                found[e["bm"]] = _roman(i - front0 + 1) if i < body0 else str(i - body0 + 1)
                cursor = i
                break
        else:
            print(f"warning: 목차 항목의 면을 찾지 못함: {e['text']}")
    return found, len(pages)


# ===================================================================== 문서 조립
def build(pages=None):
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = BODY_FONT
    st.font.size = Pt(BODY_SIZE)
    st._element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)

    numbers = number_embeds([T.MATERIALS, T.RESULTS, T.DISCUSSION])
    number_embeds([T.APPENDIX], prefix="A", numbers=numbers)
    CITE_NUMBERS.clear()
    CITE_NUMBERS.update(
        number_cites(
            [T.INTRO, T.PURPOSE, T.MATERIALS, T.RESULTS, T.DISCUSSION, T.CONCLUSION, T.APPENDIX]
        )
    )
    toc = collect_toc(numbers)
    for e in toc.entries:
        e["page"] = (pages or {}).get(e["bm"], "○")

    # ------------------------------------------------ 표지 (예시1)
    sec = doc.sections[0]
    setup_section(sec)
    footer_page_number(sec, show=False)
    center(doc, f"{PLACE['degree_kr']}학위논문", 14, bold=True, before=40)
    center(doc, PLACE["title_kr"], 22, bold=True, before=28, after=20)
    center(doc, PLACE["school_kr"], 16, bold=True, before=30)
    center(doc, PLACE["dept_kr"], 14, before=8)
    center(doc, PLACE["author_kr"], 14, before=8)
    center(doc, f"지도교수　{PLACE['advisor_kr']}", 14, before=20)
    center(doc, PLACE["date_kr"].split("년")[0] + "년", 14, before=40)

    # ------------------------------------------------ 표제지 (예시2)
    doc.add_section(WD_SECTION.NEW_PAGE)
    center(doc, PLACE["title_kr"], 22, before=60, after=20)
    center(doc, PLACE["school_kr"], 16, before=24)
    center(doc, PLACE["dept_kr"], 14, before=8)
    center(doc, PLACE["author_kr"], 14, before=8)
    center(doc, f"이 논문을 {PLACE['degree_kr']}논문으로 제출함", 16, before=26)
    center(doc, f"지도교수　{PLACE['advisor_kr']}", 14, before=10)
    center(doc, PLACE["date_kr"], 14, before=14)

    # ------------------------------------------------ 인준서 (예시3)
    doc.add_section(WD_SECTION.NEW_PAGE)
    center(
        doc,
        f"{PLACE['author_kr'].replace('　', '')}의 {PLACE['degree_kr']}학위논문을 인정함.",
        21,
        before=50,
        after=24,
    )
    for role in ["위원장", "위　원", "위　원"]:
        center(doc, f"{role}____________印", 14, before=12)
    center(doc, PLACE["school_kr"], 16, before=26)
    center(doc, PLACE["date_kr"], 14, before=12)

    # ------------------------------------------------ 전문부(목차 → 국문초록 → 영문초록): 로마 소문자 면번호
    sec = doc.add_section(WD_SECTION.NEW_PAGE)
    setup_section(sec, numbering="lowerRoman", start=1)
    footer_page_number(sec, show=True)
    write_toc(doc, toc)

    doc.add_page_break()
    toc.mark(center(doc, "국문초록", 14, bold=True, before=6, after=10), "국문초록")
    center(doc, PLACE["title_kr"], 13, bold=True, after=14)
    center(doc, PLACE["author_kr"], 11)
    center(doc, f"(지도교수 : {PLACE['advisor_kr']})", 11)
    center(doc, PLACE["dept_kr"].replace(" 전공", ""), 11)
    center(doc, PLACE["school_kr"], 11, after=12)
    for t in T.ABSTRACT_KR:
        body(doc, t)
    para(doc, f"Key Words：{T.KEYWORDS_KR}", before=14)

    doc.add_page_break()
    toc.mark(center(doc, "ABSTRACT", 14, bold=True, before=6, after=10), "ABSTRACT")
    center(doc, PLACE["title_en"], 13, bold=True, after=14)
    center(doc, PLACE["author_en"], 11)
    center(doc, f"(Advisor：{PLACE['advisor_en']})", 11)
    center(doc, PLACE["dept_en"], 11)
    center(doc, PLACE["school_en"], 11, after=12)
    for t in T.ABSTRACT_EN:
        body(doc, t)
    para(doc, f"Key Words：{T.KEYWORDS_EN}", before=14)

    # ------------------------------------------------ 본문: 아라비아 면번호
    sec = doc.add_section(WD_SECTION.NEW_PAGE)
    setup_section(sec, numbering="decimal", start=1)
    footer_page_number(sec, show=True)

    pending = False  # 직전 항목이 표/그림이면 다음 제목·문단은 새 면에서
    toc.mark(heading(doc, "Ⅰ. 서　　론", 1), "Ⅰ. 서　　론")
    for sub, paras in T.INTRO:
        toc.mark(heading(doc, sub, 2), sub)
        for t in paras:
            if is_subheading(t):
                toc.mark(heading(doc, t, 3), t)
            else:
                body(doc, t)

    toc.mark(heading(doc, "Ⅱ. 연구목적", 1), "Ⅱ. 연구목적")
    for sub, paras in T.PURPOSE:
        toc.mark(heading(doc, sub, 2), sub)
        for t in paras:
            body(doc, t)

    for title, chapter in CHAPTERS:
        toc.mark(heading(doc, title, 1, new_page=pending), title)
        pending = False
        for sub, paras in chapter:
            # 절 제목 바로 뒤가 표/그림이면 제목을 그 표/그림의 면 머리에 둔다
            owns = pending or starts_with_embed(paras)
            toc.mark(heading(doc, sub, 2, new_page=owns), sub)
            pending = chapter_items(doc, paras, numbers, toc, heading_on_new_page=owns)

    toc.mark(heading(doc, "Ⅵ. 결　　론", 1, new_page=pending), "Ⅵ. 결　　론")
    for t in T.CONCLUSION:
        body(doc, t)

    toc.mark(heading(doc, "참고문헌", 1, new_page=True), "참고문헌")
    for key, n in CITE_NUMBERS.items():
        para(
            doc,
            f"{n}. {T.REFERENCES[key]}",
            size=9.5,
            spacing=1.3,
            after=3,
            align=WD_ALIGN_PARAGRAPH.LEFT,
        )

    toc.mark(heading(doc, "부　　록", 1, new_page=True), "부　　록")
    pending = False
    for sub, paras in T.APPENDIX:
        owns = pending or starts_with_embed(paras)
        toc.mark(heading(doc, sub, 2, new_page=owns), sub)
        pending = chapter_items(doc, paras, numbers, toc, heading_on_new_page=owns)
    assert toc._i == len(toc.entries), (toc._i, len(toc.entries))
    return doc, numbers, toc


def main():
    out = PAPER / "thesis.docx"
    doc, numbers, toc = build()
    with tempfile.TemporaryDirectory() as tmp:
        draft = Path(tmp) / "thesis.docx"
        doc.save(draft)
        found = page_map(draft, toc)
    pages, n_pages = found if found else ({}, None)
    doc, numbers, toc = build(pages)
    doc.save(out)
    n_fig = sum(k == "Fig" for k, _ in numbers)
    print(
        f"saved: {out}  (figures {n_fig}, tables {len(numbers) - n_fig}, "
        f"references {len(CITE_NUMBERS)}, pages {n_pages})"
    )


if __name__ == "__main__":
    main()
