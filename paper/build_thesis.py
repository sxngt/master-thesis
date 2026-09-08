#!/usr/bin/env python
"""인제대학교 대학원 학위논문 생성기 → paper/thesis.docx

규정: CLAUDE.md '학위논문 작성 규정' 절. 19x26cm, 신명조 10.5pt, 줄간격 200%,
전문 면번호 로마 소문자 / 본문 아라비아, 항목번호 예시B(I. 1. A. 1)).
미확정 정보는 PLACE 딕셔너리의 ○○○ 유지.

실행:  uv run python paper/build_thesis.py
"""

from docx import Document
from docx.enum.section import WD_SECTION
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


def set_font(run, size=10.5, bold=False, font=BODY_FONT):
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    r = run._element.get_or_add_rPr()
    rf = r.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        r.append(rf)
    rf.set(qn("w:eastAsia"), font)


def para(doc, text="", size=10.5, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
         spacing=2.0, before=0, after=0, indent=None, font=BODY_FONT):
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = spacing
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    if indent:
        pf.first_line_indent = Cm(indent)
    for i, chunk in enumerate(text.split("\n")):
        if i:
            p.add_run().add_break()
        set_font(p.add_run(chunk), size=size, bold=bold, font=font)
    return p


def center(doc, text, size, bold=False, before=0, after=0, spacing=1.5):
    return para(doc, text, size=size, bold=bold, align=WD_ALIGN_PARAGRAPH.CENTER,
                spacing=spacing, before=before, after=after)


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


def heading(doc, text, level=1):
    sizes = {1: 14, 2: 12, 3: 11}
    return para(doc, text, size=sizes.get(level, 11), bold=True,
                align=WD_ALIGN_PARAGRAPH.LEFT if level > 1 else WD_ALIGN_PARAGRAPH.CENTER,
                before=18 if level == 1 else 12, after=8)


def body(doc, text):
    return para(doc, text, indent=0.75)


# ===================================================================== 본문 내용
ABSTRACT_KR = [
"사족보행 로봇의 험지 보행 능력은 재난 현장 탐색, 산업 설비 점검, 야지 정찰 등 실제 응용의 "
"핵심 요건이나, 비정형 지형에서의 접촉 동역학은 전통적 모델 기반 제어의 한계를 드러낸다. 심층 "
"강화학습(이하 RL이라 함)은 이에 대한 유력한 대안으로 부상하였으나, 주요 RL 알고리즘들이 험지 "
"보행이라는 특정 과제에서 보이는 상대적 성능, 학습 신뢰성, 효율성에 대한 동일 조건의 체계적 "
"비교는 여전히 부족하다. 본 연구의 목표는 대표적 RL 알고리즘 6종(PPO, TRPO, A3C, SAC, TD3, "
"DDPG)을 단일한 실험 조건에서 구현·학습·평가하는 재현 가능한 비교 프레임워크를 구축하고, 지형 "
"난이도에 따른 각 알고리즘의 성능 프로파일과 그 원인을 알고리즘의 구조적 원리로부터 규명하는 "
"것이다.",
"연구 재료 및 방법으로는 GPU 병렬 물리 시뮬레이터(Isaac Lab) 상에 Unitree A1 사족보행 로봇을 "
"구현하고, 평지, 계단, 불규칙 요철 지형의 3종 지형에서 동일한 보상 함수·관측 체계·학습 예산 "
"규약 아래 각 알고리즘을 무작위 시드 3개로 반복 학습하였다. 평가는 결정론적 정책에 대하여 "
"운동 성능(전진 속도, 성공률), 안정성(전복 빈도, 자세 요동), 효율성(운반 비용), 학습 효율"
"(임계 성능 도달 표본 수·벽시계 시간)의 4개 축으로 수행하였으며, 분산분석과 Tukey 사후검정 및 "
"효과크기로 통계적 유의성을 검증하였다.",
"연구 성적으로, 평지에서는 PPO와 TRPO가 목표 속도를 충족하며 시드 간 표준편차 0.01~0.02 "
"수준의 높은 재현성을 보인 반면, 불규칙 요철 지형에서는 구도가 역전되어 SAC와 TD3가 전 시드 "
"완주에 도달하였고 특히 SAC는 시드 간 표준편차 0.002의 일관성과 최저 운반 비용을 기록하였다. "
"DDPG는 모든 조건에서 보행 획득에 실패하였으며, 탐험 방식 개선은 부분적 개선에 그치고 TD3의 "
"구조적 수정을 통해서만 회복됨을 확인하였다. 학습 효율에서는 표본 효율(off-policy 우위)과 "
"벽시계 효율(on-policy 우위)이 상반되는 결과를 얻었다.",
"이상으로부터 험지 보행을 위한 RL 알고리즘 선택은 단일 우위가 아니라 지형 난이도, 배포 병목"
"(시뮬레이션 대 실기), 재현성 요구에 따라 달라져야 하며, 지형이 거칠어질수록 최대 엔트로피 "
"기반 off-policy 계열의 수렴 신뢰성이 우위를 갖는다는 결론을 얻었다. 본 연구의 프레임워크와 "
"공개 데이터는 후속 연구의 비교 기준선으로 활용될 수 있다.",
]
KEYWORDS_KR = "사족보행 로봇, 강화학습, 험지 보행, 물리 시뮬레이션, 학습 신뢰성, 알고리즘 비교"

ABSTRACT_EN = [
"Rough-terrain locomotion is a key requirement for deploying quadruped robots in disaster "
"response, industrial inspection, and field exploration, yet contact-rich dynamics on "
"unstructured ground exposes the limits of traditional model-based control. Deep reinforcement "
"learning (RL) has emerged as a compelling alternative; however, a systematic comparison of "
"major RL algorithms under identical conditions for this specific task remains scarce. This "
"study builds a reproducible benchmarking framework that implements, trains, and evaluates six "
"representative RL algorithms (PPO, TRPO, A3C, SAC, TD3, and DDPG) under a single protocol, "
"and seeks to explain the observed performance profiles from the structural principles of each "
"algorithm.",
"A Unitree A1 quadruped was simulated in a GPU-parallel physics simulator (Isaac Lab) over "
"three terrains: flat ground, stairs, and irregular bumpy ground. Each algorithm was trained "
"with three random seeds under identical reward, observation, and budget protocols. "
"Deterministic policies were evaluated along four axes: locomotion performance (forward "
"velocity, success rate), stability (fall frequency, attitude fluctuation), efficiency (cost "
"of transport), and learning efficiency (samples and wall-clock time to threshold). "
"Statistical significance was assessed with ANOVA, Tukey HSD post-hoc tests, and effect sizes.",
"On flat ground, PPO and TRPO met the commanded velocity with near-deterministic seed "
"reproducibility. On irregular rough ground the picture inverted: SAC and TD3 completed the "
"course in every seed, with SAC showing a seed standard deviation of 0.002 and the lowest cost "
"of transport, whereas one third of on-policy seeds failed to converge within budget. DDPG "
"failed to acquire locomotion in all conditions; improved exploration yielded only partial "
"recovery, while the structural corrections of TD3 restored walking. Sample efficiency favored "
"off-policy methods by an order of magnitude, while wall-clock efficiency favored on-policy "
"methods under massive parallel simulation.",
"We conclude that algorithm selection for rough-terrain locomotion should be conditioned on "
"terrain difficulty, deployment bottleneck, and reproducibility requirements rather than on a "
"single ranking, and that maximum-entropy off-policy methods gain a reliability advantage as "
"terrain roughness increases. The framework and data are released as an open baseline for "
"subsequent studies.",
]
KEYWORDS_EN = ("quadruped robot, reinforcement learning, rough-terrain locomotion, "
               "physics simulation, training reliability, algorithm comparison")

INTRO = [
("1. 연구 과제의 의의와 중요성", [
"보행 로봇은 바퀴형 이동체가 접근할 수 없는 계단, 잔해, 요철 지면을 통과할 수 있다는 점에서 "
"재난 현장 탐색, 산업 설비 점검, 야지 운송 등 다양한 응용의 핵심 플랫폼으로 주목되어 왔다. "
"그중 사족보행 로봇은 정적 안정성과 동적 기동성의 균형이 우수하여 가장 활발히 연구되는 형태이다. "
"그러나 험지 보행은 발끝 접촉의 생성과 소멸이 반복되는 불연속 동역학, 지형과의 상호작용에서 "
"오는 큰 불확실성, 고차원 연속 행동 공간이라는 세 가지 난제를 동시에 내포한다. 전통적 모델 기반 "
"제어는 정밀한 동역학 모델과 접촉 일정의 사전 규정을 요구하므로, 모델링이 어려운 비정형 지형"
"에서는 그 성능이 급격히 저하된다1). 강화학습(이하 RL이라 함)은 시행착오 데이터로부터 제어 "
"정책을 직접 학습함으로써 이러한 모델링 부담을 우회하는 접근으로, 기초과학적으로는 접촉이 "
"많은 비선형 시스템에서의 학습 이론 검증의 장이 되고, 응용과학적으로는 실용적 보행 제어기의 "
"새로운 설계 방법론이 된다는 점에서 이중의 의의를 갖는다.",
]),
("2. 관련 지견의 분석", [
"심층 RL을 이용한 사족보행 제어는 최근 수년간 급속히 발전하였다. Hwangbo 등2)은 액추에이터 "
"신경망을 결합한 시뮬레이션 학습 정책을 실기 ANYmal 로봇에 전이하여 민첩한 운동 능력을 "
"실증하였고, Lee 등3)은 지형 정보 없이 고유수용감각만으로 험지를 주파하는 정책을, Miki 등4)은 "
"외부감각을 융합한 강인한 야지 보행을 보고하였다. Rudin 등5)은 GPU 병렬 시뮬레이션으로 수천 "
"개의 환경을 동시 구동하여 수 분 내 보행 학습이 가능함을 보였으며, 이는 본 연구가 채택한 "
"대규모 병렬 학습 체계의 근간이 된다.",
"한편 이들 성과의 대부분은 근사 정책 최적화(PPO)6) 단일 알고리즘에 기초한다. RL 알고리즘 "
"군에는 신뢰영역 기반의 TRPO7), 최대 엔트로피 기반의 SAC8), 결정론적 정책 경사 계열의 "
"DDPG10)와 그 개선인 TD39), 비동기 병렬 학습의 A3C11) 등 상이한 구조적 원리를 갖는 대안들이 "
"존재하나, 이들이 험지 보행이라는 특정 과제에서 보이는 상대적 성능에 대한 동일 조건의 비교 "
"연구는 드물다. 일반 연속 제어 벤치마크에서의 비교14)는 존재하나 보행 특유의 접촉 동역학과 "
"지형 난이도 축을 다루지 못하며, RL 연구 전반의 재현성 문제, 특히 무작위 시드에 따른 성능 "
"변동이 결론을 좌우할 수 있다는 지적15)은 시드 반복과 통계 검정을 갖춘 비교 프레임워크의 "
"필요성을 시사한다.",
"한편 RL의 성패는 알고리즘 못지않게 보상 함수의 설계에 좌우된다. 보행 보상은 속도 추종, "
"에너지, 자세, 접촉 등 십여 개 성분의 가중합으로 구성되는데, 이 가중치는 관행적으로 연구자가 "
"반복 시행으로 손질하며 그 과정은 논문에 기록되지 않는 경우가 많다. 인간 선호를 보상으로 "
"통합하는 연구12,13)가 언어 모델 분야에서 성과를 보인 이래, 대형 언어 모델(이하 LLM이라 함)을 "
"보상 설계자로 쓰는 시도가 이어졌다. Kwon 등22)은 LLM을 선호 기반 보상 모델로, Yu 등20)은 "
"자연어 지시를 보상 코드로 번역하는 매개로, Ma 등19)과 Xie 등21)은 학습 결과를 되먹임하여 보상 "
"코드를 진화적으로 개선하는 탐색기로 각각 활용하였다. 그러나 이들은 학습이 끝난 뒤 보상 "
"함수를 통째로 다시 쓰는 외부 루프이므로 한 세대마다 전체 학습을 반복해야 하며, LLM의 제안이 "
"틀렸을 때 학습을 보호하는 장치가 없고, LLM 없이 같은 예산으로 보상을 조정하는 대조군과의 "
"비교가 드물다. 학습 도중 보상 가중치와 과제 난이도(커리큘럼)24)를 함께 조정하되 그 결정을 "
"통계적 판단층이 검증하는 구조, 그리고 그 효과를 동일 시드의 짝 비교와 무정보 대조군으로 "
"검정하는 실험은 본 연구가 다루는 공백이다.",
]),
("3. 추구 내용의 유도", [
"이상의 분석으로부터 다음의 연구 요소가 유도된다. 첫째, 알고리즘 간 비교가 유의미하려면 보상 "
"함수, 관측 체계, 학습 예산, 평가 절차가 단일 규약으로 통제되어야 한다. 둘째, 험지 보행의 "
"본질을 반영하려면 지형 난이도를 독립 변수로 하는 실험 설계가 필요하다. 셋째, 시드 간 변동이 "
"큰 RL의 특성상 반복 실험과 통계적 유의성 검정, 그리고 성능 평균뿐 아니라 재현성(시드 분산) "
"자체를 1급 평가 지표로 다루어야 한다. 넷째, 표본 예산이 상이한 on-policy와 off-policy 계열의 "
"공정한 비교를 위해서는 임계 성능 도달 비용으로 정규화한 학습 효율 분석이 요구된다. 다섯째, "
"보상 설계를 LLM에 위임하는 접근이 유효하려면 LLM의 제안이 학습을 훼손하지 않도록 검증·"
"복구하는 판단층이 필요하고, 그 효과는 같은 알고리즘·같은 시드에서 코치의 유무만을 달리한 "
"짝 비교와, 같은 개입 예산을 무정보로 쓰는 대조군에 대하여 검정되어야 한다.",
]),
("4. 구체적 연구 주제", [
"이에 본 연구는 다음을 구체적 주제로 설정한다. GPU 병렬 시뮬레이터 상에 사족보행 로봇 "
"Unitree A1과 3종 지형(평지, 계단, 불규칙 요철)을 구현하고, RL 알고리즘 6종(PPO, TRPO, A3C, "
"SAC, TD3, DDPG)을 단일 규약으로 학습시켜, 운동 성능·안정성·효율성·학습 효율의 4축 평가 "
"체계와 통계 검정으로 지형별 성능 프로파일을 도출한다. 나아가 관찰된 성능 차이를 각 알고리즘의 "
"목적함수와 갱신 규칙의 구조적 원리로부터 해석하고, 특히 보행 획득에 실패하는 알고리즘의 실패 "
"기전을 실험적으로 분해한다. 이어서 비교 결과 재현성이 가장 높았던 PPO를 대상으로, 학습 "
"도중 주기적으로 학습 상태 보고서를 읽고 보상 가중치·지령 속도·학습률을 유계 범위에서 조정하는 "
"LLM 보상 코치와 그 제안을 검증·복구하는 통계적 판단층을 구현한다. 코치의 효과는 불규칙 요철, "
"계단, 그리고 보행이 학습되지 않도록 고의로 빈약하게 설계한 보상의 3개 설정에서 코치 없는 "
"PPO와의 동일 시드 짝 비교, LLM 없는 무작위·언덕오르기 코치와의 비교로 검정하며, 코치 자체를 "
"측정 결과에 근거하여 개정하는 자가 진화 절차를 병행한다. LLM 추론은 외부 API 없이 연구실 "
"장비에서 구동하는 공개 가중치 모델로만 수행한다.",
]),
("5. 가정 및 용어의 정의", [
"본 연구는 다음의 가정과 범위에서 수행된다. 첫째, 정책은 지형 형상 정보 없이 고유수용감각만을 "
"사용하는 맹목 보행(blind locomotion)을 가정한다. 둘째, 본 논문의 주 결과는 물리 시뮬레이션 "
"환경에서의 학습과 평가에 한정하며, 실기 전이는 후속 과제로 남긴다. 셋째, 로봇에는 전방 목표 "
"속도 1.0 m/s의 지령이 주어지며, 20초의 제한 시간 내에 시작점으로부터 5 m 이상 진행하는 것을 "
"과제 성공으로 정의한다. 이하 본문에서 근사 정책 최적화는 PPO, 신뢰영역 정책 최적화는 TRPO, "
"연성 행위자-비평자는 SAC, 쌍둥이 지연 심층 결정론적 정책 경사는 TD3, 심층 결정론적 정책 "
"경사는 DDPG, 비동기 이점 행위자-비평자는 A3C로 각각 약칭한다. 학습 도중 보상 파라미터를 "
"조정하는 외부 절차를 보상 코치(이하 코치라 함), 코치가 한 보고 시점에 적용한 파라미터 변경의 "
"집합을 개입, 코치의 성과를 재는 고정 평가 하의 척도 J=성공률+0.5×전진 속도를 목적함수, LLM "
"코치를 결합한 PPO를 LLM-PPO라 한다. 코치의 명령 속도 조정과 같이 과제 난이도를 바꾸는 "
"파라미터는 커리큘럼 레버라 부른다.",
]),
]

PURPOSE = [
("1. 궁극적 연구 목적", [
"본 연구의 궁극적 목적은 사족보행 로봇의 험지 보행 과제에서 강화학습 알고리즘의 선택 기준을 "
"실증적 근거 위에 정립하는 것이다. 즉 특정 알고리즘의 단일 우열이 아니라, 지형 난이도와 배포 "
"조건에 따라 어떤 구조적 원리가 우위를 갖는지를 규명함으로써, 향후 험지 보행 제어기 개발에서 "
"합리적 알고리즘 선택과 학습 체계 설계를 가능하게 하는 일반적 지침을 도출하고자 한다. 나아가 "
"알고리즘 선택 이후에 남는 보상 설계의 부담을 LLM에 위임할 수 있는지, 즉 학습 상태를 읽고 "
"보상과 난이도를 조정하는 LLM 코치가 통계적 판단층의 보호 아래 코치 없는 학습보다 우수한 "
"정책을 더 신뢰성 있게 얻게 하는지를 실증하고, 그 이득이 LLM의 판단에서 오는지 단순한 "
"조정 예산에서 오는지를 분리하고자 한다.",
]),
("2. 실행 목표", [
"위 목적을 실측 가능한 형태로 유도한 실행 목표는 다음과 같다.",
"1) 동일한 보상·관측·평가 규약 아래 RL 알고리즘 6종을 구현하고, 물리 시뮬레이션 기반의 "
"재현 가능한 험지 보행 비교 프레임워크를 구축한다.",
"2) 평지, 계단, 불규칙 요철의 3종 지형에서 알고리즘별 운동 성능, 안정성, 효율성 지표를 "
"무작위 시드 3회 반복으로 측정하고 분산분석과 사후검정으로 차이의 유의성을 검정한다.",
"3) 시드 간 성능 분산으로 정의되는 학습 재현성과, 임계 성능 도달에 요구되는 표본 수 및 "
"벽시계 시간으로 정의되는 학습 효율을 지형별로 산출하여 on-policy와 off-policy 계열을 "
"정규화 비교한다.",
"4) 보행 획득에 실패하는 알고리즘에 대하여 탐험 방식과 가치 추정 구조를 분리한 대조 실험으로 "
"실패 기전을 규명한다.",
"5) 이상의 성적을 각 알고리즘의 목적함수·갱신 규칙과 연관지어 해석하고, 지형 난이도에 따른 "
"알고리즘 선택 지침을 제시한다.",
"6) 학습 중 보고서에 근거하여 보상 가중치·커리큘럼 레버·학습률을 유계 조정하는 LLM 보상 "
"코치와, 잡음 인지 롤백·커리큘럼 복귀 불변식·개입 효과 원장으로 구성된 판단층을 구현한다.",
"7) 불규칙 요철, 계단, 빈약 보상의 3개 설정 × 시드 3개에서 코치 없는 PPO와 LLM-PPO를 동일 "
"시드 짝 비교로 검정하고, 동일 판단층 위에서 무작위·언덕오르기 코치를 대조군으로 두어 LLM의 "
"기여를 분리한다.",
"8) 코치의 프롬프트와 판단층 설정을 하나의 버전으로 다루어, 측정 결과에 근거하여 다음 버전을 "
"제안·검정·채택하는 자가 진화 절차를 구축하고 그 채택·기각 이력을 보고한다.",
]),
]


MATERIALS = [
("1. 연구 설계 개요", [
"본 연구는 강화학습 알고리즘 6종을 독립 변수의 한 축으로, 지형 3종을 다른 한 축으로 하는 "
"요인 설계(factorial design)를 채택하였다. 알고리즘과 지형의 각 조합에 대하여 무작위 시드를 "
"달리한 3회의 독립 학습을 수행하였으며, 학습된 정책은 학습에 사용되지 않은 평가 시드 아래 "
"결정론적으로 구동하여 관찰 항목을 측정하였다. 전체 절차는 실험 환경 구축, 알고리즘 구현 및 "
"검증, 본 학습, 평가, 통계 분석의 순으로 진행되었고, 모든 실험 조건은 계층적 설정 파일로 "
"기록되어 임의 조합의 재현이 가능하도록 하였다. 본 장의 2절과 3절은 알고리즘 비교 실험(제1부)"
"의 재료와 방법을, 4절은 그 결과에 기초하여 PPO를 대상으로 수행한 LLM 보상 코치 실험(제2부)의 "
"재료와 방법을 기술한다.",
]),
("2. 연구 재료", [
"A. 시뮬레이션 환경",
"물리 시뮬레이션에는 GPU 병렬 로봇 학습 프레임워크인 Isaac Lab 0.50.2(Isaac Sim 5.1, NVIDIA, "
"미국)를 사용하였다16). 물리 엔진은 PhysX로서 적분 시간 간격은 0.005초(200 Hz)로 하였고, 제어 "
"정책은 4스텝마다 호출되어 50 Hz로 동작하였다. 병렬 환경 수는 on-policy 계열의 학습에서 "
"4,096개, off-policy 계열에서 128개로 하였는데, 이는 두 계열의 표본 수집과 갱신 빈도의 균형이 "
"상이한 점을 고려한 것이다. 모든 연산은 단일 계산 장비(GPU：GeForce RTX 4080 16 GB, NVIDIA, "
"미국; CPU：Core i5-13400, Intel, 미국; Ubuntu 22.04)에서 수행하였다.",
"B. 로봇 모델",
"연구 대상 로봇은 소형 사족보행 로봇 Unitree A1(Unitree Robotics, 중국)으로, 질량 약 12 kg, "
"작동 관절 12개(다리당 고관절 외전·굴곡, 무릎 굴곡 각 1개), 관절 토크 한계 33.5 N·m의 "
"제원을 갖는다. 시뮬레이션에는 제조사 형상에 기반한 공식 모델을 사용하였고, 구동계는 직류 "
"모터 모델(강성 25, 감쇠 0.5의 관절 위치 비례-미분 제어)로 하였다. 정책의 출력 행동은 기본 "
"기립 자세 관절각에 0.25 배율로 더해지는 관절 위치 목표로 변환된다.",
"C. 지형",
"지형은 평지, 계단, 불규칙 요철의 3종으로 하였다. 계단은 단 높이 5 cm, 단 깊이 30 cm의 "
"피라미드형이며, 불규칙 요철 지형은 격자 높이장(height field)에 ±5 cm 범위의 균일 무작위 "
"높이를 부여하고 약 25 cm 간격으로 표본화하여 발 크기 규모의 요철이 형성되도록 생성하였다. "
"지면 마찰계수는 0.7~1.0 범위로 하였고, 지형 생성의 무작위성은 학습 시드와 함께 고정하여 "
"동일 조건의 재현이 가능하도록 하였다.",
"D. 연구 대상의 규정",
"본 연구가 상정하는 이론적 모집단은 고유수용감각만으로 험지를 보행하는 소형 사족보행 로봇의 "
"제어 정책 전체이다. 비교 대상 알고리즘은 정책 경사 계열의 대표성이 확립된 6종(PPO, TRPO, "
"A3C, SAC, TD3, DDPG)으로 하였으며, 각 조합의 학습 시드는 0, 1, 2의 3개로 고정하였다. 보행 "
"획득에 실패함이 사전 확인된 조건이라도 실패 기전 분석에 필요한 경우 제외하지 않고 성적에 "
"포함하였다.",
]),
("3. 연구 방법", [
"A. 과제 정의와 관측·행동 공간",
"과제는 전방 목표 속도 1.0 m/s의 지령 추종 보행으로 하였다. 에피소드 길이는 20초이며, 시작 "
"위치로부터 직선 거리 5 m 이상 진행하면 과제 성공으로 판정한다. 몸통이 지면에 접촉하거나 "
"기울기가 임계(중력 투영 성분 기준)를 초과하면 전복으로 판정하고 에피소드를 종료한다. 관측은 "
"몸통 선속도(3), 각속도(3), 중력 투영 벡터(3), 지령(3), 관절 위치 편차(12), 관절 속도(12), "
"직전 행동(12)의 48차원 고유수용감각으로 구성하였고, 각 성분에는 선속도 2.0, 각속도 0.25, "
"관절 속도 0.05의 정규화 배율을 적용하였다. 행동은 12차원 연속 벡터이다.",
"B. 보상 함수",
"보상은 다음 성분의 가중합으로 정의하였다. 속도 추종 보상 exp{-4(v-1.0)²}(v는 몸통 전방 "
"속도), 에너지 벌점 -2.5×10⁻⁵Στ²(τ는 관절 토크), 자세 벌점 -0.5(roll²+pitch²), 발 미끄럼 "
"벌점 -0.1, 관절 한계 벌점 -0.2, 행동 변화율 벌점 -0.01, 생존 보상 0.1, 전복 벌점 -10, "
"횡방향 속도 벌점 -1.0v_y², 회전 속도 벌점 -0.5ω_z², 그리고 유각기 지속 보상(발이 접지하는 "
"순간 공중 체류 시간과 0.5초의 차에 비례)이다. 유각기 지속 보상은 속도 추종만으로는 탐험 "
"잡음에 의존하는 퇴행적 보행이 학습되는 현상을 방지하기 위한 최소한의 보행 유도 장치로 "
"도입하였으며5), 이 구성은 모든 알고리즘에 동일하게 적용되었다.",
"C. 학습 절차",
"여섯 알고리즘은 모두 동일한 심층 신경망 구조를 사용하였다. 행위자 신경망은 은닉층 "
"512-256-128의 다층 퍼셉트론, 비평자 신경망은 512-512-256-128로 하였다. 공통 하이퍼파라미터는 "
"할인율 0.99, 학습률 3×10⁻⁴(A3C, DDPG는 1×10⁻⁴)로 하였고, on-policy 계열은 일반화 이점 "
"추정17)(λ=0.95)을 사용하였다. PPO는 비율 절단 0.2와 적응형 KL 벌점을 병용하였고6), TRPO는 "
"켤레기울기법 10회 반복과 KL 제약 0.01의 역추적 선형 탐색을7), SAC는 자동 온도 조절과 이중 "
"Q-신경망을8), TD3는 목표 정책 평활화(잡음 0.2, 절단 0.5)와 2회당 1회의 지연 정책 갱신을9), "
"DDPG는 Ornstein-Uhlenbeck 잡음과 파라미터 공간 잡음의 두 탐험 변형을10) 각각 원전에 따라 "
"구현하였다. 학습 예산은 on-policy 5×10⁷ 스텝, off-policy 5×10⁶ 스텝으로 하였는데, 이는 두 "
"계열의 표본 처리 구조 차이를 반영한 것으로 계열 간 비교는 학습 효율 분석에서 임계 도달 "
"비용으로 정규화하였다. 시간 제한에 의한 에피소드 절단은 진종결과 구분하여 가치 추정치를 "
"보상에 환류하는 방식으로 처리하였다5).",
"D. 평가 절차와 관찰 항목",
"평가는 학습 종료 시점의 정책을 결정론적(확률 정책의 평균 행동)으로 구동하여 에피소드 20회에 "
"대해 수행하였고, 평가용 무작위 시드는 학습 시드와 분리하였다. 관찰 항목은 다음과 같이 "
"정의하였다. 운동 성능：평균 전진 속도(시점-종점 변위/시간), 과제 성공률, 경로 효율(변위/"
"이동 경로 길이). 안정성：분당 전복 빈도, 자세 요동(roll·pitch 표준편차의 제곱합 제곱근), "
"접촉력 분산. 효율성：운반 비용 CoT=E/(mgd)(E는 관절 기계적 일률 |τ·ω|의 시간 적분, m은 "
"질량, g는 중력가속도, d는 이동 거리), 이동 거리당 제곱평균제곱근 토크. 학습 효율：평가 "
"속도 0.8 m/s 최초 도달까지의 환경 스텝 수와 벽시계 시간, 학습 곡선의 정규화 곡선하면적. "
"예산 내 임계 미도달 시드는 중도절단(censored)으로 처리하였다.",
"E. 자료 분석 기법",
"조합별 성적은 시드 3회의 평균과 표준편차로 요약하고 t-분포 기반 95% 신뢰구간을 병기하였다. "
"알고리즘 간 차이는 일원배치 분산분석으로 검정하고 효과크기 η²을 산출하였으며, 유의한 경우 "
"Tukey HSD 사후검정과 쌍별 Cohen’s d18)를 적용하였다. 유의수준은 0.05로 하였다. 모든 분석 "
"코드는 실험 코드와 함께 관리하여 성적 산출 과정 전체의 재현이 가능하도록 하였다.",
"F. 재현성 확보 체계",
"구현에는 Python과 PyTorch를 사용하였고, 난수 발생원은 단일 진입점에서 일괄 고정하였다. 각 "
"학습 실행은 해석이 완료된 전체 설정 파일과 학습 곡선, 정책 점검점(checkpoint)을 저장하며, "
"전체 코드와 실험 산출물은 공개 저장소를 통해 제공한다. 표와 그림을 포함한 모든 성적은 저장된 "
"원자료로부터 스크립트로 재생성된다.",
]),
("4. LLM 보상 코치 연구", [
"A. 설계 개요",
"제2부는 알고리즘을 PPO로 고정하고 코치 조건을 독립 변수로 하는 짝 설계(paired design)를 "
"채택하였다. PPO는 제1부에서 평지·계단의 시드 간 재현성과 벽시계 학습 효율이 가장 높아 보상 "
"설계의 효과를 알고리즘 변동과 분리하기에 적합하였다. 코치 조건은 코치 없음(PPO), LLM 코치"
"(LLM-PPO), 그리고 LLM 없이 같은 개입 주기·범위·판단층을 쓰는 무작위 코치와 언덕오르기 코치의 "
"4수준이며, 각 조건은 동일한 무작위 시드(0, 1, 2)로 학습하여 (설정, 시드) 쌍 단위로 비교하였다. "
"설정은 불규칙 요철 지형 상급(요철 높이 8 cm)과 계단 중급(단 높이 12 cm)의 2개 험지에 제1부와 "
"동일한 보상을 적용한 2개 설정, 그리고 요철 지형 중급(5 cm)에 보행이 학습되지 않도록 고의로 "
"빈약하게 설계한 보상(에너지 벌점을 80배 강화하고 유각기 지속 보상과 방향 벌점을 제거하여 정지가 "
"최적이 되는 보상)을 적용한 1개 설정의 합계 3개로 하였다. 세 번째 설정은 코치가 결함 있는 보상을 "
"학습 도중 복구할 수 있는지를 보는 보상 복구 실험이다.",
"B. 실험 장비와 학습 예산",
"제2부는 4장의 GPU(GeForce RTX 4090 24 GB, NVIDIA, 미국)를 갖춘 연구실 서버에서 수행하였다. "
"3장은 LLM 추론 전용으로, 1장은 물리 시뮬레이션 전용으로 배정하여 시뮬레이션 GPU에서 학습 3개를 "
"동시에 진행하였다. 시뮬레이션은 Isaac Sim 4.5 / Isaac Lab 2.1.1로 하였고 환경 수 4,096, 학습 "
"예산 4×10⁷ 스텝, 정책 평가 간격 1×10⁶ 스텝으로 하였다. 정책 평가는 초기 상태를 고정한 256개 "
"환경의 첫 에피소드(5 m 주행로, 20초 제한)에 대한 결정론적 구동으로 하였으며, 학습 시드와 무관한 "
"고정 평가이므로 조건 간 직접 비교가 가능하다. 최종 성적은 학습 종료 시점의 동일 평가로 얻었다. "
"PPO의 적응형 KL 벌점 계수는 재현성 확보를 위하여 [2⁻⁶, 8]의 범위로 유계하고, 미니배치 KL 발산에 "
"의한 조기 종료는 미니배치 평균 KL로 판정하도록 하였다(범위 하한이 없을 때 계수가 10⁻³⁰ 수준으로 "
"소실되어 사실상 KL 제약이 꺼지는 현상을 예비 실험에서 관찰하였다).",
"C. 코치의 구조",
"코치는 학습 루프 바깥의 외부 루프로서, 준비 기간 2×10⁶ 스텝 이후 2×10⁶ 스텝마다 다음 절차를 "
"수행한다. (1) 보고：직전 평가의 성능 지표(성공률, 전진 속도, 전복 빈도, 운반 비용 등), 보상 "
"성분별 기여도, 보행 기술자(유각기 시간, 접촉 패턴), PPO 학습 통계(KL 발산, 엔트로피, 절단 "
"비율, 학습률), 목적함수 J의 궤적과 잡음 추정치, 개입 이력과 그 효과, 학습곡선 적합에 의한 "
"예산 말 예상 J, 항별 개선 여지(headroom)를 하나의 문서로 정리한다. (2) 제안：코치가 보고서를 "
"읽고 조정 대상 파라미터 최대 3개의 새 값, 진단, 예상 ΔJ를 구조화된 JSON으로 제안한다. 조정 "
"대상은 보상 성분 가중치 11개(속도 추종의 지령 속도 포함), 유각기 목표 시간, 그리고 PPO의 학습률과 "
"엔트로피 계수이며, 각 파라미터에는 부호를 넘지 않는 유계 범위와 개입당 변화 한도(선형 ±30%, "
"로그 척도 ×/÷3)가 부여된다. (3) 검증：판단층이 제안을 범위·한도·단계 규칙에 따라 절단 또는 "
"기각한 뒤 적용한다. (4) 판정：다음 평가에서 J의 변화를 검정하여 손상이면 정책과 파라미터를 "
"개입 직전 상태로 롤백한다. 모든 보고서, 제안 원문, 판정은 실행별 기록 파일에 남긴다.",
"D. 판단층",
"판단층은 코치의 종류(LLM, 무작위, 언덕오르기)와 무관하게 공유되며 다음 장치로 구성된다. "
"1) 잡음 인지 롤백：평가 k의 관측 J_k=J*_k+ε_k로 보고, 측정 분산(성공률의 이항 표준오차와 "
"속도의 표준오차의 가중 결합)과 공정 잡음(J 궤적의 2차 차분에 대한 중앙절대편차 추정)의 최댓값을 "
"σ로 하여 허용치 τ=min{max(0.05, 2√2σ), 0.3}을 정한다. ΔJ<−τ이면 롤백하고, −τ≤ΔJ<−0.5τ의 "
"모호 구간이면 새 에피소드로 1회 확인 재평가한 풀링 평균으로 재판정한다. 최선 정책은 연속 2회 "
"평가의 평균이 최대인 시점으로 정의하고, 한 스냅샷의 복원은 1회로 제한하며 롤백 후 1회의 "
"개입 기회를 건너뛴다. 2) 추세 제거 효과 귀속과 원장：개입 전 4회 평가의 선형 추세를 뺀 순효과 "
"(J_after−J_before)−β̂Δt를 (파라미터, 방향)별로 정규-정규 갱신하여 사후분포로 유지하고 보고서에 "
"제시한다. 보상 가중치 이동은 3회 이상 관측에서 원장이 음의 효과로 판정하면 판단층이 기각한다"
"(원장 거부권). 커리큘럼 레버의 효과는 직후 평가가 구조적으로 음으로 치우치므로 2회 보고 뒤의 "
"값으로 원장에 기입한다(정착 효과). 3) 단계 스케줄：예산의 50%까지는 탐색, 50~85%는 원장에 "
"증거가 있는 이동을 우선하는 활용, 85% 이후는 보상 변경을 동결하고 커리큘럼 레버와 학습률만 "
"허용하는 공고화 단계로 한다. 4) 커리큘럼 복귀 불변식과 잠금：지령 속도는 과제의 난이도를 "
"정의하므로, 코치가 초기에 낮춘 지령 속도는 진행률 50% 이후 최근 2회 보고의 성공률이 0.3 이상"
"이면 판단층이 기준값까지 단계적으로 되올리고(복귀 불변식), 복귀 단계에서는 어떤 코치도 이를 "
"다시 낮출 수 없다(커리큘럼 잠금). 5) 항별 개선 여지：속도항은 관측된 (지령, 측정) 쌍으로 추정한 "
"추종률로 상한을 계산하여 코치에게 각 파라미터가 J에 줄 수 있는 최대 이득을 수치로 제시한다.",
"E. LLM 코치와 실행 간 플레이북",
"LLM 코치는 시스템 프롬프트(과제·성분·파라미터 표·규칙)와 사용자 프롬프트(보고서)의 두 "
"템플릿으로 보고서를 구성하고, 응답 JSON을 스키마로 검증하여 실패 시 해당 개입을 폐기한다. LLM "
"추론은 외부 API를 사용하지 않고 공개 가중치 모델 Qwen3.8-27B25)(4비트 양자화, 다중 토큰 예측 "
"디코딩)를 GPU 3장에 1장당 1개의 독립 서버로 구동하여 수행하였으며, 응답 지연(41~87초)은 같은 "
"GPU에서 동시에 학습되는 다른 두 실행이 흡수하므로 학습 처리량 손실은 없었다. 코치의 증거는 "
"기본적으로 현재 실행 안에서만 계산되므로, 완료된 모든 코치 실행에서 정책이 성공률 0에서 "
"벗어난 시점과 그 전에 적용된 (파라미터, 방향) 이동, 보행 중 각 이동의 유지·롤백과 순효과를 "
"색인한 플레이북을 매 배치 전에 재구성하여 보고서에 같은 과제의 실행 간 증거로 제시하였다.",
"F. 무작위·언덕오르기 대조군",
"LLM의 기여를 개입 예산 자체의 효과와 분리하기 위하여 같은 개입 주기·파라미터 범위·변화 "
"한도·판단층(잠금, 원장 거부권, 정착 효과 포함)을 공유하되 제안 규칙만 다른 두 대조군을 두었다. "
"무작위 코치는 매 개입에서 파라미터 3개를 무작위로 골라 한도 내 균등 분포로 이동시키고, "
"언덕오르기 코치는 파라미터를 순환하며 한 번에 하나를 이동시키는 (1+1) 진화 전략으로서 개선된 "
"이동은 같은 방향을 반복하고 개선되지 않은 이동은 되돌린 뒤 방향을 바꾼다. 두 대조군은 LLM "
"코치와 정확히 같은 목적함수 되먹임을 받는다.",
"G. 코치 버전의 자가 진화",
"코치의 프롬프트 템플릿과 판단층 설정을 하나의 버전으로 관리하고, 세대마다 현 버전(인컴번트)"
"과 후보 버전을 같은 3개 설정 × 3개 시드에서 학습하여 (설정, 시드) 쌍별 최종 J의 차이 ΔJ로 "
"판정하였다. 채택 기준은 사전 등록하였다：단측 짝 t-검정 p≤0.2, 어느 설정의 평균 ΔJ도 −0.05 "
"미만이 아닐 것, 유효 쌍 6개 이상. 후보는 메타 LLM(코치와 같은 모델)이 인컴번트의 측정 결과, "
"개입 진단, 예측 보정치, 정체 실행 요약을 읽고 제안하되, 자리표시자·파라미터 이름·범위를 "
"검증한 뒤에만 버전이 되고 코드 변경 제안은 사람이 검토한다. 후보가 판단층 설정을 명시하지 "
"않으면 인컴번트의 설정을 상속한다. 각 버전의 채택·기각과 근거는 변경 기록으로 남겼으며, 사람이 "
"직접 작성한 버전(플레이북 도입, 커리큘럼 잠금)도 같은 절차로 판정하였다.",
"H. 관찰 항목",
"제1부의 관찰 항목(전진 속도, 성공률, 경로 효율, 전복 빈도, 자세 요동, 운반 비용)에 더하여 "
"목적함수 J, 성공률 0.5 최초 도달 스텝 수(학습 속도), 그리고 코치 실행에 한하여 개입 수, 롤백 "
"수, 최초 보고 시점의 개입 조합과 성공률 0 탈출 여부, 코치의 ΔJ 예측 부호 정확도를 기록하였다.",
"I. 자료 분석 기법",
"조건 간 비교는 (설정, 시드) 쌍의 ΔJ에 대한 짝 t-검정을 주 검정으로 하고, 정규성 가정에 "
"의존하지 않는 Wilcoxon 부호순위 검정23)을 병기하였다. 설정별(n=3)과 3개 설정을 통합한 전체"
"(n=9)에 대하여 평균±표준편차, ΔJ의 95% 신뢰구간, 짝 Cohen’s d를 보고하였고, 빈약 보상 "
"설정의 성공률 0 탈출 여부와 같은 이진 결과는 Fisher 정확검정으로 검정하였다. 시드 3개의 "
"설정별 검정은 검정력이 낮으므로 결론은 통합 검정과 효과크기에 근거하며, 설정별 값은 효과의 "
"방향과 크기를 기술하는 용도로 제한하였다.",
]),
]


# Ⅳ~Ⅵ 골격. "[…]" 문단은 자리표시자 — 표/그림은 paper/tables, paper/figures에서
# 1개당 1면으로 삽입(제목：표는 상단, 그림은 하단, 모두 영문). 표 6~9·그림 7~8은
# 제2부 결과가 확정되는 대로 scripts/analyze.py 산출물로 채운다.
RESULTS = [
("1. 강화학습 알고리즘 비교", [
"A. 지형별 최종 성능",
"[Table 1. Final Performance of Six RL Algorithms on Three Terrains — tables/table1_final_performance.csv]",
"[Fig. 2. Forward Velocity and Success Rate by Terrain — figures/fig2_velocity_success]",
"B. 통계 검정",
"[Table 2. ANOVA, Tukey HSD and Effect Sizes — tables/table2_statistics.csv]",
"C. 학습 곡선과 학습 효율",
"[Fig. 1. Learning Curves — figures/fig1_learning_curves]",
"[Table 3. Learning Efficiency：Samples and Wall-Clock Time to Threshold — tables/table3_learning_efficiency.csv]",
"[Fig. 4. Learning Efficiency — figures/fig4_learning_efficiency]",
"D. 시드 간 재현성",
"[Fig. 3. Seed Reliability — figures/fig3_seed_reliability]",
"E. 지형 난이도 스케일링",
"[Table 4. Difficulty Scaling — tables/table4_difficulty_scaling.csv]",
"[Fig. 6. Difficulty Scaling — figures/fig6_difficulty_scaling]",
"F. DDPG 실패 기전의 분해",
"[Fig. 5. DDPG Failure Decomposition — figures/fig5_ddpg_failure_decomposition]",
"G. ANYmal C 이식",
"[Table 5. Transfer of the Protocol to ANYmal C — tables/table5_anymal_transfer.csv]",
]),
("2. LLM 보상 코치", [
"A. 설정별 PPO 대 LLM-PPO",
"[Table 6. Paired Comparison of PPO and LLM-PPO per Setting (n=3 seeds each) — J, success rate, "
"velocity, CoT, falls/min, path efficiency, attitude, steps to success 0.5; paired Δ, p, d]",
"B. 통합 비교",
"[Table 7. Pooled Paired Comparison over Three Settings (n=9) — Δ with 95% CI, paired t and Wilcoxon p, Cohen's d]",
"[Fig. 7. Objective J during Training, PPO vs LLM-PPO, per Setting and Seed]",
"C. 빈약 보상의 복구",
"[Fig. 8. Success Rate during Training under the Naive Reward, with Coach Interventions Marked]",
"D. LLM 없는 코치와의 비교",
"[Table 8. LLM-PPO vs Random and Hill-Climbing Coaches on the Same Decision Layer (n=9 pairs) — 절제 실험 결과 대기]",
"E. 코치 버전의 진화",
"[Table 9. Coach Versions：Change, Paired ΔJ vs Incumbent, p, Verdict — v5, v5p, v6, v7, v7p, v8, v9, v10, …]",
"F. 개입 분석",
"[Table 10. First-Report Intervention Recipe and Escape from the Naive-Reward Floor (Fisher's exact test)]",
]),
]

DISCUSSION = [
("1. 연구 방법의 타당성과 신뢰성", [
"[재료 선택：Unitree A1·3 지형·맹목 보행의 대표성과 한계, 시뮬레이터 단일(Isaac Lab) 검증의 비뚤림]",
"[관찰 항목：J=성공률+0.5×속도가 전복·효율을 직접 포함하지 않는 점(계단에서 속도–전복 교환), "
"고정 256 환경 평가의 이점과 한계]",
"[시드 수 3의 검정력, 통합 n=9 검정의 가정(설정 간 이질성), 코치 실험이 PPO 한 알고리즘에 한정된 점]",
"[LLM 코치의 비결정성과 로컬 모델 양자화, 코치 로그 전량 공개에 의한 재현성]",
]),
("2. 성적의 해석과 추론", [
"[알고리즘 비교：지형 난이도에 따른 on/off-policy 역전, DDPG 실패 기전, 표본 효율 대 벽시계 효율]",
"[코치 효과의 원천：결정층·플레이북 대 LLM 보고별 판단(ΔJ 예측 부호 정확도 ≈0.5), 무작위·언덕오르기 대조군과의 차이]",
"[빈약 보상 복구：최초 보고의 지령 속도 하향+에너지 벌점 완화 조리법과 플레이북의 역할]",
"[계단：속도 이득은 있으나 성공률 불변, 커리큘럼 상향 시점(성공률 0.5~0.6)의 붕괴와 롤백]",
"[코치 진화：채택 2·기각 3의 이력에서 판단층 개정과 프롬프트 개정의 효과 차이, 상속 결함의 교훈]",
]),
("3. 미해석 부분과 후속 과제", [
"[계단 성공률을 높이는 커리큘럼 방향 게이트, 전복을 포함하는 목적함수, 시드 확대(n≥5)]",
"[PyBullet/Gazebo 교차 검증(sim-to-sim gap), 다른 알고리즘(SAC)에의 코치 적용, Unitree A1 실기 전이]",
]),
]

CONCLUSION = [
"[알고리즘 선택 지침：단일 우위 없음, 지형 난이도·배포 병목·재현성 요구에 따른 조건부 선택]",
"[LLM 보상 코치：통계적 판단층 위에서 코치 없는 PPO보다 높은 목적함수·학습 속도·시드 신뢰성, 결함 보상의 학습 중 복구]",
"[LLM 기여의 분리(대조군 결과에 따라 기술), 후속 제언]",
]

# TODO(최종): 본문 어깨번호 인용 순서대로 재번호 (현재 19~25는 추가분을 뒤에 붙인 상태).
REFERENCES = [
"Raibert MH. Legged robots that balance. Cambridge, MIT Press, 1986：1~233.",
"Hwangbo J, Lee J, Dosovitskiy A, et al. Learning agile and dynamic motor skills for legged robots. Sci Robot, 2019, 4：eaau5872.",
"Lee J, Hwangbo J, Wellhausen L, et al. Learning quadrupedal locomotion over challenging terrain. Sci Robot, 2020, 5：eabc5986.",
"Miki T, Lee J, Hwangbo J, et al. Learning robust perceptive locomotion for quadrupedal robots in the wild. Sci Robot, 2022, 7：eabk2822.",
"Rudin N, Hoeller D, Reist P, et al. Learning to walk in minutes using massively parallel deep reinforcement learning. Proc Conf Robot Learn, 2022, 164：91~100.",
"Schulman J, Wolski F, Dhariwal P, et al. Proximal policy optimization algorithms. arXiv preprint, 2017：arXiv:1707.06347.",
"Schulman J, Levine S, Abbeel P, et al. Trust region policy optimization. Proc Int Conf Mach Learn, 2015, 37：1889~1897.",
"Haarnoja T, Zhou A, Abbeel P, et al. Soft actor-critic：off-policy maximum entropy deep reinforcement learning with a stochastic actor. Proc Int Conf Mach Learn, 2018, 80：1861~1870.",
"Fujimoto S, van Hoof H, Meger D. Addressing function approximation error in actor-critic methods. Proc Int Conf Mach Learn, 2018, 80：1587~1596.",
"Lillicrap TP, Hunt JJ, Pritzel A, et al. Continuous control with deep reinforcement learning. Proc Int Conf Learn Represent, 2016.",
"Mnih V, Badia AP, Mirza M, et al. Asynchronous methods for deep reinforcement learning. Proc Int Conf Mach Learn, 2016, 48：1928~1937.",
"Christiano PF, Leike J, Brown TB, et al. Deep reinforcement learning from human preferences. Adv Neural Inf Process Syst, 2017, 30：4299~4307.",
"Ouyang L, Wu J, Jiang X, et al. Training language models to follow instructions with human feedback. Adv Neural Inf Process Syst, 2022, 35：27730~27744.",
"Duan Y, Chen X, Houthooft R, et al. Benchmarking deep reinforcement learning for continuous control. Proc Int Conf Mach Learn, 2016, 48：1329~1338.",
"Henderson P, Islam R, Bachman P, et al. Deep reinforcement learning that matters. Proc AAAI Conf Artif Intell, 2018, 32：3207~3214.",
"Mittal M, Yu C, Yu Q, et al. Orbit：a unified simulation framework for interactive robot learning environments. IEEE Robot Autom Lett, 2023, 8：3740~3747.",
"Schulman J, Moritz P, Levine S, et al. High-dimensional continuous control using generalized advantage estimation. Proc Int Conf Learn Represent, 2016.",
"Cohen J. Statistical power analysis for the behavioral sciences. 2nd ed. Hillsdale, Lawrence Erlbaum Associates, 1988：19~74.",
"Ma YJ, Liang W, Wang G, et al. Eureka：human-level reward design via coding large language models. Proc Int Conf Learn Represent, 2024.",
"Yu W, Gileadi N, Fu C, et al. Language to rewards for robotic skill synthesis. Proc Conf Robot Learn, 2023, 229：374~404.",
"Xie T, Zhao S, Wu CH, et al. Text2Reward：reward shaping with language models for reinforcement learning. Proc Int Conf Learn Represent, 2024.",
"Kwon M, Xie SM, Bullard K, et al. Reward design with language models. Proc Int Conf Learn Represent, 2023.",
"Wilcoxon F. Individual comparisons by ranking methods. Biometrics Bull, 1945, 1：80~83.",
"Narvekar S, Peng B, Leonetti M, et al. Curriculum learning for reinforcement learning domains：a framework and survey. J Mach Learn Res, 2020, 21：1~50.",
"Yang A, Li A, Yang B, et al. Qwen3 technical report. arXiv preprint, 2025：arXiv:2505.09388.",
]

TOC = [
("국문초록", "ⅰ"), ("영문초록", "ⅲ"), ("목차", "ⅴ"),
("Ⅰ. 서론", "1"), ("Ⅱ. 연구목적", "○"), ("Ⅲ. 연구재료 및 방법", "○"),
("Ⅳ. 연구성적", "○"), ("Ⅴ. 고찰", "○"), ("Ⅵ. 결론", "○"),
("참고문헌", "○"), ("부록", "○"),
]


def build():
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = BODY_FONT
    st.font.size = Pt(10.5)
    st._element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)

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
    center(doc, f"{PLACE['author_kr'].replace('　','')}의 {PLACE['degree_kr']}학위논문을 인정함.",
           21, before=50, after=24)
    for role in ["위원장", "위　원", "위　원"]:
        center(doc, f"{role}____________印", 14, before=12)
    center(doc, PLACE["school_kr"], 16, before=26)
    center(doc, PLACE["date_kr"], 14, before=12)

    # ------------------------------------------------ 전문부: 로마 소문자 면번호
    sec = doc.add_section(WD_SECTION.NEW_PAGE)
    setup_section(sec, numbering="lowerRoman", start=1)
    footer_page_number(sec, show=True)

    # 국문초록
    center(doc, "국문초록", 14, bold=True, before=6, after=10)
    center(doc, PLACE["title_kr"], 13, bold=True, after=14)
    center(doc, PLACE["author_kr"], 11)
    center(doc, f"(지도교수 : {PLACE['advisor_kr']})", 11)
    center(doc, PLACE["dept_kr"].replace(" 전공", ""), 11)
    center(doc, PLACE["school_kr"], 11, after=12)
    for t in ABSTRACT_KR:
        body(doc, t)
    para(doc, f"Key Words：{KEYWORDS_KR}", before=14)

    # 영문초록
    doc.add_page_break()
    center(doc, "ABSTRACT", 14, bold=True, before=6, after=10)
    center(doc, PLACE["title_en"], 13, bold=True, after=14)
    center(doc, PLACE["author_en"], 11)
    center(doc, f"(Advisor：{PLACE['advisor_en']})", 11)
    center(doc, PLACE["dept_en"], 11)
    center(doc, PLACE["school_en"], 11, after=12)
    for t in ABSTRACT_EN:
        body(doc, t)
    para(doc, f"Key Words：{KEYWORDS_EN}", before=14)

    # 목차
    doc.add_page_break()
    center(doc, "목　　차", 14, bold=True, before=6, after=14)
    for item, page in TOC:
        p = para(doc, "", spacing=1.6)
        set_font(p.add_run(item), size=11)
        tab = OxmlElement("w:ptab")
        tab.set(qn("w:alignment"), "right")
        tab.set(qn("w:relativeTo"), "margin")
        tab.set(qn("w:leader"), "dot")
        r = p.add_run()
        r._element.append(tab)
        set_font(p.add_run(str(page)), size=11)

    # ------------------------------------------------ 본문: 아라비아 면번호
    sec = doc.add_section(WD_SECTION.NEW_PAGE)
    setup_section(sec, numbering="decimal", start=1)
    footer_page_number(sec, show=True)

    heading(doc, "Ⅰ. 서　　론", 1)
    for sub, paras in INTRO:
        heading(doc, sub, 2)
        for t in paras:
            body(doc, t)

    heading(doc, "Ⅱ. 연구목적", 1)
    for sub, paras in PURPOSE:
        heading(doc, sub, 2)
        for t in paras:
            body(doc, t)

    for title, chapter in [
        ("Ⅲ. 연구재료 및 방법", MATERIALS),
        ("Ⅳ. 연구성적", RESULTS),
        ("Ⅴ. 고　　찰", DISCUSSION),
    ]:
        heading(doc, title, 1)
        for sub, paras in chapter:
            heading(doc, sub, 2)
            for t in paras:
                if len(t) < 40 and t[1:3] == ". ":
                    heading(doc, t, 3)
                else:
                    body(doc, t)

    heading(doc, "Ⅵ. 결　　론", 1)
    for t in CONCLUSION:
        body(doc, t)

    heading(doc, "참고문헌", 1)
    for i, ref in enumerate(REFERENCES, 1):
        p = para(doc, f"{i}. {ref}", spacing=1.6, after=4,
                 align=WD_ALIGN_PARAGRAPH.LEFT)

    out = __file__.replace("build_thesis.py", "thesis.docx")
    doc.save(out)
    print(f"saved: {out}")


if __name__ == "__main__":
    build()
