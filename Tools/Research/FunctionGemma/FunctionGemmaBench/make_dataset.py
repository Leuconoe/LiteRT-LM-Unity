#!/usr/bin/env python3
"""Synthetic Korean training data for fine-tuning FunctionGemma on the
VoiceCommand-WebServer ToolsPrompt.txt contract (44 tools incl. DefaultResponse).

Stdlib only and deterministic (fixed seed): a rerun reproduces its output byte
for byte. Every record is validated against the tool schema, and no utterance of
the held-out sets is ever emitted — compared after whitespace/punctuation
normalisation and again with every space removed. v1 filters eval_cases.json,
v2 adds heldout_cases.json, v3 heldout2_cases.json, v4 heldout3_cases.json.

    python make_dataset.py                                  # v6: data_v5_empty/ + 2 classes → data_v6_empty/
    python make_dataset.py --version 4                      # → data_v4_empty/
    python make_dataset.py --version 3                      # → data_v3_empty/
    python make_dataset.py --version 2 --default-args text  # → data_v2/
    python make_dataset.py --version 2                      # → data_v2_empty/
    python make_dataset.py --v1                             # the first set → data/
    python make_dataset.py --stats                          # also print per-tool counts and samples

v2 adds what the v1-trained model missed on colloquial speech: a situational
clause before ~15 % of records ('화면이 복잡하니까 …', '됐고, …') — positives and
DefaultResponse alike, so a clause never implies "execute" — and indirect verb
forms per tool family ('다른 걸로 추천해 줘', '추천은 됐어, 취소해', '원래대로 돌려 놔').
v3 adds three DefaultResponse classes of ~4 % each (negated commands, state
questions, unsupported near-domain actions), native number words in every
numeric slot ('마커 넷', '드론 둘'), 꺼내 (bring up) vs 꺼 (off), '<team> 전원 …',
and ASR spacing noise on ~15 % of records. v3 overshot into false refusals, so
v4 brings DefaultResponse back to ~20 %, puts leading (20 %) and trailing
('<command>, 시야 가린다', 15 %) clauses on executable and DefaultResponse records
alike, adds slot-less/deictic negatives ('그 팀 복귀', '저 마커 지워'), window on/off
verbs ('필터 창 꺼 줘' = CloseWindow, '추천 창 꺼' = CancelTacticsRecommendation)
and pause-is-not-stop. v5 (data_v5_empty/) is v4 plus radio-style LLM paraphrases,
built by paraphrase/paraphrase.py; v6 reads it and adds two-action negatives
('A하고 B', 'A, 그리고 B' …) and the contract's STT number codes ('마커 E', '이번
프리셋', '투번 마커', '이오기'), then trims template negatives back to 20 %.
Older versions skip every newer random draw, so each --version still
reproduces its earlier output exactly.

Record: {"text": utterance, "function": tool, "arguments": {...}}. Optional
arguments (team, soldierName) appear only when the utterance says them.
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_TOOLS_PROMPT = HERE.parents[4] / "VoiceCommand-WebServer" / "assets" / "ToolsPrompt.txt"
SEED = 20261016
VAL_FRACTION = 0.05
DEFAULT_RATIO = 0.21   # DefaultResponse records per positive record → ~17 % of all
STT_P = 0.33           # chance to corrupt a record that has correctable vocabulary
                       # (0.45 left ~41 % of 제어권/프리셋 lines corrupted — too dominant)
CLAUSE_P = 0.15        # v2: situational clause before the command
INDIRECT_P = 0.25      # v2: indirect phrasing instead of the base generator, where a tool has one
V3_EXTRA_P = 0.2       # v3: '꺼내 줘' / '<team> 전원 …' phrasing, where a tool has one
V3_DEFAULT_SHARE = 0.04  # v3: negated / state-question / near-domain records, each, of the final set
SPACE_NOISE_P = 0.15   # v3: ASR spacing — a space inside a word, or one dropped between words
V3 = False             # main() sets it; every v3 draw sits behind it, so v1/v2 replay unchanged
# v4 rebalances after v3 traded false actions for false refusals: DefaultResponse back to ~20 %,
# and clauses (leading 20 %, trailing 15 %) on every class alike, so a clause never decides the verdict.
DEFAULT_RATIO_V4 = 0.12
V4_CLASSES = [("negated", 0.025), ("state", 0.025), ("near", 0.025), ("deictic", 0.025), ("pause", 0.005)]
LEAD_P_V4, TRAIL_P_V4 = 0.20, 0.15
V4_EXTRA_P = 0.2       # v4: '<창> 켜 줘 / 꺼 줘' window on/off phrasing
V4 = False

R = random.Random(SEED)
pick = R.choice


def chance(p):
    return R.random() < p


# --------------------------------------------------------------- vocabulary

TEAMS = [("Alpha", "알파"), ("Bravo", "브라보"), ("Charlie", "찰리"), ("Delta", "델타"),
         ("Echo", "에코"), ("Foxtrot", "폭스트롯"), ("Golf", "골프"), ("Hotel", "호텔"),
         ("India", "인디아"), ("Juliet", "줄리엣"), ("Kilo", "킬로"), ("Lima", "리마"),
         ("Mike", "마이크"), ("November", "노벰버"), ("Oscar", "오스카"), ("Papa", "파파"),
         ("Quebec", "퀘벡"), ("Romeo", "로미오"), ("Sierra", "시에라"), ("Tango", "탱고"),
         ("Uniform", "유니폼"), ("Victor", "빅터"), ("Whiskey", "위스키"), ("X-ray", "엑스레이"),
         ("Yankee", "양키"), ("Zulu", "줄루")]
TEAM_ENUM = {en for en, _ in TEAMS}
TEAM_ALIASES = {en: [ko, en] for en, ko in TEAMS}
TEAM_ALIASES["Juliet"] += ["줄리에", "줄리의"]          # STT forms of 줄리엣
TEAM_TYPES = [("정찰", ["정찰"]), ("공격", ["공격", "투하", "폭격"]), ("자폭", ["자폭"]),
              ("UGV", ["UGV", "유지브이", "지상"])]
WINDOWS = {"알림 상세": ["알림 상세", "알림 상세 창", "알림 상세창", "알림 상세 화면"],
           "필터": ["필터", "필터 창", "필터 창", "필터창", "필터 메뉴"],
           "설정": ["설정", "설정 창", "설정창", "설정 화면", "설정 메뉴"],
           "전술 지도": ["전술지도", "전술 지도", "전술 지도 창", "전술지도 화면", "전술 지도 화면"]}
CONFIRMS = ["제어권 요청", "제어권 반납", "제어권 회수", "팀 해체", "드론봇 팀 제외",
            "자폭 공격", "전술 확정", "편집 취소", "게임 종료", "시뮬레이션 종료"]
FILTERS = ["드론 시야", "배터리", "팀 연결선", "장애물", "마커", "전장의 안개", "위협",
           "적 은엄폐", "아군 은엄폐", "주요 지형지물", "기동성", "기동로"]
SINO = dict(enumerate("일이삼사오육칠팔구십", 1))
NATIVE = dict(enumerate(["한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉", "열"], 1))
ORDINAL = {1: "첫", 2: "두", 3: "세", 4: "네", 5: "다섯"}
NATIVE_WORD = dict(enumerate(["하나", "둘", "셋", "넷", "다섯", "여섯", "일곱", "여덟", "아홉", "열"], 1))
NUMBER_TABLES = (SINO, NATIVE, NATIVE_WORD, ORDINAL)
NUMBER_WORDS = {w for t in NUMBER_TABLES for w in t.values()}
SURNAMES = list("김이박최정강조윤장임한오서신권황안송홍유")
RANKS = ["일병", "상병", "병장", "하사", "중사", "상사", "소위", "중위", "대위"]
BAD_TEAMS = ["독수리", "호랑이", "사과", "백두", "한라", "레드", "블루", "오메가", "감마",
             "베타", "피닉스", "코브라", "늑대", "태양", "번개", "용"]
DIRS = ["좌측", "우측", "전방", "후방", "북쪽", "남쪽", "동쪽", "서쪽", "왼쪽", "오른쪽"]

# The ToolsPrompt's own STT-correction table, applied in reverse.
STT_SUBS = [
    (r"제어권", ["제억원", "제역원", "지역원", "제혁원", "제어건"]),
    (r"이양", ["이항"]),
    (r"회수", ["횟수"]),
    (r"줄리엣", ["줄리에", "줄리의"]),
    (r"프리셋", ["프리새"]),
    (r"재추천", ["제추천"]),
    (r"귀환", ["기환", "기완"]),
    (r"복귀", ["복구"]),
    (r"개시", ["게시"]),
    (r"새 팀", ["새틴"]),
    (r"(?<!\d)(?:2|이)호기", ["이오기"]),
    (r"(?<!\d)(?:3|삼)호기", ["사모기"]),
    (r"(?<!\d)(?:4|사)호기", ["사오기"]),
    (r"(?<=마커 )2(?![\d번])", ["E", "이"]),
    (r"(?<=프리셋 )2(?![\d번])", ["E", "이"]),
    (r"닫아", ["다다", "달아"]),
    (r"지워", ["지어", "지요", "지호"]),
    (r"꺼(?= |$|줘)", ["거"]),
]
FILLERS = ["어 ", "음 ", "자 ", "저기 ", "아 ", "그 "]

# ------------------------------------------------------------------ helpers

_DIGIT_READING = "영일이삼사오육칠팔구"
_JOSA = {"을": ("를", "을"), "은": ("는", "은"), "과": ("와", "과"), "이나": ("나", "이나"),
         "이랑": ("랑", "이랑"), "으로": ("로", "으로")}


def j(word, josa):
    """Attach a particle in its right allomorph: j('알파', '을') → '알파를'."""
    ch = word[-1]
    if ch.isdigit():
        ch = _DIGIT_READING[int(ch)]          # '10' reads 십; 영 has a final too, so 0 works
    code = ord(ch) - 0xAC00
    final = code % 28 if 0 <= code < 11172 else 0   # Latin (UGV, E) reads vowel-final
    without, with_ = _JOSA[josa]
    if josa == "으로" and final == 8:          # ㄹ-final takes 로
        return word + "로"
    return word + (with_ if final else without)


HADA = ["", "", "", "", "", "해", "해 줘", "해줘", "해 주세요", "해주세요", "하자", "해라",
        " 부탁해", " 부탁합니다", " 해 줘", "할게", "합니다", " 바람", " 좀 해 줘"]


def expand(phrase):
    """'선택~' → '선택' / '선택해 줘' / ... (a trailing ~ marks a 하다 noun)."""
    return phrase[:-1] + pick(HADA) if phrase.endswith("~") else phrase


def attach(subject, phrase):
    """Join subject + predicate. '^x' puts x first; '을 x'/'은 x'/'으로 x' adds a particle."""
    phrase = expand(phrase)
    if phrase.startswith("^"):
        return f"{phrase[1:]} {subject}"
    for josa in ("을", "은", "으로"):
        if phrase.startswith(josa + " "):
            return j(subject, josa) + phrase[len(josa):]
    return f"{subject} {phrase}".strip()


def team():
    en, ko = pick(TEAMS)
    r = R.random()
    s = (ko + "팀" if r < 0.62 else ko + " 팀" if r < 0.72 else ko if r < 0.92
         else en + pick(["팀", " 팀"]))
    return s, en


def number():
    """Mostly 1-10 (what people say), sometimes up to 16 to keep digits general."""
    return R.randint(1, 10) if chance(0.9) else R.randint(11, 16)


def bare(n):
    if V3 and 2 <= n <= 10 and chance(0.12):     # '마커 넷', '프리셋 셋'; 하나 reads as a count
        return NATIVE_WORD[n]
    return SINO[n] if n <= 10 and chance(0.35) else str(n)


def bun(n):
    if n <= 10 and chance(0.35):
        return SINO[n] + pick(["번", "번", " 번"])
    return str(n) + pick(["번", "번", "번", " 번"])


def drone(with_team=None):
    """→ (surface, number, team or None). Forms: N번 드론, 드론 N번, N호기, <team> N호기 ..."""
    n = number()
    ho = (SINO[n] if n <= 10 and chance(0.3) else str(n)) + "호기"
    has_team = chance(0.35) if with_team is None else with_team
    if V3 and 2 <= n <= 10 and chance(0.08):    # '드론 둘'; '드론 하나' stays "some drone" (missing)
        if not has_team:
            return f"드론 {NATIVE_WORD[n]}", n, None
        en, ko = pick(TEAMS)
        return f"{pick([ko, ko + '팀', en + '팀'])} 드론 {NATIVE_WORD[n]}", n, en
    if not has_team:
        if n <= 5 and chance(0.08):
            return f"{ORDINAL[n]} 번째 드론", n, None
        return pick([f"{bun(n)} 드론"] * 3 + [f"드론 {bun(n)}", ho, ho]), n, None
    en, ko = pick(TEAMS)
    s = pick([ko, ko, ko + "팀", ko + " 팀", en + "팀"])
    if n <= 5 and chance(0.06):
        return f"{s} {ORDINAL[n]} 번째 드론", n, en
    return pick([f"{s} {ho}", f"{s} {ho}", f"{s} {bun(n)} 드론", f"{s} 드론 {bun(n)}",
                 f"{s} 소속 {bun(n)} 드론"]), n, en


def soldier():
    return pick(SURNAMES) + pick(RANKS)


def count_phrase(n, drone_word=True):
    if V3 and 2 <= n <= 10 and chance(0.15):    # '넷으로 팀 짜', '드론 셋'
        return pick([NATIVE_WORD[n], f"드론 {NATIVE_WORD[n]}"]) if drone_word else NATIVE_WORD[n]
    forms = [f"{n}기", f"{n}대", f"{n} 기", f"{NATIVE[n]} 대", f"{NATIVE[n]}대", f"{NATIVE[n]} 기"]
    if drone_word:
        forms += [f"드론 {NATIVE[n]} 대", f"드론 {n}대", f"드론 {n}기"]
    return pick(forms)


# --------------------------------------------------------- tool generators

# Generators that name a team, drone or number take an optional `subject`: v4 passes a deictic
# ('그 팀', '저 드론') to build missing-slot negatives. Without it the random draws are unchanged.

def team_tool(phrases):
    def gen(subject=None):
        s, en = team() if subject is None else (subject, None)
        return attach(s, pick(phrases)), {"team": en}
    return gen


def drone_tool(phrases):
    def gen(subject=None):
        d, n, en = drone() if subject is None else (subject, None, None)
        args = {"droneNumber": n}
        if en:
            args["team"] = en
        return attach(d, pick(phrases)), args
    return gen


VIEW_MOVE = ["이동", "이동해", "이동해 줘", "전환", "전환해 줘", "가 줘", "바꿔 줘", "옮겨 줘"]


def view_phrase(subject, views):
    base = f"{subject} {pick(views)}"
    return pick([base, base, j(base, "으로") + " " + pick(VIEW_MOVE)])


def g_team_view(subject=None):
    s, en = team() if subject is None else (subject, None)
    return view_phrase(s, ["팀 뷰", "팀 뷰", "뷰", "시점", "숄더 뷰", "팀 시점", "리더 시점",
                           "리더 숄더 뷰"]), {"team": en}


def g_team_overview(subject=None):
    s, en = team() if subject is None else (subject, None)
    if chance(0.2):
        return attach(s, pick(["전체가 보이게", "전체 보이게 해 줘", "줌 투 핏", "팀 전체 보여 줘",
                               "전체가 다 보이게 맞춰 줘"])), {"team": en}
    return view_phrase(s, ["팀 전체 뷰", "팀 전체 뷰", "전체 뷰", "전체 시점", "팀 전체 보기",
                           "전체 보기", "팀 전체 시점"]), {"team": en}


def g_drone_view(subject=None):
    d, n, en = drone() if subject is None else (subject, None, None)
    args = {"droneNumber": n}
    if en:
        args["team"] = en
    return view_phrase(d, ["드론 뷰", "드론 뷰", "뷰", "시점", "시점", "숄더 뷰", "드론 시점"]), args


def g_assign(subject=None):
    s, en = team() if subject is None else (subject, None)
    if not chance(0.4):
        return attach(s, pick(["제어권 지정~", "제어권 지정~", "제어권 이양~", "제어권 이양~",
                               "제어권 넘겨", "제어권 넘겨 줘", "제어권 넘겨줘", "제어권 넘길게",
                               "제어권을 이양~", "제어권을 넘겨 줘", "제어권 이양 창 열어 줘"])), {"team": en}
    who = soldier()
    to = who + pick(["한테", "에게"])
    t = pick([f"{s} {who} 제어권 " + expand(pick(["지정~", "이양~"])),
              f"{s} 제어권 {to} " + expand(pick(["넘겨", "넘겨 줘", "넘겨줘", "넘길게", "이양~", "지정~"])),
              f"{to} {s} 제어권 " + pick(["넘겨", "넘겨 줘", "이양해", "이양해 줘"]),
              f"{s} 제어권 {j(who, '으로')} " + expand("지정~")])
    return t, {"team": en, "soldierName": who}


MAKE = ["생성~", "생성~", "만들어", "만들어 줘", "만들어줘", "편성~", "구성~", "만들자"]


def g_create_team():
    canon, words = pick(TEAM_TYPES)
    ty, n = pick(words), pick([1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 8, 9, 10])
    c, c0 = count_phrase(n), count_phrase(n, False)
    t = pick([f"{ty} {c} 팀 {pick(MAKE)}", f"{ty} {c} 팀 {pick(MAKE)}",
              f"{ty} 드론 {j(c0, '으로')} 팀 {pick(MAKE)}", f"{c} {ty} 팀 {pick(MAKE)}",
              f"{ty}팀 {j(c, '으로')} {pick(MAKE)}", f"{ty} {c0}짜리 팀 {pick(MAKE)}",
              f"드론 {j(c0, '으로')} {ty} 팀 {pick(MAKE)}", f"{ty} 팀 {c} {pick(MAKE)}"])
    return expand(t), {"teamType": canon, "memberCount": n}


ENTER_TEAM = ["팀 생성~", "팀 생성~", "새 팀 만들기", "새 팀 생성~", "팀 편성 화면", "팀 하나 만들어",
              "팀 만들어", "팀 만들어 줘", "팀 만들기", "새 팀 만들어 줘", "새로운 팀 만들어",
              "새로운 팀 생성~", "팀 생성 화면", "팀 생성 화면 열어 줘", "팀 편성~", "팀 편성 화면 보여 줘",
              "팀 추가~", "새 팀 추가~", "팀 하나 추가~", "팀 새로 만들자", "팀 구성 화면", "새 팀 편성~"]


def g_enter_team():
    """Contrast with CreateTeam: type alone or count alone is still the creation screen."""
    r = R.random()
    if r < 0.2:
        t = pick([f"{pick(pick(TEAM_TYPES)[1])} 팀 {pick(MAKE)}", f"{pick(pick(TEAM_TYPES)[1])}팀 {pick(MAKE)}"])
    elif r < 0.32:
        n = R.randint(2, 8)
        t = pick([f"드론 {j(count_phrase(n, False), '으로')} 팀 {pick(MAKE)}",
                  f"{count_phrase(n)} 팀 {pick(MAKE)}", f"{count_phrase(n, False)}짜리 팀 {pick(MAKE)}"])
    else:
        t = pick(ENTER_TEAM)
    return expand(t), {}


MOVE = ["이동~", "이동~", "시점", "보여 줘", "보여줘", "가 줘", "시점으로", "시점으로 이동~"]
GO = ["이동~", "가", "가 줘", "가자", "이동해 줘"]
DELETE = ["삭제~", "삭제~", "지워", "지워 줘", "지워줘", "제거~", "없애", "없애 줘"]


def numbered(nouns, key, move):
    def gen(subject=None):
        if subject is None:
            n, noun = number(), pick(nouns)
            b, bn = bare(n), bun(n)
            subj = pick([f"{noun} {b}", f"{noun} {b}", f"{bn} {noun}", f"{noun} {bn}"])
        else:
            n, subj = None, subject
        if move:
            t = pick([f"{subj} {pick(MOVE)}", f"{subj} {pick(MOVE)}", j(subj, "으로") + " " + pick(GO)])
        else:
            t = pick([f"{subj} {pick(DELETE)}", f"{subj} {pick(DELETE)}", j(subj, "을") + " " + pick(DELETE)])
        return expand(t), {key: n}
    return gen


OPEN = ["호출~", "호출~", "열어", "열어 줘", "열어줘", "열기", "보여 줘", "보여줘", "띄워", "띄워 줘",
        "오픈~", "열어 주세요"]
CLOSE = ["닫기", "닫기", "닫아", "닫아 줘", "닫아줘", "닫아 주세요", "닫자", "치워", "치워 줘", "없애 줘"]


def window_tool(verbs):
    def gen():
        name = pick(list(WINDOWS))
        s = pick(WINDOWS[name])
        t = pick([f"{s} {pick(verbs)}", f"{s} {pick(verbs)}", j(s, "을") + " " + pick(verbs)])
        return expand(t), {"windowName": name}
    return gen


ACCEPT = ["승인~", "승인~", "승인~", "확인", "확인", "확인해", "확인해 줘", "수락~", "수락할게",
          "승인 바람", "승인할게", "오케이 승인"]
REJECT = ["거절~", "거절~", "거절~", "거부~", "거부할게", "취소", "취소해", "취소해 줘", "거절 바람",
          "안 돼 거절"]


def confirm_tool(verbs):
    def gen():
        c = pick(CONFIRMS)
        t = pick([f"{c} {pick(verbs)}"] * 3 + [j(c, "을") + " " + pick(verbs), f"{c} 창 {pick(verbs)}"])
        return expand(t), {"confirmName": c}
    return gen


FILTER_ON = ["설정~", "설정~", "켜", "켜", "켜 줘", "켜줘", "켜 주세요", "켜기", "표시~", "표시~",
             "보이게 해 줘", "활성화~"]
FILTER_OFF = ["해제~", "해제~", "꺼", "꺼", "꺼 줘", "꺼줘", "꺼 주세요", "끄기", "숨겨", "숨겨 줘",
              "숨기기", "안 보이게 해 줘", "비활성화~"]


def filter_tool(verbs):
    def gen():
        names = R.sample(FILTERS, pick([1, 1, 1, 2, 2, 3]))
        style = pick(["comma", "comma", "space", "space", "josa"]) if len(names) > 1 else "space"
        if style == "comma":
            said = pick([", ", ","]).join(names)
        elif style == "space":
            said = " ".join(names)
        else:
            said = names[0]
            for name in names[1:]:
                said = pick([j(said, "이랑"), j(said, "과"), said + "하고"]) + " " + name
        mid = pick(["", "", "", "", " 필터", " 표시"])
        verb = pick([v for v in verbs if not (mid == " 표시" and v.startswith("표시"))])
        return expand(f"{said}{mid} {verb}"), {"filterNames": ", ".join(names)}
    return gen


def fixed_tool(phrases):
    return lambda: (expand(pick(phrases)), {})


TOOLS = {  # name: (generator, target record count before the val split)
    "SelectTeam": (team_tool(["", "", "선택~", "선택~", "선택~", "을 선택~", "골라", "골라 줘", "잡아",
                              "잡아 줘", "선택할게", "으로 선택~", "으로 바꿔", "으로 전환~", "^선택"]), 300),
    "EnterTeamCreation": (g_enter_team, 200),
    "CreateTeam": (g_create_team, 300),
    "ReturnTeam": (team_tool(["복귀~", "복귀~", "귀환~", "귀환~", "돌아와", "돌아와 줘", "돌아오세요",
                              "돌아오게 해", "복귀시켜", "복귀시켜 줘", "귀환시켜 줘", "즉시 복귀~", "바로 복귀~",
                              "기지로 복귀~", "기지로 돌아와", "은 복귀해", "을 복귀시켜", "^복귀"]), 300),
    "DisbandTeam": (team_tool(["해체~", "해체~", "해산~", "팀 해체~", "팀 해산~", "해체시켜", "해체시켜 줘",
                               "해산시켜 줘", "을 해체~", "은 해산~"]), 300),
    "StartTeamTactics": (team_tool(["전술 시작~", "전술 시작~", "전술 개시~", "전술 운용 시작~", "작전 시작~",
                                    "작전 개시~", "전술 실행~", "전술 들어가", "전술 시작하자", "은 전술 시작~"]), 300),
    "EditTeamTactics": (team_tool(["전술 편집~", "전술 편집~", "전술 수정~", "전술 변경~", "작전 편집~", "전술 고쳐",
                                   "전술 고칠게", "전술 바꿀게", "전술 편집 모드", "전술 편집 화면", "은 전술 수정~"]), 300),
    "SelectDrone": (drone_tool(["선택~", "선택~", "선택~", "을 선택~", "골라", "골라 줘", "잡아", "잡아 줘",
                                "선택할게", "으로 선택~", "^선택"]), 300),
    "EditDroneTactics": (drone_tool(["전술 편집~", "전술 편집~", "전술 수정~", "전술 변경~", "전술 고쳐",
                                     "전술 편집 모드", "은 전술 수정~"]), 300),
    "ReturnDrone": (drone_tool(["복귀~", "복귀~", "귀환~", "돌아와", "돌아와 줘", "복귀시켜", "복귀시켜 줘",
                                "귀환시켜", "즉시 복귀~", "기지로 복귀~", "은 복귀해", "을 복귀시켜"]), 300),
    "DetachDrone": (drone_tool(["팀 제외", "팀 제외~", "팀에서 제외~", "내보내기", "내보내", "내보내 줘",
                                "팀에서 빼 줘", "팀에서 빼", "빼 줘", "팀에서 내보내", "을 팀에서 빼 줘", "을 내보내",
                                "개별 팀으로 편성~", "팀에서 분리~"]), 300),
    "StartManualFlight": (drone_tool(["개별제어", "개별제어", "개별 제어", "개별제어~", "개별 제어 시작~",
                                      "수동 제어~", "수동 비행~", "수동 조종~", "직접 조종할게", "수동으로 몰게",
                                      "개별제어 진입", "수동 제어 모드", "은 내가 직접 조종할게"]), 300),
    "AssignTeamControl": (g_assign, 300),
    "RevokeTeamControl": (team_tool(["제어권 회수~", "제어권 회수~", "제어권 가져와", "제어권 가져와 줘",
                                     "제어권 가져올게", "제어권 뺏어", "제어권 뺏어 와", "제어권 다시 가져와",
                                     "제어권 회수할게", "은 제어권 회수~", "제어권을 회수~", "제어권을 가져와"]), 300),
    "RequestTeamControl": (team_tool(["제어권 요청~", "제어권 요청~", "제어권 주세요", "제어권 줘",
                                      "제어권 좀 주세요", "제어권 주십시오", "제어권 요청할게", "제어권 요청해요",
                                      "제어권을 요청~", "제어권 저한테 주세요", "제어권 나한테 줘"]), 300),
    "ReturnTeamControl": (team_tool(["제어권 반납~", "제어권 반납~", "제어권 돌려줄게", "제어권 돌려드릴게요",
                                     "제어권 돌려줄게요", "제어권 돌려 드립니다", "제어권 반납하겠습니다",
                                     "제어권을 반납~", "제어권 반납할게"]), 300),
    "MoveToMarker": (numbered(["마커"] * 12 + ["LM"], "markerNumber", True), 220),
    "MoveToPreset": (numbered(["프리셋"], "presetNumber", True), 220),
    "EnterMarkerCreation": (fixed_tool([
        "마커 생성~", "마커 생성~", "마커 추가~", "마커 추가해 줘", "마커 찍을게", "마커 찍어", "마커 찍어 줘",
        "마커 만들어", "마커 만들어 줘", "새 마커", "새 마커 추가~", "마커 하나 추가~", "마커 하나 찍어",
        "마커 생성 모드", "마커 놓을게", "여기 마커 찍어 줘", "여기에 마커 추가~", "마커 등록~",
        "LM 생성~", "LM 추가~", "새 마커 생성~"]), 140),
    "CreatePreset": (fixed_tool([
        "프리셋 생성~", "프리셋 생성~", "프리셋 추가~", "현재 시점 프리셋으로 저장~", "지금 시점 프리셋으로 저장해 줘",
        "프리셋 저장~", "지금 화면 프리셋으로 저장해 줘", "이 시점 저장~", "이 시점 프리셋으로 등록~",
        "프리셋 만들어", "프리셋 만들어 줘", "프리셋 하나 추가~", "프리셋 새로 만들어", "현재 뷰 저장~",
        "시점 저장해 줘", "프리셋 등록~", "현재 시점 저장~"]), 140),
    "DeleteMarker": (numbered(["마커"] * 12 + ["LM"], "markerNumber", False), 220),
    "DeletePreset": (numbered(["프리셋"], "presetNumber", False), 220),
    "RequestTacticsRecommendation": (fixed_tool([
        "전술 추천~", "전술 추천~", "AI 전술 추천~", "전술 재추천~", "다시 추천해 줘", "다른 전술 추천~",
        "전술 추천 받아 줘", "전술 다시 추천~", "새로 추천해 줘", "다른 거 추천해 줘", "작전 추천~",
        "AI 추천~", "인공지능 전술 추천~", "추천 다시", "추천 전술 보여 줘", "전술 추천 창 열어 줘",
        "추천해 줘", "전술 하나 추천해 줘", "재추천~"]), 140),
    "CancelTacticsRecommendation": (fixed_tool([
        "추천 취소~", "추천 취소~", "전술 추천 취소~", "추천 창 닫아 줘", "전술 추천 창 닫아", "추천 닫기",
        "추천 그만", "전술 추천 그만", "추천 안 할래", "전술 비교 창 닫아", "추천 창 닫기", "전술 추천 닫아",
        "추천 그만해", "전술 비교 추천 창 닫아 줘", "추천 창 치워 줘"]), 140),
    "OpenWindow": (window_tool(OPEN), 250),
    "CloseWindow": (window_tool(CLOSE), 250),
    "ShowTeamCamera": (team_tool(["카메라", "카메라", "카메라 보여 줘", "카메라 보여줘", "카메라 띄워",
                                  "카메라 띄워 줘", "카메라 창", "카메라 창 열어 줘", "카메라 화면", "영상",
                                  "영상 보여 줘", "리더 카메라", "리더 드론 카메라", "카메라 켜 줘"]), 300),
    "ShowDroneCamera": (drone_tool(["카메라", "카메라", "영상", "영상", "카메라 보여 줘", "영상 보여 줘",
                                    "카메라 띄워", "영상 띄워 줘", "카메라 창", "카메라 화면", "카메라 열어 줘",
                                    "카메라 켜 줘", "영상 틀어 줘"]), 300),
    "ResetCameraGimbal": (fixed_tool([
        "카메라 영점 조정~", "짐벌 초기화~", "카메라 정면으로", "카메라 영점", "짐벌 리셋~", "카메라 각도 초기화~",
        "카메라 각도 원래대로", "짐벌 영점 맞춰", "카메라 정면 보게 해 줘", "카메라 정면으로 돌려",
        "짐벌 원래대로", "카메라 리셋~", "짐벌 영점 조정~", "카메라 정면으로 맞춰 줘"]), 140),
    "EnlargeCameraWindow": (fixed_tool([
        "카메라창 확대~", "카메라 창 확대~", "카메라 창 크게", "카메라 창 크게 해 줘", "카메라 창 키워",
        "카메라 창 키워 줘", "카메라창 크게", "카메라 화면 크게 해 줘", "카메라 창 늘려", "카메라 창 최대로",
        "카메라창 크게 만들어"]), 140),
    "ShrinkCameraWindow": (fixed_tool([
        "카메라창 축소~", "카메라 창 축소~", "카메라 창 작게", "카메라 창 작게 해 줘", "카메라 창 줄여",
        "카메라 창 줄여 줘", "카메라창 작게", "카메라 화면 작게 해 줘", "카메라 창 작게 만들어",
        "카메라창 줄여"]), 140),
    "ApplyDisplayFilters": (filter_tool(FILTER_ON), 450),
    "ClearDisplayFilters": (filter_tool(FILTER_OFF), 450),
    "AcceptConfirmWindow": (confirm_tool(ACCEPT), 300),
    "RejectConfirmWindow": (confirm_tool(REJECT), 300),
    "EnableVideoThrough": (fixed_tool([
        "비디오 쓰루 설정~", "비디오 쓰루 켜", "비디오 쓰루 켜 줘", "패스스루 켜", "패스스루 켜 줘",
        "비디오 쓰루 활성화~", "패스스루 활성화~", "비디오 스루 켜 줘", "패스 스루 켜", "비디오 쓰루 켜기",
        "패스스루 모드 켜", "비디오 쓰루 모드 켜 줘", "패스스루 설정~", "패스스루 켜줘"]), 140),
    "DisableVideoThrough": (fixed_tool([
        "비디오 쓰루 해제~", "비디오 쓰루 꺼", "비디오 쓰루 꺼 줘", "패스스루 꺼", "패스스루 꺼 줘",
        "비디오 쓰루 비활성화~", "패스스루 해제~", "비디오 스루 꺼 줘", "패스 스루 꺼", "패스스루 끄기",
        "비디오 쓰루 끄기", "패스스루 모드 꺼", "패스스루 꺼줘"]), 140),
    "MoveToTeamView": (g_team_view, 300),
    "MoveToTeamOverview": (g_team_overview, 300),
    "MoveToDroneView": (g_drone_view, 300),
    "ResetPanelPositions": (fixed_tool([
        "패널 원위치", "패널 원위치~", "패널 위치 초기화~", "패널 위치 리셋~", "패널 원래 자리로",
        "패널 위치 원래대로", "패널들 제자리로", "2D 패널 원위치", "패널 제자리", "패널 위치 되돌려",
        "패널 원위치 시켜 줘", "패널 초기화~", "패널들 원래 위치로 돌려 줘"]), 140),
    "StartRecording": (fixed_tool([
        "녹화 시작~", "녹화 시작~", "녹화해 줘", "전술 녹화 시작~", "녹화 켜", "녹화 개시~", "레코딩 시작~",
        "녹화 시작할게", "녹화 들어가", "지금부터 녹화해", "녹화 켜 줘"]), 140),
    "StopRecording": (fixed_tool([
        "녹화 중지~", "녹화 중지~", "녹화 멈춰", "녹화 그만", "녹화 종료~", "녹화 꺼", "전술 녹화 중지~",
        "녹화 끝", "레코딩 중지~", "녹화 정지~", "녹화 멈춰 줘", "녹화 그만해", "녹화 꺼 줘"]), 140),
}

# ---------------------------------------------------------- DefaultResponse

UNSUPPORTED = [
    "화면 밝게", "화면 어둡게", "밝기 올려", "밝기 내려", "밝기 낮춰 줘", "화면 밝기 올려 줘", "밝기 조절~",
    "볼륨 올려", "볼륨 내려", "볼륨 업", "볼륨 다운", "소리 키워", "소리 키워 줘", "소리 줄여", "음량 높여 줘",
    "음량 낮춰", "음소거~", "소리 꺼", "볼륨 조절~", "오늘 날씨 알려 줘", "오늘 날씨 어때", "지금 몇 시야",
    "안녕", "안녕하세요", "고마워", "수고했어", "너 누구야", "노래 틀어 줘", "전화 걸어 줘", "메시지 보내 줘",
    "알람 맞춰 줘", "사진 찍어 줘", "와이파이 켜", "블루투스 꺼", "전원 꺼", "시스템 재부팅~", "스크린샷 찍어",
    "화면 캡처~", "멀티모달 데이터 목록 보여 줘", "멀티모달 데이터 꺼", "멀티모달 데이터 표시~",
    "상황도 정렬~", "오늘 상황인지 결과 열람~", "상황인지 결과 보여 줘", "위협평가 결과 보여 줘",
    "타격 스케줄링 결과 보여 줘", "스케줄링 결과 열람~", "모니터링 바 열어 줘", "환경 미니맵 보여 줘",
    "팀 정보 패널 열어", "음성 명령 패널 닫아", "지도 이동 모드 켜", "카메라 줌 인", "카메라 줌 아웃",
    "줌 인", "줌 아웃", "지도 확대~", "지도 축소~", "오늘 점심 뭐야", "배고프다", "이거 어떻게 해", "잘 들려",
    "테스트 테스트", "마이크 테스트", "잠깐만", "알겠습니다", "네", "아니요", "전술 평가 결과 보고~",
    "게임 저장~", "일시 정지~", "다음", "이전", "뒤로 가기", "도움말", "전체 화면", "언어 바꿔 줘",
    "영어로 바꿔", "녹화 시작했다가 바로 중지해", "패스스루 켰다가 꺼 줘", "카메라창 확대했다가 축소해",
    "마커 만들고 바로 지워", "적 위치 알려 줘", "남은 탄약 몇 발이야", "배터리 몇 퍼센트야",
]
MISSING = [
    "드론 선택~", "드론 하나 선택~", "드론 복귀~", "드론 귀환~", "드론 카메라", "드론 영상 보여 줘", "드론 뷰",
    "드론 시점으로 이동~", "드론 개별제어", "드론 수동 제어~", "드론 전술 편집~", "드론 팀에서 빼 줘",
    "드론 내보내기", "팀 선택~", "팀 복귀~", "팀 해체~", "팀 전술 시작~", "팀 전술 편집~", "팀 카메라", "팀 뷰",
    "팀 전체 뷰", "제어권 회수~", "제어권 지정~", "제어권 이양~", "제어권 가져와", "제어권 넘겨 줘",
    "제어권 요청~", "제어권 반납~", "제어권 주세요", "마커 삭제~", "마커 지워 줘", "마커 이동~", "마커로 이동~",
    "마커 시점으로", "프리셋 이동~", "프리셋 삭제~", "프리셋 지워", "프리셋으로 이동~", "창 열어 줘", "창 닫아",
    "창 호출~", "승인~", "거절~", "확인", "거부~", "카메라", "카메라 보여 줘", "시점 이동~", "뷰 전환~",
    "전술 편집~", "전술 시작~", "복귀~", "선택~",
]
POSITIVE = list(TOOLS)
_BARE_PREDICATE = re.compile(r"(선택|복귀|귀환|시작|중지|해체|해산|편집|회수|요청|반납|지정|이양|이동|삭제"
                             r"|생성|추가|확대|축소|호출|초기화|승인|거절|확인|취소|해제|설정|표시)$")


def g_multi():
    """Two executable actions in one utterance."""
    while True:
        a_tool, b_tool = pick(POSITIVE), pick(POSITIVE)
        if a_tool == b_tool and a_tool.endswith("DisplayFilters"):
            continue        # 'A 켜고 B 켜' could fairly be one call with two filters
        a, b = TOOLS[a_tool][0]()[0], TOOLS[b_tool][0]()[0]
        if a == b or (a_tool == "SelectTeam" and not re.search("선택|골라|잡아|바꿔|전환", a)):
            continue        # bare '알파' + '2호기 선택' reads as one drone selection
        break
    if _BARE_PREDICATE.search(a):
        return a + pick(["하고 ", "하고 ", "한 다음 ", "하고 나서 "]) + b
    return a + pick([" 그리고 ", ", ", " 그 다음 ", " 그리고 나서 "]) + b


def g_deprecated():
    """Move commands removed from the client contract on 2026-09-23."""
    s, _ = team()
    d, _, _ = drone(False)
    n, m, dr = number(), pick(["10", "20", "30", "50", "100", "삼십", "오십", "백"]), pick(DIRS)
    return pick([f"{s} 웨이포인트 {n} 이동~", f"{j(s, '을')} {n}번 웨이포인트로 이동해 줘",
                 f"{s} {n}번 경유지로 보내", f"{s} 경유지 {j(str(n), '으로')} 이동~", f"{d} {m}미터 {dr} 이동~",
                 f"{d} {j(dr, '으로')} {m}미터 이동해", f"{s} {j(dr, '으로')} 이동~",
                 f"{s} {j(dr, '으로')} {m}미터 가", f"{d} {j(dr, '으로')} 이동해 줘", f"{s} {n}번 작전 지역으로 이동~",
                 f"{d} 고도 {m}미터 올려", f"{s} 전진", f"{d} 앞으로 가", f"{s} {dr} 방향으로 이동~"])


def g_ambiguous():
    (sa, ea), (sb, eb) = team(), team()
    while eb == ea:
        sb, eb = team()
    d1, n1, _ = drone(False)
    d2, n2, _ = drone(False)
    act = pick(["선택", "복귀", "해체", "전술 시작", "카메라", "제어권 회수", "팀 뷰"])
    forms = [f"{j(sa, '이나')} {sb} {act}", f"{sa} 또는 {sb} {act}", f"{j(sa, '과')} {sb} 중 하나 {act}",
             f"{j(sa, '과')} {sb} {act}"]
    if n1 != n2:
        forms += [f"{j(d1, '이나')} {d2} 선택", f"{d1} 아니면 {d2} 복귀"]
    return pick(forms)


def g_invalid_team():
    return f"{pick(BAD_TEAMS)}팀 " + expand(pick(["선택~", "복귀~", "해체~", "전술 시작~", "카메라", "팀 뷰",
                                                  "제어권 회수~", "제어권 요청~", "전술 편집~"]))


def g_default():
    r = R.random()
    if r < 0.30:
        return g_multi(), {}
    if r < 0.60:
        return expand(pick(UNSUPPORTED)), {}
    if r < 0.78:
        return expand(pick(MISSING_ACTIVE)), {}
    if r < 0.90:
        return expand(g_deprecated()), {}
    if r < 0.95:
        return g_ambiguous(), {}
    return g_invalid_team(), {}


# ------------------------------------------------------------ v2 additions
# Clauses carry no callsign, number, filter/window/tool name or command verb, so
# they never change the label (validate() asserts the callsign part).

CLAUSES = [
    "화면이 너무 복잡해,", "화면이 잘 안 보여,", "전력이 얼마 안 남았어,", "전력이 간당간당하네,",
    "카메라가 엉뚱한 데를 보고 있어,", "영상이 좀 흔들리네,", "팀이 너무 흩어졌어,", "적이 접근하고 있어,",
    "상황이 바뀌었어,", "작전이 거의 끝나 가,", "시간이 별로 없어,", "지금 급해,", "빨리빨리,", "서둘러,",
    "아 진짜 답답하네,", "이거 왜 이래,", "좀 헷갈리네,", "아까부터 계속 이러네,", "좋아,", "알겠어,",
    "됐고,", "오케이,", "그래,", "응,", "자 이제,", "그럼,", "다음은,", "일단,", "아 맞다,", "근데,",
    "미안한데,", "부탁인데,", "화면이 복잡하니까", "잘 안 보이니까", "시간이 없으니까", "급하니까",
    "상황이 바뀌었으니까", "적이 가까우니까", "전력이 부족하니까", "다시 봐야 하니까", "작전 준비해야 하니까",
    "위험하니까", "정신없으니까", "화면이 답답해서", "잘 안 보여서", "상황 파악하려고", "다음 작전 때문에",
    "이제 다 끝났으니까", "그건 됐으니까", "방금 건 잘 됐고,", "좀 정리하고 싶은데", "한번 보자,",
]


def clause():
    c = pick(CLAUSES)
    return c + ("" if c.endswith(",") else pick(["", ","])) + " "


def g_enter_team_v2():
    n = R.randint(2, 8)
    c0 = count_phrase(n, False)
    return pick([f"{j(c0, '으로')} 팀 짜", f"드론 {j(c0, '으로')} 팀 짜 줘", f"{j(c0, '으로')} 팀 하나 꾸려 줘",
                 "팀 하나 짜 줘", "새로 팀 꾸려 줘", "팀 꾸리자", "팀 짜자", "팀 새로 짜 줘",
                 f"{c0}짜리 팀 꾸려", "팀 하나 꾸려 봐"]), {}


def g_create_team_v2():
    canon, words = pick(TEAM_TYPES)
    ty, n = pick(words), R.randint(1, 8)
    c, c0 = count_phrase(n), count_phrase(n, False)
    return pick([f"{ty} {j(c, '으로')} 팀 짜 줘", f"{ty} 드론 {j(c0, '으로')} 팀 꾸려", f"{j(c0, '으로')} {ty} 팀 짜",
                 f"{ty} {c} 팀 꾸려 줘", f"{ty} {c0}짜리 팀 하나 짜 줘"]), {"teamType": canon, "memberCount": n}


INDIRECT = {
    "RequestTacticsRecommendation": fixed_tool([
        "다른 걸로 추천해 줘", "다른 걸로 다시 추천~", "새로 추천해 봐", "새로 추천받자", "다른 전술은 없어",
        "다른 전술안 보여 줘", "이 전술 말고 다른 걸로 추천해 줘", "다른 안으로 추천~", "추천 새로 받아 줘",
        "다른 방법 추천해 줘", "전술 하나 더 추천해 줘", "더 나은 전술 추천해 줘", "다른 걸로 다시 뽑아 줘",
        "추천 한 번 더"]),
    "CancelTacticsRecommendation": fixed_tool([
        "추천은 됐어", "추천 됐어, 취소해", "추천은 이제 됐어", "됐어, 추천 취소~", "추천 필요 없어", "추천 안 받을래",
        "추천 창은 치워", "추천은 그만 보여 줘", "됐어, 추천 창 닫아", "전술 추천은 됐어, 취소해",
        "추천은 됐으니까 닫아 줘", "추천 이제 필요 없어, 닫아"]),
    "ResetCameraGimbal": fixed_tool([
        "카메라 원래대로 돌려 놔", "카메라 정면으로 돌려 놔", "짐벌 원래대로 돌려 놔", "카메라 각도 원래대로 해 줘",
        "카메라 다시 정면 보게 해", "카메라 앞으로 돌려 놔", "카메라 처음 각도로 돌려", "짐벌 정면으로 돌려 줘",
        "카메라 각도 되돌려 줘", "카메라 시선 정면으로"]),
    "ResetPanelPositions": fixed_tool([
        "패널 원래대로 돌려 놔", "패널 제자리로 돌려 놔", "패널들 원래 자리에 돌려 놔", "패널 위치 처음대로",
        "패널 다 원래 위치로", "흩어진 패널 원위치", "패널들 처음 자리로 돌려 줘"]),
    "EnterTeamCreation": g_enter_team_v2,
    "CreateTeam": g_create_team_v2,
    "EnterMarkerCreation": fixed_tool(["마커 하나 박아", "마커 하나 박아 줘", "여기 마커 하나 놔 줘", "마커 하나 꽂아",
                                       "새 마커 하나 놓자"]),
    "CreatePreset": fixed_tool(["지금 화면 기억해 둬", "이 시점 프리셋으로 남겨 둬", "현재 시점 저장해 둬",
                                "지금 보는 거 프리셋으로", "이 화면 프리셋에 넣어 줘"]),
    "StartRecording": fixed_tool(["녹화 떠 줘", "지금부터 기록해 둬", "녹화 돌려 줘", "녹화 걸어 줘"]),
    "StopRecording": fixed_tool(["녹화 끊어", "녹화 이제 됐어", "녹화 그만 떠", "녹화 꺼 버려"]),
    "EnlargeCameraWindow": fixed_tool(["카메라 창 좀 크게 봐야겠어", "카메라 창 키워 봐", "카메라 창 넓게"]),
    "ShrinkCameraWindow": fixed_tool(["카메라 창 좀 작게 봐", "카메라 창 줄여 봐", "카메라 창 좁게"]),
    "EnableVideoThrough": fixed_tool(["패스스루 좀 켜 봐", "비디오 쓰루로 바꿔 줘", "패스스루로 전환~"]),
    "DisableVideoThrough": fixed_tool(["패스스루 그만", "비디오 쓰루 이제 됐어", "패스스루 끊어"]),
    "SelectTeam": team_tool(["으로 할게", "잡자", "으로 바꿔 줘"]),
    "ReturnTeam": team_tool(["들어와", "이제 들어와", "그만 들어와", "불러들여", "불러들여 줘", "철수시켜", "다시 데려와",
                             "은 이제 들어와"]),
    "DisbandTeam": team_tool(["팀 풀어", "팀 풀어 줘", "해체해 버려", "은 해체하자", "팀 해체시켜 버려"]),
    "StartTeamTactics": team_tool(["작전 들어가", "전술대로 움직여", "전술 돌입~", "은 전술대로 가", "전술 개시하자"]),
    "EditTeamTactics": team_tool(["전술 좀 손볼게", "전술 다시 짜자", "전술 손봐 줘"]),
    "RevokeTeamControl": team_tool(["제어권 내가 가져갈게", "제어권 이리 가져와", "제어권 다시 내놔",
                                    "제어권은 내가 회수할게"]),
    "RequestTeamControl": team_tool(["제어권 저 주세요", "제어권 제가 받을게요"]),
    "ReturnTeamControl": team_tool(["제어권 다시 드릴게요", "제어권 이제 돌려드려요", "제어권 이제 안 쓸게요", "제어권 놓을게"]),
    "ShowTeamCamera": team_tool(["카메라 좀 보자", "카메라로 보여 줘", "영상 좀 보자", "리더 카메라 띄워 봐"]),
    "MoveToTeamView": team_tool(["쪽으로 시점 옮겨 줘", "쪽 시점으로", "쪽 뷰로 넘겨 줘", "리더 어깨 너머 시점으로",
                                 "쪽으로 뷰 바꿔"]),
    "MoveToTeamOverview": team_tool(["다 보이게 해 줘", "전체가 한눈에 보이게", "전부 보이게 줌 아웃",
                                     "다 나오게 화면 맞춰 줘"]),
    "SelectDrone": drone_tool(["으로 바꿔", "잡자", "으로 할게", "으로 선택 바꿔"]),
    "ReturnDrone": drone_tool(["들어와", "이제 들어와", "불러들여", "다시 데려와", "철수시켜", "은 들어오게 해"]),
    "DetachDrone": drone_tool(["따로 빼", "따로 빼 줘", "팀에서 떼어 내", "팀에서 떼어 줘", "은 따로 움직이게 빼 줘"]),
    "StartManualFlight": drone_tool(["내가 몰게", "직접 몰아 볼게", "은 내가 조종할게", "내가 직접 조작할게"]),
    "EditDroneTactics": drone_tool(["전술 좀 손볼게", "전술 다시 짜자", "전술 손봐 줘"]),
    "ShowDroneCamera": drone_tool(["카메라 좀 보자", "영상 좀 보자", "카메라로 보여 줘"]),
    "MoveToDroneView": drone_tool(["쪽 시점으로", "시점으로 넘겨 줘", "어깨 너머로 보여 줘"]),
    "OpenWindow": window_tool(["좀 볼게", "들어가", "들어가 줘", "펼쳐", "펼쳐 줘", "보자"]),
    "CloseWindow": window_tool(["나가", "그만 볼래", "내려 줘", "이제 닫아도 돼", "치워 버려"]),
    "ApplyDisplayFilters": filter_tool(["보여 줘", "좀 보자", "띄워 줘", "다시 보여 줘"]),
    "ClearDisplayFilters": filter_tool(["안 봐도 돼", "그만 보여 줘", "치워 줘", "치워", "없애 줘", "안 보이게"]),
    "AcceptConfirmWindow": confirm_tool(["받아 줘", "허락해", "그래 승인", "진행해", "좋아 승인해"]),
    "RejectConfirmWindow": confirm_tool(["안 돼", "받지 마", "하지 마", "반려~", "안 할래"]),
}
# Indirect phrasing with no object is still a missing argument — so '됐어' or
# '돌려 놔' alone must not learn to mean a particular cancel or reset.
MISSING_V2 = ["됐어 취소해", "그건 됐어", "다른 걸로 해 줘", "원래대로 돌려 놔", "돌려 놔", "이제 됐어", "그거 치워",
              "다른 걸로 바꿔", "좀 보자", "다시 해 줘"]
MISSING_ACTIVE = MISSING   # main() widens this for v2


# ------------------------------------------------------------ v3 additions

V3_EXTRA = {
    # 꺼내 = take out / bring up. Never '꺼' (off): the STT rule and space noise both skip it.
    "OpenWindow": window_tool(["꺼내 줘", "꺼내 봐", "꺼내", "꺼내 줄래"]),
    "ShowTeamCamera": team_tool(["카메라 꺼내 줘", "카메라 꺼내 봐", "영상 꺼내 줘", "카메라 화면 꺼내", "카메라 꺼내"]),
    "ShowDroneCamera": drone_tool(["카메라 꺼내 줘", "카메라 꺼내 봐", "영상 꺼내 줘", "카메라 화면 꺼내", "카메라 꺼내"]),
    # 전원 / 다 = every member of the team — not 제어권, and not 전원 (power) as in '전원 꺼'.
    "ReturnTeam": team_tool(["전원 복귀~", "전원 귀환~", "다 복귀~", "다 돌아와", "전원 돌아와", "다 들어와",
                             "전원 철수시켜", "전원 즉시 복귀~"]),
    "StartTeamTactics": team_tool(["전원 전술 시작~", "다 전술 시작~", "전원 작전 개시~", "전원 전술대로 움직여"]),
    "SelectTeam": team_tool(["전원 선택~", "다 선택~", "전원 잡아"]),
    "DisbandTeam": team_tool(["전원 해산~", "다 해산~", "전원 해체~"]),
    "MoveToTeamOverview": team_tool(["전원 보이게 해 줘", "다 보이게 맞춰 줘", "전원 다 나오게"]),
    "RevokeTeamControl": team_tool(["전원 제어권 회수~", "다 제어권 가져와"]),
}
MISSING_V3 = ["그대로 있어", "가만히 있어", "아직 하지 마", "하지 마", "그거 하지 마", "잠깐 기다려"]

# Verb endings a command can carry, and the stem that takes -지 마 / -면.
NEG_STEMS = [("돌아와", "돌아오"), ("가져와", "가져오"), ("만들어", "만들"), ("내보내", "내보내"), ("열어", "열"),
             ("닫아", "닫"), ("지워", "지우"), ("보여", "보여 주"), ("띄워", "띄우"), ("숨겨", "숨기"),
             ("넘겨", "넘기"), ("치워", "치우"), ("골라", "고르"), ("잡아", "잡"), ("없애", "없애"),
             ("시켜", "시키"), ("켜", "켜"), ("꺼", "끄"), ("빼", "빼"), ("가", "가"), ("해", "하")]
             # 시켜 before 켜: '해체시켜' must become '해체시키지 마', not '해체시켜지 마'
# '<확인 창> 하지 마' already means RejectConfirmWindow (v2), so confirm tools are not negated here.
NEG_TOOLS = [t for t in TOOLS if t not in ("AcceptConfirmWindow", "RejectConfirmWindow")]


def _cond(stem):
    code = ord(stem[-1]) - 0xAC00
    final = code % 28 if 0 <= code < 11172 else 0
    return stem + ("으면" if final and final != 8 else "면")


def g_negated():
    """'<command> 하지 마' and friends: a command that is explicitly not to be run."""
    for _ in range(50):
        text = TOOLS[pick(NEG_TOOLS)][0]()[0]
        base = re.sub(r"\s*(줘|주세요|줘요)$", "", text)
        if base.endswith("부탁해"):
            continue
        if _BARE_PREDICATE.search(base):
            stem = base + pick(["", " "]) + "하"
            break
        stem = next((base[:-len(e)] + s for e, s in NEG_STEMS if base.endswith(e)), None)
        if stem:
            break
    else:
        return "그거 하지 마"
    return pick([f"{stem}지 마", f"{stem}지 말아", f"{stem}지 마라", f"아직 {stem}지 마", f"{stem}지 마, 그대로 있어",
                 _cond(stem) + " 안 돼", f"절대 {stem}지 마", f"{stem}지 말고 기다려", f"지금은 {stem}지 마"])


STATE_MISC = ["녹화 중이야", "녹화되고 있어", "지금 녹화 중", "몇 번 드론이 선택됐어", "지금 선택된 팀이 뭐야",
              "지금 선택된 드론 몇 번이야", "배터리 몇 퍼센트야", "전력 얼마나 남았어", "전술 추천 끝났어",
              "추천 결과 나왔어", "마커 몇 개 있어", "프리셋 몇 개 저장돼 있어", "지금 몇 번 마커야",
              "카메라 정면 보고 있어", "제어권 나한테 있어", "누가 제어권 가지고 있어", "적 몇 명이야",
              "남은 시간 얼마야", "지금 어느 시점이야", "지금 무슨 뷰야", "패널 원위치 돼 있어",
              "카메라 창 확대돼 있어", "팀 몇 개야", "드론 몇 대 남았어"]
STATE_THINGS = FILTERS + ["비디오 쓰루", "패스스루", "카메라 창", "녹화", "전술 추천 창"]


def g_state():
    """A question about state, not a command."""
    s, _ = team()
    d, _, _ = drone()
    thing, w = pick(STATE_THINGS), pick(pick(list(WINDOWS.values())))
    q = pick([f"{s} 지금 어디 있어", f"{s} 어디야", f"{s} 상태 어때", f"{s} 몇 기 남았어", f"{s} 전력 얼마나 남았어",
              f"{s} 제어권 누구한테 있어", f"{s} 전술 시작했어", f"{s} 복귀했어", f"{s} 아직 살아 있어",
              f"{s} 지금 뭐 하고 있어", f"{d} 어디 있어", f"{d} 배터리 몇 퍼센트야", f"{d} 상태 어때",
              f"{d} 복귀했어", f"{d} 선택됐어", f"{d} 카메라 켜져 있어", f"{thing} 켜져 있나", f"{thing} 켜져 있어",
              f"{thing} 꺼져 있어", f"{thing} 지금 표시되고 있어", f"{thing} 켜졌어", f"{w} 열려 있어",
              f"{w} 열려 있나", f"{w} 닫혔어", f"{w} 떠 있어", pick(STATE_MISC), pick(STATE_MISC)])
    if q.endswith("어") and chance(0.25):
        q += "요"
    return q + pick(["?", "?", "?", ""])


def g_near():
    """Unsupported actions worded like the real tools: zoom, formation, map scale, speed, altitude,
    heading, targeting, weapons, waypoints."""
    s, _ = team()
    d, _, _ = drone()
    x, m = pick(["2", "3", "두", "세"]), pick(["50", "100", "백", "이백"])
    return expand(pick([
        f"{d} 카메라 줌 인~", f"{d} 카메라 줌 아웃~", f"{s} 카메라 줌 당겨 줘", f"카메라 배율 {x}배로 올려",
        f"{d} 영상 줌 인~", f"카메라 줌 {x}배", "영상 배율 낮춰",
        f"{s} 대형 바꿔", f"{s} 횡대 대형으로 전환~", f"{s} 쐐기 대형~", f"{s} 종대로 정렬~", f"{s} 대형 유지~",
        f"{s} 산개 대형으로",
        "지도 축척 바꿔", "지도 축척 키워 줘", "지도 줌 인~", "지도 줌 아웃~", "전술지도 확대~", "전술 지도 축소~",
        "지도 더 크게 보여 줘", "지도 배율 올려",
        f"{d} 속도 올려", f"{s} 속도 줄여", f"{d} 최고 속도로", f"{s} 천천히 이동~", f"{d} 속도 두 배로",
        f"{d} 고도 올려", f"{d} 고도 {m}미터로", f"{s} 고도 낮춰", f"{d} 상승~", f"{d} 하강~", f"{d} 착륙~",
        f"{s} 이륙~",
        f"{d} 오른쪽으로 회전~", f"{s} 방향 틀어", f"{d} 기수 북쪽으로", f"{d} 좌로 돌아", f"{s} 방향 전환~",
        f"{d} 180도 회전~",
        f"{s} 타격 대상 지정~", "타격 대상 지정~", f"{d} 저 표적 공격~", f"{s} 적 전차 공격~", f"{d} 목표 타격~",
        f"{d} 자폭 공격 실행~", f"{s} 공격 개시~",
        f"{d} 무장 활성화~", f"{d} 무장 해제~", f"{s} 사격 개시~", f"{d} 미사일 발사~", "폭탄 투하~",
        f"{d} 탄약 재장전~",
        f"{s} 웨이포인트 추가~", f"{d} 경유지 설정~", "웨이포인트 삭제~", f"{s} 정찰 경로 생성~", f"{d} 경로 저장~",
        f"{s} 경로 표시~"]))


def space_noise(text):
    """ASR spacing: a space inside a word ('프 리셋') or one dropped between words ('설정호출').
    Never inside a digit run, never after 꺼/켜 (so 꺼내 and 꺼져 never read as the off/on verb), and
    never leaving a fragment that is a number word ('프리 셋 2번' would put a 3 next to the 2)."""
    for _ in range(pick([1, 1, 2])):
        if chance(0.6):
            spots = [i for i in range(1, len(text))
                     if "가" <= text[i - 1] <= "힣" and "가" <= text[i] <= "힣" and text[i - 1] not in "꺼켜"
                     and text[:i].split(" ")[-1] not in NUMBER_WORDS and text[i:].split(" ")[0] not in NUMBER_WORDS]
            if spots:
                i = pick(spots)
                text = text[:i] + " " + text[i:]
        else:
            spaces = [k.start() for k in re.finditer(" ", text)]
            if spaces:
                i = pick(spaces)
                text = text[:i] + text[i + 1:]
    return text


# ------------------------------------------------------------ v4 additions

# '<command>, <reason>': the observation/reason clauses of CLAUSES read as afterthoughts too
# (acknowledgements like '좋아,' do not), plus afterthought-only ones.
LEAD_ONLY = {"좋아,", "알겠어,", "됐고,", "오케이,", "그래,", "응,", "자 이제,", "그럼,", "다음은,", "일단,",
             "아 맞다,", "근데,", "미안한데,", "부탁인데,", "한번 보자,", "빨리빨리,", "서둘러,", "방금 건 잘 됐고,"}
TRAIL_CLAUSES = [c.rstrip(",") for c in CLAUSES if c not in LEAD_ONLY] + [
    "시야 가린다", "시야를 가려", "작아서 안 보여", "너무 작아", "너무 커", "가려서 안 보여", "방해돼", "거슬려",
    "지금 급해", "빨리", "지금 바로", "얼른", "당장", "다시 보려고", "잠깐 보게", "정리 좀 하게", "헷갈려",
    "적 온다", "시간 없어", "급한 상황이야", "답답하네", "잘 안 보이네"]


def trail():
    return pick([", ", ", ", ",", " "]) + pick(TRAIL_CLAUSES)


def window_switch(on):
    """'<창> 켜 줘' / '<창> 꺼 줘' — the on/off verbs on a window. 필터 always with 창, since
    '<필터 명> 켜/꺼' is a display filter."""
    verbs = ["켜", "켜 줘", "켜줘", "켜 주세요", "켜기"] if on else ["꺼", "꺼 줘", "꺼줘", "꺼 주세요", "끄기"]

    def gen():
        name = pick(list(WINDOWS))
        s = pick(["필터 창", "필터창"] if name == "필터" else WINDOWS[name])
        return f"{s} {pick(verbs)}", {"windowName": name}
    return gen


V4_EXTRA = {
    "OpenWindow": window_switch(True),
    "CloseWindow": window_switch(False),
    "RequestTacticsRecommendation": fixed_tool([
        "전술 추천 창 켜 줘", "전술 추천 창 켜", "추천 창 켜", "추천 창 켜 줘", "추천 창 열어 줘", "전술 추천 창 열어",
        "전술 추천창 띄워 줘", "추천창 켜줘", "추천 창 다시 켜", "전술 추천 창 열어 줘"]),
    "CancelTacticsRecommendation": fixed_tool([
        "전술 추천 창 꺼", "전술 추천 창 꺼 줘", "추천 창 꺼 줘", "추천 창 꺼", "전술 추천창 꺼줘", "추천창 끄기",
        "전술 추천 창 닫아 줘", "추천 창 꺼 주세요", "전술 추천 창 끄기"]),
}

DEICTIC_TEAM = ["그 팀", "저 팀", "아까 그 팀", "이 팀", "그쪽 팀", "다른 팀", "해당 팀", "방금 그 팀"]
DEICTIC_DRONE = ["그 드론", "저 드론", "아까 그 드론", "이 드론", "그 기체", "저 기체", "해당 드론", "방금 그 드론",
                 "다른 드론"]
NUMBER_TOOLS = ["MoveToMarker", "DeleteMarker", "MoveToPreset", "DeletePreset"]


def g_deictic():
    """A team/drone/number tool with its slot omitted or pointed at ('그 팀', '아까 그 드론',
    '저 마커'): nothing to resolve the argument from, so it must not be guessed."""
    kind = pick(["team"] * 4 + ["drone"] * 4 + ["number"] * 3 + ["thing"])
    if kind == "thing":
        return expand(pick(["그거 지워 줘", "저거 선택~", "이거 꺼 줘", "그거 열어 줘", "저거 삭제~",
                            "아까 그거 다시 보여 줘", "그걸로 이동~", "저기로 이동~", "그거 복귀시켜", "저거 카메라"]))
    if kind == "team":
        tool, subj = pick(TEAM_ONLY + ["AssignTeamControl"]), pick(DEICTIC_TEAM)
    elif kind == "drone":
        tool, subj = pick(DRONE), pick(DEICTIC_DRONE)
    else:
        tool = pick(NUMBER_TOOLS)
        noun = "마커" if "Marker" in tool else "프리셋"
        subj = pick([f"그 {noun}", f"저 {noun}", f"아까 그 {noun}", f"이 {noun}", f"해당 {noun}",
                     f"방금 그 {noun}", noun, noun] + ([f"{noun} 하나"] if tool.startswith("Delete") else []))
    gens = [TOOLS[tool][0]] + ([INDIRECT[tool]] if tool in INDIRECT else [])
    return pick(gens)(subj)[0]


def g_pause():
    """Pause is not stop: there is no pause tool."""
    return expand(pick(["녹화 일시정지~", "녹화 일시 정지~", "녹화 잠깐 멈춰", "녹화 잠시 멈춰 줘", "녹화 일시 중지~",
                        "녹화 잠깐 중지~", "잠깐 멈춰", "일시정지~", "녹화 잠깐만 멈춰", "녹화 멈췄다가 다시 해",
                        "녹화 잠깐 쉬어", "녹화 잠시 중단~", "잠깐 정지~", "녹화 일시정지 걸어 줘"]))


V4_GENERATORS = {"deictic": g_deictic, "pause": g_pause}


# --------------------------------------------------------------- validation

TEAM_ONLY = ["SelectTeam", "ReturnTeam", "DisbandTeam", "StartTeamTactics", "EditTeamTactics",
             "RevokeTeamControl", "RequestTeamControl", "ReturnTeamControl", "ShowTeamCamera",
             "MoveToTeamView", "MoveToTeamOverview"]
DRONE = ["SelectDrone", "EditDroneTactics", "ReturnDrone", "DetachDrone", "StartManualFlight",
         "ShowDroneCamera", "MoveToDroneView"]
NOARG = ["EnterTeamCreation", "EnterMarkerCreation", "CreatePreset", "RequestTacticsRecommendation",
         "CancelTacticsRecommendation", "ResetCameraGimbal", "EnlargeCameraWindow", "ShrinkCameraWindow",
         "EnableVideoThrough", "DisableVideoThrough", "ResetPanelPositions", "StartRecording", "StopRecording"]
SCHEMA = {t: ({"team": TEAM_ENUM}, {}) for t in TEAM_ONLY}       # tool: (required, optional)
SCHEMA |= {t: ({"droneNumber": int}, {"team": TEAM_ENUM}) for t in DRONE}
SCHEMA |= {t: ({}, {}) for t in NOARG}
SCHEMA |= {
    "AssignTeamControl": ({"team": TEAM_ENUM}, {"soldierName": str}),
    "CreateTeam": ({"teamType": {c for c, _ in TEAM_TYPES}, "memberCount": int}, {}),
    "MoveToMarker": ({"markerNumber": int}, {}), "DeleteMarker": ({"markerNumber": int}, {}),
    "MoveToPreset": ({"presetNumber": int}, {}), "DeletePreset": ({"presetNumber": int}, {}),
    "OpenWindow": ({"windowName": set(WINDOWS)}, {}), "CloseWindow": ({"windowName": set(WINDOWS)}, {}),
    "AcceptConfirmWindow": ({"confirmName": set(CONFIRMS)}, {}),
    "RejectConfirmWindow": ({"confirmName": set(CONFIRMS)}, {}),
    "ApplyDisplayFilters": ({"filterNames": str}, {}), "ClearDisplayFilters": ({"filterNames": str}, {}),
    "DefaultResponse": ({"text": str}, {}),
}
ALL_ALIASES = [a for aliases in TEAM_ALIASES.values() for a in aliases]


def validate(name, text, args):
    req, opt = SCHEMA[name]
    assert set(req) <= set(args) <= set(req) | set(opt), (name, text, args)
    for key, value in args.items():
        spec = req.get(key) or opt.get(key)
        if spec is int:
            assert type(value) is int and value > 0, (name, text, args)
        elif spec is str:
            assert isinstance(value, str) and value, (name, text, args)
        else:
            assert value in spec, (name, text, args)
    flat = text.replace(" ", "")
    if name == "DefaultResponse":
        assert args.get("text", text) == text
    if "team" in args:
        assert any(a in flat for a in TEAM_ALIASES[args["team"]]), (name, text, args)
    elif name in DRONE:
        assert not any(a in flat for a in ALL_ALIASES), (name, text, args)
    if "soldierName" in args:
        assert args["soldierName"] in flat, (name, text, args)
    for key in ("droneNumber", "markerNumber", "presetNumber", "memberCount"):
        if key in args:   # the number must be said in some form (weak for 이/일, catches a dropped slot)
            assert any(w in flat for w in number_words(args[key])), (name, text, args)
    if name.endswith("DisplayFilters"):
        pos = [flat.find(n.replace(" ", "")) for n in args["filterNames"].split(", ")]
        assert -1 not in pos and pos == sorted(pos), (name, text, args)


ENGLISH_NUM = {1: "원", 2: "투", 3: "쓰리", 4: "포"}   # v6: '투번 마커', '원 번 프리셋'


def number_words(n):
    words = {str(n)} | {t[n] for t in NUMBER_TABLES if n in t} | ({ENGLISH_NUM[n]} if n in ENGLISH_NUM else set())
    return words | {2: {"이오기", "E"}, 3: {"사모기"}, 4: {"사오기"}}.get(n, set())


def selfcheck(tools_prompt):
    reading = {}   # every number word reads as exactly one value: 셋/삼 = 3, 넷/사 = 4, 칠/일곱 = 7 ...
    for table in NUMBER_TABLES:
        for n, word in table.items():
            assert reading.setdefault(word, n) == n, (word, n, reading[word])
    assert _cond("닫") == "닫으면" and _cond("열") == "열면" and _cond("끄") == "끄면"
    assert set(V3_EXTRA) <= set(TOOLS) and set(V4_EXTRA) <= set(TOOLS) and LEAD_ONLY <= set(CLAUSES)
    assert j("알파팀", "을") == "알파팀을" and j("알파", "을") == "알파를"
    assert j("마커 2", "으로") == "마커 2로" and j("마커 3", "으로") == "마커 3으로"
    assert j("마커 1", "으로") == "마커 1로" and j("세 대", "으로") == "세 대로"
    assert j("좌측", "으로") == "좌측으로" and j("UGV", "으로") == "UGV로" and j("10", "으로") == "10으로"
    if tools_prompt.exists():   # the contract and this generator must name the same tools
        declared = set(re.findall(r"^  (\w+): \(", tools_prompt.read_text(encoding="utf-8"), re.M))
        assert declared == set(SCHEMA), declared ^ set(SCHEMA)
    assert set(SCHEMA) == set(TOOLS) | {"DefaultResponse"} and set(INDIRECT) <= set(TOOLS)


# --------------------------------------------------------------------- main

def norm(text):
    return re.sub(r"\s+", " ", text).strip().rstrip(".,!?~ ").strip()


def finish(text, space_ok=True):
    """Add what ASR does to a command: STT confusions, fillers, a dropped space, a period."""
    stt = False
    subs = [(p, reps) for p, reps in STT_SUBS if re.search(p, text)]
    if subs and chance(STT_P):
        for p, reps in R.sample(subs, 2 if len(subs) > 1 and chance(0.15) else 1):
            text = re.sub(p, pick(reps), text, count=1)
        stt = True
    spaced = V3 and space_ok and chance(SPACE_NOISE_P)
    if spaced:
        text = space_noise(text)
    if " " in text and chance(0.05):    # before the filler, so '자 호텔' never fuses to '자호텔'
        i = pick([m.start() for m in re.finditer(" ", text)])
        text = text[:i] + text[i + 1:]
    if chance(0.08):
        text = pick(FILLERS) + text
    if chance(0.04):
        text += "."
    return text, stt, spaced


# ------------------------------------------------------------ v6 (on top of v5)
# v5 = data_v5_empty/, built outside this script (paraphrase/paraphrase.py: v4 plus two-teacher-
# filtered LLM paraphrases in radio style). v6 reads it as-is and adds two classes the v5 model
# regressed on, using the v4 generators.

MULTI_V6, STT_V6, DEFAULT_SHARE_V6 = 700, 300, 0.20
RADIO_LEAD = ["어,", "음,", "즉시", "지금", "자,"]
RADIO_TRAIL = [", 오버", " 오버", ", 바람", " 바람", ", 이상", ", 빨리", ", 서둘러", ", 즉시", ", 지금", ", 당장"]
STT_MARKER = {2: ["마커 E"] * 3 + ["마커 이"] * 2 + ["투번 마커", "투 번 마커", "마커 투"],
              1: ["원 번 마커", "원번 마커", "마커 원"], 3: ["쓰리 번 마커"], 4: ["포 번 마커"]}
STT_PRESET = {2: ["이번 프리셋"] * 2 + ["이 번 프리셋"] * 2 + ["프리셋 E"] * 2 + ["프리셋 이", "투번 프리셋", "프리셋 투"],
              1: ["원 번 프리셋", "원번 프리셋", "프리셋 원"], 3: ["쓰리 번 프리셋"], 4: ["포 번 프리셋"]}
STT_DRONE = {"이오기": 2, "사모기": 3, "사오기": 4}


def dress(text):
    """The remarks the radio-style paraphrases carry: a lead clause or filler, urgency, a sign-off."""
    for slot in R.sample(["lead", "radio_lead", "trail", "radio_trail"], pick([1, 1, 2])):
        if slot == "lead":
            text = clause() + text
        elif slot == "radio_lead":
            text = f"{pick(RADIO_LEAD)} {text}"
        elif slot == "trail":
            text += trail()
        else:
            text += pick(RADIO_TRAIL)
    return text


def any_phrasing(tool):
    return pick([TOOLS[tool][0]] + ([INDIRECT[tool]] if tool in INDIRECT else []))


_HADA_TAILS = sorted({e for e in HADA if e}, key=len, reverse=True)


def as_noun(text):
    """'알파팀 선택해 줘' → '알파팀 선택' (the 하다 noun under its ending), else None."""
    for tail in [""] + _HADA_TAILS:
        if text.endswith(tail):
            noun = text[:len(text) - len(tail)]
            if _BARE_PREDICATE.search(noun):
                return noun
    return None


def g_multi_v6():
    """Two different executable commands joined the way people say them → DefaultResponse."""
    while True:
        a_tool, b_tool = R.sample(POSITIVE, 2)
        a, b = any_phrasing(a_tool)()[0], any_phrasing(b_tool)()[0]
        if not (a_tool == "SelectTeam" and not re.search("선택|골라|잡아|바꿔|전환", a)):
            break           # bare '알파' + '2호기 선택' reads as one drone selection
    # Weights: the -고 forms are how people actually chain commands, so they lead when A allows them.
    joins = [(", ", f"{a}, {b}")] * 2 + [("그리고", pick([f"{a}, 그리고 {b}", f"{a} 그리고 {b}"]))]
    a_noun, b_noun = as_noun(a), as_noun(b)
    if a_noun:                                          # '알파팀 선택하고 …'
        joins += [("하고", f"{a_noun}하고 {b}")] * 4 + [
            ("한 다음", pick([f"{a_noun} 한 다음 {b}", f"{a_noun}한 다음 {b}"])), ("후에", f"{a_noun} 후에 {b}")]
        if b_noun:
            joins += [("랑…해", f"{j(a_noun, '이랑')} {b_noun} " + pick(["해", "해 줘", "해라"]))] * 3
    else:                                               # '위협 켜 줘' → '위협 켜고 …'
        base = re.sub(r"\s*(줘|주세요|줘요)$", "", a)
        stem = next((base[:-len(e)] + s for e, s in NEG_STEMS if base.endswith(e)), None)
        if stem and not base.endswith("부탁해"):
            joins += [("고", f"{stem}고 {b}")] * 5
    kind, text = pick(joins)
    return text, "DefaultResponse", {}, kind


def g_stt_v6():
    """The contract's STT code forms for numbers: 마커 E / 이번 프리셋 / 투번 마커 / 이오기 …"""
    if chance(0.6):
        tool = pick(NUMBER_TOOLS)
        marker = "Marker" in tool
        n = pick([2] * 6 + [1] * 2 + [3, 4])
        form = pick((STT_MARKER if marker else STT_PRESET)[n])
        return TOOLS[tool][0](form)[0], tool, {("markerNumber" if marker else "presetNumber"): n}, form
    tool, code = pick(DRONE), pick(list(STT_DRONE))
    args, form = {"droneNumber": STT_DRONE[code]}, code
    if chance(0.3):
        en, ko = pick(TEAMS)
        form, args["team"] = f"{pick([ko, ko + '팀', en + '팀'])} {code}", en
    return any_phrasing(tool)(form)[0], tool, args, code


def build_v6(banned, banned_ns, show_stats):
    load = lambda d, s: [json.loads(line) for line in open(HERE / d / f"{s}.jsonl", encoding="utf-8")]
    base = {s: load("data_v5_empty", s) for s in ("train", "val")}
    template = {r["text"] for s in ("train", "val") for r in load("data_v4_empty", s)}
    n_base = sum(len(v) for v in base.values())

    owner, filtered, duplicates = {}, 0, 0
    for s in base:          # the base is filtered too: heldout4 is newer than v5
        kept = []
        for r in base[s]:
            key = norm(r["text"])
            if key in banned or key.replace(" ", "") in banned_ns:
                filtered += 1
                continue
            if key in owner:    # v5 carries one exact duplicate
                duplicates += 1
                continue
            owner[key] = r["function"]
            kept.append(r)
        base[s] = kept

    new, dressed, forms = defaultdict(list), Counter(), defaultdict(Counter)
    for tag, gen, target in (("multi", g_multi_v6, MULTI_V6), ("stt", g_stt_v6, STT_V6)):
        tries = 0
        while len(new[tag]) < target and tries < target * 60:
            tries += 1
            text, name, fargs, form = gen()
            is_dressed = chance(0.5)
            if is_dressed:
                text = dress(text)
            text, _, _ = finish(text)
            key = norm(text)
            if key in banned or key.replace(" ", "") in banned_ns or key in owner:
                continue
            validate(name, text, fargs)
            owner[key] = name
            dressed[tag] += is_dressed
            forms[tag][form] += 1
            new[tag].append({"text": text, "function": name, "arguments": fargs})

    # Back to DEFAULT_SHARE_V6 by dropping a uniform sample of v5's template-made DefaultResponse
    # records (never the paraphrases), so the negative classes keep their proportions.
    rows = [r for s in base for r in base[s]] + [r for v in new.values() for r in v]
    n_default = sum(r["function"] == "DefaultResponse" for r in rows)
    excess = max(0, round((n_default - DEFAULT_SHARE_V6 * len(rows)) / (1 - DEFAULT_SHARE_V6)))
    pool = [(s, i) for s in base for i, r in enumerate(base[s])
            if r["function"] == "DefaultResponse" and r["text"] in template]
    drop = set(R.sample(pool, min(excess, len(pool))))
    for s in base:
        base[s] = [r for i, r in enumerate(base[s]) if (s, i) not in drop]

    train, val = list(base["train"]), list(base["val"])
    for tag in new:
        recs = new[tag]
        k = max(1, round(len(recs) * VAL_FRACTION))
        val += recs[:k]
        train += recs[k:]
    R.shuffle(train)
    R.shuffle(val)
    out = HERE / "data_v6_empty"
    out.mkdir(exist_ok=True)
    for fname, part in (("train.jsonl", train), ("val.jsonl", val)):
        with open(out / fname, "w", encoding="utf-8", newline="\n") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(train) + len(val)
    every = train + val
    n_def = sum(r["function"] == "DefaultResponse" for r in every)
    para_single = sum(r["function"] != "DefaultResponse" and r["text"] not in template for s in base for r in base[s])
    print(f"train {len(train)}  val {len(val)}  total {total}  -> {out}")
    print(f"base v5 {n_base}, dropped by the exact-text filter {filtered}, duplicates {duplicates}, "
          f"template DefaultResponse trimmed {len(drop)}")
    for tag, label in (("multi", "multi-action joins"), ("stt", "STT code forms")):
        print(f"{label}: " + ", ".join(f"{k} {v}" for k, v in forms[tag].most_common()))
    print(f"new: multi-action DefaultResponse {len(new['multi'])} ({dressed['multi']} dressed), "
          f"STT code forms {len(new['stt'])} ({dressed['stt']} dressed): "
          + ", ".join(f"{k} {v}" for k, v in sorted(Counter(r['function'] for r in new['stt']).items())))
    print(f"DefaultResponse share {n_def / total:.1%}   paraphrase single commands {para_single} "
          f"vs dressed multi-action {dressed['multi']}")
    if show_stats:
        tc = Counter(r["function"] for r in every)
        print("\n" + "\n".join(f"{k:32} {v:5}" for k, v in sorted(tc.items())))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stats", action="store_true", help="print per-tool counts and samples")
    ap.add_argument("--tools-prompt", type=Path, default=DEFAULT_TOOLS_PROMPT)
    ap.add_argument("--version", type=int, choices=[1, 2, 3, 4, 6], default=6,
                    help="5 is data_v5_empty/, built by paraphrase/paraphrase.py; 6 reads it")
    ap.add_argument("--v1", dest="version", action="store_const", const=1, help="same as --version 1")
    ap.add_argument("--default-args", choices=["text", "empty"],
                    help="'empty' (default from v2 on) writes DefaultResponse with {} — the app fills text "
                         "with the transcript, so the model no longer spends decode steps copying it")
    args = ap.parse_args()
    version = args.version
    default_args = args.default_args or ("text" if version == 1 else "empty")
    if version == 1 and default_args == "empty":
        ap.error("--v1 reproduces data/, which has text arguments")
    if version == 6 and default_args == "text":
        ap.error("v6 builds on data_v5_empty/, which has empty DefaultResponse arguments")
    augment = version >= 2
    global MISSING_ACTIVE, V3, V4
    V3, V4 = version >= 3, version >= 4
    selfcheck(args.tools_prompt)
    if augment:
        MISSING_ACTIVE = MISSING + MISSING_V2 + (MISSING_V3 if V3 else [])
    if default_args == "empty":
        SCHEMA["DefaultResponse"] = ({}, {})

    filters = ["eval_cases.json", "heldout_cases.json", "heldout2_cases.json", "heldout3_cases.json",
               "heldout4_cases.json"][:min(version, 5)]
    banned = {norm(c["text"]) for f in filters
              for c in json.loads((HERE / f).read_text(encoding="utf-8"))["cases"]}
    banned_ns = {b.replace(" ", "") for b in banned}
    if version == 6:
        print(f"exact-text filter: {' + '.join(filters)}")
        return build_v6(banned, banned_ns, args.stats)

    owner, records, stt_texts, aug_texts = {}, defaultdict(list), set(), {}
    collisions, conflicts, shortfall = 0, [], {}

    aug_n = Counter()

    def collect(name, gen, target, tag=None):
        nonlocal collisions
        tries = 0
        while len(records[name]) < target and tries < target * 60:
            tries += 1
            # Each v2/v3 draw sits behind `augment and` / `V3 and`, so an older version consumes the
            # same random stream as before. The caps matter for tools with few base phrasings: dedupe
            # rejects their repeats, so uncapped augmented variants crowd in (ShrinkCameraWindow
            # reached 55 % clauses).
            indirect = (augment and name in INDIRECT and aug_n[name, "indirect"] < INDIRECT_P * target
                        and chance(INDIRECT_P))
            extra = (V3 and not indirect and name in V3_EXTRA and aug_n[name, "v3extra"] < V3_EXTRA_P * target
                     and chance(V3_EXTRA_P))
            extra4 = (V4 and not indirect and not extra and name in V4_EXTRA
                      and aug_n[name, "v4extra"] < V4_EXTRA_P * target and chance(V4_EXTRA_P))
            text, fargs = (INDIRECT[name] if indirect else V3_EXTRA[name] if extra
                           else V4_EXTRA[name] if extra4 else gen)()
            trailing = False
            if V4:   # one draw picks leading / trailing / none
                r = R.random()
                with_clause = r < LEAD_P_V4 and aug_n[name, "clause"] < LEAD_P_V4 * target
                trailing = (LEAD_P_V4 <= r < LEAD_P_V4 + TRAIL_P_V4 and aug_n[name, "trail"] < TRAIL_P_V4 * target
                            and not text.endswith("?"))
            else:
                with_clause = augment and aug_n[name, "clause"] < CLAUSE_P * target and chance(CLAUSE_P)
            if with_clause:
                text = clause() + text
            if trailing:
                text += trail()
            text, stt, spaced = finish(text, aug_n[name, "spacing"] < SPACE_NOISE_P * target)
            key = norm(text)
            if key in banned or key.replace(" ", "") in banned_ns:
                collisions += 1
                continue
            if key in owner:
                if owner[key] != name:
                    conflicts.append((key, owner[key], name))
                continue
            if name == "DefaultResponse":
                fargs = {"text": text} if default_args == "text" else {}
            validate(name, text, fargs)
            owner[key] = name
            if stt:
                stt_texts.add(text)
            kinds = [k for k, on in (("indirect", indirect), ("v3extra", extra), ("v4extra", extra4),
                                     ("clause", with_clause), ("trail", trailing), ("spacing", spaced),
                                     (tag, tag)) if on]
            if kinds:
                aug_texts[text] = "+".join(kinds)
                aug_n.update((name, k) for k in kinds)
            records[name].append({"text": text, "function": name, "arguments": fargs})
        if len(records[name]) < target:
            shortfall[name] = (len(records[name]), target)

    for name, (gen, target) in TOOLS.items():
        collect(name, gen, target)
    positives = sum(len(v) for v in records.values())
    collect("DefaultResponse", g_default, round(positives * (DEFAULT_RATIO_V4 if V4 else DEFAULT_RATIO)))
    if V4:   # each class is its share of the final set, on top of the base negatives
        final = len(owner) / (1 - sum(share for _, share in V4_CLASSES))
        gens = {"negated": g_negated, "state": g_state, "near": g_near} | V4_GENERATORS
        for tag, share in V4_CLASSES:
            collect("DefaultResponse", lambda g=gens[tag]: (g(), {}),
                    len(records["DefaultResponse"]) + round(share * final), tag)
    elif V3:   # each class ≈ V3_DEFAULT_SHARE of the final set, on top of the v2 negatives
        per_class = round(V3_DEFAULT_SHARE * len(owner) / (1 - 3 * V3_DEFAULT_SHARE))
        for tag, gen in (("negated", g_negated), ("state", g_state), ("near", g_near)):
            collect("DefaultResponse", lambda g=gen: (g(), {}), len(records["DefaultResponse"]) + per_class, tag)

    train, val = [], []
    for name in sorted(records):
        recs = records[name]
        R.shuffle(recs)
        k = max(1, round(len(recs) * VAL_FRACTION))
        val += recs[:k]
        train += recs[k:]
    R.shuffle(train)
    R.shuffle(val)

    out = HERE / ("data" if version == 1 else f"data_v{version}" + ("_empty" if default_args == "empty" else ""))
    out.mkdir(exist_ok=True)
    for fname, rows in (("train.jsonl", train), ("val.jsonl", val)):
        with open(out / fname, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(train) + len(val)
    print(f"train {len(train)}  val {len(val)}  total {total}  -> {out}")
    print(f"collisions with {' + '.join(filters)} dropped: {collisions}")
    if augment:
        kinds = Counter(aug_texts.values())
        print(f"augmented {len(aug_texts)} ({len(aug_texts) / total:.1%}): " +
              ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())))
        clause_default = sum(1 for r in train + val if r["function"] == "DefaultResponse"
                             and "clause" in aug_texts.get(r["text"], ""))
        print(f"DefaultResponse records with a clause: {clause_default}")
    print(f"label conflicts (same text, two tools; first kept): {len(conflicts)}")
    for c in conflicts[:20]:
        print("   ", c)
    for name, (got, want) in shortfall.items():
        print(f"shortfall {name}: {got}/{want}")
    if not args.stats:
        return
    tc, vc = Counter(r["function"] for r in train), Counter(r["function"] for r in val)
    print(f"\n{'tool':32} train  val  clause indirect v3extra spacing")
    for name in sorted(SCHEMA):
        print(f"{name:32} {tc[name]:5} {vc[name]:4} {aug_n[name, 'clause']:6} {aug_n[name, 'indirect']:8} "
              f"{aug_n[name, 'v3extra']:7} {aug_n[name, 'spacing']:7}")
    if V3:
        tags = [t for t, _ in V4_CLASSES] if V4 else ["negated", "state", "near"]
        print("DefaultResponse classes: " + ", ".join(
            f"{t} {aug_n['DefaultResponse', t]} ({aug_n['DefaultResponse', t] / total:.1%})" for t in tags))
        print(f"spacing noise {sum(v for (_, k), v in aug_n.items() if k == 'spacing') / total:.1%} of records")
    if V4:
        n_def = tc["DefaultResponse"] + vc["DefaultResponse"]
        n_exe = total - n_def
        for kind, label in (("clause", "leading"), ("trail", "trailing")):
            exe = sum(v for (t, k), v in aug_n.items() if k == kind and t != "DefaultResponse")
            print(f"{label} clause: executable {exe / n_exe:.1%}, DefaultResponse "
                  f"{aug_n['DefaultResponse', kind] / n_def:.1%}")
    pos = {k: v for k, v in tc.items() if k != "DefaultResponse"}
    print(f"\nper-tool train min {min(pos.values())} ({min(pos, key=pos.get)}), "
          f"max {max(pos.values())} ({max(pos, key=pos.get)})")
    n_default = tc["DefaultResponse"] + vc["DefaultResponse"]
    print(f"DefaultResponse share {n_default / total:.1%}   STT share {len(stt_texts) / total:.1%}")
    by_tool = defaultdict(list)
    for r in train:
        by_tool[r["function"]].append(r)
    print()
    for name in sorted(by_tool):
        for r in by_tool[name][:3]:
            print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
