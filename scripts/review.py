#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
어항 디자인 리뷰 자동화 — Gemini Vision API (GitHub Actions용)

photos/tank-N/ 에 새 사진이 올라오면(파일명 YYYY-MM-DD_HHMM.jpg) 감지하여,
해당 어항 데이터(aquarium-data.json)를 맥락으로 함께 Gemini 2.5 Flash에 전달하고,
구도/색감/밸런스/밀도/장비노출 5개 항목을 평가받아 데이터와 대시보드를 갱신한다.

- 새 사진이 없으면 변경 없이 종료(exit 0, changed=false)
- 의존성 없음(파이썬 표준 라이브러리만 사용)
"""

import os
import re
import sys
import glob
import json
import base64
import datetime
import urllib.request
import urllib.error

# ---- 경로/상수 -------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "aquarium-data.json")
HTML_PATH = os.path.join(ROOT, "index.html")
PHOTOS_DIR = os.path.join(ROOT, "photos")

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{MODEL}:generateContent"
)

GRADES = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C"]
GRADE_RANK = {g: i for i, g in enumerate(GRADES)}  # 작을수록 높은 등급
CATEGORIES = ["구도", "색감", "밸런스", "밀도", "장비노출"]
TODAY = datetime.date.today().isoformat()

PHOTO_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{4})\.jpg$")


# ---- 사진 감지 -------------------------------------------------------------
def latest_photo(tank_id):
    """tank-N 폴더에서 latest.jpg 를 제외한 가장 최신 타임스탬프 사진 경로/날짜 반환."""
    folder = os.path.join(PHOTOS_DIR, f"tank-{tank_id}")
    if not os.path.isdir(folder):
        return None, None
    best = None  # (datekey, filename, path)
    for path in glob.glob(os.path.join(folder, "*.jpg")):
        name = os.path.basename(path)
        if name == "latest.jpg":
            continue
        m = PHOTO_RE.match(name)
        if not m:
            continue
        datekey = m.group(1)  # YYYY-MM-DD
        if best is None or (datekey, name) > (best[0], best[1]):
            best = (datekey, name, path)
    if best is None:
        return None, None
    return best[2], best[0]


# ---- Gemini 호출 -----------------------------------------------------------
def build_context(tank):
    """어항 데이터를 사람이 읽는 맥락 텍스트로 정리."""
    def fmt_fauna(items):
        return ", ".join(
            f"{x.get('name')} {x.get('count')}마리"
            + (f"({x.get('note')})" if x.get("note") else "")
            for x in (items or [])
        ) or "없음"

    prev = tank.get("designReview", {}) or {}
    prev_details = prev.get("details", {}) or {}
    prev_lines = "\n".join(
        f"    - {c}: {prev_details.get(c, {}).get('grade', '?')} "
        f"/ {prev_details.get(c, {}).get('comment', '')}"
        for c in CATEGORIES
    )
    lines = [
        f"[어항] {tank.get('id')}번 · {tank.get('name')} · 컨셉: {tank.get('concept')}",
        f"용량: {tank.get('volume')} / 바닥재: {tank.get('substrate')}",
        f"여과: {tank.get('filter')}",
        f"조명: {tank.get('light')}",
        f"이탄(CO2): {tank.get('co2') or '없음'}",
        f"장비: {', '.join(tank.get('equipment', []) or []) or '없음'}",
        f"하드스케이프: {tank.get('hardscape') or '명시 없음'}",
        f"수초(flora): {', '.join(tank.get('flora', []) or []) or '없음'}",
        f"생체(fauna): {fmt_fauna(tank.get('fauna'))}",
        f"투입 예정: {fmt_fauna(tank.get('plannedFauna'))}",
        "",
        f"[직전 리뷰] 종합 {prev.get('grade', '없음')} (리뷰일 {prev.get('reviewDate', '없음')})",
        f"  요약: {prev.get('summary', '없음')}",
        "  항목별:",
        prev_lines,
    ]
    return "\n".join(lines)


def category_schema():
    return {
        "type": "OBJECT",
        "properties": {
            "grade": {"type": "STRING", "enum": GRADES},
            "comment": {"type": "STRING"},
        },
        "required": ["grade", "comment"],
    }


def response_schema():
    return {
        "type": "OBJECT",
        "properties": {
            "grade": {"type": "STRING", "enum": GRADES},
            "summary": {"type": "STRING"},
            "details": {
                "type": "OBJECT",
                "properties": {c: category_schema() for c in CATEGORIES},
                "required": CATEGORIES,
            },
            "suggestions": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
            },
            "changes": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
            },
        },
        "required": ["grade", "summary", "details", "suggestions", "changes"],
    }


PROMPT = """당신은 세계 최고 수준의 아쿠아스케이핑(수초 어항 레이아웃) 전문 심사위원입니다.
첨부된 어항 사진을 아래 어항 데이터와 함께 종합적으로 평가하세요.

평가 규칙:
- 5개 항목을 각각 채점합니다: 구도, 색감, 밸런스, 밀도, 장비노출.
- 등급 체계: A+, A, A-, B+, B, B-, C+, C (A+가 최고).
- 색감/밀도 코멘트에서는 데이터에 적힌 수초·생체 종을 실제 이름으로 언급하세요
  (예: "루드위지아 슈퍼레드의 발색이...", "미크란테뭄 카펫의 밀도가...").
- 장비노출은 데이터의 장비 목록과 교차 확인하세요. 예를 들어 히터가 외부여과기 내장형이면
  히터 노출로 감점하지 마세요. 데이터상 보일 수밖에 없는 장비만 평가하세요.
- 사진에 데이터에 없는 종/장비가 보이면 그 사실을 코멘트에 언급하세요.
- 직전 리뷰와 비교하여 개선/후퇴된 점을 changes 배열에 적으세요
  (예: "장비노출 B- → B+: 히터를 외부여과기 내장형으로 교체하여 노출 장비 감소").
  의미 있는 변화가 없으면 changes는 빈 배열 []로 두세요.
- suggestions에는 특정 수초/장비 이름을 언급한 실행 가능한 개선안 2~4개를 적으세요.
- 모든 텍스트는 한국어로 작성합니다.

종합 등급(grade)은 5개 항목을 종합한 전체 인상입니다.

아래는 이 어항의 데이터입니다:
---
{context}
---
"""


def call_gemini(image_path, context):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    body = {
        "contents": [
            {
                "parts": [
                    {"text": PROMPT.format(context=context)},
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": response_schema(),
            "temperature": 0.4,
        },
    }
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": API_KEY,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        sys.stderr.write(f"Gemini HTTP {e.code}: {e.read().decode('utf-8', 'replace')}\n")
        raise
    text = payload["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text)


# ---- 데이터/대시보드 갱신 --------------------------------------------------
def apply_review(tank, review):
    """Gemini 결과를 tank 객체에 반영."""
    grade = review["grade"]
    summary = review["summary"]
    details = review["details"]
    suggestions = review.get("suggestions", [])
    changes = review.get("changes", [])

    tank["designReview"] = {
        "grade": grade,
        "summary": summary,
        "reviewDate": TODAY,
        "details": {
            c: {
                "grade": details[c]["grade"],
                "comment": details[c]["comment"],
            }
            for c in CATEGORIES
        },
        "suggestions": suggestions,
        "changes": changes,
    }
    tank.setdefault("ratings", {})
    tank["ratings"]["디자인"] = {"grade": grade, "comment": summary}


def update_html(data):
    with open(HTML_PATH, "r", encoding="utf-8") as f:
        html = f.read()
    minified = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    new_html, n = re.subn(
        r"const AQUARIUM_DATA=\{.*?\};",
        lambda _m: "const AQUARIUM_DATA=" + minified + ";",
        html,
        count=1,
        flags=re.DOTALL,
    )
    if n != 1:
        raise RuntimeError("index.html 에서 AQUARIUM_DATA 블록을 찾지 못했습니다.")
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(new_html)


def set_output(name, value):
    out = os.environ.get("GITHUB_OUTPUT")
    if not out:
        return
    with open(out, "a", encoding="utf-8") as f:
        f.write(f"{name}={value}\n")


# ---- 메인 -----------------------------------------------------------------
def main():
    if not API_KEY:
        sys.stderr.write("GEMINI_API_KEY 환경변수가 없습니다.\n")
        sys.exit(1)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    reviewed = []  # (id, old_grade, new_grade)
    for tank in data.get("tanks", []):
        tid = tank.get("id")
        path, photo_date = latest_photo(tid)
        if path is None:
            continue
        prev_date = (tank.get("designReview", {}) or {}).get("reviewDate", "")
        if not (photo_date > prev_date):  # YYYY-MM-DD 문자열 비교
            continue
        old_grade = (tank.get("designReview", {}) or {}).get("grade")
        print(f"[리뷰] {tid}번 — 새 사진 {os.path.basename(path)} (직전 리뷰 {prev_date or '없음'})")
        review = call_gemini(path, build_context(tank))
        apply_review(tank, review)
        reviewed.append((tid, old_grade, review["grade"]))
        print(f"       → 종합 {old_grade or '없음'} → {review['grade']}")

    if not reviewed:
        print("새 사진 없음 — 리뷰 변경 없음")
        set_output("changed", "false")
        return

    data.setdefault("meta", {})["lastUpdated"] = TODAY
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    update_html(data)

    # 커밋 메시지 구성: "디자인 리뷰 갱신: N번 어항 (등급[ → 등급])"
    ids = "·".join(str(t[0]) for t in reviewed)
    if len(reviewed) == 1:
        _, old_g, new_g = reviewed[0]
        if old_g and old_g != new_g:
            grade_part = f"{old_g} → {new_g}"
        else:
            grade_part = f"{new_g} 유지" if old_g else new_g
    else:
        grade_part = ", ".join(
            f"{t[0]}번 {t[1]}→{t[2]}" if t[1] and t[1] != t[2] else f"{t[0]}번 {t[2]}"
            for t in reviewed
        )
    commit_msg = f"디자인 리뷰 갱신: {ids}번 어항 ({grade_part})"
    print(commit_msg)
    set_output("changed", "true")
    set_output("commit_msg", commit_msg)


if __name__ == "__main__":
    main()
