# services/rag.py
from __future__ import annotations
import os, glob, csv, re, calendar
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
from datetime import date, timedelta
from typing import Optional, List, Tuple, Dict

import pandas as pd
from chromadb import PersistentClient
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# ---------------- OpenAI 클라이언트 ----------------
try:
    from openai import OpenAI as _OpenAI
    import httpx
except Exception:
    _OpenAI = None
    httpx = None

def _make_openai_client() -> Optional["_OpenAI"]:
    if not _OpenAI:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return _OpenAI(api_key=api_key, http_client=httpx.Client(timeout=30))

_OPENAI_CLIENT = _make_openai_client()
_OPENAI_EMB_MODEL = os.getenv("EMB_MODEL", "text-embedding-3-small")

# ---------------- 임베딩 함수 (충돌 해결 버전) ----------------
try:
    from chromadb import EmbeddingFunction as _EmbeddingFunctionBase
except ImportError:
    _EmbeddingFunctionBase = object

class OpenAIEmbedder(_EmbeddingFunctionBase):
    """chromadb용 임베더: EmbeddingFunction 프로토콜 구현"""
    def __init__(self, client, model_name: str):
        self.client = client
        self.model_name = model_name
        self._name = f"openai:{model_name}"

    def name(self) -> str:
        return self._name

    def __call__(self, input: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model_name, input=input)
        return [d.embedding for d in resp.data]

def _embedding_function():
    if _OPENAI_CLIENT:
        return OpenAIEmbedder(_OPENAI_CLIENT, os.getenv("EMB_MODEL", "text-embedding-3-small"))
    else:
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

# ---------------- 경로 설정 ----------------
_BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DBDIR  = os.path.join(_BASE_DIR, "data", "vectorstore", "chroma")
FAQDIR = os.path.join(_BASE_DIR, "data", "vectorstore")

HIDDEN_DIR     = os.path.join(FAQDIR, "hidden")
CORRECTION_DIR = os.path.join(FAQDIR, "correction")

def _read_hidden_data() -> tuple[str, str]:
    hidden_path = os.path.join(HIDDEN_DIR, "hidden_rules.txt")
    correction_path = os.path.join(CORRECTION_DIR, "correction_info.txt")
    hidden_context = ""
    correction_rules = ""
    if os.path.exists(hidden_path):
        with open(hidden_path, encoding="utf-8") as f:
            hidden_context = f.read()
    if os.path.exists(correction_path):
        with open(correction_path, encoding="utf-8") as f:
            correction_rules = f.read()
    return hidden_context, correction_rules

def _chroma_client():
    os.makedirs(DBDIR, exist_ok=True)
    return PersistentClient(path=DBDIR)

# ---------------- 파일 리더 ----------------
def _read(path: str) -> str:
    if path.lower().endswith((".md", ".txt")):
        return open(path, "r", encoding="utf-8").read()
    try:
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    except Exception:
        return ""

def _read_qa_csv(path: str) -> list[tuple[str, str]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            q = (r.get("question") or "").strip()
            a = (r.get("answer") or "").strip()
            if q and a:
                rows.append((q, a))
    return rows

def _read_qa_md(path: str) -> list[tuple[str, str]]:
    # 매우 단순: "### Q:" / "A:" 포맷
    txt = open(path, "r", encoding="utf-8").read()
    pairs = []
    for block in txt.split("### Q:")[1:]:
        q, _, rest = block.partition("\n")
        a_tag = "A:"
        a = rest.split("\n", 1)[0] if a_tag not in rest else rest.split(a_tag, 1)[1].strip()
        q, a = q.strip(), a.strip()
        if q and a:
            pairs.append((q, a))
    return pairs

# ---------------- 텍스트 청킹 ----------------
def _chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> list[str]:
    """긴 텍스트를 겹치는 청크로 분할"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += max_chars - overlap
    return chunks

# ---------------- 인덱스 빌드 ----------------
def build_index() -> None:
    os.makedirs(DBDIR, exist_ok=True)
    cli = _chroma_client()
    col = cli.get_or_create_collection("faqs", embedding_function=_embedding_function())

    # 1) PDF/MD/TXT 문서 색인 (청크 단위)
    files = glob.glob(f"{FAQDIR}/*.md") + glob.glob(f"{FAQDIR}/*.txt") + glob.glob(f"{FAQDIR}/*.pdf")
    ids, docs, metas = [], [], []
    for f in files:
        txt = _read(f)
        if not txt.strip():
            continue
        chunks = _chunk_text(txt)
        basename = os.path.basename(f)
        for i, chunk in enumerate(chunks):
            ids.append(f"{basename}::chunk{i}")
            docs.append(chunk)
            metas.append({"path": f, "chunk": i})
    if ids:
        col.upsert(ids=ids, documents=docs, metadatas=metas)

    # 2) CSV/MD Q&A 색인
    qa_files = glob.glob(f"{FAQDIR}/*.csv") + glob.glob(f"{FAQDIR}/faq_qa.md") + glob.glob(f"{FAQDIR}/*_qa.md")
    q_ids, q_docs, q_metas = [], [], []
    for f in qa_files:
        pairs = _read_qa_csv(f) if f.endswith(".csv") else _read_qa_md(f)
        for idx, (q, a) in enumerate(pairs):
            q_ids.append(f"{os.path.basename(f)}::Q{idx+1}")
            q_docs.append(f"[Q] {q}\n[A] {a}")
            q_metas.append({"path": f, "type": "qa"})
    if q_ids:
        col.upsert(ids=q_ids, documents=q_docs, metadatas=q_metas)

    try:
        cli.persist()
    except AttributeError:
        pass  # 최신 ChromaDB는 자동 persist

# ---------------- 마감일 규칙 & 영업일 보정 ----------------
ADJUST_TO_BUSINESS_DAY = os.getenv("RAG_DEADLINE_BUSINESS_DAY", "1") in ("1", "true", "True")
HOLIDAYS_FILE = os.getenv("RAG_HOLIDAYS_FILE", "data/vectorstore/holidays_kr.csv")

_HOLIDAYS: set[date] = set()
def _load_holidays() -> set[date]:
    days: set[date] = set()
    try:
        if os.path.exists(HOLIDAYS_FILE):
            with open(HOLIDAYS_FILE, encoding="utf-8") as f:
                for line in f:
                    s = line.strip().split(",")[0]
                    if not s:
                        continue
                    try:
                        y, m, d = s.split("-")
                        days.add(date(int(y), int(m), int(d)))
                    except:
                        pass
    except:
        pass
    return days
_HOLIDAYS = _load_holidays()

def _adjust_to_business_day(d: date, prefer: str = "previous") -> date:
    step = -1 if prefer == "previous" else +1
    cur = d
    while cur.weekday() >= 5 or cur in _HOLIDAYS:  # 5=토, 6=일
        cur = cur + timedelta(days=step)
    return cur

_DEADLINE_ANY = re.compile(r"(마감일|마감|말일)", re.I)
_THIS_MONTH   = re.compile(r"(이번\s*달|이번달)", re.I)
_NEXT_MONTH   = re.compile(r"(다음\s*달|익월)", re.I)
_PREV_MONTH   = re.compile(r"(지난\s*달|전월)", re.I)
_YEAR_MONTH   = re.compile(r"(?:(?P<y>\d{4})\s*[./-]?\s*(년)?\s*)?(?P<m>1[0-2]|0?[1-9])\s*(월)?")

def _shift_month(dt: date, delta: int) -> date:
    y, m = dt.year, dt.month + delta
    while m < 1:
        y -= 1; m += 12
    while m > 12:
        y += 1; m -= 12
    return date(y, m, 1)

def _end_of_month(dt: date) -> date:
    last = calendar.monthrange(dt.year, dt.month)[1]
    return date(dt.year, dt.month, last)

def _parse_target_month_from_text(q: str, today: date) -> Optional[date]:
    t = q.strip()
    if _THIS_MONTH.search(t): return date(today.year, today.month, 1)
    if _NEXT_MONTH.search(t): return _shift_month(today, +1)
    if _PREV_MONTH.search(t): return _shift_month(today, -1)
    m = _YEAR_MONTH.search(t)
    if m:
        yy = m.group("y")
        mm = int(m.group("m"))
        yyyy = int(yy) if yy else today.year
        return date(yyyy, mm, 1)
    return None

# ---------------- 질의 응답 ----------------
import os
import re
from datetime import date, datetime, timedelta


# ---------------- 질의 응답 (통합) ----------------
def rag_answer(question: str, k: int = 3, show_sources: bool = False) -> tuple[str, list[dict]]:
    """
    Old rag_answer + safe_rag_query 기능 통합.
    - 마감일 특수처리 (기존 rag_answer)
    - hidden context / 정정 자료 / 조직도 / 업무추진비 자동계산 (기존 safe_rag_query)
    - 출처 표시 옵션 (기존 safe_rag_query)
    """

    # ──────────────────────────────────────────────
    # 0) 마감일 계열 특수처리 (RAG 전에 결정론)
    # ──────────────────────────────────────────────
    if _DEADLINE_ANY.search(question):
        today = date.today()
        target = _parse_target_month_from_text(question, today) or date(today.year, today.month, 1)
        eom = _end_of_month(target)
        eom_adj = (
            _adjust_to_business_day(eom, prefer=os.getenv("RAG_DEADLINE_PREFER", "previous"))
            if ADJUST_TO_BUSINESS_DAY
            else eom
        )
        y, m = target.year, target.month
        msg = (
            "지출결의서 마감일은 일반적으로 매월 말일입니다.\n"
            f"요청하신 기간: **{y}년 {m}월**\n"
            f"기본 말일: {eom.strftime('%Y-%m-%d')}\n"
            f"영업일 보정: **{eom_adj.strftime('%Y년 %m월 %d일')}**"
        )
        return msg, []

    # ──────────────────────────────────────────────
    # 1) 문서/QA RAG 검색
    # ──────────────────────────────────────────────
    cli = _chroma_client()
    col = cli.get_or_create_collection("faqs", embedding_function=_embedding_function())
    r = col.query(query_texts=[question], n_results=k)

    ctx = "\n\n".join(r["documents"][0]) if r and r.get("documents") else ""
    metas = r["metadatas"][0] if r and r.get("metadatas") else []

    # ──────────────────────────────────────────────
    # 2) 날짜·시간 정보
    # ──────────────────────────────────────────────
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d (%A)")
    week_start = (now - timedelta(days=now.weekday())).strftime("%Y-%m-%d")
    week_end = (now + timedelta(days=6 - now.weekday())).strftime("%Y-%m-%d")
    month_str = now.strftime("%Y-%m")

    # ──────────────────────────────────────────────
    # 3) Hidden 데이터 · 정정 자료 로드
    # ──────────────────────────────────────────────
    hidden_context, correction_rules = _read_hidden_data()

    # 직급별 업무추진비 단가 파싱
    designated_amounts = {}
    for line in hidden_context.splitlines():
        m = re.match(r"(\S+)\s*(\d+(?:,\d+)*)원", line)
        if m:
            rank, amt = m.groups()
            designated_amounts[rank] = int(amt.replace(",", ""))

    # ──────────────────────────────────────────────
    # 4) 질문에서 이름·직급 추출 → 조직도 조회
    # ──────────────────────────────────────────────
    # 이름+직급 패턴 먼저 시도
    match_name_rank = re.search(
        r"([가-힣A-Za-z]+)\s*(팀장|부장|임원|차장|과장|대리|사원)", question
    )
    user_name, user_rank = match_name_rank.groups() if match_name_rank else (None, None)

    # 직급 없이 이름만 있는 경우: 조직도 팀장/팀원 목록에서 검색
    org_path = os.path.join(FAQDIR, "org_info.csv")
    not_leader_msg = ""
    if not user_name:
        try:
            org_df = pd.read_csv(org_path, encoding="utf-8-sig")
            # 팀장 목록에서 검색
            for leader in org_df["팀장"].astype(str).str.strip():
                if leader in question:
                    user_name = leader
                    user_rank = "팀장"
                    break
            # 팀장에 없으면 팀원 목록에서 검색
            if not user_name:
                for _, row in org_df.iterrows():
                    members = str(row.get("팀원", "")).strip()
                    for m in members.split(","):
                        m = m.strip()
                        if m and m in question:
                            user_name = m
                            not_leader_msg = f"\n\n> {m} 님은 팀장이 아닌 것으로 보입니다. 김도희 사원에게 문의해주세요."
                            break
                    if user_name:
                        break
        except Exception:
            pass

    member_count, team_name = (
        _count_members_from_org(org_path, user_name) if user_name and not not_leader_msg else (None, None)
    )

    # ──────────────────────────────────────────────
    # 5) 조직도 텍스트 · 업무추진비 자동 계산
    # ──────────────────────────────────────────────
    team_info_text, auto_calc_text = "", ""
    if member_count and team_name:
        team_info_text = f"{team_name} ({user_rank or '팀장'} {user_name})의 팀원 수는 {member_count}명입니다."

    # 팀 이름 부분 검색: 질문에 팀 키워드가 있으면 관련 팀 목록 제공
    team_keyword_match = re.search(r"([가-힣A-Za-z0-9\-]+)팀", question)
    if team_keyword_match and os.path.exists(org_path):
        try:
            org_df = pd.read_csv(org_path, encoding="utf-8-sig")
            keyword = team_keyword_match.group(1)
            matched_teams = org_df[org_df["팀명"].astype(str).str.contains(keyword, na=False)]
            if not matched_teams.empty:
                team_lines = []
                for _, row in matched_teams.iterrows():
                    team_lines.append(f"- {row['팀명']} (팀장: {row['팀장']}, 팀원수: {row['팀원수']}명)")
                team_info_text += f"\n\n'{keyword}'이(가) 포함된 팀 목록:\n" + "\n".join(team_lines)
                if len(matched_teams) > 1:
                    team_info_text += f"\n\n'{keyword}팀'에 해당하는 팀이 여러 개입니다. 위 목록을 보여주고 정확한 팀 이름을 알려달라고 안내하세요."
        except Exception:
            pass

    if "업무추진비" in question:
        base = designated_amounts.get(user_rank) if user_rank else None
        # 팀장인데 designated_amounts에 없으면 기본 단가 20,000원
        if base is None and member_count:
            base = 20000
        if base and member_count:
            total = base * member_count
            auto_calc_text = (
                f"자동 계산: {user_rank or '팀장'} {user_name} - "
                f"팀원 {member_count}명 x {base:,}원 = {total:,}원"
            )
        elif base:
            auto_calc_text = f"자동 계산: {user_rank or '팀장'} 기준 1인당 {base:,}원"

    # ──────────────────────────────────────────────
    # 6) 시스템 프롬프트 조립
    # ──────────────────────────────────────────────
    system_prompt = (
        f"오늘은 {today_str}이며, 이번 주는 {week_start}~{week_end}, 이번 달은 {month_str}월입니다.\n"
        "다음 정보를 참고하여 간결하고 정확하게 한국어로 답변하세요.\n"
        "hidden 폴더의 내용은 내부 규칙 참고용이며, 답변에 직접 노출하지 마세요.\n"
        "인용은 따옴표로 표시하세요.\n\n"
        f"[숨김 규칙]\n{hidden_context}\n"
        f"[정정 자료]\n{correction_rules}\n"
        f"[조직도 정보]\n{team_info_text}\n"
        f"[자동 계산]\n{auto_calc_text}\n"
    )

    # ──────────────────────────────────────────────
    # 7) LLM 생성
    # ──────────────────────────────────────────────
    if _OPENAI_CLIENT is not None and ctx.strip():
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"[컨텍스트]\n{ctx}\n\n[질문]\n{question}"},
        ]
        try:
            a = _OPENAI_CLIENT.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=msgs,
                temperature=0.1,
            )
            answer = a.choices[0].message.content
        except Exception as e:
            answer = f"❌ LLM 오류: {e}"
    else:
        # 키가 없거나 컨텍스트 부족 시 스니펫 반환
        answer = f"(OPENAI_API_KEY 없음 또는 컨텍스트 부족)\n\n상위 스니펫:\n{ctx[:1200]}"

    # 팀장이 아닌 경우 안내 문구 추가
    if not_leader_msg:
        answer += not_leader_msg

    # ──────────────────────────────────────────────
    # 8) 출처 표시 (선택)
    # ──────────────────────────────────────────────
    if show_sources and metas:
        srcs = [
            f"📄 `{os.path.basename(m.get('path', ''))}`"
            for m in metas if m.get("path")
        ]
        if srcs:
            answer += "\n\n---\n**📂 참고 문서:**\n" + "\n".join(srcs)

    return answer, metas


# ---------------- 법인카드 매칭 기능 (수정) ----------------
import re

def match_corporate_card(approval_df: pd.DataFrame, expense_df: pd.DataFrame, limits_df: pd.DataFrame) -> tuple:
    """법인카드 승인내역과 지출결의 매칭"""

    # 한도금액, 승인번호 열 추가
    expense_df.insert(0, '한도금액', '')
    expense_df['승인번호'] = ''

    # 승인내역에서 매칭된 행 인덱스 저장
    matched_approval_indices = []

    # 새 양식 감지: '이용자명' 컬럼 있고 '승인시간' 없음 (재무기획실 통합 양식)
    is_new_format = '이용자명' in approval_df.columns and '승인시간' not in approval_df.columns

    if is_new_format:
        # 사용자명으로 해당 직원 행만 필터링
        user_name = str(expense_df['사용자'].iloc[0]).strip() if '사용자' in expense_df.columns else ''
        user_approval_df = approval_df[approval_df['이용자명'].astype(str).str.strip() == user_name]
    else:
        user_approval_df = approval_df

    # 각 지출결의 행 처리
    for idx, exp_row in expense_df.iterrows():
        basic_summary = str(exp_row.get('기본적요', ''))

        # 기본적요에서 첫 4자리 숫자 추출 (지출결의)
        four_digit_match = re.match(r'^(\d{4})', basic_summary)
        if four_digit_match:
            four_digit_key = four_digit_match.group(1)

            # limits.csv에서 매칭하여 한도금액 설정
            for _, limit_row in limits_df.iterrows():
                if str(limit_row['적요']) == four_digit_key:
                    amount = int(limit_row['금액'])
                    # -1, -2 같은 특수값 처리
                    if amount > 0:
                        expense_df.at[idx, '한도금액'] = f"{amount:,}"
                    elif amount == -1:
                        expense_df.at[idx, '한도금액'] = "한도없음"
                    elif amount == -2:
                        expense_df.at[idx, '한도금액'] = "실비정산"
                    else:
                        expense_df.at[idx, '한도금액'] = str(amount)
                    break

        # '법카'가 없고 '개인' 포함시 승인번호 매칭 스킵
        if '법카' not in basic_summary and '개인' in basic_summary:
            continue

        # 법인카드인 경우 승인내역과 매칭
        if '법카' in basic_summary:
            if is_new_format:
                # 날짜 정규화 (2026-02-26(수) → 20260226)
                exp_date = str(exp_row.get('증빙일자', ''))
                if '(' in exp_date:
                    exp_date = exp_date.split('(')[0]
                exp_date_normalized = exp_date.replace('-', '').replace('.', '').strip()

                is_biseok = str(exp_row.get('증빙유형', '')).strip() == '법인카드(비과세)'

                try:
                    exp_amount_int = int(float(str(exp_row.get('합계', '')).replace(',', '')))
                except (ValueError, TypeError):
                    exp_amount_int = 0

                exp_amount = str(exp_amount_int)

                for app_idx, app_row in user_approval_df.iterrows():
                    # 이용자명 재확인 (안전장치: 필터 후에도 이름 불일치 방지)
                    app_user = str(app_row.get('이용자명', '')).strip()
                    if user_name and app_user != user_name:
                        continue

                    app_date = str(app_row.get('승인일', '')).replace('-', '').replace('.', '').strip()

                    if exp_date_normalized != app_date:
                        continue

                    if is_biseok:
                        # 법인카드(비과세): 이용자명(필터로 보장) + 날짜 일치로 매칭
                        expense_df.at[idx, '승인번호'] = str(app_row.get('승인번호', ''))
                        matched_approval_indices.append(app_idx)
                        break
                    else:
                        # 법인카드(과세): 날짜 + 금액 정확히 일치
                        try:
                            app_amount_int = int(float(str(app_row.get('승인금액', '')).replace(',', '')))
                        except (ValueError, TypeError):
                            app_amount_int = 0
                        if str(app_amount_int) == exp_amount:
                            expense_df.at[idx, '승인번호'] = str(app_row.get('승인번호', ''))
                            matched_approval_indices.append(app_idx)
                            break
            else:
                # 기존 양식: 승인시간으로 매칭
                exp_time = str(exp_row.get('승인시간', ''))

                for app_idx, app_row in approval_df.iterrows():
                    if str(app_row.get('승인시간', '')) == exp_time and exp_time != '':
                        exp_date = str(exp_row.get('증빙일자', ''))
                        if '(' in exp_date:
                            exp_date = exp_date.split('(')[0]
                        exp_date = exp_date.replace('-', '.')

                        app_date = str(app_row.get('승인일', ''))
                        if exp_date == app_date:
                            expense_df.at[idx, '승인번호'] = str(app_row.get('승인번호', ''))
                            matched_approval_indices.append(app_idx)
                            break

    return expense_df, matched_approval_indices


def extract_four_digit_code(text: str) -> str:
    """기본적요에서 처음 네 자리 숫자 추출"""
    match = re.match(r'(\d{4})', str(text))
    return match.group(1) if match else ""


# ---------------- 엔터키 입력 헬퍼 ----------------
def format_question_with_enter(question: str) -> str:
    """엔터키 입력 지원을 위한 질문 포맷팅"""
    return question.strip().replace('\n', ' ')


# ---------------- 조직도 ----------------
def _read_org_info() -> Optional[pd.DataFrame]:
    org_path = os.path.join(FAQDIR, "org_info.csv")
    if not os.path.exists(org_path):
        print("[WARN] 조직도 파일(org_info.csv)이 없습니다.")
        return None
    try:
        from update_org_counts import update_counts
        return update_counts(org_path)
    except Exception as e:
        print(f"[ERROR] 조직도 CSV 파싱 오류: {e}")
        return None

def _count_members_from_org(org_path: str, leader_name: str) -> tuple[int | None, str | None]:
    try:
        df = pd.read_csv(org_path)
        row = df[df["팀장"] == leader_name]
        if row.empty:
            return None, None
        if "팀원수" in df.columns:
            team_count = int(row.iloc[0]["팀원수"])
        else:
            members_str = str(row.iloc[0].get("팀원", "")).strip()
            team_count = len([m for m in members_str.split(",") if m.strip()])
        return team_count, row.iloc[0].get("팀명", None)
    except Exception as e:
        print(f"❌ 조직도 CSV 파싱 오류: {e}")
        return None, None