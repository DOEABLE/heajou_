import streamlit as st
from datetime import datetime
import pandas as pd
import os
import re
from dotenv import load_dotenv
import gspread
from google.oauth2.service_account import Credentials
from rag import rag_answer, build_index, _read_org_info, match_corporate_card, format_question_with_enter
from html_to_csv import extract_expense_data_from_html

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
GOOGLE_CREDENTIALS_PATH = os.path.join(
    os.path.dirname(__file__), "..", os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json")
)
FEEDBACK_IMAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "feedback_images")
os.makedirs(FEEDBACK_IMAGE_DIR, exist_ok=True)

def get_gsheet():
    scopes = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_file(GOOGLE_CREDENTIALS_PATH, scopes=scopes)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(GOOGLE_SHEET_ID).sheet1
    return sheet

def append_feedback(row: list):
    try:
        sheet = get_gsheet()
        sheet.append_row(row)
        return True
    except Exception as e:
        st.error(f"Google Sheets 전송 오류: {e}")
        return False

def yellow_highliter(cardlog, doclog):
    #cardlog 데이터프레임에 "승인번호" 행과 doclog 데이터프레임에 "승인번호"가 일치하는 경우
     #cardlog의 "승인금액" 열과 doclog의 "합계"가 일치하지 않는 경우
      #doclog의 "승인번호"를 가진 행에 노란색 하이라이트
     # 그렇지 않다면
      #pass
    #그렇지 않다면 pass
    print("new>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    #print(cardlog)
    print("new>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    #print(doclog)

    cardlog['승인번호'] = cardlog['승인번호'].astype(str)
    doclog['승인번호'] = doclog['승인번호'].astype(str)

    merged = pd.merge(cardlog, doclog, on="승인번호", how='inner')
    #result_test = merged[merged['승인번호']=="30009885"]
    #print(result_test)
    yellow_data = []
    for _, row in merged.iterrows():
        try:
            total = str(row['합계']).replace(",", "").strip()
            approved = str(row['승인금액']).replace(",", "").strip()
            try:
                approved = str(int(float(approved)))
            except (ValueError, TypeError):
                pass
            if total != approved:
                print(row['승인번호'])
                yellow_data.append(row['승인번호'])
        except Exception as e:
            print(f"[WARN] yellow_highliter 행 처리 오류: {e}")

    return yellow_data


def blue_highliter(doclog):
    blue_data = []
    blue_data = doclog.loc[
        doclog['기본적요'].astype(str).str.contains('개인', na=False),
        '기본적요'
    ].tolist()
    print("blue_data >>>", blue_data)

    return blue_data
    


def team_leader_finder(team_leader_name):
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "vectorstore", "org_info.csv")

    # CSV 파일 읽기 (팀원수 자동 재계산)
    from update_org_counts import update_counts
    df = update_counts(csv_path)

    # 입력한 이름과 일치하는 팀장 검색 (공백 제거 후 비교)
    team_leader_name = str(team_leader_name).strip()
    df["팀장"] = df["팀장"].astype(str).str.strip()
    matched = df[df["팀장"] == team_leader_name]

    if matched.empty:
        print(f"[WARN] '{team_leader_name}' 이름의 팀장을 찾을 수 없습니다.")
        return None

    # 팀원수 열 값 추출
    team_member_count = int(matched["팀원수"].iloc[0])

    print(f"[INFO] {team_leader_name} 팀장님의 팀원 수: {team_member_count}명")
    return team_member_count

EXEC_ONLY_CODES = {"8529", "8365", "8049"}

SPECIAL_TEAM_KEYWORDS = {"사업개발", "AX지원팀", "솔루션현장지원팀", "모빌리티지원팀"}

def get_comm_limit_8363(user_name):
    """8363(통신비) 한도 반환
    - 특수팀(사업개발 포함/AX지원팀/솔루션현장지원팀/모빌리티지원팀) 팀장·팀원 → 70,000
    - 일반팀 팀장 → 60,000
    - 일반팀 팀원 → None (한도 미적용)
    """
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "vectorstore", "org_info.csv")
    try:
        from update_org_counts import update_counts
        df = update_counts(csv_path)
    except Exception:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')

    user_name = str(user_name).strip()
    df["팀장"] = df["팀장"].astype(str).str.strip()

    leader_match = df[df["팀장"] == user_name]
    if not leader_match.empty:
        team_name = str(leader_match.iloc[0]["팀명"])
        if any(kw in team_name for kw in SPECIAL_TEAM_KEYWORDS):
            return 70000
        return 60000

    for _, row in df.iterrows():
        members = [m.strip() for m in str(row.get("팀원", "")).split(",")]
        if user_name in members:
            team_name = str(row["팀명"])
            if any(kw in team_name for kw in SPECIAL_TEAM_KEYWORDS):
                return 70000
            return None

    return None


def test_all_data(pd_data):
    df = pd_data

    if "사용자" not in df.columns:
        raise ValueError("'사용자' 열이 없습니다.")

    user_name_col = str(df["사용자"].iloc[0]).strip()
    print(f"[DEBUG] 사용자 값: '{user_name_col}'")
    all_team_num = team_leader_finder(user_name_col)
    is_leader = all_team_num is not None

    # 팀장이 아닌 경우: 보직자 전용 항목(8529/8365/8049) 포함 여부 체크
    if not is_leader:
        has_exec_only = df["기본적요"].astype(str).apply(
            lambda x: x[:4] in EXEC_ONLY_CODES
        ).any()
        if has_exec_only:
            return 2  # 보직자 전용 항목 포함 경고
        return 3  # 팀장 아님 + 보직자 항목 없음 → 알림 없음

    # 업무추진비 조건 필터링
    valid_conditions = [
        "8029 / (판) 법카 - 업무추진비(기타)",
        "8031 / (판) 법카 - 업무추진비(식대)"
    ]
    filtered = df[df["기본적요"].isin(valid_conditions)]

    if "합계" not in filtered.columns:
        raise ValueError("'합계'라는 열이 없습니다.")

    hapgye_col = (
        filtered["합계"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

    total_sum = hapgye_col.sum()
    team_num = all_team_num if all_team_num else 0

    if (team_num * 20000) >= total_sum:
        return 1
    else:
        return 0
    
# 적요의 한도금액이 사용 금액보다 작은지 확인하는 함수    
def _parse_amount(v):
    """합계/공급가액 문자열에서 숫자만 추출해 정수로 변환. 파싱 실패 시 None 반환."""
    s = re.sub(r'[^\d.]', '', str(v).replace(",", ""))
    if not s:
        return None
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


def test_csv_data_valid(pd_data):
    df = pd_data

    yellow_index = []

    # 통신비 합산 한도 체크 대상 코드 (행별이 아닌 합산으로 비교)
    # 8363: 일반팀 팀장만 60,000 적용 (특수팀은 70,000이므로 한도금액으로 구분)
    # 6363: limits.csv 기준 60,000 (일반팀 통신비)
    CUMULATIVE_COMM_CODES = {"8363", "6363"}

    for code4 in CUMULATIVE_COMM_CODES:
        target_rows = df[df["기본적요"].astype(str).str.strip().str[:4] == code4]
        if target_rows.empty:
            continue
        limit_60_rows = target_rows[
            target_rows["한도금액"].astype(str).str.replace(",", "", regex=False).str.strip() == "60000"
        ]
        if limit_60_rows.empty:
            print(f"[DEBUG] {code4}: limit_60_rows 없음 (한도금액 설정 확인 필요)")
            continue
        amounts = []
        for v_합계, v_공급 in zip(limit_60_rows["합계"], limit_60_rows.get("공급가액", [""] * len(limit_60_rows))):
            amt = _parse_amount(v_합계)
            if amt is None:
                amt = _parse_amount(v_공급)  # 합계 파싱 실패 시 공급가액 대체
            if amt is not None:
                amounts.append(amt)
        total_comm = sum(amounts)
        print(f"[DEBUG] {code4} 합산: {amounts} → 합계={total_comm}, 한도=60000")
        if total_comm > 60000:
            print(f"[INFO] {code4} 통신비 합산 한도초과: 합계={total_comm}, 한도=60000")
            yellow_index.extend(limit_60_rows.index.tolist())

    for idx, df_row in df.iterrows():
        if df_row["한도금액"] == "" or pd.isna(df_row["한도금액"]):
            pass
        else:
            try:
                desc = str(df_row.get("기본적요") or "")
                # 합산 처리 대상 코드(8363/6363)의 60,000 한도 행은 위에서 처리했으므로 스킵
                if desc.strip()[:4] in CUMULATIVE_COMM_CODES and str(df_row["한도금액"]).replace(",", "").strip() == "60000":
                    continue
                limit_val = _parse_amount(df_row["한도금액"])
                total_val = _parse_amount(df_row["합계"])
                if limit_val is None or total_val is None:
                    continue
                if limit_val < 0:
                    pass  # 음수 한도(-1, -2)는 체크 제외
                elif limit_val >= total_val:
                    pass
                else:
                    print(f"[INFO] 한도초과: idx={idx}, 합계={total_val}, 한도={limit_val}")
                    yellow_index.append(idx)
            except (ValueError, TypeError):
                pass

    return yellow_index


def make_highlight_func(red_inx_arr, yellow_idx_arr, blue_idx_arr, biseok_yellow_idx=None):

    def highlight_over_limit(row):
        # 승인번호가 비어있는 경우(지출결의서 등) yellow 체크 스킵 → 한도초과 red가 가려지는 것 방지
        if str(row.승인번호).strip() and row.승인번호 in yellow_idx_arr:
            return ["background-color: #f6bd5a; color: #ffffff"] * len(row)

        if biseok_yellow_idx and row.name in biseok_yellow_idx:
            return ["background-color: #f6bd5a; color: #ffffff"] * len(row)

        # 한도초과(red)는 개인카드(blue)보다 우선 적용
        if row.name in red_inx_arr:
            return ["background-color: #fa4747; color: #ffffff"] * len(row)

        if row.기본적요 in blue_idx_arr:
            return ["background-color: #1e73be; color: #ffffff"] * len(row)

        else:
            return [""] * len(row)
    return highlight_over_limit


st.set_page_config(page_title="FAQ & 법인카드 감사시스템", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Black+Han+Sans&display=swap');
.title-font {
    font-family: 'Black Han Sans', sans-serif;
    text-align: center;
    font-size: 48px;
    margin-bottom: 40px;
    color: #FF6B00;
}
/* 메인 탭 버튼만 스타일 적용 (첫 번째 stHorizontalBlock) */
div[data-testid="stHorizontalBlock"]:first-of-type button[data-testid="stBaseButton-secondary"],
div[data-testid="stHorizontalBlock"]:first-of-type button[data-testid="stBaseButton-primary"] {
    height: 140px !important;
    font-size: 40px !important;
    font-weight: 900 !important;
    font-family: 'Black Han Sans', sans-serif !important;
    border-radius: 12px !important;
    transition: all 0.2s !important;
}
div[data-testid="stHorizontalBlock"]:first-of-type button[data-testid="stBaseButton-secondary"] {
    border: 2px solid #ddd !important;
}
div[data-testid="stHorizontalBlock"]:first-of-type button[data-testid="stBaseButton-secondary"]:hover {
    border-color: #2c5fc7 !important;
    color: #2c5fc7 !important;
}
</style>
""", unsafe_allow_html=True)

_duck_path = os.path.join(os.path.dirname(__file__), "..", "data", "duck.png")
_left_col, _title_col, _right_col = st.columns([1, 5, 1])
with _left_col:
    if os.path.exists(_duck_path):
        st.image(_duck_path)
with _title_col:
    st.markdown("<h1 class='title-font'>🌐 얼마남지 않았어, 아무도 야근하지 않는 방법을 찾아야 해.⏱️</h1>", unsafe_allow_html=True)
with _right_col:
    if os.path.exists(_duck_path):
        st.image(_duck_path)

# 탭 상태 관리
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "매칭"

# 탭 버튼
tabs_config = [
    ("연구소", "🐦 경영지원연구소"),
    ("매칭", "💳 법인카드 매칭"),
    ("피드백", "📝 피드백"),
    ("관리자", "🛠 관리자"),
]
with st.container():
    st.markdown('<div class="main-tabs">', unsafe_allow_html=True)
    btn_cols = st.columns(len(tabs_config))
    for i, (key, label) in enumerate(tabs_config):
        with btn_cols[i]:
            is_active = st.session_state.active_tab == key
            btn_type = "primary" if is_active else "secondary"
            if st.button(label, use_container_width=True, key=f"tab_btn_{i}", type=btn_type):
                st.session_state.active_tab = key
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# Tab 1: 경영지원연구소
if st.session_state.active_tab == "연구소":
    if "history" not in st.session_state:
        st.session_state.history = []

    st.markdown("<p style='font-size: 20px; font-weight: 800;'> 🐦 슈어에 대한 모든 것 🐦 (Ctrl+Enter: 검색)</p>", unsafe_allow_html=True)
    with st.form(key='question_form'):
        question = st.text_area("", height=100, label_visibility="collapsed", placeholder="ex) 경조사 청구에 대해 알려줘.")
        col1, col2 = st.columns([1, 5])
        with col1:
            submit = st.form_submit_button("🔍 검색", use_container_width=True)

    if submit and question:
        question = format_question_with_enter(question)
        answer, _ = rag_answer(question)
        st.session_state.history.append((question, answer))
        st.markdown("### 🤖 답변")
        st.markdown(f"<div style='font-size:18px; line-height:1.8'>{answer}</div>", unsafe_allow_html=True)
        st.divider()

    if st.session_state.history:
        st.markdown("### 📜 이전 대화 기록")
        for i, (q, a) in enumerate(st.session_state.history, 1):
            with st.expander(f"Q{i}: {q}", expanded=False):
                st.markdown(f"<div style='font-size:18px; line-height:1.8'>{a}</div>", unsafe_allow_html=True)

# Tab 2: 법인카드 매칭
elif st.session_state.active_tab == "매칭":
    st.markdown("### 💳 법인카드 승인내역 매칭")
    
    # 세션 상태 초기화
    if "approval_df" not in st.session_state:
        st.session_state.approval_df = None
    if "expense_df" not in st.session_state:
        st.session_state.expense_df = None
    if "matched_indices" not in st.session_state:
        st.session_state.matched_indices = []
    
    # 화면을 좌우로 분할
    left_col, right_col = st.columns(2)
    #left_placeholder = left_col.empty()
    
    with left_col:
        st.markdown("#### 📄 법인카드 승인내역")

        approval_file = st.file_uploader(
            "승인내역 파일 업로드 (xlsx/xls/csv)", 
            type=['xlsx', 'xls', 'csv'],
            key="approval"
        )
        left_placeholder = st.empty()
        if approval_file:
            # 파일 읽기
            try:
                if approval_file.name.endswith('.csv'):
                    st.session_state.approval_df = pd.read_csv(approval_file, encoding='utf-8-sig')
                elif approval_file.name.endswith('.xls'):
                    import xlrd, io
                    file_bytes = approval_file.read()
                    wb = xlrd.open_workbook(file_contents=file_bytes, encoding_override='cp949')
                    ws = wb.sheet_by_index(0)
                    headers = ws.row_values(0)
                    data = [ws.row_values(i) for i in range(1, ws.nrows)]
                    _approval = pd.DataFrame(data, columns=headers)
                    # xlrd float → 정수 변환 (예: 3100.0 → 3100)
                    for col in _approval.columns:
                        try:
                            converted = _approval[col].apply(lambda x: int(x) if isinstance(x, float) and not pd.isna(x) and x == int(x) else x)
                            _approval[col] = converted
                        except Exception:
                            pass
                    _drop_cols = ['할인금액', '청구회차', '잔여회차', '수수료', '연체원금', '연체수수료']
                    _approval.drop(columns=[c for c in _drop_cols if c in _approval.columns], inplace=True)
                    st.session_state.approval_df = _approval
                else:
                    st.session_state.approval_df = pd.read_excel(approval_file)

                # 승인금액 등 정수형이어야 할 컬럼 float → int 변환 (csv/xlsx는 NaN 포함 시 float64로 읽힘)
                for _col in ['승인금액', '이용금액']:
                    if _col in st.session_state.approval_df.columns:
                        try:
                            st.session_state.approval_df[_col] = (
                                pd.to_numeric(st.session_state.approval_df[_col], errors='coerce')
                                .fillna(0)
                                .astype(int)
                            )
                        except Exception:
                            pass

                # 매칭된 행 하이라이트 함수
                def highlight_matched_approval(row):
                    if row.name in st.session_state.get("matched_indices", []) and str(row['승인번호']) in st.session_state.get("matched_expense_no_set", set()):
                        return ['background-color: #5d6d7e; color: #ffffff'] * len(row)
                    return [''] * len(row)
                
                # 데이터 표시
                if st.session_state.matched_indices:
                    #st.dataframe(
                    #    st.session_state.approval_df.style.apply(highlight_matched_approval, axis=1),
                    #    use_container_width=True, height=400
                    #)

                    # (교체)
                    left_placeholder.dataframe(
                        st.session_state.approval_df.style.apply(highlight_matched_approval, axis=1),
                        use_container_width=True, height=400
                    )

                else:
                    #st.dataframe(st.session_state.approval_df, use_container_width=True, height=400)
                    # (교체)
                    left_placeholder.dataframe(st.session_state.approval_df, use_container_width=True, height=400)
                    
            except Exception as e:
                st.error(f"파일 읽기 오류: {str(e)}")
    
    with right_col:
        st.markdown("#### 📝 지출결의")
        expense_file = st.file_uploader(
            "지출결의 파일 업로드 (HTML 또는 CSV)",
            type=['html', 'csv'],
            key="expense"
        )

        if expense_file:
            try:
                if expense_file.name.lower().endswith('.html'):
                    html_content = expense_file.read().decode('utf-8')
                    expense_df = extract_expense_data_from_html(html_content)
                    st.info(f"HTML → CSV 변환 완료: {len(expense_df)}건")
                else:
                    expense_df = pd.read_csv(expense_file, encoding='utf-8-sig')
                
                # limits.csv 로드
                limits_path = os.path.join(os.path.dirname(__file__), "..", "data", "vectorstore", "correction", "limits.csv")
                if os.path.exists(limits_path):
                    limits_df = pd.read_csv(limits_path, encoding='utf-8-sig')
                    #st.success(f"✅ limits.csv 로드 완료 ({len(limits_df)}개 항목)")
                else:
                    st.warning("⚠️ data/vectorstore/limits.csv 파일이 없습니다.")
                    limits_df = pd.DataFrame(columns=['적요', '직급', '금액'])
                
                # 매칭 실행 버튼
                if st.button("🔄 매칭 실행", type="primary", use_container_width=True):
                    if st.session_state.approval_df is not None:
                        with st.spinner("매칭 중..."):
                            # 매칭 실행
                            result_df, matched_approval_indices = match_corporate_card(
                                st.session_state.approval_df, 
                                expense_df.copy(),  # 원본 보존을 위해 복사본 사용
                                limits_df
                            )
                            
                            # ✅ 추가: 승인번호 set 생성 (왼쪽 표 색칠 및 대조 기준)
                            matched_ids = set(
                                st.session_state.approval_df.loc[matched_approval_indices]['승인번호'].astype(str)
                            ) if matched_approval_indices else set()
                            st.session_state.matched_ids_set = matched_ids
                            # expense_df에 기록된 승인번호 set (승인번호 일치 추가 검증용)
                            st.session_state.matched_expense_no_set = set(
                                result_df['승인번호'].astype(str).replace('', pd.NA).dropna()
                            )
                            
                            #추가
                                                        # 매칭 결과 저장
                            st.session_state.expense_df = result_df
                            st.session_state.matched_indices = matched_approval_indices
                            
                            def highlight_matched_approval(row):
                                if row.name in st.session_state.get("matched_indices", []) and str(row['승인번호']) in st.session_state.get("matched_expense_no_set", set()):
                                    return ['background-color: #5d6d7e; color: #ffffff'] * len(row)
                                return [''] * len(row)
                            
                            left_placeholder.dataframe(
                                st.session_state.approval_df.style.apply(highlight_matched_approval, axis=1),
                                use_container_width=True, height=400
                            )

                            # 매칭 결과 저장
                            st.session_state.expense_df = result_df
                            st.session_state.matched_indices = matched_approval_indices

                            # --- 🔢 한도금액 매칭 (하이라이트 판정 전에 실행) ---
                            try:
                                if '기본적요' in st.session_state.expense_df.columns and not limits_df.empty:
                                    _limit_df = st.session_state.expense_df.copy()
                                    _limit_df['한도금액'] = ""

                                    # 통신비(8363) 한도 판단을 위한 사용자명 추출
                                    _comm_user = ""
                                    if "사용자" in _limit_df.columns and len(_limit_df) > 0:
                                        _comm_user = str(_limit_df["사용자"].iloc[0]).strip()

                                    for idx, row in _limit_df.iterrows():
                                        desc = str(row.get('기본적요') or '').strip()
                                        if len(desc) < 4:
                                            continue
                                        code4 = desc[:4]

                                        # 통신비(8363) 특수 처리: 팀/직급에 따라 한도 차등 적용
                                        if code4 == "8363":
                                            comm_limit = get_comm_limit_8363(_comm_user)
                                            if comm_limit:
                                                _limit_df.at[idx, '한도금액'] = f"{comm_limit:,}"
                                            continue

                                        matched_rows = limits_df[limits_df['적요'].apply(lambda x: str(int(float(x))) if pd.notna(x) else '').str.strip() == code4]

                                        if not matched_rows.empty:
                                            raw = matched_rows.iloc[0].get('금액')
                                            amount = str(raw).replace(",", "").strip()
                                            if re.match(r"^-?\d+(\.\d+)?$", amount):
                                                amount_int = int(float(amount))
                                                if amount_int == -1:
                                                    _limit_df.at[idx, '한도금액'] = "한도없음"
                                                elif amount_int == -2:
                                                    _limit_df.at[idx, '한도금액'] = "실비정산"
                                                elif amount_int > 0:
                                                    _limit_df.at[idx, '한도금액'] = f"{amount_int:,}"
                                                else:
                                                    _limit_df.at[idx, '한도금액'] = str(amount_int)

                                    st.session_state.expense_df = _limit_df
                                else:
                                    print("[WARN] limits_df 비어있거나 기본적요 컬럼 없음")
                            except Exception as e:
                                st.error(f"[ERROR] 한도금액 매칭 오류: {e}")

                            result = test_all_data(st.session_state.expense_df)
                            if result == 1:
                                st.success("✅ 업무추진비 체크 완료, 사용 금액이 제한 금액 이내")
                            elif result == 2:
                                st.error("⚠️ 보직자 이상만 올릴 수 있는 내역이 포함되어있습니다.")
                            elif result == 0:
                                st.error("❌ 업무추진비 정합성 오류, 제한 금액 초과")
                            # result == 3: 팀장 아님 + 보직자 항목 없음 → 알림 없음

                            red_inx_arr = test_csv_data_valid(st.session_state.expense_df)
                            yellow_inx_arr = yellow_highliter(st.session_state.approval_df, st.session_state.expense_df)
                            blue_inx_arr = blue_highliter(st.session_state.expense_df)

                            df_coler = st.session_state.expense_df
                            # 법인카드(비과세) 행 인덱스 추출 → 노랑 처리
                            biseok_yellow_idx = list(
                                st.session_state.expense_df[
                                    st.session_state.expense_df['증빙유형'].astype(str).str.strip() == '법인카드(비과세)'
                                ].index
                            )
                            highlight_func = make_highlight_func(red_inx_arr, yellow_inx_arr, blue_inx_arr, biseok_yellow_idx)

                            df_style = df_coler.style.apply(highlight_func, axis=1)
                            st.dataframe(df_style)


                        # 매칭 통계 표시
                        total_expense = len(result_df)
                        matched_count = len(matched_approval_indices)
                        limit_count = sum(1 for val in result_df['한도금액'] if val != '')
                        
                        st.success(f"""✅ 매칭 완료!
                        - 승인번호 매칭: {matched_count}/{total_expense}건
                        - 한도금액 설정: {limit_count}/{total_expense}건""")

                    else:
                        st.error("⚠️ 먼저 승인내역 파일을 업로드해주세요.")
                else:
                    # 매칭 전 원본 표시
                    st.info("매칭 실행 버튼을 클릭하여 처리를 시작하세요.")
            
                # 다운로드 버튼
                if st.session_state.expense_df is not None:
                    csv = st.session_state.expense_df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        "📥 결과 다운로드",
                        csv,
                        f"matched_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                
            except Exception as e:
                st.error(f"파일 처리 오류: {str(e)}")

# Tab 3: 피드백
elif st.session_state.active_tab == "피드백":
    st.markdown("### 📝 피드백 제출")
    st.markdown("불편하신 점이나 개선 요청사항을 남겨주세요. 개발팀이 검토 후 반영하겠습니다.")

    with st.form(key="feedback_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            fb_dept = st.text_input("부서 *", placeholder="예) 재무기획실")
        with col2:
            fb_feature = st.selectbox("사용 기능 *", ["경영지원연구소", "법인카드 매칭", "기타"])

        fb_issue_type = st.selectbox(
            "문제 유형 *",
            ["오류 발생", "사용 불편", "속도 느림", "기능 요청", "문의"]
        )
        fb_content = st.text_area("의견 및 문제 설명 *", height=150, placeholder="구체적으로 작성해주실수록 빠른 처리가 가능합니다.")
        #fb_image = st.file_uploader("화면 첨부 (선택)", type=["png", "jpg", "jpeg", "gif", "webp"])

        submitted = st.form_submit_button("📨 제출", use_container_width=True, type="primary")

    if submitted:
        if not fb_dept.strip() or not fb_content.strip():
            st.error("부서와 의견 및 문제 설명은 필수 항목입니다.")
        else:
            # 이미지 저장
            image_path = ""
            if fb_image:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_name = f"{ts}_{fb_image.name}"
                save_path = os.path.join(FEEDBACK_IMAGE_DIR, save_name)
                with open(save_path, "wb") as f:
                    f.write(fb_image.read())
                image_path = save_path

            row = [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                fb_dept.strip(),
                fb_feature,
                fb_content.strip(),
                fb_issue_type,
                image_path,
                "New"
            ]
            if append_feedback(row):
                st.success("✅ 피드백이 성공적으로 제출되었습니다. 감사합니다!")

# Tab 4: 관리자
elif st.session_state.active_tab == "관리자":
    st.markdown("### 🛠 관리자 기능")

    # 📋 사용설명서 다운로드
    spec_path = os.path.join(os.path.dirname(__file__), "..", "data", "docs", "feature_spec.html")
    if os.path.exists(spec_path):
        with open(spec_path, "r", encoding="utf-8") as f:
            spec_html = f.read()
        st.download_button(
            label="📋 사용설명서 다운로드 (HTML)",
            data=spec_html,
            file_name="feature_spec.html",
            mime="text/html",
        )
    else:
        st.warning("⚠️ 기능명세서 파일이 없습니다.")

    st.divider()

    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 색인 재빌드", use_container_width=True):
            with st.spinner("재빌드 중..."):
                build_index()
                st.success("✅ 색인 재빌드 완료")
    
    with col2:
        if st.button("📁 히든/정정 파일 확인", use_container_width=True):
            hidden_dir = "../data/vectorstore/hidden"
            correction_dir = "../data/vectorstore/correction"
            
            if os.path.exists(hidden_dir):
                st.info(f"Hidden: {os.listdir(hidden_dir)}")
            else:
                st.info("Hidden: 폴더 없음")
                
            if os.path.exists(correction_dir):
                st.info(f"Correction: {os.listdir(correction_dir)}")
            else:
                st.info("Correction: 폴더 없음")
    
    with col3:
        # 테스트 질의
        test_query = st.text_input("테스트 질의:")
        if st.button("🧪 테스트", use_container_width=True):
            if test_query:
                result = rag_answer(test_query, show_sources=True)
                st.write(result)
    
    st.divider()
    
    # limits.csv 미리보기
    st.markdown("### 💰 한도 설정")
    limits_path = os.path.join(os.path.dirname(__file__), "..", "data", "vectorstore", "correction", "limits.csv")
    if os.path.exists(limits_path):
        limits_preview = pd.read_csv(limits_path, encoding='utf-8-sig')
        st.dataframe(limits_preview, use_container_width=True, height=200)
    else:
        st.warning("limits.csv 파일이 없습니다.")
    
    st.divider()
    
    # 조직도 관리
    st.markdown("### 👥 조직도 관리")
    
    org_df = _read_org_info()
    if org_df is not None:
        # 검색 기능
        search_term = st.text_input("🔍 팀/팀장 검색:")
        
        if search_term:
            filtered_df = org_df[
                org_df['팀명'].str.contains(search_term, na=False) |
                org_df['팀장'].str.contains(search_term, na=False)
            ]
        else:
            filtered_df = org_df
        
        # 펼침 보기
        for idx, row in filtered_df.iterrows():
            with st.expander(f"{row['팀명']} ({row['팀장']} 팀장)"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("팀원 수", f"{row['팀원수']}명")
                with col2:
                    st.write(f"**팀원:** {row['팀원']}")
    else:
        st.warning("⚠️ 조직도 파일(org_info.csv)이 없습니다.")
    
    # 시스템 정보
    st.divider()
    st.caption(f"🕒 서버 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

