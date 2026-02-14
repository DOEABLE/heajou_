import streamlit as st
from datetime import datetime
import pandas as pd
import os
import re
from services.rag import safe_rag_query, build_index_all, _read_org_info, match_corporate_card, format_question_with_enter

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
    for row in merged.itertuples(index=False):
        if (row.합계.replace(",", "") != str(row.승인금액)):
            print(row.승인번호)#30010024
            yellow_data.append(row.승인번호)
    
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
    csv_path = "/Users/tarrtarr/Desktop/programming/corpcardAudit/data/vectorstore/org_info.csv"
    
    # CSV 파일 읽기
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    # 입력한 이름과 일치하는 팀장 검색
    matched = df[df["팀장"] == team_leader_name]

    if matched.empty:
        print(f"⚠️ '{team_leader_name}' 이름의 팀장을 찾을 수 없습니다.")
        return 0

    # 팀원수 열 값 추출
    team_member_count = int(matched["팀원수"].iloc[0])

    print(f"✅ {team_leader_name} 팀장님의 팀원 수: {team_member_count}명")
    return team_member_count

def test_all_data(pd_data):
    df = pd_data

    # 조건 필터링
    valid_conditions = [
        "8029 / (판) 법카 - 업무추진비(기타)",
        "8031 / (판) 법카 - 업무추진비(식대)"
    ]
    filtered = df[df["기본적요"].isin(valid_conditions)]

    # '합계' 열만 추출
    if "합계" not in filtered.columns:
        raise ValueError("'합계'라는 열이 없습니다.")
    
    hapgye_col = (
        filtered["합계"]
        .astype(str)  # 혹시 숫자 외 값이 있을 수도 있으므로 문자열화
        .str.replace(",", "", regex=False)  # 쉼표 제거
        .astype(float)  # 숫자 변환
    )

    if "사용자" not in df.columns:
        raise ValueError("'사용자' 열이 없습니다.")
    else:
        user_name_col = df["사용자"].iloc[0]  # 첫 번째 행 값

    all_team_num = team_leader_finder(user_name_col)
    
    # 합계 계산
    total_sum = hapgye_col.sum()

    if (all_team_num * 20000) >= total_sum:
        return 1
    else:
        return 0
    
# 적요의 한도금액이 사용 금액보다 작은지 확인하는 함수    
def test_csv_data_valid(pd_data):
    df = pd_data

    yellow_index = []
    
    for idx, df_row in df.iterrows():
        if df_row["한도금액"] == "":
            pass
        else:
            if df_row["한도금액"] >= df_row["합계"]:
                pass
            else:
                df.style.apply(["background-color: #ffcccc"]*len(df_row))
                print(df_row)
                yellow_index.append(idx)

    return yellow_index


def make_highlight_func(red_inx_arr, yellow_idx_arr, blue_idx_arr):

    def highlight_over_limit(row):
        if row.승인번호 in yellow_idx_arr:
            print("yellow>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            return ["background-color: #ffff99"] * len(row)
        
        if row.기본적요 in blue_idx_arr:
            return ["background-color: #9dbefa"] * len(row)

        if row.name in red_inx_arr:
            return ["background-color: #ffcccc"] * len(row)
        
        elif row["합계"] > row["한도금액"]:
            return ["background-color: #ffeeba"] * len(row)
        
        else:
            return [""] * len(row)
    return highlight_over_limit


st.set_page_config(page_title="RAG FAQ 시스템", layout="wide")
st.title("📚 \"알려줘\"")

# 탭 구성
tab1, tab2, tab3 = st.tabs(["💬 일반 사용자", "💳 법인카드 매칭", "🛠 관리자"])

# Tab 1: 일반 사용자
with tab1:
    if "history" not in st.session_state:
        st.session_state.history = []
    
    # 엔터키로 제출 가능한 form
    with st.form(key='question_form'):
        question = st.text_area("질문을 입력하세요 (Enter로 실행):", height=100)
        col1, col2 = st.columns([1, 5])
        with col1:
            submit = st.form_submit_button("🔍 검색", use_container_width=True)
    
    if submit and question:
        question = format_question_with_enter(question)
        answer = safe_rag_query(question)
        st.session_state.history.append((question, answer))
        st.markdown("### 🤖 답변")
        st.write(answer)
        st.divider()

    if st.session_state.history:
        st.markdown("### 📜 이전 대화 기록")
        for i, (q, a) in enumerate(st.session_state.history, 1):
            with st.expander(f"Q{i}: {q}", expanded=False):
                st.markdown(f"**A:** {a}")

# Tab 2: 법인카드 매칭
with tab2:
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
                else:
                    st.session_state.approval_df = pd.read_excel(approval_file)
                
                # 매칭된 행 하이라이트 함수
                def highlight_matched_approval(row):
                    #if row.name in st.session_state.matched_indices:
                    if str(row['승인번호']) in st.session_state.get("matched_ids_set", set()):
                        return ['background-color: #cccccc'] * len(row)
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
            "지출결의 CSV 파일 업로드",
            type=['csv'],
            key="expense"
        )
        
        if expense_file:
            try:
                expense_df = pd.read_csv(expense_file, encoding='utf-8-sig')
                
                # limits.csv 로드
                limits_path = "data/vectorstore/limits.csv"
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
                                st.session_state.approval_df.iloc[matched_approval_indices]['승인번호'].astype(str)
                            )
                            st.session_state.matched_ids_set = matched_ids  # ← 이제 NameError 안 남
                            
                            #추가
                                                        # 매칭 결과 저장
                            st.session_state.expense_df = result_df
                            st.session_state.matched_indices = matched_approval_indices
                            
                            def highlight_matched_approval(row):
                                if str(row['승인번호']) in st.session_state.get("matched_ids_set", set()):
                                    return ['background-color: #cccccc'] * len(row)
                                return [''] * len(row)
                            
                            left_placeholder.dataframe(
                                st.session_state.approval_df.style.apply(highlight_matched_approval, axis=1),
                                use_container_width=True, height=400
                            )

                            # 매칭 결과 저장
                            st.session_state.expense_df = result_df
                            st.session_state.matched_indices = matched_approval_indices

                            if test_all_data(st.session_state.expense_df) == 1:
                                st.success(f"✅ 업무추진비 체크 완료, 사용 금액이 제한 금액 이내")
                            else:
                                st.success(f"❌ 업무추진비 정합성 오류, 제한 금액 초과")

                            red_inx_arr = test_csv_data_valid(st.session_state.expense_df)
                            yellow_inx_arr = yellow_highliter(st.session_state.approval_df, st.session_state.expense_df)
                            blue_inx_arr = blue_highliter(st.session_state.expense_df)

                            df_coler = st.session_state.expense_df
                            highlight_func = make_highlight_func(red_inx_arr, yellow_inx_arr, blue_inx_arr)

                            df_style = df_coler.style.apply(highlight_func, axis=1)
                            st.dataframe(df_style)

                                                        
                        # --- 🔢 한도금액 매칭 추가 (기본적요 첫 4자리 숫자 기준) ---
                        try:
                            if '기본적요' in st.session_state.expense_df.columns and not limits_df.empty:
                                st.session_state.expense_df['한도금액'] = ""

                                for idx, row in st.session_state.expense_df.iterrows():
                                    # 매 줄마다 승인번호 기준으로 금액 체크
                                    

                                    desc = str(row.get('기본적요') or '').strip()
                                    if len(desc) < 4:
                                        continue
                                    code4 = desc[:4]  # 기본적요 앞 4자리 숫자
                                    # limits.csv의 적요 열 값과 비교
                                    matched_rows = limits_df[limits_df['적요'].astype(str).str.strip() == code4]

                                    if not matched_rows.empty:
                                        amount = str(matched_rows.iloc[0].get('금액') or '').replace(",", "").strip()
                                        # 숫자인 경우만 천단위 표시
                                        if re.match(r"^-?\d+$", amount):
                                            amount_fmt = f"{int(amount):,}"
                                            st.session_state.expense_df.at[idx, '한도금액'] = amount_fmt
                                            print(f"✅ [{code4}] 한도금액 {amount_fmt}원 설정 완료 (기본적요: {desc})")
                                        else:
                                            print(f"⚠️ [{code4}] 금액이 숫자가 아님: {amount}")
                                    else:
                                        print(f"⚠️ [{code4}] limits.csv에 해당 코드 없음 (기본적요: {desc})")
                            else:
                                print("⚠️ limits_df 비어있거나 기본적요 컬럼 없음")
                        except Exception as e:
                            print(f"❌ 한도금액 매칭 오류: {e}")


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

# Tab 3: 관리자
with tab3:
    st.markdown("### 🛠 관리자 기능")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 색인 재빌드", use_container_width=True):
            with st.spinner("재빌드 중..."):
                build_index_all()
                st.success("✅ 색인 재빌드 완료")
    
    with col2:
        if st.button("📁 히든/정정 파일 확인", use_container_width=True):
            hidden_dir = "data/vectorstore/hidden"
            correction_dir = "data/vectorstore/correction"
            
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
                result = safe_rag_query(test_query, show_sources=True)
                st.write(result)
    
    st.divider()
    
    # limits.csv 미리보기
    st.markdown("### 💰 한도 설정 (limits.csv)")
    limits_path = "data/vectorstore/limits.csv"
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

