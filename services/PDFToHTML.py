import re
import fitz  # PyMuPDF
import pandas as pd

ENTRY_START = re.compile(r"\b(?P<code>\d{3,5})\s*/\s*\(판\)\s*", re.MULTILINE)

def _extract_document_header(full_text: str):
    """
    문서 상단 공통 메타(지급요청일/결의금액/부서/사용자) 최대한 가져오되,
    없으면 빈 값으로 둠. (서식 변형 파일 대비)
    """
    # 공백 정규화
    t = re.sub(r"[ \t]+", " ", full_text)

    pay_request = re.search(r"지급요청일\s*([0-9\-()]+)", t)
    total_amount = re.search(r"결의금액\s*([\d,]+)", t)
    dept = re.search(r"소\s*속\s*([^\s]+)", t)
    drafter = re.search(r"기\s*안\s*자\s*([가-힣A-Za-z]+)", t)

    return {
        "지급요청일": pay_request.group(1) if pay_request else "",
        "결의금액": total_amount.group(1) if total_amount else "",
        "부서": dept.group(1) if dept else "",
        "사용자": drafter.group(1) if drafter else "",
    }

def _read_pdf_as_ordered_text(path: str) -> str:
    """
    페이지별로 block을 y,x 좌표 기준 정렬해 문단 순서를 최대한 보장.
    줄 구분은 '\n'로 유지하여 후단 정규식 안정화.
    """
    doc = fitz.open(path)
    parts = []
    for page in doc:
        blocks = page.get_text("blocks")  # [(x0,y0,x1,y1, text, ...), ...]
        # y,x로 정렬
        blocks = sorted(blocks, key=lambda b: (round(b[1], 1), round(b[0], 1)))
        for b in blocks:
            text = b[4].replace("\xa0", " ").strip()
            if text:
                parts.append(text)
        parts.append("\n")  # 페이지 경계
    # 여러 줄 공백 정리
    txt = "\n".join(parts)
    # 가독성 향상을 위해 다중 공백 최소화(숫자 사이 공백 제거는 금지)
    txt = re.sub(r"[ \t]{2,}", " ", txt)
    return txt

def _split_entries(text: str):
    """
    항목 시작(예: '8029 / (판) ...') 위치를 모두 찾아 다음 시작 전까지를 한 블록으로 분리.
    '상세내용'이 없더라도 항목 경계로 동작.
    """
    spans = []
    starts = [m.start() for m in ENTRY_START.finditer(text)]
    for i, s in enumerate(starts):
        e = starts[i+1] if i+1 < len(starts) else len(text)
        spans.append(text[s:e].strip())
    return spans

def _parse_entry_block(block: str, header_meta: dict):
    """
    한 항목 블록 안에서 필드 추출.
    - 법인/개인 공통 처리
    - 승인시간/사용카드 optional
    - 거래처/금액은 금액 패턴을 기준으로 역으로 분리
    - 상세내용은 존재 시만 추출
    """
    rec = {
        "기본적요": "",
        "증빙유형": "",
        "증빙일자": "",
        "승인시간": "",
        "사용카드": "",
        "거래처": "",
        "공급가액": "",
        "부가세": "",
        "합계": "",
        "상세내용": "",
        "지급요청일": header_meta.get("지급요청일", ""),
        "결의금액": header_meta.get("결의금액", ""),
        "부서": header_meta.get("부서", ""),
        "사용자": header_meta.get("사용자", ""),
    }

    # 기본적요: '#### / (판) ...' ~ '법인카드|개인카드' 앞까지
    m = re.search(r"(?P<기본>^\d{3,5}\s*/\s*\(판\)\s*.+?)-(?:\s*)?(?=.*?(법인카드|개인카드))", block, re.DOTALL | re.MULTILINE)
    if m:
        # ' ... - ' 앞까지를 깔끔히
        base = m.group("기본").strip()
        # 뒤에 붙은 '법카 - ' 같은 접두 보정(있으면 유지)
        rec["기본적요"] = re.sub(r"\s+", " ", base)

    # 증빙유형 (+과세/비과세)
    m = re.search(r"(법인카드|개인카드)(?:\((과세|비과세)\))?", block)
    if m:
        rec["증빙유형"] = m.group(1) + (f"({m.group(2)})" if m.group(2) else "")

    # 증빙일자(YYYY-MM-DD)
    m = re.search(r"(\d{4}-\d{2}-\d{2})", block)
    if m:
        rec["증빙일자"] = m.group(1)

    # 승인시간 (optional)
    m = re.search(r"\b(\d{2}:\d{2}:\d{2})\b", block)
    if m:
        rec["승인시간"] = m.group(1)

    # 사용카드 (optional)
    m = re.search(r"(국민|신한|하나|삼성|BC)\([^)]*\)-?\d*", block)
    if m:
        rec["사용카드"] = m.group(0)

    # 상세내용 (optional) - 현재 블록 안의 '상세내용 ...' ~ 끝 또는 다음 항목 시작 직전
    m = re.search(r"상세내용\s*(.+)$", block, re.DOTALL)
    if m:
        # 다음 항목 시작표현 제거
        detail = re.split(r"\n?\d{3,5}\s*/\s*\(판\)\s*", m.group(1).strip())[0]
        rec["상세내용"] = re.sub(r"\s+", " ", detail).strip()

    # 거래처 + 금액(공급가액/부가세/합계)
    # 금액 3개 연속 패턴을 먼저 찾고, 그 앞 단어들을 거래처로 간주(개인카드는 '조규원' 등)
    money = re.search(r"(?P<공급>[\d]{1,3}(?:,\d{3})*)\s+(?P<부가세>[\d]{1,3}(?:,\d{3})*)\s+(?P<합계>[\d]{1,3}(?:,\d{3})*)", block)
    if money:
        rec["공급가액"] = money.group("공급")
        rec["부가세"] = money.group("부가세")
        rec["합계"] = money.group("합계")

        # 금액 덩어리 앞쪽을 잘라 거래처 도출
        left = block[:money.start()].strip()
        # 마지막 줄/토큰에서 거래처 후보 추출
        # (한글/영문/숫자/특수 일부 허용)
        cand = re.search(r"([가-힣A-Za-z0-9()_./&\-\s]{1,60})$", left)
        if cand:
            vendor = re.sub(r"\s+", " ", cand.group(1)).strip()
            # 너무 일반적인 꼬리표 제거(예: 카드번호가 남은 경우)
            vendor = re.sub(r"(국민|신한|하나|삼성|BC)\([^)]*\)-?\d*$", "", vendor).strip()
            # 지나치게 짧은 단어 보정(예: '판교'만 남는 이슈) - 앞쪽 몇 단어 더 탐색
            if len(vendor) <= 2:
                cand2 = re.search(r"([가-힣A-Za-z0-9()_./&\-\s]{1,100})$", left[:max(0, len(left)-80)])
                if cand2:
                    vendor = (cand2.group(1) + " " + vendor).strip()
            rec["거래처"] = vendor

    return rec

def extract_expense_data_from_pdf(pdf_path: str, output_csv: str = None) -> pd.DataFrame:
    """
    ✅ 항목 시작 패턴으로 블록 분리 + 블록 내 필드 인식
    ✅ '상세내용'이 없어도 동작 (있으면 채움)
    ✅ 개인/법인 공통 처리
    ✅ 문서 헤더(결의금액 표)는 자연스레 제외
    """
    text = _read_pdf_as_ordered_text(pdf_path)
    header = _extract_document_header(text)
    blocks = _split_entries(text)

    records = []
    for blk in blocks:
        rec = _parse_entry_block(blk, header)
        # 유효성 체크: 합계/거래처/증빙유형 중 1개 이상 존재하면 항목으로 인정
        if any(rec.get(k) for k in ("합계", "거래처", "증빙유형")):
            records.append(rec)

    df = pd.DataFrame.from_records(records)

    # 후처리: 공백/포맷 정리
    for col in ["기본적요", "증빙유형", "사용카드", "거래처", "상세내용"]:
        if col in df.columns:
            df[col] = df[col].fillna("").map(lambda s: re.sub(r"\s+", " ", str(s)).strip())

    # 금액 합계 검증용 숫자 컬럼(옵션)
    for col in ["공급가액", "부가세", "합계"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", "", regex=False).str.strip()
            df[col] = df[col].where(df[col].str.fullmatch(r"\d+").fillna(False), "")
            df[col+"_num"] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    if output_csv:
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    return df

# 사용 예시:
df = extract_expense_data_from_pdf("전자결재 - 슈어소프트 그룹웨어.pdf", "지출결의서_최종_2.csv")
print(len(df), df["합계_num"].sum())