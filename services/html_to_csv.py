"""그룹웨어(AMARANTH10) 지출결의서 HTML → DataFrame 변환"""
from __future__ import annotations
import pandas as pd
from bs4 import BeautifulSoup


def _find_td_by_class(row, cls: str) -> str:
    """row 안에서 특정 CSS class를 가진 td의 텍스트 반환"""
    td = row.find("td", class_=cls)
    return td.get_text(strip=True) if td else ""


def _extract_header(tables: list) -> dict:
    """헤더 테이블에서 부서/사용자/결의금액/지급요청일 추출"""
    header = {"부서": "", "사용자": "", "결의금액": "", "지급요청일": ""}
    for table in tables:
        td = table.find("td", class_="useDeptName")
        if td:
            header["부서"] = td.get_text(strip=True)
        td = table.find("td", class_="userName")
        if td:
            header["사용자"] = td.get_text(strip=True)
        td = table.find("td", class_="amtTot")
        if td:
            header["결의금액"] = td.get_text(strip=True)
        td = table.find("td", class_="payReqDate")
        if td:
            header["지급요청일"] = td.get_text(strip=True)
        if all(header.values()):
            break
    return header


def extract_expense_data_from_html(html_content: str) -> pd.DataFrame:
    """
    그룹웨어 지출결의서 HTML을 파싱하여 DataFrame으로 변환.
    각 항목은 3개 행으로 구성:
      Row A (debitRmk): 기본적요, 증빙유형, 증빙일자, 승인시간, 사용카드
      Row B (dztrade):  프로젝트, 거래처, 업무용차량, 공급가액, 부가세, 합계
      Row C (summary):  상세내용
    """
    soup = BeautifulSoup(html_content, "html.parser")
    tables = soup.find_all("table")
    header = _extract_header(tables)

    # 본문 테이블: debitRmk class를 포함하는 테이블 찾기
    data_table = None
    for t in tables:
        if t.find("td", class_="debitRmk"):
            data_table = t
            break
    if data_table is None:
        return pd.DataFrame()

    rows = data_table.find_all("tr")

    # debitRmk가 있는 행의 인덱스를 먼저 수집
    debit_indices = []
    for idx, row in enumerate(rows):
        if row.find("td", class_="debitRmk"):
            debit_indices.append(idx)

    records = []
    for di in debit_indices:
        row_a = rows[di]
        debit_td = row_a.find("td", class_="debitRmk")

        rec = {
            "기본적요": debit_td.get_text(strip=True),
            "증빙유형": _find_td_by_class(row_a, "taxInvestigationDivision"),
            "증빙일자": _find_td_by_class(row_a, "taxDate"),
            "승인시간": _find_td_by_class(row_a, "apprTime"),
            "사용카드": _find_td_by_class(row_a, "cardUser"),
            "프로젝트": "",
            "거래처": "",
            "업무용차량": "",
            "공급가액": "",
            "부가세": "",
            "합계": "",
            "상세내용": "",
            "지급요청일": header["지급요청일"],
            "결의금액": header["결의금액"],
            "부서": header["부서"],
            "사용자": header["사용자"],
        }

        # Row B: 거래처, 프로젝트, 업무용차량, 금액
        if di + 1 < len(rows):
            row_b = rows[di + 1]
            trade_td = row_b.find("td", class_="dztrade")
            if trade_td:
                rec["거래처"] = trade_td.get_text(strip=True)
            project_td = row_b.find("td", class_="dzproject")
            if project_td:
                rec["프로젝트"] = project_td.get_text(strip=True)
            workcar_td = row_b.find("td", class_="workcar")
            if workcar_td:
                rec["업무용차량"] = workcar_td.get_text(strip=True)
            rec["공급가액"] = _find_td_by_class(row_b, "valueOfSupply")
            rec["부가세"] = _find_td_by_class(row_b, "valueOfVAT")
            rec["합계"] = _find_td_by_class(row_b, "valueOfTot")

        # Row C: 상세내용
        if di + 2 < len(rows):
            row_c = rows[di + 2]
            summary_td = row_c.find("td", class_="summary")
            if summary_td:
                rec["상세내용"] = summary_td.get_text(strip=True)

        # 빈 항목 필터 (기본적요가 비어있거나 합계가 없는 경우 제외)
        if not rec["기본적요"] or (not rec["합계"] and not rec["공급가액"]):
            continue

        records.append(rec)

    columns = [
        "기본적요", "증빙유형", "증빙일자", "승인시간", "사용카드", "프로젝트",
        "거래처", "업무용차량", "공급가액", "부가세", "합계", "상세내용",
        "지급요청일", "결의금액", "부서", "사용자",
    ]
    return pd.DataFrame(records, columns=columns)
