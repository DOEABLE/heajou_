"""org_info.csv의 팀원수/실원수를 팀원 컬럼 기반으로 자동 재계산"""
import os
import pandas as pd

ORG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "vectorstore", "org_info.csv")


def update_counts(path: str = ORG_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")

    # 팀원수 = 팀원 콤마 구분 인원수 + 1 (팀장 포함)
    def count_members(row):
        members_str = str(row.get("팀원", "")).strip()
        if not members_str or members_str == "nan":
            return 1  # 팀장만
        return len([m for m in members_str.split(",") if m.strip()]) + 1

    df["팀원수"] = df.apply(count_members, axis=1)

    # 실원수 = 같은 실의 팀원수 합계 + 실장 1명 (실장이 팀장을 겸하지 않는 경우)
    if "실명" in df.columns and "실장" in df.columns:
        for sil_name in df["실명"].dropna().unique():
            sil_mask = df["실명"] == sil_name
            sil_total = df.loc[sil_mask, "팀원수"].sum()

            # 실장이 팀장을 겸하는지 확인
            sil_leader = df.loc[sil_mask, "실장"].iloc[0] if sil_mask.any() else ""
            leader_is_team_leader = sil_leader in df.loc[sil_mask, "팀장"].values
            if not leader_is_team_leader and sil_leader:
                sil_total += 1  # 실장 본인 추가

            df.loc[sil_mask, "실원수"] = int(sil_total)

        df["실원수"] = df["실원수"].astype("Int64")

    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"완료: {path} ({len(df)}행 업데이트)")
    return df


if __name__ == "__main__":
    df = update_counts()
    print(df[["실명", "실장", "팀명", "팀장", "팀원수", "실원수"]].to_string()
          if "실명" in df.columns
          else df[["팀명", "팀장", "팀원수"]].to_string())
