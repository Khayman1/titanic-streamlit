# modules/passenger_filter.py

import streamlit as st
from utils import load_train_data

def run_passenger_filter():
    st.header("🔎 탑승자 데이터 검색")
    st.markdown("""
선택한 조건(성별, 객실 등급, 나이대, 생존 여부)에 따라 탑승자 데이터를 실시간으로 필터링하여 조회할 수 있습니다.

- **성별**: 남성 / 여성  
- **객실 등급**: 1, 2, 3등석  
- **생존 여부**: 생존 / 사망  
- **나이대**: 0~9세부터 80세 이상, 기타까지 구간별 선택 가능

검색 결과는 아래 표에 자동으로 반영되며, 총 인원 수도 함께 표시됩니다.
""")


    # 📦 데이터 불러오기 (맨 위에서 수행)
    df = load_train_data()

    # 📊 나이 그룹화 함수
    def get_age_group(age):
        try:
            age = float(age)
            if age < 10: return "0-9세"
            elif age < 20: return "10-19세"
            elif age < 30: return "20-29세"
            elif age < 40: return "30-39세"
            elif age < 50: return "40-49세"
            elif age < 60: return "50-59세"
            elif age < 70: return "60-69세"
            elif age < 80: return "70-79세"
            elif age >= 80: return "80세 이상"
            else: return "기타"
        except:
            return "기타"

    # 연령대 컬럼 생성
    df['AgeGroup'] = df['Age'].apply(get_age_group)

    # 🔎 필터 UI
    sex_filter = st.multiselect("성별 선택", options=df['Sex'].unique(), default=df['Sex'].unique())
    pclass_filter = st.multiselect("객실 등급 선택", options=sorted(df['Pclass'].unique()), default=sorted(df['Pclass'].unique()))
    survived_filter = st.multiselect("생존 여부 선택", options=[0, 1], format_func=lambda x: "사망" if x == 0 else "생존", default=[0, 1])
    age_group_options = sorted(df['AgeGroup'].unique().tolist())
    selected_groups = st.multiselect("나이대 선택", options=age_group_options, default=age_group_options)

    # ✅ 필터 적용
    filtered_df = df[
        (df['Sex'].isin(sex_filter)) &
        (df['Pclass'].isin(pclass_filter)) &
        (df['AgeGroup'].isin(selected_groups)) &
        (df['Survived'].isin(survived_filter))
    ]

    # 결과 출력
    st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)
    st.success(f"🔍 검색 결과: 총 {len(filtered_df)}명")
