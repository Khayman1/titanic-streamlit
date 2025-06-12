import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils import load_train_data

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def format_pct(pct, total):
    count = int(round(pct * total / 100.0))
    return f"{pct:.1f}%\n({count}명)"

def run_passenger_analysis():
    st.header("🚢 탑승자 분석 대시보드")
    st.markdown("""
이 대시보드는 타이타닉호 탑승자 데이터를 시각적으로 분석하여  
탑승자의 ( **성별**, **나이대**, **탑승 위치**, **요금**, **가족 동반 여부** )에 따른 분포를 보여줍니다.
각 분석 항목은 상단의 버튼을 클릭하여 선택할 수 있으며, 선택된 항목에 따라 관련된 그래프와 함께 설명이 함께 제공됩니다.
                
이를 통해 탑승자의 다양한 특성이 생존 여부와 어떤 관계가 있었는지를  
직관적으로 이해하고, 데이터 기반의 인사이트를 얻을 수 있습니다.
""")
    df = load_train_data()

    # 공통 전처리
    df['Sex_Cat'] = df['Sex'].where(df['Sex'].isin(['male', 'female']), other='기타').fillna('기타')
    sex_counts = df['Sex_Cat'].value_counts().sort_index()
    sex_labels = sex_counts.index.tolist()
    sex_sizes = sex_counts.values
    sex_total = sum(sex_sizes)
    sex_colors = {'male': "#1083e0", 'female': "#9ec9f4", '기타': '#bdbdbd'}
    pie_colors = [sex_colors.get(label, '#ccc') for label in sex_labels]

    # 나이대
    age_bins = [0, 20, 40, 60, 80, float('inf')]
    age_labels = ['0-19세', '20-39세', '40-59세', '60-79세', '80세 이상']
    df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
    df['AgeGroup'] = df['AgeGroup'].cat.add_categories('기타').fillna('기타')
    age_group_counts = df['AgeGroup'].value_counts().sort_index()

    # 요금대
    fare_bins = [0, 10, 30, 100, 250, float('inf')]
    fare_labels = ['0-10', '10-30', '30-100', '100-250', '250+']
    df['FareGroup'] = pd.cut(df['Fare'], bins=fare_bins, labels=fare_labels, right=False)
    fare_group_counts = df['FareGroup'].value_counts().sort_index()

    # 강조된 제목 박스
    st.markdown("""
    <div style='
        background-color: #f0f4ff;
        border-left: 6px solid #1e88e5;
        padding: 10px 14px;
        margin-bottom: 0px;
        border-radius: 5px;
        font-size: 15px;
        '>
    📌 <span style="color:#0d47a1;">아래에서 분석 항목을 선택하세요:</span>
    </div>
    """, unsafe_allow_html=True)

    # 수평 radio 메뉴
    chart_type = st.radio(
        label="",
        options=("탑승자 분포 & 나이대 분포", "탑승 위치 분포 & 요금 분포", "가족 동반 여부"),
        horizontal=True
    )



    if chart_type == "탑승자 분포 & 나이대 분포":
        col1, col2 = st.columns(2)

        # 왼쪽: 성별 탑승자 수
        with col1:
            st.markdown("#### 👤 성별 탑승자 수")
            fig1, ax1 = plt.subplots(figsize=(5, 4))  # 크기 약간 키움
            sex_counts = df['Sex'].value_counts()
            sns.barplot(x=sex_counts.index, y=sex_counts.values, palette="pastel", ax=ax1)
            for i, val in enumerate(sex_counts.values):
                ax1.text(i, val * 0.95, f'{val}명', ha='center', va='top', fontsize=11, color='black')
            ax1.set_ylabel("탑승자 수")
            ax1.set_xlabel("성별")
            ax1.set_title("성별 탑승자 분포")
            st.pyplot(fig1)
            st.info("""
- 전체 승객 중 **남성이 가장 많고**, 여성이 그보다 적은 수로 탑승하였습니다.  
- 이는 당대 사회 구조에서 **남성이 주요 이동 주체**였음을 시사합니다.  
- 하지만 생존률은 여성이 훨씬 높으므로, 단순 인원 수만으로 구조 우선 순위를 판단해선 안 됩니다.
""")
        with col2:
            st.markdown("#### 📊 나이대 탑승자 수")
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            highlight_label = '20-39세'
            colors = ['#1565C0' if label == highlight_label else '#cfd8dc' for label in age_group_counts.index]

            sns.barplot(x=age_group_counts.index, y=age_group_counts.values, palette=colors, ax=ax2)

            # 텍스트 라벨
            for i, (label, value) in enumerate(zip(age_group_counts.index, age_group_counts.values)):
                offset = value * 0.02 if value > 20 else 1.5
                color = 'white' if label == highlight_label else 'black'
                ax2.text(i, value - offset, f"{value}명", ha='center', va='top', fontsize=9, color=color)

            ax2.set_title("나이대별 탑승자 분포 (결측 포함)")
            ax2.set_ylabel("탑승자 수")
            ax2.set_xlabel("나이대")
            st.pyplot(fig2)
            st.warning("""
            - **20–39세** 구간에 가장 많은 승객이 분포되어 있습니다.  
            - 이는 경제 활동 인구 및 이민 목적 탑승 가능성을 시사합니다.  
            - **결측치(기타)**도 상당수 존재하므로 주의가 필요합니다.
            """)

    elif chart_type == "탑승 위치 분포 & 요금 분포":
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🚏 승객 탑승 위치")
            embarked_counts = df['Embarked'].value_counts().sort_index()
            fig_embarked, ax_embarked = plt.subplots(figsize=(4, 3))
            sns.countplot(data=df, x='Embarked', order=embarked_counts.index, palette='Blues', ax=ax_embarked)
            for i, count in enumerate(embarked_counts):
                ax_embarked.text(i, count - 5, f"{count}명", ha='center', va='top', fontsize=9, color='white')
            ax_embarked.set_title("탑승지별 승객 수")
            ax_embarked.set_xlabel("탑승 위치")
            ax_embarked.set_ylabel("탑승자 수")
            st.pyplot(fig_embarked)
            st.info("""
            - **S(Southampton)**: 는 출발 항구로, 탑승자의 과반수가 이곳에서 승선.  
            - **Q(Queenstown)**: 대부분 3등석 이민자, 생존률 낮음.  
            - **C(Cherbourg)**: 1등석 비중 높아 생존률과 연관될 수 있음.
            """)

        with col2:
            st.markdown("#### 💸 요금(Fare) 분포")
            fig_fare, ax_fare = plt.subplots(figsize=(4.5, 3.5))
            sns.barplot(x=fare_group_counts.index, y=fare_group_counts.values, palette='Blues', ax=ax_fare)
            for i, v in enumerate(fare_group_counts.values):
                if v < 15:
                    ax_fare.text(i, v + 2, f"{v}명", ha='center', va='bottom', fontsize=9, color='black')  # 막대 위
                else:
                    ax_fare.text(i, v - 3, f"{v}명", ha='center', va='top', fontsize=9, color='white')  # 막대 안쪽
            ax_fare.set_title("요금 그룹별 승객 수")
            ax_fare.set_xlabel("요금 구간 ($)")
            ax_fare.set_ylabel("탑승자 수")
            st.pyplot(fig_fare)
            st.info("""
            - 대부분 승객은 **30달러 이하** 요금을 지불.  
            - 이는 **3등석 승객** 비중이 높다는 점을 시사합니다.
            """)


    elif chart_type == "가족 동반 여부":

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 👤 형제자매 / 배우자 수")
            fig_sibsp, ax_sibsp = plt.subplots(figsize=(5, 4))
            sns.countplot(data=df, x='SibSp', palette='Blues', ax=ax_sibsp)
            for container in ax_sibsp.containers:
                ax_sibsp.bar_label(container, fmt='%d명', fontsize=9)
            ax_sibsp.set_title("형제자매/배우자 수")
            st.pyplot(fig_sibsp)

        with col2:
            st.markdown("#### 👶 부모 / 자녀 수")
            fig_parch, ax_parch = plt.subplots(figsize=(5, 4))
            sns.countplot(data=df, x='Parch', palette='Blues', ax=ax_parch)
            for container in ax_parch.containers:
                ax_parch.bar_label(container, fmt='%d명', fontsize=9)
            ax_parch.set_title("부모/자녀 수")
            st.pyplot(fig_parch)

        st.info("""
        - 대부분 승객은 **혼자 또는 배우자/형제자매 1명과 함께 탑승**했습니다.  
        - 부모/자녀 동반 승객은 비교적 적으며, **어린이 또는 가족 단위 탑승 여부는 생존율과 연관**될 수 있습니다.
        """)
    