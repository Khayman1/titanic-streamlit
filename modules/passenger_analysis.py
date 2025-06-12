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
    st.markdown("""
### 🚢 탑승자 데이터 분석 시각화

본 대시보드는 타이타닉 탑승자 데이터를 시각적으로 분석하여  
탑승자의 **성별, 나이, 요금, 탑승 위치 및 가족 동반 여부**에 따른 분포를 확인할 수 있습니다.  
각 항목은 드롭다운 메뉴에서 선택하여 자세한 그래프와 함께 확인 가능합니다.  
이를 통해 탑승자 특성과 생존율 간의 관계 분석에 활용할 수 있는 인사이트를 얻을 수 있습니다.
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

    # 탭 선택
    chart_type = st.selectbox(
        "분석 항목 선택",
        ("성별 분포", "나이대 분포", "탑승 위치", "요금 분포", "가족 동반 여부")
    )

    if chart_type == "성별 분포":
        st.markdown("### 👤 탑승자 분포 (남성 / 여성)")
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        wedges, texts, autotexts = ax.pie(
            sex_sizes,
            labels=sex_labels,
            autopct=lambda pct: format_pct(pct, sex_total),
            startangle=90,
            colors=pie_colors
        )
        ax.axis('equal')
        st.pyplot(fig)
        st.info("- 전체 승객 중 **남성이 가장 많고**, 여성이 그보다 적은 수로 탑승하였습니다.")

    elif chart_type == "나이대 분포":
        st.markdown("### 📊 나이 구간별 탑승자 수")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=age_group_counts.index, y=age_group_counts.values, palette='Blues', ax=ax)
        for i, v in enumerate(age_group_counts.values):
            ax.text(i, v + 3, f"{v}명", ha='center', va='bottom', fontsize=7)
        ax.set_title("나이대별 탑승자 분포 (결측 포함)")
        st.pyplot(fig)
        st.warning("""
        - **20–39세** 구간에 가장 많은 승객이 분포되어 있습니다.  
        - 이는 경제 활동 인구 및 이민 목적 탑승 가능성을 시사합니다.  
        - **결측치(기타)**도 상당수 존재하므로 주의가 필요합니다.
        """)

    elif chart_type == "탑승 위치":
        st.markdown("### 🚏 승객 탑승 위치(Embarked)")
        embarked_counts = df['Embarked'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        sns.countplot(data=df, x='Embarked', order=embarked_counts.index, palette='Blues', ax=ax)
        for i, count in enumerate(embarked_counts):
            ax.text(i, count + 2, f"{count}명", ha='center', va='bottom', fontsize=5)
        ax.set_title("탑승지별 승객 수")
        st.pyplot(fig)
        st.info("""
        - **S(Southampton)**는 출발 항구로, 탑승자의 과반수가 이곳에서 승선.  
        - **Q(Queenstown)**: 대부분 3등석 이민자, 생존률 낮음.  
        - **C(Cherbourg)**: 1등석 비중 높아 생존률과 연관될 수 있음.
        """)

    elif chart_type == "요금 분포":
        st.markdown("### 💸 요금(Fare) 분포")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=fare_group_counts.index, y=fare_group_counts.values, palette='Blues', ax=ax)
        for i, v in enumerate(fare_group_counts.values):
            ax.text(i, v + 5, f"{v}명", ha='center', va='bottom', fontsize=7)
        ax.set_title("Fare Group 분포")
        st.pyplot(fig)
        st.info("""
        - 대부분 승객은 **30달러 이하** 요금을 지불.  
        - 이는 **3등석 승객** 비중이 높다는 점을 시사합니다.
        """)

    elif chart_type == "가족 동반 여부":
        st.markdown("### 👪 형제자매 / 배우자 & 부모 / 자녀 수별 탑승자 수")

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
    