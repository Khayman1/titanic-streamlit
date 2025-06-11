import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from utils import load_train_data

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def run_passenger_analysis():
    st.markdown("### 👤 성별 탑승자 분포 (남성 / 여성 / 기타)")
    df = load_train_data()

    df['Sex_Cat'] = df['Sex'].where(df['Sex'].isin(['male', 'female']), other='기타').fillna('기타')
    sex_counts = df['Sex_Cat'].value_counts().sort_index()
    labels = sex_counts.index.tolist()
    sizes = sex_counts.values
    total = sum(sizes)
    colors = {'male': '#64b5f6', 'female': '#f06292', '기타': '#bdbdbd'}
    pie_colors = [colors.get(label, '#ccc') for label in labels]

    def format_pct(pct):
        count = int(round(pct * total / 100.0))
        return f"{pct:.1f}%\n({count}명)"

    fig1, ax1 = plt.subplots(figsize=(4.5, 4.5))
    ax1.pie(sizes, labels=labels, autopct=format_pct, startangle=90, colors=pie_colors)
    ax1.axis('equal')
    st.pyplot(fig1)

    st.info("""
    - 전체 승객 중 **남성이 가장 많고**, 여성은 그보다 적은 수로 탑승하였습니다.
    """)
    st.markdown("---")

    st.markdown("### 📊 나이 구간별 탑승자 수")
    age_bins = [0, 20, 40, 60, 80, float('inf')]
    age_labels = ['0-19세', '20-39세', '40-59세', '60-79세', '80세 이상']
    df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
    df['AgeGroup'] = df['AgeGroup'].cat.add_categories('기타').fillna('기타')
    age_group_counts = df['AgeGroup'].value_counts().sort_index()
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.barplot(x=age_group_counts.index, y=age_group_counts.values, palette='Blues', ax=ax2)
    for i, v in enumerate(age_group_counts.values):
        ax2.text(i, v + 3, f"{v}명", ha='center', va='bottom', fontsize=10)
    ax2.set_title("나이대별 탑승자 분포 (결측 포함)")
    st.pyplot(fig2)

    st.warning("""
    - **20–39세** 구간에 가장 많은 승객이 분포되어 있습니다.
    - 이는 경제 활동 인구 및 이민 목적 탑승 가능성을 시사합니다.
    - **결측치(기타)**도 상당수 존재하므로 주의가 필요합니다.
    """)
    st.markdown("---")

    st.markdown("### 🚏 승객 탑승 위치(Embarked)")
    embarked_counts = df['Embarked'].value_counts().sort_index()
    fig3, ax3 = plt.subplots(figsize=(5.5, 3.8))
    sns.countplot(data=df, x='Embarked', order=embarked_counts.index, palette='Blues', ax=ax3)
    for i, count in enumerate(embarked_counts):
        ax3.text(i, count + 2, f"{count}명", ha='center', va='bottom', fontsize=10)
    ax3.set_title("탑승지별 승객 수")
    st.pyplot(fig3)

    st.info("""
            - S(Southampton)은 타이타닉의 출발 항구로, 전체 탑승자의 과반수가 이곳에서 승선했습니다.  
            - 특히 3등석 승객이 다수를 차지하며, 다양한 계층(1~3등석)의 승객이 혼합되어 탑승했습니다.
            - Q(Queenstown)에서는 3등석 탑승자가 대부분으로, 저렴한 요금을 낸 이민자 계층이 많았고, 생존률은 낮은 편이었습니다.
            - C(Cherbourg)는 1등석 승객 비중이 높아, 생존률과의 관계 분석에 활용될 수 있습니다.  
            """)
    st.markdown("---")

    st.markdown("### 💸 요금(Fare) 분포")
    fare_bins = [0, 10, 30, 100, 250, float('inf')]
    fare_labels = ['0-10', '10-30', '30-100', '100-250', '250+']
    df['FareGroup'] = pd.cut(df['Fare'], bins=fare_bins, labels=fare_labels, right=False)
    fare_group_counts = df['FareGroup'].value_counts().sort_index()
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    sns.barplot(x=fare_group_counts.index, y=fare_group_counts.values, palette='Blues', ax=ax4)
    for i, v in enumerate(fare_group_counts.values):
        ax4.text(i, v + 5, f"{v}명", ha='center', va='bottom', fontsize=9)
    ax4.set_title("Fare Group 분포")
    st.pyplot(fig4)

    st.info("""
    - 다수 승객은 **30달러 이하**의 요금을 지불한 것으로 나타났습니다.
    - 이는 **3등석 승객**이 많음을 의미하며, 요금과 객실 등급 간 강한 연관성이 있습니다.
    """)

    st.markdown("### 👪 형제자매 / 배우자 수별 탑승자 수")
    fig_sibsp, ax_sibsp = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x='SibSp', palette='Blues', ax=ax_sibsp)
    for container in ax_sibsp.containers:
        ax_sibsp.bar_label(container, fmt='%d명', fontsize=9)
    ax_sibsp.set_title("형제자매/배우자 수별 탑승자 분포")
    st.pyplot(fig_sibsp)

    st.info("""
    - 대부분 승객은 **혼자** 탑승했거나 **형제자매/배우자 1명과 함께**였습니다.
    - 동반자 수는 생존률과 직접적인 영향을 줄 수 있는 요인입니다.
    """)

    st.markdown("### 👨‍👧 부모 / 자녀 수별 탑승자 수")
    fig_parch, ax_parch = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x='Parch', palette='Blues', ax=ax_parch)
    for container in ax_parch.containers:
        ax_parch.bar_label(container, fmt='%d명', fontsize=9)
    ax_parch.set_title("부모/자녀 수별 탑승자 분포")
    st.pyplot(fig_parch)

    st.info("""
    - **부모/자녀와 동반한 승객**은 소수이며, 대부분은 **단독 또는 부부 단위** 탑승입니다.
    - 아이 동반 여부는 **생존 우선순위와 관련** 있을 수 있습니다.
    """)
