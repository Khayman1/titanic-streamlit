# survival_data.py

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_train_data

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def run_survival_data():
    st.title("📊 생존 여부 통계 및 시각 자료")

    df = load_train_data()

    st.subheader("✅ 생존자 / 사망자 수")
    count_data = df['Survived'].value_counts().sort_index()
    labels = ['사망', '생존']
    colors = ['#ff9999', '#66b3ff']
    total = count_data.sum()

    def format_autopct(pct):
        count = int(round(pct * total / 100.0))
        return f"{pct:.1f}%\n({count}명)"
    fig1, ax1 = plt.subplots()
    ax1.pie(count_data, labels=labels, autopct=format_autopct, startangle=90, colors=colors)
    ax1.set_title("전체 생존 비율")
    ax1.axis('equal')
    st.pyplot(fig1)

    st.subheader("👥 성별 생존/사망 인원 수")
    sex_survival = df.groupby(['Sex', 'Survived']).size().unstack().fillna(0)
    sex_survival.columns = ['사망자', '생존자']
    plot_df = sex_survival.reset_index().melt(id_vars='Sex', var_name='생존여부', value_name='명수')
    fig2, ax = plt.subplots()
    sns.barplot(data=plot_df, x='Sex', y='명수', hue='생존여부', ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt='%d명', label_type='edge', fontsize=9)
    ax.set_title("성별에 따른 생존/사망 인원 수")
    st.pyplot(fig2)

    st.subheader("🎟️ 객실 등급별 생존/사망 인원 수")
    pclass_survival = df.groupby(['Pclass', 'Survived']).size().unstack().fillna(0)
    pclass_survival.columns = ['사망자', '생존자']
    plot_df = pclass_survival.reset_index().melt(id_vars='Pclass', var_name='생존여부', value_name='명수')
    fig3, ax = plt.subplots()
    sns.barplot(data=plot_df, x='Pclass', y='명수', hue='생존여부', ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt='%d명', label_type='edge', fontsize=9)
    ax.set_title("객실 등급(Pclass)에 따른 생존/사망 인원 수")
    st.pyplot(fig3)
