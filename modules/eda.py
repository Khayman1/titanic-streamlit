import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_train_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def run_survival_data():
    st.title("📊 생존 여부 통계 및 시각 자료")
    df = load_train_data()

    # ✅ 생존 / 사망 비율 (원형 차트)
    st.subheader("✅ 생존자 / 사망자 수")
    count_data = df['Survived'].value_counts().sort_index()
    labels = ['사망', '생존']
    colors = ['#ff4d4d', "#48db6b"]
    total = count_data.sum()

    def format_autopct(pct):
        count = int(round(pct * total / 100.0))
        return f"{pct:.1f}%\n({count}명)"

    fig1, ax1 = plt.subplots()
    ax1.pie(count_data, labels=labels, autopct=format_autopct, startangle=90, colors=colors)
    ax1.set_title("전체 생존 비율")
    ax1.axis('equal')
    st.pyplot(fig1)
    st.info("""
    - 전체적으로 사망자가 생존자보다 많습니다.
    - 약 38%만이 생존했으며, 이는 객실 등급, 성별, 나이와 밀접한 관련이 있습니다.
    """)
    st.markdown("---")


    # ✅ 성별 생존/사망
    st.subheader("👥 성별 생존/사망 인원 수")
    sex_survival = df.groupby(['Sex', 'Survived']).size().unstack().fillna(0)
    sex_survival.columns = ['사망자', '생존자']
    plot_df = sex_survival.reset_index().melt(id_vars='Sex', var_name='생존여부', value_name='명수')

    palette = {'생존자': '#48db6b', '사망자': '#ff4d4d'}
    hue_order = ['사망자', '생존자']

    fig2, ax2 = plt.subplots()
    sns.barplot(data=plot_df, x='Sex', y='명수', hue='생존여부', hue_order=hue_order, palette=palette, ax=ax2)
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%d명', label_type='edge', fontsize=9)
    ax2.set_title("성별에 따른 생존/사망 인원 수")
    st.pyplot(fig2)
    st.info("""
    - 여성 생존률이 남성보다 압도적으로 높습니다.
    - 이는 '여성과 어린이 우선 구조' 규칙의 영향일 수 있습니다.
    """)
    st.markdown("---")

    # ✅ 객실 등급별 생존/사망
    st.subheader("🎟️ 객실 등급별 생존/사망 인원 수")
    pclass_survival = df.groupby(['Pclass', 'Survived']).size().unstack().fillna(0)
    pclass_survival.columns = ['사망자', '생존자']
    plot_df = pclass_survival.reset_index().melt(id_vars='Pclass', var_name='생존여부', value_name='명수')

    fig3, ax3 = plt.subplots()
    sns.barplot(data=plot_df, x='Pclass', y='명수', hue='생존여부', hue_order=hue_order, palette=palette, ax=ax3)
    for container in ax3.containers:
        ax3.bar_label(container, fmt='%d명', label_type='edge', fontsize=9)
    ax3.set_title("객실 등급(Pclass)에 따른 생존/사망 인원 수")
    st.pyplot(fig3)
    st.info("""
    - 1등석 탑승자는 높은 생존률을 보였으며, 3등석은 생존률이 매우 낮았습니다.
    - 객실 등급은 사회적 계층과 구조 우선순위에 영향을 주는 중요한 요소입니다.
    """)
    st.markdown("---")

    # ✅ 간단 예측 분석
    st.subheader("🧠 생존 예측 모델 정확도")

    # ✅ 성별을 숫자로 바꿔줌 (남자: 1, 여자: 0)
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

    # ✅ 특성 선택
    X = df[['Sex', 'Pclass']]
    y = df['Survived']

    # ✅ 모델 훈련
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    # ✅ 정확도 출력
    accuracy = accuracy_score(y_val, y_pred)
    st.metric("🎯 예측 정확도", f"{accuracy:.2%}")

    # ✅ 예측 결과 해석
    st.markdown("### 🔍 예측 결과 해석")
    st.success(f"""
    이 모델은 탑승자의 성별(`Sex`)과 객실 등급(`Pclass`)이라는 단 두 가지 변수만으로 타이타닉 탑승자의 생존 여부를 약 **{accuracy:.2%}** 정확도로 예측하였습니다.

    - **성별**: 여성이 남성보다 생존률이 높다는 사실을 반영  
    - **객실 등급**: 1등석 승객이 구조 우선순위에 있었음을 반영  

    이처럼 단순한 변수만으로도 예측 정확도가 높다는 것은,  
    **탑승자의 생존에 성별과 계층이 큰 영향을 미쳤다**는 사회적 해석도 가능합니다.
    """)