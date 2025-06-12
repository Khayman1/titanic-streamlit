import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_train_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
import joblib, os
from streamlit_option_menu import option_menu

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def run_survival_data():
    st.header("📊 생존 여부 시각화 예측 모델링")

    st.markdown("""
이 대시보드는 타이타닉 탑승자 데이터를 기반으로 **성별, 객실 등급** 등의 변수와 **생존 여부 간의 관계를 시각화**하고, 간단한 머신러닝 모델을 통해 생존 여부를 예측합니다.

**카드형 메뉴**에서 분석 항목을 선택하면 해당 항목에 대한 시각화와 해석이 나타납니다. 
각 시각화 아래에는 관련 요약 해설과 시사점이 함께 제공되어 탑승자 특성과 생존 여부 간의 관계를 더욱 직관적으로 이해할 수 있습니다.
""")

    df = load_train_data()

    # 카드형 수평 메뉴
    selected = option_menu(
        menu_title=None,
        options=["전체 생존/사망 비율", "성별/객실 생존 분석", "예측 모델 정확도"],
        icons=["heart-pulse", "bar-chart", "cpu"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#f9f9f9"},
            "icon": {"color": "#c94343", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "font-weight": "600",
                "text-align": "center",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {
                "background-color": "#e3f2fd",
                "color": "#0d47a1",
                "font-weight": "bold",
                "border": "1px solid #90caf9",
                "border-radius": "5px",
            },
        }
    )

    if selected == "전체 생존/사망 비율":
        st.subheader("✅ 생존자 / 사망자 수")
        count_data = df['Survived'].value_counts().sort_index()
        labels = ['사망', '생존']
        colors = ["#f86f8f", "#82f99e"]
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

    elif selected == "성별/객실 생존 분석":
        palette = {'생존자': '#48db6b', '사망자': '#ff4d4d'}
        hue_order = ['사망자', '생존자']

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<p style='font-size:20px; font-weight:bold; color:#373737'>👥 성별 생존/사망 인원 수</p>", unsafe_allow_html=True)
            sex_survival = df.groupby(['Sex', 'Survived']).size().unstack().fillna(0)
            sex_survival.columns = ['사망자', '생존자']
            plot_df_sex = sex_survival.reset_index().melt(id_vars='Sex', var_name='생존여부', value_name='명수')

            fig_sex, ax_sex = plt.subplots()
            sns.barplot(data=plot_df_sex, x='Sex', y='명수', hue='생존여부', hue_order=hue_order, palette=palette, ax=ax_sex)
            for container in ax_sex.containers:
                ax_sex.bar_label(container, fmt='%d명', label_type='edge', fontsize=9)
            ax_sex.set_title("성별에 따른 생존/사망 인원 수")
            st.pyplot(fig_sex)

            st.info("""
            - 여성 생존률이 남성보다 압도적으로 높습니다.
            - 이는 '여성과 어린이 우선 구조' 규칙의 영향일 수 있습니다.
            """)

        with col2:
            st.markdown("<p style='font-size:20px; font-weight:bold; color:#373737'>🎟️ 객실 등급별 생존/사망 인원 수</p>", unsafe_allow_html=True)
            pclass_survival = df.groupby(['Pclass', 'Survived']).size().unstack().fillna(0)
            pclass_survival.columns = ['사망자', '생존자']
            plot_df_pclass = pclass_survival.reset_index().melt(id_vars='Pclass', var_name='생존여부', value_name='명수')

            fig_pclass, ax_pclass = plt.subplots()
            sns.barplot(data=plot_df_pclass, x='Pclass', y='명수', hue='생존여부', hue_order=hue_order, palette=palette, ax=ax_pclass)
            for container in ax_pclass.containers:
                ax_pclass.bar_label(container, fmt='%d명', label_type='edge', fontsize=9)
            ax_pclass.set_title("객실 등급(Pclass)에 따른 생존/사망 인원 수")
            st.pyplot(fig_pclass)

            st.info("""
            - 1등석 탑승자는 높은 생존률을 보였으며, 3등석은 생존률이 매우 낮았습니다.
            - 객실 등급은 사회적 계층과 구조 우선순위에 영향을 주는 중요한 요소입니다.
            """)

    elif selected == "예측 모델 정확도":
        st.subheader("🧠 생존 예측 모델 정확도")

        # 인코딩 및 피처 정의
        df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
        features = ['Sex', 'Pclass', 'SibSp', 'Parch', 'Fare']
        X = df[features]
        y = df['Survived']

        # 학습 및 검증
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)

        # 모델 저장
        os.makedirs("model", exist_ok=True)
        joblib.dump(model, "model/survival_model.pkl")
        # st.info("✅ 모델이 `model/survival_model.pkl` 로 저장되어 있습니다.")

        # 정확도 gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=accuracy * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "예측 정확도 (%)"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "green"}}
        ))
        st.plotly_chart(fig_gauge)

        # 해석
        st.markdown("### 🔍 예측 결과 해석")
        st.success(f"""
        이 모델은 다음의 5가지 변수를 사용했습니다:

        - **성별(Sex)**  
        - **객실 등급(Pclass)**  
        - **형제/배우자 수(SibSp)**  
        - **부모/자녀 수(Parch)**  
        - **탑승 요금(Fare)**

        이 변수들을 기반으로 **약 {accuracy:.2%}의 정확도**로 생존 여부를 예측했습니다.

        📌 **시사점**  
        - **여성 승객**은 구조 우선 대상이었을 가능성이 높습니다.  
        - **1등석 및 고요금 승객**은 더 빠른 구조 혜택을 받았을 수 있습니다.  
        - **가족 동반 여부**(SibSp, Parch)는 생존 가능성과 관계가 있을 수 있습니다.  
        가족과 함께한 승객은 구조 시 보호를 받았거나, 반대로 구조가 더 어려웠을 가능성도 고려됩니다.

        이 결과는 **사회적 지위, 가족 구조, 요금 수준 등 여러 요인이 생존에 영향을 미쳤다**는 것을 보여줍니다.
        """)
