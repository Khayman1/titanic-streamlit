import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_train_data, load_test_data, load_gender_submission_data
# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def run_home():
    st.header("🚢 타이타닉 생존자 대시보드")

    st.markdown("""
    이 대시보드는 **Kaggle 타이타닉 생존자 예측 경진대회**를 기반으로 제작되었습니다.  
    탐색적 자료 분석(EDA)과 머신러닝 모델을 통해 승객의 생존 여부를 시각적으로 분석합니다.
    """)

    # 📦 데이터 불러오기
    train = load_train_data()
    test = load_test_data()
    gender_submission = load_gender_submission_data()
    df = train.copy()

    # 📁 데이터셋 요약
    with st.expander("📁 데이터셋 개요 보기", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.metric("학습 데이터", f"{train.shape[0]}행", f"{train.shape[1]}열")
        col2.metric("테스트 데이터", f"{test.shape[0]}행", f"{test.shape[1]}열")
        col3.metric("제출 예시", f"{gender_submission.shape[0]}행", f"{gender_submission.shape[1]}열")

    # 📊 주요 생존 통계 계산
    total = len(train)
    survived = train['Survived'].sum()
    dead = total - survived
    survival_rate = survived / total * 100
    death_rate = dead / total * 100

    # 📊 카드 형태의 주요 통계 요약
    st.subheader("📊 탑승자 주요 통계 요약")

    card_css = """
    <style>
    .card-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    .card {
        flex: 1;
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        margin: 5px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .card h4 {
        color: #888;
        margin-bottom: 10px;
        font-size: 17px;
    }
    .card h2 {
        font-size: 32px;
        margin: 0;
    }
    .delta-up {
        color: green;
        font-size: 13px;
        margin-top: 5px;
    }
    .delta-down {
        color: red;
        font-size: 13px;
        margin-top: 5px;
    }
    </style>
    """

    card_html = f"""
    <div class="card-container">
        <div class="card">
            <h4>👥 총 탑승자 수</h4>
            <h2 style="color:#1f77b4">{total:,}명</h2>
        </div>
        <div class="card">
            <h4>🟢 생존자 수</h4>
            <h2 style="color:green">{survived:,}명</h2>
        </div>
        <div class="card">
            <h4>🔴 사망자 수</h4>
            <h2 style="color:red">{dead:,}명</h2>
        </div>
    </div>
    """

    st.markdown(card_css + card_html, unsafe_allow_html=True)

    # (선택) 📈 상관관계 히트맵
    with st.expander("📈 수치형 변수 간 상관관계 보기"):
        numeric_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
        corr_matrix = df[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
        ax.set_title("상관관계 히트맵")
        st.pyplot(fig)

        st.info("""
        - `Fare`와 `Pclass`: 강한 음의 상관관계 (요금 ↑ ↔ 등급 낮음)
        - `SibSp`와 `Parch`: 가족 수 간 다소 양의 상관
        - `Survived`와 관련성 있는 변수: `Fare`, `Pclass`, `Parch`
        """)

    # 🔍 샘플 데이터
    with st.expander("🔍 샘플 데이터 미리보기"):
        st.dataframe(df.sample(20, random_state=42), use_container_width=True)
