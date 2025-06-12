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
    이 대시보드는 **Kaggle 타이타닉 생존자 예측 경진대회** 기반으로 만들어졌습니다.  
    탐색적 자료 분석(EDA)과 머신러닝 모델을 통해 승객의 생존 여부를 시각적으로 분석합니다.
    """)

    # 📦 데이터 불러오기
    train = load_train_data()
    test = load_test_data()
    gender_submission = load_gender_submission_data()
    df = train.copy()

    # 📁 데이터셋 요약
    with st.expander("📁 데이터셋 개요 보기"):
        col1, col2, col3 = st.columns(3)
        col1.metric("학습 데이터", f"{train.shape[0]}행", f"{train.shape[1]}열")
        col2.metric("테스트 데이터", f"{test.shape[0]}행", f"{test.shape[1]}열")
        col3.metric("제출 예시", f"{gender_submission.shape[0]}행", f"{gender_submission.shape[1]}열")

    st.markdown("---")
    # 전체 통계 계산
    total = len(train)
    survived = train['Survived'].sum()
    dead = total - survived
    # 📊 주요 생존 정보 카드
    st.subheader("🧾 탑승자 주요 통계 요약")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 👥 총 탑승자 수")
        st.markdown(f"<h2 style='text-align:center; color:#1f77b4'>{total:,}명</h2>", unsafe_allow_html=True)

    with col2:
        st.markdown("### 🟢 생존자 수")
        st.markdown(f"<h2 style='text-align:center; color:green'>{survived:,}명</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center;'>비율: <b>{survived / total:.1%}</b></p>", unsafe_allow_html=True)

    with col3:
        st.markdown("### 🔴 사망자 수")
        st.markdown(f"<h2 style='text-align:center; color:red'>{dead:,}명</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center;'>비율: <b>{dead / total:.1%}</b></p>", unsafe_allow_html=True)

    st.divider()

    # # 📈 상관관계 히트맵
    # st.subheader("📈 수치형 변수 간 상관관계 분석")

    # numeric_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    # corr_matrix = df[numeric_cols].corr()

    # fig, ax = plt.subplots(figsize=(4, 4))
    # sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f", square=True, ax=ax)
    # ax.set_title("상관관계 히트맵", fontsize=2)
    # st.pyplot(fig)

    # with st.expander("🔎 해석 보기"):
    #     st.info("""
    #     - `Fare`와 `Pclass`: 강한 음의 상관관계 (요금 ↑ ↔ 객실등급 1등석=1)
    #     - `SibSp`와 `Parch`: 가족 수 간 다소 양의 상관
    #     - `Survived`와 가장 관련된 변수: `Fare`, `Pclass`, `Parch`
    #     """)

    # # 🔍 샘플 데이터 보기
    # st.subheader("🔍 샘플 데이터 미리보기")
    # st.caption("※ 학습 데이터 중 무작위로 추출한 20개 행을 표시합니다.")
    # st.dataframe(train.sample(20, random_state=42), use_container_width=True)

    # st.markdown("---")
