import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_train_data, load_test_data, load_gender_submission_data

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def run_home():
    st.title("🚢 타이타닉 생존자 예측 프로젝트")
    st.markdown("""
        이 대시보드는 **Kaggle 타이타닉 생존자 예측** 경진대회를 기반으로 만들어졌습니다.  
        탐색적 자료 분석(EDA)과 머신러닝 모델을 통해 승객의 생존 여부를 예측합니다.
    """)

    # ✅ 데이터 로딩
    train = load_train_data()
    test = load_test_data()
    gender_submission = load_gender_submission_data()

    # 📋 데이터셋 정보
    st.subheader("📋 데이터셋 요약")
    st.markdown(f"""
    - 학습 데이터: {train.shape[0]}행 × {train.shape[1]}열  
    - 테스트 데이터: {test.shape[0]}행 × {test.shape[1]}열  
    - 제출 예시 데이터: {gender_submission.shape[0]}행 × {gender_submission.shape[1]}열
    """)

    # 📊 대시보드 메트릭 카드
    st.subheader("📊 주요 통계 요약")
    col1, col2, col3 = st.columns(3)

    total = len(train)
    survived = train['Survived'].sum()
    dead = total - survived
    male_surv = train[(train['Sex'] == 'male') & (train['Survived'] == 1)].shape[0]
    female_surv = train[(train['Sex'] == 'female') & (train['Survived'] == 1)].shape[0]

    with col1:
        st.metric("전체 생존자 수", f"{survived:,}", delta=f"{survived / total:.1%}")
    with col2:
        st.metric("남성 생존자 수", f"{male_surv:,}")
    with col3:
        st.metric("여성 생존자 수", f"{female_surv:,}")

    total = len(train)
    survived = train['Survived'].sum()
    dead = total - survived
    male_survived = train[(train['Sex'] == 'male') & (train['Survived'] == 1)].shape[0]
    female_survived = train[(train['Sex'] == 'female') & (train['Survived'] == 1)].shape[0]
    male_total = (train['Sex'] == 'male').sum()
    female_total = (train['Sex'] == 'female').sum()

    st.markdown(f"""<br>
    <div style='font-size:18px; line-height:1.6'>
    🚢 총 승객 수는 <b>{total:,}명</b>이며,  
    <span style='color:green'><b>{survived:,}명</b>이 생존</span>했고,  
    <span style='color:red'><b>{dead:,}명</b>이 사망</span>했습니다.  
    <br>
    👨‍🦱 남성은 <b>{male_total:,}명</b> 중 <span style='color:green'><b>{male_survived:,}명</b></span> 생존  
    👩 여성은 <b>{female_total:,}명</b> 중 <span style='color:green'><b>{female_survived:,}명</b></span> 생존
    </div><br>
    """, unsafe_allow_html=True)

    st.markdown("## 🔍 수치형 변수 간 상관관계 분석")
    df = load_train_data()
    numeric_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    corr_matrix = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f", square=True, ax=ax)
    ax.set_title("📈 수치형 변수 간 상관관계 히트맵")
    st.pyplot(fig)

    st.info("""
    - `Fare`와 `Pclass`는 음의 상관관계를 보입니다 → 요금이 높을수록 등급이 낮은 수치(1등석 = 1)
    - `SibSp`와 `Parch`는 다소 양의 상관관계 → 가족이 많은 승객의 특성
    - `Survived`와 가장 관련 있는 변수는 `Fare`, `Pclass`, `Parch` 정도로 추정됩니다.
    - 강한 상관관계는 0.7 이상, 약한 관계는 ±0.3 이하로 판단할 수 있습니다.
    """)

    # 🔍 샘플 데이터
    st.subheader("🔍 샘플 데이터 미리보기")
    st.dataframe(train.sample(100, random_state=42))

    st.markdown("<hr>", unsafe_allow_html=True)

