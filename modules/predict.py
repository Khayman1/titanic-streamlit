import streamlit as st
import pandas as pd

def run_predict():
    st.title("🚢 탑승자 생존 예측")

    st.markdown("아래 정보를 입력하면 생존 여부를 예측합니다.")

    sex = st.selectbox("성별", ["male", "female"])
    pclass = st.selectbox("객실 등급 (1=1등석, 3=3등석)", [1, 2, 3])
    age = st.slider("나이", 0, 100, 25)
    fare = st.number_input("운임 요금", min_value=0.0, value=32.0)

    # 입력값 전처리
    input_df = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [1 if sex == "female" else 0],  # 인코딩
        "Age": [age],
        "Fare": [fare]
    })

    st.write("입력된 데이터:")
    st.dataframe(input_df)

    # # 예측
    # if st.button("예측하기"):
    #     model_path = "model/titanic_model.pkl"
    #     if not os.path.exists(model_path):
    #         st.error("모델 파일이 존재하지 않습니다. 모델을 먼저 학습 후 저장하세요.")
    #         return

    #     model = joblib.load(model_path)
    #     try:
    #         prediction = model.predict(input_df)
    #         result = "🎉 생존" if prediction[0] == 1 else "☠ 사망"
    #         st.success(f"예측 결과: **{result}**")
    #     except Exception as e:
    #         st.error(f"예측 중 오류 발생: {e}")
