# data_page.py

import streamlit as st
import pandas as pd
from utils import load_train_data, load_test_data, load_gender_submission_data

def run_data_download():
    st.title("📁 데이터 다운로드")

    st.markdown("""
    이 페이지에서는 타이타닉 프로젝트에서 사용되는 원본 CSV 파일 3종을 다운로드할 수 있습니다.  
    데이터를 직접 탐색하거나 외부에서 분석/모델링 테스트에 활용해보세요.
    """)

    # 데이터 로드
    train = load_train_data()
    test = load_test_data()
    gender = load_gender_submission_data()

    # 세 컬럼으로 다운로드 버튼 배치
    col1, col2, col3 = st.columns(3)

    with col1:
        csv_train = train.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 train.csv",
            data=csv_train,
            file_name="train.csv",
            mime="text/csv"
        )

    with col2:
        csv_test = test.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 test.csv",
            data=csv_test,
            file_name="test.csv",
            mime="text/csv"
        )

    with col3:
        csv_gender = gender.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 gender_submission.csv",
            data=csv_gender,
            file_name="gender_submission.csv",
            mime="text/csv"
        )

    st.markdown("---")
    st.markdown("📊 각 데이터는 EDA, 예측 모델링, 제출 파일 생성 등에 사용할 수 있습니다.")
