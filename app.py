import streamlit as st
import pandas as pd
from home import run_home
from survival_data import run_survival_data 
# from predict import run_predict
from streamlit_option_menu import option_menu

def main():
    with st.sidebar:
        selected = option_menu(
            "대시보드 메뉴",
            ["홈", "생존 여부 자료"],  # ✔ 메뉴명 통일
            icons=["house", "file-bar-graph"],
            menu_icon="cast", 
            default_index=0,
        )

    if selected == "홈":
        run_home()
    elif selected == "생존 여부 자료": 
        run_survival_data()  
    # elif selected == "생존자 예측":
    #     run_predict()
    else:
        st.error("알 수 없는 메뉴입니다.")
    
if __name__ == "__main__":
    main()
