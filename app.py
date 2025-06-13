import streamlit as st
from streamlit_option_menu import option_menu
from modules.home import run_home
from modules.eda import run_survival_data
from modules.passenger_analysis import run_passenger_analysis
from modules.passenger_filter import run_passenger_filter
from modules.data_page import run_data_download

def main():
    with st.sidebar:
        st.markdown("<h2 style='color:black'>🚢 타이타닉 대시보드</h2>", unsafe_allow_html=True)
        st.markdown("<hr style='margin-top:0;'>", unsafe_allow_html=True)
        st.markdown("<p style='color:gray;'>생존과 사망 데이터를<br>시각적으로 분석합니다.</p>", unsafe_allow_html=True)

        # 메뉴
        selected = option_menu(
            None,
            ["홈", "탑승자 분석","탑승자 데이터 검색","생존 여부 예측 모델","데이터 다운로드"],  # 메뉴 항목
            icons=["house-fill", "people-fill", "search","bar-chart-line-fill", "cloud-download-fill"],
            menu_icon="cast",
            default_index=0,
            # orientation="vertical",
            styles={
                "container": {
                    "padding": "0!important",
                },
                "nav-link": {
                    "font-size": "16px",
                    "padding": "12px 20px",
                    "white-space": "nowrap",  # 🔹 줄바꿈 방지
                },
                "nav-link-selected": {
                    "background-color": "#ff6666",
                    "color": "#ffffff",
                    "font-weight": "bold"
                },
            }
        )
    # 메뉴 선택에 따른 페이지 전환
    if selected == "홈":
        run_home()
    elif selected == "탑승자 분석":
        run_passenger_analysis()
    elif selected == "생존 여부 예측 모델":
        run_survival_data()
    elif selected == "탑승자 데이터 검색":
        run_passenger_filter()
    elif selected == "데이터 다운로드":
        run_data_download()
    else:
        st.error("⚠️ 알 수 없는 메뉴입니다.")
    
    # Footer
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:gray;'>© 2025 Titanic Dashboard by Min Khay Man</p>", unsafe_allow_html=True)
# 실행
if __name__ == "__main__":
    main()
