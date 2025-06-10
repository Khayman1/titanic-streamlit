# utils.py
import pandas as pd
import streamlit as st

@st.cache_data
def load_train_data():
    return pd.read_csv('data/train.csv')

@st.cache_data
def load_test_data():
    return pd.read_csv('data/test.csv')

@st.cache_data
def load_gender_submission_data():
    return pd.read_csv('data/gender_submission.csv')
