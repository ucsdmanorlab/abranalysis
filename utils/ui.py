import streamlit as st

def initialize_progress_bar():
    progress_bar = st.progress(0)
    status_text = st.empty()
    count = 0
    return progress_bar, status_text, count

def update_progress_bar(progress_bar, status_text, count, total, status_string):
    progress_value = min(count / total, 1.0) if total > 0 else 0.0
    progress_bar.progress(progress_value)
    status_text.text(status_string)
    return count + 1

def clear_status_bar(progress_bar, status_text):
    progress_bar.empty()
    status_text.empty()