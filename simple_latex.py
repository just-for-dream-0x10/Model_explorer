"""
简化的LaTeX渲染工具
"""

import streamlit as st


def display_latex(formula, display_mode=True):
    """
    显示LaTeX公式
    """
    if display_mode:
        st.markdown(f"$$ {formula} $$")
    else:
        st.markdown(f"$ {formula} $")


def display_formula_box(formula, title=""):
    """
    显示带标题的公式框
    """
    if title:
        st.markdown(f"**{title}**")
    st.markdown(f"$$ {formula} $$")


def display_math_content(content, title=""):
    """
    显示数学内容
    """
    if title:
        st.markdown(f"**{title}**")
    st.markdown(content)
