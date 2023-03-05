"""
streamlit run D:/Code/PythonCode/streamlit_tutorial/demo.py
"""
import streamlit as st
from srcs.main_page import showMainPage
from srcs.tec_doc import main_docs
from srcs.vis_tutorial import visPage
import srcs.multipage as mt

# 设置网页标题，以及使用宽屏模式
st.set_page_config(
    page_title="个人主页",
    layout="wide",
    page_icon=":seedling:",  # icon
    initial_sidebar_state="auto"  # 侧边栏
)

# 隐藏右边的菜单以及页脚
hide_streamlit_style = """
<style>
MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# 左边导航栏
sidebar = st.sidebar.selectbox(
    "功能汇总",
    ("个人主页", "技术文档", "学习资料", '可视化平台')
)

if sidebar == "技术文档":
    main_docs(st)
elif sidebar == "可视化平台":
    visPage(st)
else:
    showMainPage(st)
