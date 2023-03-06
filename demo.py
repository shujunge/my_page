"""
streamlit run D:/Code/PythonCode/streamlit_tutorial/demo.py
"""
import streamlit as st
from streamlit_ace import st_ace
import streamlit_echarts as st_echarts


import srcs.multipage as mt
from srcs.multipage import MultiPage

from srcs.main_page import showMainPage
from srcs.tec_doc import main_docs
from srcs.vis_tutorial import visPage

def landing_page(st):
    st.title("This is a Multi Page Application")
    st.write("Feel free to leave give a star in the Github Repo")


def footer(st):
    st.write("Developed by [ELC](https://github.com/shujunge)")


def header(st):
    st.write("This app is free to use")


def sidebar(st):
    st.button("Donate (Dummy)")


def input_page(st, **state):
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


app = MultiPage()
app.st = st

app.start_button = "Go to the main page"
app.navbar_name = "Other Pages:"
app.next_page_button = "Next Chapter"
app.previous_page_button = "Previous Chapter"
app.reset_button = "Delete Cache"
app.navbar_style = "SelectBox"

app.header = header
app.footer = footer
app.navbar_extra = sidebar

app.hide_menu = True
app.hide_navigation = True

app.add_app("Landing", landing_page, initial_page=True)
app.add_app("个人主页", input_page)
app.add_app("技术文档", main_docs)
app.add_app("可视化平台", visPage)

app.run()
