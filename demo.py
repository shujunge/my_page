"""
streamlit run D:/Code/PythonCode/streamlit_tutorial/demo.py
"""
import streamlit as st
from srcs.multipage import MultiPage

from srcs.main_page import showMainPage
from srcs.tec_doc import devops_tutorial, data_analysis
from srcs.vis_tutorial import my_vis_page


def landing_page(st):
    st.title("This is a Multi Page Application")
    st.write("Feel free to leave give a star in the Github Repo")


def footer(st):
    st.write("Developed by [shujunge](https://github.com/shujunge)")


def header(st):
    st.write("This app is free to use")


def sidebar(st):
    st.button("Donate (Dummy)")


app = MultiPage()
app.st = st

app.start_button = "Go to the main page"
app.navbar_name = "Other Pages:"
app.next_page_button = "Next Chapter"
app.previous_page_button = "Previous Chapter"
app.reset_button = "Delete Cache"
app.navbar_style = "SelectBox"

# app.header = header
# app.footer = footer
# app.navbar_extra = sidebar

app.hide_menu = True
app.hide_navigation = True

app.add_app("Landing", landing_page, initial_page=True)
app.add_app("个人主页", showMainPage)

app.add_app("数据分析实战", data_analysis)
app.add_app("可视化教程", my_vis_page)
app.add_app("开发教程", devops_tutorial)

app.run()
