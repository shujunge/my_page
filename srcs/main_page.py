import pandas as pd
import pandas_profiling
from streamlit_ace import st_ace
import streamlit_echarts as st_echarts
from srcs.multipage import MultiPage

def showMainPage(st, **state):
    st.write("欢迎使用")
    # content = st_ace()
    # # Display editor's content as you type
    #
    # if content:
    #     st.write(content)
    #     MultiPage.save({"content": content})
    # else:
    #     if content in state:
    #         st.write(state["content"])
