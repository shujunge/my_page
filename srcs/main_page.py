import pandas as pd
import pandas_profiling
from streamlit_ace import st_ace
import streamlit_echarts as st_echarts


def showMainPage(st):
    st.write("欢迎使用")
    content = st_ace()

    # Display editor's content as you type
    st.write(content)

    # nlp = spacy.load("en_core_web_sm")
    # doc = nlp("Sundar Pichai is the CEO of Google.")
    # visualize_ner(doc, labels=nlp.get_pipe("ner").labels)

    # df = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")
    # pr = df.profile_report()

    # st_profile_report(pr)

    # project_name = st.sidebar.selectbox(
    #     "请选择项目",
    #     ["项目A", "项目B"]
    # )
    # st.write("search_term:", project_name)
    #
    # with open("output/index.html", "r", encoding="gb2312") as f:
    #     st.write(f.read(), unsafe_allow_html=True)
