import numpy as np
import pandas as pd
import streamlit_echarts as st_echarts
import graphviz
from docs.my_pyecharts_tutorial import pyecharts_tutorial


def showGraph():
    graph = graphviz.Digraph()
    graph.edge('grandfather', 'father')
    graph.edge('grandmother', 'father')
    graph.edge('maternal grandfather', 'mother')
    graph.edge('maternal grandmother', 'mother')
    graph.edge('father', 'brother')
    graph.edge('mother', 'brother')
    graph.edge('father', 'me')
    graph.edge('mother', 'me')
    graph.edge('brother', 'nephew')
    graph.edge('Sister-in-law', 'nephew')
    graph.edge('brother', 'niece')
    graph.edge('Sister-in-law', 'niece')
    graph.edge('me', 'son')
    graph.edge('me', 'daughter')
    graph.edge('where my wife?', 'son')
    graph.edge('where my wife?', 'daughter')
    return graph


def my_vis_page(st, **state):
    valueList = ['pyecharts教程', 'graphviz教程', 'matplotlib教程' ]
    result = st.sidebar.selectbox("可视化汇总", valueList)
    if result == 'graphviz教程':
        graph = showGraph()
        st.write(graph)
    elif result == 'pyecharts教程':
        pyecharts_tutorial(st)
    else:
        pass
