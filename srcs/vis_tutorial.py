import numpy as np
import pandas as pd
import streamlit_echarts as st_echarts
import graphviz


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


def visPage(st):
    st.title("权限管理")
    graph = showGraph()
    st.write(graph)
