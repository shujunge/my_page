import os
import pandas as pd
import numpy as np
import pyecharts.options as opts
from pyecharts.charts import Bar, Line, Grid
from pyecharts.faker import Faker
from pyecharts.commons.utils import JsCode
from pyecharts.charts import Graph

from streamlit_echarts import st_pyecharts


def show_bar():
    import numpy as np
    bar = (
        Bar(init_opts=opts.InitOpts(
            bg_color='rgba(255,255,255,1)',
            width='800px',
            height='600px'
        ))
            .add_xaxis(Faker.days_attrs)
            .add_yaxis("商家A", Faker.days_values, color=Faker.rand_color())
            .set_global_opts(
            title_opts=opts.TitleOpts(title="Bar-DataZoom（slider+inside）"),
            datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")],
        )
    )
    return bar


def show_graph():
    nodes = [
        {"name": "结点1", "symbolSize": 10},
        {"name": "结点2", "symbolSize": 20},
        {"name": "结点3", "symbolSize": 30},
        {"name": "结点4", "symbolSize": 40},
        {"name": "结点5", "symbolSize": 50},
        {"name": "结点6", "symbolSize": 40},
        {"name": "结点7", "symbolSize": 30},
        {"name": "结点8", "symbolSize": 20},
    ]
    links = []
    for i in nodes:
        for j in nodes:
            links.append({"source": i.get("name"), "target": j.get("name")})
    graph = (
        Graph(init_opts=opts.InitOpts(
            bg_color='rgba(255,255,255,1)',
            width='800px',
            height='600px'
        ))
            .add("", nodes, links, repulsion=8000)
            .set_global_opts(title_opts=opts.TitleOpts(title="Graph-基本示例"))
    )
    return graph


def pyecharts_tutorial(st, **state):
    st.markdown("""
    ### 安装pyecharts包
    * pip install pyecharts
    * pip install pyecharts-jupyter-installer
    
    地图相关的包
    
    * pip install echarts-countries-pypkg  
    * pip install echarts-china-provinces-pypkg 
    * pip install echarts-china-cities-pypkg 
    * pip install echarts-china-counties-pypkg
    * pip install echarts-china-misc-pypkg 
    * pip install echarts-united-kingdom-pypkg
    """)
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['a', 'b', 'c'])

    st.line_chart(chart_data)

    bar = show_bar()
    st_pyecharts(bar)
    graph = show_graph()
    st_pyecharts(graph)
