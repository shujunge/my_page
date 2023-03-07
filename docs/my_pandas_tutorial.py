import streamlit_echarts as st_echarts
from streamlit_ace import st_ace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def pandas_plot(st, **state):
    st.code("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

test_dict = {'销售量':[1000,2000,5000,2000,4000,3000],'收藏':[1500,2300,3500,2400,1900,3000]}
df = pd.DataFrame(test_dict,index=['一月','二月','三月','四月','五月','六月'])
)
df.head(3).style.highlight_max(axis=0)
""", language='python')

    test_dict = {'销售量': [1000, 2000, 5000, 2000, 4000, 3000], '收藏': [1500, 2300, 3500, 2400, 1900, 3000]}
    df = pd.DataFrame(test_dict, index=['一月', '二月', '三月', '四月', '五月', '六月'])
    st.dataframe(df.head(3).style.highlight_max(axis=0))

    st.markdown("### 绘制箱型图")
    st.code("""
# 设置绘图的大小
text_kwargs = dict(figsize=(6, 3), fontsize=6,
               color=dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray'),
               sym='r+')  
ax = df.plot(title='box chart', kind='box', vert=False, **text_kwargs)
fig = ax.get_figure()
fig
    """)
    text_kwargs = dict(figsize=(6, 3), fontsize=9,
                       color=dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray'),
                       sym='r+')
    ax = df.plot(title='box chart', kind='box', vert=False, **text_kwargs)
    fig = ax.get_figure()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### 绘制折线图")
    st.code("""
# 设置绘图的大小
text_kwargs = dict(figsize=(6, 3), fontsize =9, color=['y','b','purple','gray', 'g', 'b', ])
ax = df.plot(title='line chart', kind='line', **text_kwargs)
fig = ax.get_figure()
        """)
    text_kwargs = dict(figsize=(6, 3), fontsize=9, color=['gray', 'g', 'b', 'purple'])
    ax = df.plot(title='line chart', kind='line', **text_kwargs)
    fig = ax.get_figure()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### 绘制柱状图")
    st.code("""
# 设置绘图的大小
text_kwargs = dict(figsize=(6, 3), fontsize=9,  color=['y','b','purple','gray', 'g', 'b', ])
ax = df.plot(title='bar chart',kind='bar',rot=45, **text_kwargs)
fig = ax.get_figure()
        """)
    text_kwargs = dict(figsize=(6, 3), fontsize=9, color=['y', 'b', 'purple', 'gray', 'g', 'b', ])
    ax = df.plot(title='bar chart', kind='bar', rot=45, **text_kwargs)
    fig = ax.get_figure()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### 绘制堆叠柱状图")
    st.code("""
# 设置绘图的大小
text_kwargs = dict(figsize=(6, 3), fontsize=9, color=['y','b','purple','gray', 'g', 'b', ])
ax = df.plot(title='bar stacked chart', kind='bar', stacked=True, **text_kwargs)
fig = ax.get_figure()
            """)
    text_kwargs = dict(figsize=(6, 3), fontsize=9, color=['y', 'b', 'purple', 'gray', 'g', 'b', ])
    ax = df.plot(title='bar stacked chart', kind='bar', stacked=True, **text_kwargs)
    fig = ax.get_figure()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### 绘制横向柱状图")
    st.code("""
# 设置绘图的大小
text_kwargs = dict(figsize=(6, 3), fontsize=9, color=['purple', 'gray', 'g', 'b', ])
ax = df.plot(title='barh chart', kind='barh',  **text_kwargs)
fig = ax.get_figure()
                """)
    text_kwargs = dict(figsize=(6, 3), fontsize=9, color=['y', 'b', 'purple', 'gray', 'g', 'b', ])
    ax = df.plot(title='barh chart', kind='barh', **text_kwargs)
    fig = ax.get_figure()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### 绘制直方图")
    st.code("""
# 设置绘图的大小
test_dict2 = {'泊松分布': np.random.poisson(50, 100), '贝塔分布': np.random.beta(5, 1, 100) * 40}
pf = pd.DataFrame(test_dict2)
ax = pf.plot(kind='hist', alpha=0.5, stacked=False,subplots=True, bins=20, sharex=False, layout=(1, 2),figsize=(6,3), fontsize=9,)
fig = ax.all().get_figure()
                    """)
    test_dict2 = {'泊松分布': np.random.poisson(50, 100), '贝塔分布': np.random.beta(5, 1, 100) * 40}
    pf = pd.DataFrame(test_dict2)
    ax = pf.plot(kind='hist', alpha=0.5, subplots=True, stacked=False, bins=20, sharex=False, layout=(1, 2),
                 figsize=(6, 3), fontsize=9, )
    fig = ax.all().get_figure()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### 绘制核密度图")
    st.code("""
# 设置绘图的大小
test_dict2 = {'泊松分布': np.random.poisson(50, 100), '贝塔分布': np.random.beta(5, 1, 100) * 40}
pf = pd.DataFrame(test_dict2)
ax = pf.plot(kind='kde', subplots=True, grid=True,figsize=(6,3), fontsize=9,)
fig = ax.all().get_figure()
                        """)
    test_dict2 = {'泊松分布': np.random.poisson(50, 100), '贝塔分布': np.random.beta(5, 1, 100) * 40}
    pf = pd.DataFrame(test_dict2)
    ax = pf.plot(kind='kde', subplots=True, grid=True, figsize=(6, 3), fontsize=9, )
    fig = ax.all().get_figure()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### 绘制直方图和核密度图")
    st.code("""
import seaborn as sns
import numpy as np
test_dict2 = {'泊松分布': np.random.poisson(50, 100), '贝塔分布': np.random.beta(5, 1, 100) * 40}
pf = pd.DataFrame(test_dict2)
ax = sns.distplot(pf['泊松分布'])
fig = ax.get_figure()
        """)
    test_dict2 = {'泊松分布': np.random.poisson(50, 100), '贝塔分布': np.random.beta(5, 1, 100) * 40}
    pf = pd.DataFrame(test_dict2)
    fig = plt.figure(figsize=(5, 3))
    ax = sns.distplot(pf['泊松分布'], color="y", kde=True, hist=True)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""### 绘制饼状图""")
    st.code("""
    # 设置绘图的大小
test_dict = {'销售量': [1000, 2000, 5000, 2000, 4000, 3000], '收藏': [1500, 2300, 3500, 2400, 1900, 3000]}
pf = pd.DataFrame(test_dict, index=['一月', '二月', '三月', '四月', '五月', '六月'])
ax = pf.plot(kind='pie', subplots=True,figsize=(10, 8),autopct='%.2f%%',radius = 1.2,startangle = 250,legend=False,colormap='viridis')
# figsize # 图片的大小,autopct # 显示百分比, radius # 圆的半径。, startangle # 旋转角度。, colormap # 颜色。
fig = ax.all().get_figure()

                            """)
    test_dict = {'销售量': [1000, 2000, 5000, 2000, 4000, 3000], '收藏': [1500, 2300, 3500, 2400, 1900, 3000]}
    pf = pd.DataFrame(test_dict, index=['一月', '二月', '三月', '四月', '五月', '六月'])

    ax = pf.plot(kind='pie', subplots=True, figsize=(10, 8), autopct='%.2f%%', radius=1.2, startangle=250, legend=False,
                 colormap='viridis')
    fig = ax.all().get_figure()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""### 绘制散点图""")
    st.code("""
# 设置绘图的大小
test_dict = {'销售量': [1000, 2000, 5000, 2000, 4000, 3000], '收藏': [1500, 2300, 3500, 2400, 1900, 3000]}
pf = pd.DataFrame(test_dict, index=['一月', '二月', '三月', '四月', '五月', '六月'])
ax = pf.plot(kind='scatter', x=0, y=1, legend=True, label='Group 1',s=10)
pf.plot(kind='scatter', x=1, y=0,  color='DarkGreen', label='Group 2', ax=ax,s=20)
fig = ax.get_figure()
    """)
    test_dict = {'销售量': [1000, 2000, 5000, 2000, 4000, 3000], '收藏': [1500, 2300, 3500, 2400, 1900, 3000]}
    pf = pd.DataFrame(test_dict, index=['一月', '二月', '三月', '四月', '五月', '六月'])
    ax = pf.plot(kind='scatter', x=0, y=1, legend=True, label='Group 1', s=10)
    pf.plot(kind='scatter', x=1, y=0, color='DarkGreen', label='Group 2', ax=ax, s=20)
    fig = ax.get_figure()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""### 绘制**散点矩阵图**""")
    st.code("""
# 设置绘图的大小
from pandas.plotting import scatter_matrix
df = pd.DataFrame(np.random.randn(1000, 4), columns=['a', 'b', 'c', 'd'])
ax=scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')
fig = ax.all().get_figure()
    """)
    from pandas.plotting import scatter_matrix
    df = pd.DataFrame(np.random.randn(1000, 4), columns=['a', 'b', 'c', 'd'])
    ax = scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')
    fig = ax.all().get_figure()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""### 绘制折线表图""")
    st.code("""
# 设置绘图的大小
from pandas.plotting import scatter_matrix
from pandas.plotting import table

df = pd.DataFrame(np.random.random((1000, 4)), columns=['a', 'b', 'c', 'd'])
fig, ax = plt.subplots(1, 1)
table(ax, np.round(df.describe(), 2), loc='upper right', colWidths=[0.2, 0.2, 0.2, 0.2])
df.plot(ax=ax, ylim=(0, 2), legend=None)
fig = ax.get_figure()
        """)
    from pandas.plotting import scatter_matrix
    from pandas.plotting import table
    df = pd.DataFrame(np.random.random((1000, 4)), columns=['a', 'b', 'c', 'd'])
    fig, ax = plt.subplots(1, 1)
    table(ax, np.round(df.describe(), 2), loc='upper right', colWidths=[0.2, 0.2, 0.2, 0.2])
    df.plot(ax=ax, ylim=(0, 2), legend=None)
    fig = ax.get_figure()
    st.pyplot(fig)
    plt.close(fig)


def pandas_tutorial(st, **state):

    result = st.sidebar.selectbox("教程目录", ['pd.plot应用', 'mangodb技术文档'])
    if result == 'pd.plot应用':
        st.markdown("## pandas之plot使用")
        pandas_plot(st)
    else:
        pass
