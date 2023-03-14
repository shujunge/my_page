import streamlit_echarts as st_echarts
from streamlit_ace import st_ace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
import seaborn as sns
from scipy import stats
import streamlit.components.v1 as components  # 将要展示的 弄成html

fontPath = 'ttf/SimHei.ttf'
font_manager.fontManager.addfont(fontPath)
prop = font_manager.FontProperties(fname=fontPath)
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = prop.get_name()
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'


# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def pandas_plot(st, **state):
    st.code("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
import seaborn as sns

# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
fontPath = 'ttf/SimHei.ttf'
font_manager.fontManager.addfont(fontPath)
prop= font_manager.FontProperties(fname=fontPath)
mpl.rcParams['font.family']='sans-serif'
mpl.rcParams['font.sans-serif']=prop.get_name()
mpl.rcParams['font.size']=12
mpl.rcParams['axes.unicode_minus']=False
mpl.rcParams['xtick.direction']='in'
mpl.rcParams['ytick.direction']='in'


test_dict = {'销售量':[1000,2000,5000,2000,4000,3000],'收藏':[1500,2300,3500,2400,1900,3000]}
df = pd.DataFrame(test_dict,index=['一月','二月','三月','四月','五月','六月'])
)
df.head(3).style.highlight_max(axis=0)
""", language='python')

    test_dict = {'销售量': [1000, 2000, 5000, 2000, 4000, 3000], '收藏量': [1500, 2300, 3500, 2400, 1900, 3000]}
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

    st.markdown("### 绘制柱状图带数据标签")
    st.code("""
    # 设置绘图的大小
text_kwargs = dict(figsize=(8, 4), fontsize=9, color=['y', 'b', 'purple', 'gray', 'g', 'b', ])
ax = df['销售量'].plot(title='bar chart', kind='bar', rot=45, **text_kwargs)
m_ax = ax.bar(df.index.tolist(), df['销售量'].tolist())
ax.bar_label(m_ax)
fig = ax.get_figure()
st.pyplot(fig)
            """)
    text_kwargs = dict(figsize=(8, 4), fontsize=9, color=['y', 'b', 'purple', 'gray', 'g', 'b', ])
    ax = df['销售量'].plot(title='bar chart', kind='bar', rot=45, **text_kwargs)
    m_ax = ax.bar(df.index.tolist(), df['销售量'].tolist())
    ax.bar_label(m_ax)
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


def pandas_style(st, **state):
    np.random.seed(24)
    df = pd.DataFrame({'A': np.linspace(1, 10, 10)})
    df = pd.concat([df, pd.DataFrame(np.random.randn(10, 4), columns=list('BCDE'))],
                   axis=1)
    df.iloc[0, 2] = np.nan

    def color_negative_red(val):
        color = 'red' if val < 0 else 'black'
        return 'color: %s' % color

    st.code("""
import pandas as pd
import numpy as np
np.random.seed(24)
df = pd.DataFrame({'A': np.linspace(1, 10, 10)})
df = pd.concat([df, pd.DataFrame(np.random.randn(10, 4), columns=list('BCDE'))],
               axis=1)
df.iloc[0, 2] = np.nan

def color_negative_red(val):
    color = 'red' if val < 0 else 'black'
    return 'color: %s' % color
s = df.style.applymap(color_negative_red)
s.render()
""")
    s = df.style.applymap(color_negative_red)
    components.html(s.render())

    st.markdown("## 格式化输出style")
    st.code("""
df.style.format("{:.2%}")
    """)
    components.html(df.style.format("{:.2%}").render())

    st.markdown("## 格式化输出 bar style")
    st.code("""
import seaborn as sns
df.style.bar(subset=['A', 'B'], color='#d65f5f')
zz = df.style.bar(subset=['A', 'B'], align='mid', color=['#d65f5f', '#5fba7d'])
zz.render()
""")
    zz = df.style.bar(subset=['A', 'B'], align='mid', color=['#d65f5f', '#5fba7d'])
    components.html(zz.render())

    st.markdown("## 梯度展示样式")
    st.code("""
    import seaborn as sns
    cm = sns.light_palette("green", as_cmap=True)
    df.style.set_caption('Colormaps, with a caption.').background_gradient(cmap=cm)
    """)
    cm = sns.light_palette("green", as_cmap=True)
    results = df.style.set_caption('Colormaps, with a caption.').background_gradient(cmap=cm)
    components.html(results.render())


def pandas_buildin(st, **state):
    st.sidebar.radio("本章节的内容",
                     ('groupby函数用法', 'groupby nth代码', '将多层次的列转换为一列', 'agg函数重命名', 'agg函数使用其他函数', '排序函数',
                      'Multiindex/Multicolumn使用', 'explode函数使用将一行转换为多行', 'pipe函数用法', ' memory_usage使用'
                      , 'stack和unstack用法', 'query函数使用', 'eval函数使用', 'assign函数使用', 'qcut函数使用', '写入excel多个sheet页'))
    st.code("""
df = pd.DataFrame({'X': ['A', 'B', 'A', 'B'], 'Y': [1, 2, 1, 2], 'Z': [4, 4, 1, 2]})
""")
    df = pd.DataFrame({'X': ['A', 'B', 'A', 'B'], 'Y': [1, 2, 1, 2], 'Z': [4, 4, 1, 2]})
    st.dataframe(df)

    st.markdown("## groupby函数用法")
    st.code("""
zz = df.groupby(by='X').agg({'Y':['count','mean','std'],'Z':['sum']}).reset_index()
    """)
    zz = df.groupby(by='X').agg({'Y': ['count', 'mean', 'std'], 'Z': ['sum']}).reset_index()
    st.dataframe(zz)

    st.markdown("## groupby nth代码")
    st.code("""
zz = df.sort_values(by="Z", ascending=False).groupby(by="X", as_index=False).nth(1)
# GroupBy.nth()，取每一组第n行的数据，n从0开始，0代表第一行。
        """)
    zz = df.sort_values(by="Z", ascending=False).groupby(by="X", as_index=False).nth(1)
    st.dataframe(zz)

    st.markdown("## 将多层次的列转换为一列")
    st.code("""
    zz.columns= [  "_".join(x) for x in zz.columns.view()]
    zz
    """)
    zz.columns = ["_".join(x) for x in zz.columns.view()]
    st.dataframe(zz)

    st.markdown("## agg函数重命名")
    st.code("""
zz = df.groupby(by='X').agg(y_count = ('Y', 'count'), z_sum = ('Z', 'sum') 
,y_a_sum=('Z', lambda x: x[df.iloc[x.index].X == 'A'].sum())).reset_index()
        """)
    zz = df.groupby(by='X').agg(y_count=('Y', 'count'), z_sum=('Z', 'sum') \
                                , y_a_sum=('Z', lambda x: x[df.iloc[x.index].X == 'A'].sum())).reset_index()
    st.dataframe(zz)

    st.markdown("## agg函数使用其他函数")
    st.code("""
from scipy import stats
zz = df.groupby(by='Z').agg(x_concat=('X', ".".join),x_unique=('X',pd.Series.unique)
,y_mode=('Y',lambda x: stats.mod(x)[0][0])).reset_index()
            """)
    zz = df.groupby(by='Z').agg(x_concat=('X', ".".join), x_unique=('X', pd.Series.unique),
                                y_mod=('Y', lambda x: stats.mode(x)[0][0])).reset_index()
    st.dataframe(zz)

    st.markdown("## 排序函数")
    st.code("""
zz = df.sort_values('X',ascending=False).reset_index(drop=True)
            """)
    zz = df.sort_values('X', ascending=False).reset_index(drop=True)
    st.dataframe(zz)

    st.markdown("##  Multiindex/Multicolumn使用")
    st.code("""
arrays = [['bar', 'bar', 'baz', 'baz', 'foo', 'foo', ],
          ['one', 'two', 'one', 'two', 'one', 'two', ]]
tuples = list(zip(*arrays))
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
zz = pd.DataFrame(np.random.randn(6, 6), columns=arrays, index=arrays)
""")
    arrays = [['bar', 'bar', 'baz', 'baz', 'foo', 'foo', ],
              ['one', 'two', 'one', 'two', 'one', 'two', ]]
    tuples = list(zip(*arrays))
    index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
    zz = pd.DataFrame(np.random.randn(6, 6), columns=arrays, index=arrays)
    st.dataframe(zz)

    st.code("""
    zz.loc[:, [('bar', 'one'), ('baz', 'two')]],df.loc[:, 'bar']
    """)
    st.dataframe(zz.loc[:, [('bar', 'one'), ('baz', 'two')]])
    st.dataframe(zz.loc[:, 'bar'])

    st.markdown("## explode函数使用将一行转换为多行")
    st.code("""
    id = ['a', 'b', 'c']
    measurement = [4, 6, [2, 3, 8]]
    day = [1, 1, 1]
    zz = pd.DataFrame({'id': id, 'measurement': measurement, 'day': day})
    zz = zz.explode('measurement').reset_index(drop=True)
    """)
    id = ['a', 'b', 'c']
    measurement = [4, 6, [2, 3, 8]]
    day = [1, 1, 1]
    zz = pd.DataFrame({'id': id, 'measurement': measurement, 'day': day})
    zz = zz.explode('measurement').reset_index(drop=True)
    st.dataframe(zz)

    st.markdown("## explode函数使用将一行转换为多列")
    st.code(r"""
data = {"a": ["test1", "test2"], "b": ["up", "up"],
     "c": ["eth1|up|16G|32G", "eth0|up|32G|32G"]}
zz = pd.DataFrame(data)
zz = zz['c'].str.split('|',expand=True)

id = ['a', 'b', 'c']
measurement = [4, 6, [2, 3, 8]]
day = [1, 1, 1]
zz = pd.DataFrame({'id': id, 'measurement': measurement, 'day': day})
zz = zz['measurement'].apply(pd.Series)
    
    """)

    data = {"a": ["test1", "test2"], "b": ["up", "up"],
            "c": ["eth1|up|16G|32G", "eth0|up|32G|32G"]}
    zz = pd.DataFrame(data)
    zz = zz['c'].str.split('|', expand=True)
    st.dataframe(zz)
    id = ['a', 'b', 'c']
    measurement = [4, 6, [2, 3, 8]]
    day = [1, 1, 1]
    zz = pd.DataFrame({'id': id, 'measurement': measurement, 'day': day})
    zz = zz['measurement'].apply(pd.Series)
    st.dataframe(zz)

    df = pd.DataFrame({
        "id": [100, 100, 101, 102, 103, 104, 105, 106],
        "A": [1, 2, 3, 4, 5, 2, np.nan, 5],
        "B": [45, 56, 48, 47, 62, 112, 54, 49],
        "C": [1.2, 1.4, 1.1, 1.8, np.nan, 1.4, 1.6, 1.5]
    })

    def fill_missing_values(df):
        for col in df.select_dtypes(include=["int", "float"]).columns:
            val = df[col].mean()
            df[col].fillna(val, inplace=True)
        return df

    def drop_duplicates(df, column_name):
        df = df.drop_duplicates(subset=column_name)
        return df

    def remove_outliers(df, column_list):
        for col in column_list:
            avg = df[col].mean()
            std = df[col].std()
            low = avg - 2 * std
            high = avg + 2 * std
            df = df[df[col].between(low, high, inclusive=True)]
        return df

    df_processed = (
        df.pipe(fill_missing_values)
            .pipe(drop_duplicates, "id")
            .pipe(remove_outliers, ["A", "B"])
    )

    st.markdown("## pipe函数用法")
    st.code("""
df = pd.DataFrame({
    "id": [100, 100, 101, 102, 103, 104, 105, 106],
    "A": [1, 2, 3, 4, 5, 2, np.nan, 5],
    "B": [45, 56, 48, 47, 62, 112, 54, 49],
    "C": [1.2, 1.4, 1.1, 1.8, np.nan, 1.4, 1.6, 1.5]
})
def fill_missing_values(df):
    for col in df.select_dtypes(include=["int", "float"]).columns:
        val = df[col].mean()
        df[col].fillna(val, inplace=True)
    return df

def drop_duplicates(df, column_name):
    df = df.drop_duplicates(subset=column_name)
    return df

def remove_outliers(df, column_list):
    for col in column_list:
        avg = df[col].mean()
        std = df[col].std()
        low = avg - 2 * std
        high = avg + 2 * std
        df = df[df[col].between(low, high, inclusive=True)]
    return df

df_processed = (
    df.pipe(fill_missing_values)
        .pipe(drop_duplicates, "id")
        .pipe(remove_outliers, ["A", "B"])
)
    """)
    st.dataframe(df_processed)

    st.markdown("## memory_usage使用")
    st.code("""
df_large = pd.DataFrame({'A': np.random.randn(100),
                 'B': np.random.randint(100, size=100)})
df_large.memory_usage()

df_large.info(memory_usage='deep')
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 10000 entries, 0 to 9999
# Data columns (total 2 columns):
# #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
# 0   A       10000 non-null  float64
# 1   B       10000 non-null  int64
# dtypes: float64(1), int64(1)
# memory usage: 156.4 KB
    """)
    df_large = pd.DataFrame({'A': np.random.randn(10000),
                             'B': np.random.randint(100, size=10000)})
    st.dataframe(df_large.memory_usage())

    st.markdown("## stack和unstack用法")
    st.code(r"""
data = pd.DataFrame(np.arange(12).reshape((3, 4)) + 100,
        index=pd.Index(['date1', 'date2', 'date3']),
        columns=pd.Index(['store1', 'store2', 'store3', 'store4'])
        )

# store1  store2  store3  store4
# date1     100     101     102     103
# date2     104     105     106     107
# date3     108     109     110     111

stack_pf =data.stack().reset_index()
print("stack结果: \n", stack_pf)
unstack_pf = stack_pf.set_index(['level_0','level_1'])[0].unstack().reset_index()
print("unstack结果: \n", unstack_pf)
    """)
    data = pd.DataFrame(np.arange(12).reshape((3, 4)) + 100,
                        index=pd.Index(['date1', 'date2', 'date3']),
                        columns=pd.Index(['store1', 'store2', 'store3', 'store4'])
                        )

    st.markdown("### stack结果:")
    stack_pf = data.stack().reset_index()
    st.dataframe(stack_pf)
    st.markdown("### unstack结果:")
    unstack_pf = stack_pf.set_index(['level_0', 'level_1'])[0].unstack().reset_index()
    st.dataframe(unstack_pf)

    st.markdown("## query函数使用")
    st.code("""
    unstack_pf.query('store2>104').query('c=store2*store3')
    """)
    st.dataframe(unstack_pf.query('store2 > 104 and store3 >109'))

    st.markdown("## eval函数使用")
    st.code("""
    unstack_pf.query('store2 > 104 and store3 >109').eval('c=store2*store3')
    """)
    st.dataframe(unstack_pf.query('store2 > 104 and store3 >109').eval('c=store2*store3'))

    st.markdown("## assign函数使用")
    st.code("""
    unstack_pf.assign(c=unstack_pf.store2 + unstack_pf.store3)
    """)
    st.dataframe(unstack_pf.assign(c=unstack_pf.store2 + unstack_pf.store3))

    st.markdown("## qcut分箱函数")
    st.code("""
draws=np.random.randn(1000)
bins= pd.qcut(draws,4,labels=['Q1','Q2','Q3','Q4'])
bins= pd.Series(bins, name='quartile')
results= (pd.Series(draws).groupby(bins).agg(['count','min','max']).reset_index())
       """)
    draws = np.random.randn(1000)
    bins = pd.qcut(draws, 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    bins = pd.Series(bins, name='quartile')
    results = (pd.Series(draws).groupby(bins).agg(['count', 'min', 'max']).reset_index())
    st.dataframe(results)

    st.markdown("### 写入excel多个sheet页")
    st.code("""
import pandas as pd
from datetime import datetime

df1 = pd.DataFrame({"日期": [datetime(2020,1,1), datetime(2020,1,2)],"销量":[10,20]})
df2 = pd.DataFrame({"日期": [datetime(2020,2,1), datetime(2020,2,2)],"销量":[30,40]})

with pd.ExcelWriter("111.xlsx", datetime_format="YYYY-MM-DD") as writer:
    df1.to_excel(writer, sheet_name="1月")
    df2.to_excel(writer, sheet_name="2月")
""")


def pandas_check(st, **state):
    st.sidebar.radio("本章节的内容",
                     ('join操作', 'rank操作', 'Series操作', '逻辑运算&&类型转换', 'sort_values排序相关函数', '替换相关函数', '函数过滤方法', '数据迭代'))

    st.markdown("## join操作")
    st.code(r"""
    
df1 = pd.DataFrame([[1, 2], [3, 4]],columns=list('AB'))
df2 = pd.DataFrame([[3, 6], [7, 8]],columns=list('AB'))


# join 操作
pd.merge(df1, df2, on="A") # inner join
pd.merge(df1, df2, on="A", how="outer") # full join 
pd.merge(df1, df2, on="A", how="left") # left join
pd.merge(df1, df2, on="A", how="join") # join join

pd.merge(df1, df2, left_on="name1", right_on="name2") df1.name1==df2.name2
# 	name1	B_x	name2	B_y
# 0	3	4	3	6

pd.merge(df1, df2, left_on="name1", right_on="name2", suffixes=("_left", "_right"))
# 	name1	B_left	name2	B_right
# 0	3	4	3	6

# union all操作(append, concat)
df1.append(df2,ignore_index=True)

# 等价于
result= pd.concat([df1, df2], ignore_index=True)
        """)
    df1 = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB'))
    df2 = pd.DataFrame([[5, 6], [7, 8]], columns=list('AB'))
    st.dataframe(df1.append(df2, ignore_index=True))

    st.markdown(" rank操作")
    st.code(r"""

# row_number() 窗口函数
data_1 = data[data['班级']=='1班']
data_1['成绩_first'] = data_1['成绩'].rank(method='first',ascending=False)
data_1
   班级   姓名  成绩  成绩_first
0  1班  〇韩愈  50       1.0
1  1班  柳宗元  30       2.0
2  1班  欧阳修  30       3.0
3  1班  〇苏洵  20       4.0
4  1班  〇苏轼  10       5.0



# dense() 窗口函数
data_1 = data[data['班级']=='1班']
data_1['成绩_dense'] = data_1['成绩'].rank(method='dense',ascending=False)
data_1
   班级   姓名  成绩  成绩_dense
0  1班  〇韩愈  50       1.0
1  1班  柳宗元  30       2.0
2  1班  欧阳修  30       2.0
3  1班  〇苏洵  20       3.0
4  1班  〇苏轼  10       4.0


# group 分组 rank函数
data['成绩_dense']= data.groupby('班级')['成绩'].rank(method='dense')
data
班级   姓名  成绩  成绩_dense
0  1班  〇韩愈  50       4.0
1  1班  柳宗元  30       3.0
2  1班  欧阳修  30       3.0
3  1班  〇苏洵  20       2.0
4  1班  〇苏轼  10       1.0
5  2班  〇苏辙  60       3.0
6  2班  〇曾巩  60       3.0
7  2班  王安石  50       2.0
8  2班  〇张三  50       2.0
9  2班  小伍哥  40       1.0
""")

    st.markdown("## Series操作")
    st.code(r"""
 s = pd.Series([0, 1, 2])
# Series转为DataFrame，name参数用于指定转换后的字段名
s = s.to_frame(name='列名')

# 只有单列数据的DataFrame转为Series
s.squeeze() # 

""")

    st.markdown("## 逻辑运算&&类型转换")
    st.code(r"""
    
# Q1成绩大于36
df.Q1> 36
# Q1成绩不小于60分，并且是C组成员
~(df.Q1< 60) & (df['team'] == 'C')

# 表达式与切片一致
df.loc[df['Q1']> 90, 'Q1':] # Q1大于90，只显示Q1
df.loc[(df.Q1> 80) & (df.Q2 < 15)] # and关系
df.loc[(df.Q1> 90) | (df.Q2 < 90)] # or关系
df.loc[df['Q1']== 8] # 等于8
df.loc[df.Q1== 8] # 等于8
df.loc[df['Q1']> 90, 'Q1':] # Q1大于90，显示Q1及其后所有列


# 查询最大索引的值
df.Q1[lambda s: max(s.index)] # 值为21
# 计算最大值
max(df.Q1.index)
# 99
df.Q1[df.index==99]


# 以下相当于 df[df.Q1 == 60]
df[df.Q1.eq(60)]
df.ne() # 不等于 !=
df.le() # 小于等于 <=
df.lt() # 小于 <
df.ge() # 大于等于 >=
df.gt() # 大于 >


 # 直接写类型SQL where语句
df.query('Q1 > Q2 > 90')


df.filter(items=['Q1', 'Q2']) # 选择两列
df.filter(regex='Q', axis=1) # 列名包含Q的列
df.filter(regex='e$', axis=1) # 以e结尾的列
df.filter(regex='1$', axis=0) # 正则，索引名以1结尾
df.filter(like='2', axis=0) # 索引中有2的
# 索引中以2开头、列名有Q的
df.filter(regex='^2',axis=0).filter(like='Q', axis=1)

df.select_dtypes(include=['float64']) # 选择float64型数据
df.select_dtypes(include='bool')
df.select_dtypes(include=['number']) # 只取数字型
df.select_dtypes(exclude=['int']) # 排除int类型
df.select_dtypes(exclude=['datetime64'])


# 对所有字段指定统一类型
df = pd.DataFrame(data, dtype='float32')
# 对每个字段分别指定
df = pd.read_excel(data, dtype={'team':'string', 'Q1': 'int32'})

# 自动转换合适的数据类型
df.infer_objects() # 推断后的DataFrame
df.infer_objects().dtypes

# 按大体类型推定
m = ['1', 2, 3]
s = pd.to_numeric(s) # 转成数字
pd.to_datetime(m) # 转成时间
pd.to_timedelta(m) # 转成时间差
pd.to_datetime(m, errors='coerce') # 错误处理
pd.to_numeric(m, errors='ignore')
pd.to_numeric(m errors='coerce').fillna(0) # 兜底填充
pd.to_datetime(df[['year', 'month', 'day']])
# 组合成日期

    """)

    st.markdown("## sort_values排序相关函数")
    st.code(r"""
   s.sort_index() # 升序排列
df.sort_index() # df也是按索引进行排序
df.team.sort_index()s.sort_index(ascending=False)# 降序排列
s.sort_index(inplace=True) # 排序后生效，改变原数据
# 索引重新0-(n-1)排，很有用，可以得到它的排序号
s.sort_index(ignore_index=True)
s.sort_index(na_position='first') # 空值在前，另'last'表示空值在后
s.sort_index(level=1) # 如果多层，排一级
s.sort_index(level=1, sort_remaining=False) #这层不排
# 行索引排序，表头排序
df.sort_index(axis=1) # 会把列按列名顺序排列

df.Q1.sort_values()
df.sort_values('Q4')
df.sort_values(by=['team', 'name'],ascending=[True, False])


## 按值大小排序nsmallest()和nlargest()
s.nsmallest(3) # 最小的3个
s.nlargest(3) # 最大的3个
# 指定列
df.nlargest(3, 'Q1')
df.nlargest(5, ['Q1', 'Q2'])
df.nsmallest(5, ['Q1', 'Q2'])
""")

    st.markdown("## 替换相关函数")
    st.code(r"""
df.iloc[0,0] # 查询值
# 'Liver'
df.iloc[0,0] = 'Lily' # 修改值
df.iloc[0,0] # 查看结果
# 'Lily'

# 将小于60分的成绩修改为60
df[df.Q1 < 60] = 60
# 查看
df.Q1

s.replace(0, 5) # 将列数据中的0换为5
df.replace(0, 5) # 将数据中的所有0换为5
df.replace([0, 1, 2, 3], 4) # 将0～3全换成4
df.replace([0, 1, 2, 3], [4, 3, 2, 1]) # 对应修改
s.replace([1, 2], method='bfill') # 向下填充
df.replace({0: 10, 1: 100}) # 字典对应修改
df.replace({'Q1': 0, 'Q2': 5}, 100) # 将指定字段的指定值修改为100
df.replace({'Q1': {0: 100, 4: 400}}) # 将指定列里的指定值替换为另一个指定的值


## 填充空值
df.fillna(0) # 将空值全修改为0
# {'backfill', 'bfill', 'pad', 'ffill',None}, 默认为None
df.fillna(method='ffill') # 将空值都修改为其前一个值
values = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
df.fillna(value=values) # 为各列填充不同的值
df.fillna(value=values, limit=1) # 只替换第一个


## 修改索引名
df.rename(columns={"Q1":"a", "Q2": "b"}) # 对表头进行修改
df.rename(index={0: "x", 1:"y", 2: "z"}) # 对索引进行修改
df.rename(index=str) # 对类型进行修改
df.rename(str.lower, axis='columns') # 传索引类型
df.rename({1: 2, 2: 4}, axis='index')

# 对索引名进行修改
s.rename_axis("animal")
df.rename_axis("animal") # 默认是列索引
df.rename_axis("limbs",axis="columns") # 指定行索引

# 索引为多层索引时可以将type修改为class
df.rename_axis(index={'type': 'class'})

# 可以用set_axis进行设置修改
s.set_axis(['a', 'b', 'c'], axis=0)
df.set_axis(['I', 'II'], axis='columns')
df.set_axis(['i', 'ii'], axis='columns',inplace=True)


# 插入列df.insert()
# 在第三列的位置上插入新列total列，值为每行的总成绩
df.insert(2, 'total', df.sum(1))


# 指定列df.assign()
# 增加total列
df.assign(total=df.sum(1))
# 增加两列
df.assign(total=df.sum(1), Q=100)
df.assign(total=df.sum(1)).assign(Q=100)
其他使用示例：
df.assign(Q5=[100]*100) # 新增加一列Q5
df = df.assign(Q5=[100]*100) # 赋值生效
df.assign(Q6=df.Q2/df.Q1) # 计算并增加Q6
df.assign(Q7=lambda d: d.Q1 * 9 / 5 + 32) # 使用lambda# 添加一列，值为表达式结果：True或False
df.assign(tag=df.Q1>df.Q2)
# 比较计算，True为1，False为0
df.assign(tag=(df.Q1>df.Q2).astype(int))
# 映射文案
df.assign(tag=(df.Q1>60).map({True:'及格',False:'不及格'}))
# 增加多个
df.assign(Q8=lambda d: d.Q1*5,
          Q9=lambda d: d.Q8+1) # Q8没有生效，不能直接用df.Q8


## 增加行
# 新增索引为100的数据
df.loc[100] = ['tom', 'A', 88, 88, 88, 88]

## 追加合并
df = pd.DataFrame([[1, 2], [3, 4]],columns=list('AB'))
df2 = pd.DataFrame([[5, 6], [7, 8]],columns=list('AB'))
df.append(df2)

## 删除空值
df.dropna() # 一行中有一个缺失值就删除
df.dropna(axis='columns') # 只保留全有值的列
df.dropna(how='all') # 行或列全没值才删除
df.dropna(thresh=2) # 至少有两个空值时才删除
df.dropna(inplace=True) # 删除并使替换生效
""")

    st.markdown("## 函数过滤方法")
    st.code(r"""
    

# 数值大于70
df.where(df > 70)


# 小于60分为不及格
np.where(df>=60, '合格', '不合格')

# 符合条件的为NaN
df.mask(s > 80)


# 行列相同数量，返回一个array
df.lookup([1,3,4], ['Q1','Q2','Q3']) # array([36, 96, 61])
df.lookup([1], ['Q1']) # array([36])
""")

    st.markdown("## 数据迭代")
    st.code(r"""
## 迭代Series
# 迭代指定的列
for i in df.name:
      print(i)

# 迭代索引和指定的两列
for i,n,q in zip(df.index, df.name,df.Q1):
    print(i, n, q)
    

# 迭代，使用name、Q1数据
for index, row in df.iterrows():
    print(index, row['name'], row.Q1)


for row in df.itertuples():
    print(row)


# Series取前三个
for label, ser in df.items():
    print(label)
    print(ser[:3], end='\n\n')
""")


def pandas_str(st, **state):
    st.markdown("# 相关str函数")
    st.code(r"""
index = pd.Index(data=["Tom", "Bob", "Mary", "James", "Andy", "Alice"], name="name")
data = {
    "age": [18, 30, np.nan, 40, np.nan, 30],
    "city": ["Bei Jing ", "Shang Hai ", "Guang Zhou", "Shen Zhen", np.nan, " "],
    "sex": [None, "male", "female", "male", np.nan, "unknown"],
    "birth": ["2000-02-10", "1988-10-17", None, "1978-08-08", np.nan, "1988-10-17"]
}
user_info = pd.DataFrame(data=data, index=index)
user_info["birth"] = pd.to_datetime(user_info.birth)
user_info = user_info.reset_index()
	name	age	city	sex	birth
0	Tom	18.0	Bei Jing	None	2000-02-10
1	Bob	30.0	Shang Hai	male	1988-10-17
2	Mary	NaN	Guang Zhou	female	NaT
3	James	40.0	Shen Zhen	male	1978-08-08
4	Andy	NaN	NaN	NaN	NaT
5	Alice	30.0		unknown	1988-10-17

# 将文本转为小写
user_info.city=user_info.city.str.lower()

# 计算文本长度
user_info.city.str.len()

# 拼接列
user_info.city.str.replace(" ", '_').str.cat(user_info.sex)

# 替换文本符号
user_info.city.str.replace(" ", "_")
user_info.city.str.replace("^S.*", " ")
user_info.city.str.split(" ")

# 分隔成多列
pf=user_info.city.str.split(" ",expand=True) # 返回 DataFrame

# 提取第一个匹配的子串
user_info.city.str.extract("(\w+)\s+", expand=True) # 返回 DataFrame
ser_info.city.str.extract("(\w+)\s+(\w+)", expand=True)  # 返回 DataFrame

# 是否包含子串
user_info.city.str.contains("Zh")

# 生成哑变量
user_info.city.str.get_dummies(sep=" ")

# 常用函数汇总
cat()	连接字符串
split()	在分隔符上分割字符串
rsplit()	从字符串末尾开始分隔字符串
get()	索引到每个元素（检索第i个元素）
join()	使用分隔符在系列的每个元素中加入字符串
get_dummies()	在分隔符上分割字符串，返回虚拟变量的DataFrame
contains()	如果每个字符串都包含pattern / regex，则返回布尔数组
replace()	用其他字符串替换pattern / regex的出现
repeat()	重复值（s.str.repeat(3)等同于x * 3 t2 >）
pad()	将空格添加到字符串的左侧，右侧或两侧
center()	相当于str.center
ljust()	相当于str.ljust
rjust()	相当于str.rjust
zfill()	等同于str.zfill
wrap()	将长长的字符串拆分为长度小于给定宽度的行
slice()	切分Series中的每个字符串
slice_replace()	用传递的值替换每个字符串中的切片
count()	计数模式的发生
startswith()	相当于每个元素的str.startswith(pat)
endswith()	相当于每个元素的str.endswith(pat)
findall()	计算每个字符串的所有模式/正则表达式的列表
match()	在每个元素上调用re.match，返回匹配的组作为列表
extract()	在每个元素上调用re.search，为每个元素返回一行DataFrame，为每个正则表达式捕获组返回一列
extractall()	在每个元素上调用re.findall，为每个匹配返回一行DataFrame，为每个正则表达式捕获组返回一列
len()	计算字符串长度
strip()	相当于str.strip
rstrip()	相当于str.rstrip
lstrip()	相当于str.lstrip
partition()	等同于str.partition
rpartition()	等同于str.rpartition
lower()	相当于str.lower
upper()	相当于str.upper
find()	相当于str.find
rfind()	相当于str.rfind
index()	相当于str.index
rindex()	相当于str.rindex
capitalize()	相当于str.capitalize
swapcase()	相当于str.swapcase
normalize()	返回Unicode标准格式。相当于unicodedata.normalize
translate()	等同于str.translate
isalnum()	等同于str.isalnum
isalpha()	等同于str.isalpha
isdigit()	相当于str.isdigit
isspace()	等同于str.isspace
islower()	相当于str.islower
isupper()	相当于str.isupper
istitle()	相当于str.istitle
isnumeric()	相当于str.isnumeric
isdecimal()	相当于str.isdecimal
""")


def pandas_windows(st, **state):
    st.sidebar.radio("本章节的内容", ('统计函数', '移动函数rolling', 'expanding扩展窗口函数', 'ewm指数加权移动函数'))

    st.markdown("# 统计函数")
    st.code(r"""
index = pd.Index(data=["Tom", "Bob", "Mary", "James", "Andy", "Alice"], name="name")
data = {"age": [18, 40, 28, 20, 30, 35],"income": [1000, 4500 , 1800, 1800, 3000, np.nan],}
df = pd.DataFrame(data=data, index=index)

# 计算年龄与收入之间的协方差
df.age.cov(df.income)

# 计算年龄与收入之间的相关性
df.age.corr(df.income, method="kendall")
df.age.corr(df.income, method="spearman")

""")

    st.markdown("## 移动函数rolling")
    st.code(r"""
data = {
    "turnover": [12000, 18000, np.nan, 12000, 9000, 16000, 18000],
    "date": pd.date_range("2018-07-01", periods=7)
}
df2 = pd.DataFrame(data=data)

# rolling 我们可以实现，设置 window=2 来保证窗口长度为 2，设置 on="date" 来保证根据日期这一列来滑动窗口
df2.rolling(window=2, on="date", min_periods=1).sum()


count()	非空观测值数量
sum()	值的总和
mean()	价值的平均值
median()	值的算术中值
min()	最小值
max()	最大
std()	贝塞尔修正样本标准差
var()	无偏方差
skew()	样品偏斜度（三阶矩）
kurt()	样品峰度（四阶矩）
quantile()	样本分位数（百分位上的值）
apply()	通用适用
cov()	无偏协方差（二元）
corr()	相关（二进制）

## 统计窗口内的和和均值
df2.rolling(window=2, min_periods=1)["turnover"].agg({"tur_sum": np.sum, "tur_mean": np.mean}).reset_index()
""")
    data = {
        "turnover": [12000, 18000, np.nan, 12000, 9000, 16000, 18000],
        "date": pd.date_range("2018-07-01", periods=7)
    }
    df2 = pd.DataFrame(data=data)
    st.markdown(r"""st.dataframe(df2.rolling(window=2, min_periods=1)["turnover"]): """)
    st.dataframe(df2.rolling(window=2, min_periods=1)["turnover"])
    st.markdown(
        r"""df2.rolling(window=2, min_periods=1)["turnover"].agg({"tur_sum": np.sum, "tur_mean": np.mean}).reset_index(): """)
    st.dataframe(
        df2.rolling(window=2, min_periods=1)["turnover"].agg({"tur_sum": np.sum, "tur_mean": np.mean}).reset_index())

    st.markdown("## expanding扩展窗口函数")
    st.code(r"""
import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.randn(10, 4),
      index = pd.date_range('1/1/2018', periods=10),
      columns = ['A', 'B', 'C', 'D'])
#              	A	        B	        C	        D
# 2018-01-01	-0.540429	0.601402	0.209709	-0.029808
# 2018-01-02	-1.298410	1.084708	-2.401062	0.599314
# 2018-01-03	1.870927	0.543951	0.612595	0.412484
# 2018-01-04	1.118217	0.413411	-1.010912	-0.392598
# 2018-01-05	0.839430	-1.370726	1.666621	-2.033319
# 2018-01-06	-0.323914	-0.433943	-0.415155	-1.357330
# 2018-01-07	-0.550545	1.225102	-1.019251	-0.638103
# 2018-01-08	-0.305286	-1.496738	1.570793	0.925841
# 2018-01-09	0.512227	-0.760423	1.396245	-0.891062
# 2018-01-10	-1.921195	0.211140	-0.506363	1.886672


print (df.expanding(min_periods=3).mean())

设置 min_periods=3，表示至少 3 个数求一次均值，计算方式为 (index0+index1+index2)/3，而 index3 的计算方式是 (index0+index1+index2+index3)/3，依次类推。
                   A         B         C         D
2020-01-01       NaN       NaN       NaN       NaN
2020-01-02       NaN       NaN       NaN       NaN
2020-01-03 -0.567833  0.258723  0.498782  0.403639
2020-01-04 -0.384198 -0.093490  0.456058  0.459122
2020-01-05 -0.193821  0.085318  0.389533  0.552429
2020-01-06 -0.113941  0.252397  0.214789  0.455281
2020-01-07  0.147863  0.400141 -0.062493  0.565990
2020-01-08 -0.036038  0.452132 -0.091939  0.371364
2020-01-09 -0.043203  0.368912 -0.033141  0.328143
2020-01-10 -0.100571  0.349378 -0.078225  0.225649

""")

    st.markdown("## [ewm指数加权移动函数](https://www.cjavapy.com/article/451/)")
    st.code(r"""
# 表示指数加权移动。ewm() 函数先会对序列元素做指数加权运算，其次计算加权后的均值。该函数通过指定 com、span 或者 halflife 参数来实现指数加权移动
#设置com=0.5，先加权再求均值
df.ewm(com=0.5).mean()
""")


def pandas_tutorial(st, **state):
    result = st.sidebar.selectbox("教程目录", ['常用聚合函数介绍', '常用查询函数介绍', '窗口函数介绍', 'str处理介绍', 'pd.plot应用', 'style样式'])
    st.markdown("[pandas官方文档](https://pandas.pydata.org/pandas-docs/version/0.23/contributing_docstring.html)")
    st.code(r"pandas version: %s" % pd.__version__)
    if result == 'str处理介绍':
        st.markdown("# 文本处理")
        pandas_str(st)

    elif result == '窗口函数介绍':
        st.markdown("# 窗口函数介绍")
        pandas_windows(st)

    elif result == '常用查询函数介绍':
        st.markdown("# pandas之查询函数使用")
        pandas_check(st)

    elif result == '常用聚合函数介绍':
        st.markdown("# pandas之聚合函数使用")
        pandas_buildin(st)

    elif result == 'style样式':
        st.markdown("# pandas之style样式")
        pandas_style(st)

    elif result == 'pd.plot应用':
        st.markdown("# pandas之plot使用")
        pandas_plot(st)
    else:
        pass
