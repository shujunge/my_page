import streamlit_echarts as st_echarts
from streamlit_ace import st_ace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
import seaborn as sns
from scipy import stats
import streamlit.components.v1 as components        #将要展示的 弄成html

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
    zz=df.style.bar(subset=['A', 'B'], align='mid', color=['#d65f5f', '#5fba7d'])
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
    st.code("""
df = pd.DataFrame({'X': ['A', 'B', 'A', 'B'], 'Y': [1, 2, 1, 2], 'Z': [4, 4, 1, 2]})
""")
    df = pd.DataFrame({'X': ['A', 'B', 'A', 'B'], 'Y': [1, 2, 1, 2], 'Z': [4, 4, 1, 2]})
    st.dataframe(df)

    st.markdown("## groupby代码")
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


def pandas_tutorial(st, **state):
    result = st.sidebar.selectbox("教程目录", ['常用聚合函数介绍', 'pd.plot应用', 'style样式'])
    if result == '常用聚合函数介绍':
        st.markdown("# pandas之聚合函数使用")
        pandas_buildin(st)

    elif result == 'style样式':
        st.markdown("## pandas之style样式")
        pandas_style(st)
    elif result == 'pd.plot应用':
        st.markdown("## pandas之plot使用")
        pandas_plot(st)
    else:
        pass
