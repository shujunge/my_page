import numpy as np
import pandas as pd
import time
from docs.my_redis_doc import redis_document
from docs.my_mangodb_doc import mangodb_document
from docs.my_pandas_tutorial import pandas_tutorial


def devops_tutorial(st, **state):
    result = st.sidebar.selectbox("python汇总", ['redis技术文档', 'mangodb技术文档'])
    if result == 'redis技术文档':
        redis_document(st)
    elif result == 'mangodb技术文档':
        mangodb_document(st)
    else:
        pass


def data_analysis(st, **state):
    result = st.sidebar.selectbox("选择文章", ['pandas教程', 'numpy教程', 'sklearn教程'])
    if result == 'pandas教程':
        pandas_tutorial(st)
    else:
        pass