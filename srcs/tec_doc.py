import numpy as np
import pandas as pd
import time
from docs.my_redis_doc import redis_document
from docs.my_mangodb_doc import mangodb_document


def main_docs(st):
    valueList = ['redis技术文档', 'mangodb技术文档']
    result = st.sidebar.selectbox("文章汇总", valueList)
    if result == 'redis技术文档':
        redis_document(st)
    elif result == 'mangodb技术文档':
        mangodb_document(st)
    else:
        pass
