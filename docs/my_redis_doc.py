import streamlit_echarts as st_echarts

def redis_document(st):
    st.write("# redis快速使用教程")
    st.markdown("## 连接服务器")
    st.code(">>>  redis-cli.exe -h 127.0.0.1 -p 6379")
    st.markdown("## 使用Python教程")
    st.code("""
        import redis
        r = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)
        r.set('food', 'mutton', ex=3)  # key是"food" value是"mutton" , ex=3s 为过期时间
        print(r.get("foo"))
        """, language='python')

    st.markdown("## 操作hash的命令")
    st.code("""
    r.flushall() # 删除所有的值
    r.hset(hash, key, value)  # 在hash中设置(key, value)
    r.hget(hash, key)  # 通过hash, key获得对应得 value 
    r.hmget("hash1", "k1", "k2"))  # 多个(k1,K2)取hash的key对应的值
    print(r.hgetall("hash1"))  # 获得hash所有得键值对
    
    print(r.hkeys("hash1"))  # 获取name对应的hash中所有的key的值
    print(r.hvals("hash1"))  # 获取name对应的hash中所有的values的值
    
    print(r.hexists("hash1", "k4"))  # False 不存在
    print(r.hexists("hash1", "k1"))  # True 存在
    """, language='python')
