# 对cal.py的说明

1. 依赖：pandas, numpy, tqdm, warnings, pickle, PyEMD (PyEMD似乎无法在Jupyter上使用)
2. API调用：
`gen_graph(df, n_lookback_days, n_imf_use, lower_bound=0.8, upper_bound=0.9):`

3. 参数说明：
- df：数据dataframe，每一列为每家公司的收益时间序列
- n_lookback_days: 用于生成一张图的时间长度
- n_imf_use：将时间序列拆为几个子序列
- lower_bound， upper_bound：我们只考虑相关性在lower bound和upper_bound之间的相关性（这是因为某些长期停牌的股票之间会被认为存在强相关，实则不然；而太弱的相关性考虑没有意义）

4. 推荐参数：150, 5， 0.8， 0.9

5. 输出介绍
- adj_mat： 假设输入总公司数是N，则adj_mat的大小为（N，N），为邻接矩阵，每一个entry代表i公司和j公司是否相关（1:相关，0:不相关）。利用correlation来进行判断，相关与不相关的阈值为输入lower_bound和upper_bound

- imf_matries_dict: 对于adj_mat中为1的股票组合我们会利用EMD算法来将他们的时间序列进行拆分（拆分为n_imf_use个）并且再次计算他们的correlation，由此得到imf_matries_dict，大小为(n_imf_use, N, N)

6. 可能用例：
adj_mat和imf_matries_dict都是表征公司组合相关性的矩阵，区别是邻接矩阵用binary0-1来表征两公司是否有关系，而imf_matries则进一步将有关公司的关系具体呈现，其中imf_1_matix为最显著的关系大小，imf_2_matix次之，以此推类
