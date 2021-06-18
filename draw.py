import glob
import json
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
# 利用正则读取json文件,不太可取,因为文件夹里很多之前运行结果的json文件,不具备通用性
#convergency curve
# dirName = glob.glob(r'./pic/result_att.json')
# name = dirName[0]
# jsonContext = []
# for line in open(name, 'r'):
#     jsonContext.append(json.loads(line))
#
# reward_mean_att = []
#
# for i in jsonContext:
#     reward_mean_att.append(i['episode_reward_mean'])
#
# dirName = glob.glob(r'./pic/result_no_att.json')
# name = dirName[0]
# jsonContext = []
# for line in open(name, 'r'):
#     jsonContext.append(json.loads(line))
#
# reward_mean_no_att=[]
#
# for i in jsonContext:
#     reward_mean_no_att.append(i['episode_reward_mean'])
#
# reward_mean_att=reward_mean_att[0:249]
# reward_mean_no_att=reward_mean_no_att[0:249]
#
# df=pd.DataFrame(columns=["Att"], data=reward_mean_att)
# df["NoAtt"]=reward_mean_no_att
#
# df.to_csv("./reward_all.csv")
#
# plt.plot(reward_mean_att, label="Base")
# plt.plot(reward_mean_no_att, label="Base-Att")
# plt.xlabel('trial')
# plt.ylabel('rewards')
#
# plt.legend()
# plt.savefig('convergency_curve.png')

#attention draw
# df=pd.read_csv("./attention_default_model.csv")
# df=df.iloc[:,1:]
# # print(df.head())
# # print('after softmax')
# # df=df.apply(lambda x:np.exp(x)/np.sum(np.exp(x)), axis=1)
# # print(df.head())
# sns.heatmap(df)
# plt.savefig('./heatmap.png')

# #convergency curve2
# df=pd.read_csv("./reward_all.csv")
#
# df=df.loc[0:199,:]
#
# dirName = glob.glob(r'./pic/NoAttn.json')
# name = dirName[0]
# jsonContext = []
# for line in open(name, 'r'):
#     jsonContext.append(json.loads(line))
#
# reward_mean_no_att = []
#
# for i in jsonContext:
#     reward_mean_no_att.append(i['episode_reward_mean'])
# print(len(reward_mean_no_att))
# reward_mean_no_att=reward_mean_no_att[200:299]
#
#
# dirName = glob.glob(r'./pic/result_att.json')
# name = dirName[0]
# jsonContext = []
# for line in open(name, 'r'):
#     jsonContext.append(json.loads(line))
#
# reward_mean_att = []
#
# for i in jsonContext:
#     reward_mean_att.append(i['episode_reward_mean'])
#
# reward_mean_att=reward_mean_att[200:299]
#
# dirName = glob.glob(r'./pic/result_no_att.json')
# name = dirName[0]
# jsonContext = []
# for line in open(name, 'r'):
#     jsonContext.append(json.loads(line))
#
# reward_mean_drl = []
#
# for i in jsonContext:
#     reward_mean_drl.append(i['episode_reward_mean'])
#
# reward_mean_drl=reward_mean_drl[200:299]
#
#
#
# df2=pd.DataFrame()
# df2["Base"]=reward_mean_att
# df2["DRL"]=reward_mean_drl
# df2["NoAtt"]=reward_mean_no_att
#
#
# df=pd.read_csv("./reward_all2.csv")
#
# df3=pd.concat([df,df2],axis=1,ignore_index=True)
#
# plt.plot(df["Base"], label="ours")
# plt.plot(df["DRL"], label="DRL")
# plt.plot(df["NoAtt"], label="no-attention")
# plt.xlabel('trial')
# plt.ylabel('rewards')
#
# plt.legend()
#
# df["NoAtt"]=reward_mean_no_att
# df.to_csv("./reward_all2.csv")


df=pd.read_csv("./reward_all2.csv")
plt.plot(df["Base"], label="Ours")
plt.plot(df["DRL"], label="DRL")
plt.plot(df["NoAtt"], label="No-Attention")
plt.xlabel('trial')
plt.ylabel('rewards')

plt.legend()
plt.savefig('convergency_curve.png')




