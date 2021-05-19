import glob
import json
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
# 利用正则读取json文件,不太可取,因为文件夹里很多之前运行结果的json文件,不具备通用性
#convergency curve
# dirName = glob.glob(r'./results/NewReward/PPO_SeqTrafficLightEnv-v0_5753e_00000_0_2021-04-14_22-47-31/result.json')
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
# plt.plot(reward_mean_att)

# plt.plot(reward_mean_att, label="Base")
# plt.plot(reward_mean_no_att, label="Base-Att")
# plt.xlabel('trial')
# plt.ylabel('rewards')
#
# plt.legend()
# plt.savefig('exp.png')

#attention draw
# df=pd.read_csv("./attention_default_model.csv")
# df=df.iloc[:,1:]
# # print(df.head())
# # print('after softmax')
# # df=df.apply(lambda x:np.exp(x)/np.sum(np.exp(x)), axis=1)
# # print(df.head())
# sns.heatmap(df)
# plt.savefig('./heatmap.png')

#convergency curve2

dirName = glob.glob(r'/home/male/Desktop/nigger/results/Desired/PPOPlus_SeqTrafficLightEnv-v0_6e526_00000_0_2021-05-07_22-13-43/result.json')
name = dirName[0]
jsonContext = []
for line in open(name, 'r'):
    jsonContext.append(json.loads(line))

reward_mean_no_att=[]

for i in jsonContext:
    reward_mean_no_att.append(i['episode_reward_mean'])
plt.plot(reward_mean_no_att)
plt.xlabel('trial')
plt.ylabel('rewards')

plt.savefig('Niggerconvergency_curve.png')




