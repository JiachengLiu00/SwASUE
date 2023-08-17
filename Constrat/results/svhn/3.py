import matplotlib
import numpy as np
from  matplotlib import pyplot as plt
# %matplotlib inline
#通用设置
acc = {}
our = np.load('./ours.npy')*100
our1 = np.load('./ours-f.npy')*100
CoreGCN = np.load('./CoreGCN0.npy')
VAAL= np.load('./VAAL0.npy')
lloss = np.load('lloss0.npy')
CoreSet = np.load('CoreSet0.npy')
Random = np.load('Random0.npy')

acc['lloss'] = lloss
acc['VAAL'] = VAAL
acc['CoreGCN0'] = CoreGCN
acc['ours'] = our
acc['ours-f'] = our1
acc["CoreSet"] = CoreSet
acc['Random'] = Random
x = ['1000','2000','3000',"4000",'5000']
# y = acc['ours']
print(our)
# print(y)
# matplotlib.rc('axes', facecolor = 'white')
# matplotlib.rc('figure', figsize = (6, 4))
# matplotlib.rc('axes', grid = False)
#数据及线属性
plt.figure()
# for key in acc.keys():

plt.plot(x, acc['ours'],color='b',linestyle='--',label='ours')
plt.scatter(x, acc['ours'], c='b',s=15)
plt.plot(x, acc['ours-f'],color='purple',linestyle=':',label='ours-f')
plt.scatter(x, acc['ours-f'], c='purple',s=15)
plt.plot(x, acc['lloss'],color='r',linestyle='-.',label='lloss')
plt.scatter(x, acc['lloss'], c='r',s=15)
plt.plot(x, acc['VAAL'],color='g',linestyle=':',label='VAAL')
plt.scatter(x, acc['VAAL'], c='g',s=15)
plt.plot(x, acc['CoreGCN0'],color='c',linestyle='--',label='CoreGCN')
plt.scatter(x, acc['CoreGCN0'], c='c',s=15)
plt.plot(x, acc["CoreSet"],color='y',linestyle='-.',label='CoreSet')
plt.scatter(x, acc["CoreSet"], c='y',s=15)
plt.plot(x, acc['Random'],color='m',linestyle='-.',label='Random')
plt.scatter(x, acc['Random'], c='m',s=15)

plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.5)
#标题设置
plt.title('svhn')
plt.xlabel('number of data')
plt.ylabel('acc')
plt.savefig('./svhn.png')
plt.show()

