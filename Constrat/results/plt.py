import os
from  matplotlib import pyplot as plt
import numpy as np

result=os.listdir('./ablation')
acc={}
for file in result:
  ac = np.load('./ablation/'+file)*100
  name = file[:-4]
  acc[name] = ac

plt.figure()
x = ['1000','2000','3000',"4000",'5000']
for key in acc.keys():
  plt.plot(x, acc[key],linestyle='--',label=key)
  plt.scatter(x, acc[key], s=15)
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.5)
# 标题设置
plt.title('ablation')
plt.xlabel('number of data')
plt.ylabel('acc')
plt.savefig('./ablation/ablation.png')
plt.show()



