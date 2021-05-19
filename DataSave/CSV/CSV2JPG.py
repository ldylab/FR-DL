import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

data = pd.read_csv("2021-05-19-18_05_45--ResNet.csv", header=0, names = ['times', 'loss', 'acc'])
# data = pd.read_csv("2021-05-19-17_34_52--ResNet.csv",header=0)#不把第一行作列属性

#np.around(data.acc.values, 3)
print(data.acc.values)

x = data.times.values + 1
y1 = data.acc.values # np.around([], decimals=2)
y2 = data.loss.values # np.around([data.loss.values], decimals=2)
max_y1_index = np.argmax(y1) #max value index
min_y2_indx = np.argmin(y2) #min value index

print(type(y1))

plt.figure(figsize=(12.5, 10))
# 线条颜色black, 线宽2, 标记大小13, 标记填充颜色从网上找16进制好看的颜色
plt.plot(x, y1, '-o', color='black', markersize=13, markerfacecolor='#44cef6', linewidth=2, label='Acc')
plt.plot(x, y2, '-o', color='black', markersize=13, markerfacecolor='#e29c45', linewidth=2, label='Loss')

# 最大值的显示
# plt.plot(max_y1_index, y1[max_y1_index], 'ks')
# show_max='['+str(max_y1_index)+' '+str(y1[max_y1_index])+']'
# plt.annotate(show_max,xytext=(max_y1_index,y1[max_y1_index]),xy=(max_y1_index, y1[max_y1_index] + 0.1))
# 字体设置: 字体名称Times New Roman, 字体大小34
font_format = {'family':'Times New Roman', 'size':34}
plt.xlabel('Epoch', font_format)
plt.ylabel('Accuracy/Loss', font_format)
# 设置坐标轴 x范围0~3*pi, y范围-1.2~1.2
plt.axis([0, 32, 0.25, 1.0])
# 横纵坐标上的字体大小与类型(不是xlabel, 是xticks)
plt.xticks(fontproperties='Times New Roman', size=34)
plt.yticks(fontproperties='Times New Roman', size=34)
# 整个图像与展示框的相对位置
plt.subplots_adjust(left=0.19,right=0.94, bottom=0.13)
# 调整上下左右四个边框的线宽为2
ax=plt.gca()
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
plt.grid()  # 生成网格
legendfont = {'family' : 'Times New Roman',
                'weight' : 'normal',
                'size'   : 23,
              }
plt.legend(loc=0, ncol=1, prop=legendfont)
plt.show()