import numpy as np
from cvxpy import *
import cvxpy as cvx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
from matplotlib.animation import FuncAnimation
m = 6
n = 6
totals = [60, 60]
nums = []
for index, i in enumerate(totals):
    if index == 0:
        total = i
        temp = []
        for j in range(m-1):
            val = np.random.randint(0, total)
            temp.append(val)
            total -= val
        temp.append(total)
        nums.append(temp)
    elif index == 1:
        total = i
        temp = []
        for j in range(n-1):
            val = np.random.randint(0, total)
            temp.append(val)
            total -= val
        temp.append(total)
        nums.append(temp)
# print(cvx.installed_solvers())      # ['ECOS', 'ECOS_BB', 'OSQP', 'SCS']
# print(nums)
factories_output = np.asarray(nums[0])
wareHouse_demand = np.asarray(nums[1])
Flag = np.sum(factories_output) == np.sum(wareHouse_demand)
print('Flag:', Flag)
print('factories_output:', factories_output)
print('wareHouse_demand:', wareHouse_demand)
factories_location = np.random.random((m,2)) * 10
wareHouse_location = np.random.random((n,2)) * 10
print('factories_location:\n{}'.format(factories_location))
print('wareHouse_location:\n{}'.format(wareHouse_location))

fac_index = ["fac_" + str(i) for i in range(m)]
wareHouse_index = ["house_" + str(i) for i in range(n)]
# fac_anotatation = [str(i) for i in factories_output]
# print(fac_anotatation)
# plot factories_output and wareHouse_demand
#"""
# figure of Factories Products、WareHouse Demands、Location
# """
plt.figure(figsize=(30, 30))
plt.subplot(1, 2, 1)
plt.scatter(fac_index, factories_output, marker=6)
plt.title('Factories Products')
plt.xlabel('Factories')
plt.ylabel('Plant Output')

plt.subplot(1, 2, 2)
plt.scatter(wareHouse_index, wareHouse_demand, marker=7)
plt.title('WareHouse Demands')
plt.xlabel('WareHouses')
plt.ylabel('Demands')
"""
plt.subplot(2, 2, 3)
plt.scatter(factories_location[:,0], factories_location[:,1], marker='.', color='red', alpha=0.5, label='factory')
plt.scatter(wareHouse_location[:,0], wareHouse_location[:,1], marker='*', color='blue', alpha=0.5, label='wareHouse')
plt.title('Location ')
plt.xlabel('x')
plt.ylabel( 'y')
plt.legend(loc='upper right')
"""
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=0.2, hspace=0.5)
plt.show()
# """
# for i in range(len(fac_anotatation)):
#     plt.annotate(fac_anotatation[i], xy=(fac_index[i], factories_output[i]), xytext=(fac_index[i]+0.1, factories_output[i]+0.1)) # 这里xy是需要标记的坐标，xytext是对应的标签坐标

#"""
# print('factories_location:', factories_location)
# print('wareHouse_location:', wareHouse_location)
"""
def cost(solution_matrix, fac_loc, ware_loc):
    total_cost = 0
    for i in range(m):
        for j in range(n):
            total_cost += solution_matrix[i][j] * np.linalg.norm(fac_loc[i] - ware_loc[j])
    return total_cost
"""
x = cvx.Variable((m, n))    # 构建迭代变量
position_distance_matrix = np.zeros((m, n))     # 存放工厂i到j的距离数据，计算两点之间的二范数
for i in range(m):
    for j in range(n):
        position_distance_matrix[i][j] = np.linalg.norm((factories_location[i] - wareHouse_location[j]), ord=2)

constraints = []    # 存放三个迭代过程的三组限制条件
for i in range(m):
    constraints += [cvx.sum(x[i,:]) == factories_output[i]]

for j in range(n):
    constraints += [cvx.sum(x[:, j]) == wareHouse_demand[j]]

for i in range(m):
    for j in range(n):
        constraints += [x[i, j] >= 0]

# constraints = [cvx.sum(x, axis=1) == factories_output, cvx.sum(x, axis=0) == wareHouse_demand, x >= 0]
objective = cvx.Minimize(cvx.sum(multiply(x, position_distance_matrix)))    # 构建目标函数
prob = cvx.Problem(objective, constraints)   # 迭代求解问题
print("Optimal value", prob.solve(solver=cvx.SCS))
print("Optimal var")
print("status:", prob.status)
# print(x.value)  # A numpy ndarray.
# print(x.value)
result = np.rint(x.value)
print(result)
# print(result[0])
# print(np.rint(x.value))
fac_index = [i for i in range(m)]
wareHouse_index = [i for i in range(n)]
"""
color = ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
fig = plt.figure()
ax = Axes3D(fig)
for i in range(m):
    ax.scatter(fac_index, wareHouse_index, result[i], c=color[i])

ax.set_zlabel('Product', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('House', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('Fac', fontdict={'size': 15, 'color': 'red'})

plt.show()
"""




fig = plt.figure(figsize=(15, 15))
plt.scatter(factories_location[:,0], factories_location[:,1], marker='.', color='red', alpha=0.5, label='factory')
plt.scatter(wareHouse_location[:,0], wareHouse_location[:,1], marker='*', color='blue', alpha=0.5, label='wareHouse')
plt.title('Location ')
plt.xlabel('x')
plt.ylabel( 'y')
plt.legend(loc='upper right')
color = ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
lineShape = ['solid', 'dashed', 'dashdot', 'dotted','--', '-.']
# """
for i in range(m):
    x1 = factories_location[i, 0]
    y1 = factories_location[i, 1]
    for j in range(n):
        shiftNumber = result[i, j]
        if shiftNumber != 0:
            x2 = wareHouse_location[j, 0]
            y2 = wareHouse_location[j, 1]
            # ls :'-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
            plt.plot([x1, x2], [y1, y2], c=color[j],ls=lineShape[i])
            x_annotate = (x1 + x2) / 2
            y_annotate = (y1 + y2) / 2
            plt.annotate(text='%s' % shiftNumber, xy=(x_annotate, y_annotate),
                         xytext=(x_annotate, y_annotate), textcoords='offset pixels', xycoords='data',color=color[j])
# """

"""
ln, = plt.plot([], [])
def update(frame):
    for i in range(m):
        x1 = factories_location[i, 0]
        y1 = factories_location[i, 1]
        for j in range(n):
            shiftNumber = result[i, j]
            if shiftNumber != 0:
                x2 = wareHouse_location[j, 0]
                y2 = wareHouse_location[j, 1]
                xData = [x1, x2]
                yData = [y1, y2]
                ln.set_data = (xData, yData)
                ln.c = color[j]
                ln.ls = lineShape[i]
                # plt.plot([x1, x2], [y1, y2], c=color[j], ls=lineShape[i])
                x_annotate = (x1 + x2) / 2
                y_annotate = (y1 + y2) / 2
                plt.annotate(text='%s' % shiftNumber, xy=(x_annotate, y_annotate),
                             xytext=(x_annotate, y_annotate), textcoords='offset pixels', xycoords='data',
                             color=color[j])
                return ln,
ani = FuncAnimation(fig, update,blit=True, interval=1)
"""
# plt.plot([factories_location[0,0],wareHouse_location[0,0]],[factories_location[0,1],wareHouse_location[0,1]],c='blue',ls='--')
# plt.annotate(text='10',xy=((factories_location[0,0] + wareHouse_location[0,0])/2, (factories_location[0,1] + wareHouse_location[0,1]) / 2),
#              xytext= ((factories_location[0,0] + wareHouse_location[0,0])/2, (factories_location[0,1] + wareHouse_location[0,1]) / 2),
#              textcoords='offset points')
plt.show()
