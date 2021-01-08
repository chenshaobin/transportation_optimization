import numpy as np
from cvxpy import *
import cvxpy as cvx
import matplotlib.pyplot as plt
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
pi = factories_location
qj = wareHouse_location
print('factories_location:\n{}'.format(factories_location))
print('wareHouse_location:\n{}'.format(wareHouse_location))

fac_index = ["fac_" + str(i) for i in range(m)]
wareHouse_index = ["house_" + str(i) for i in range(n)]
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
    # figure factories and wareHouse locations
"""
fig = plt.figure(figsize=(15, 15))
plt.scatter(factories_location[:,0], factories_location[:,1], marker='.', color='red', alpha=0.5, label='factory')
plt.scatter(wareHouse_location[:,0], wareHouse_location[:,1], marker='*', color='blue', alpha=0.5, label='wareHouse')
plt.title('Location ')
plt.xlabel('x')
plt.ylabel( 'y')
plt.legend(loc='upper right')
"""
    # figure factories and wareHouse locations End
"""

color = ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
lineShape = ['solid', 'dashed', 'dashdot', 'dotted','--', '-.']
"""
    # figure static transportation figure
"""
for i in range(m):
    x1 = factories_location[i, 0]
    y1 = factories_location[i, 1]
    for j in range(n):
        shiftNumber = result[i, j]
        if shiftNumber != 0:
            x2 = wareHouse_location[j, 0]
            y2 = wareHouse_location[j, 1]
            plt.plot([x1, x2], [y1, y2], c=color[j],ls=lineShape[i])
            x_annotate = (x1 + x2) / 2
            y_annotate = (y1 + y2) / 2
            plt.annotate(text='%s' % shiftNumber, xy=(x_annotate, y_annotate),
                         xytext=(x_annotate, y_annotate), textcoords='offset pixels', xycoords='data',color=color[j])

"""
    # figure static transportation figure End
"""
"""
    # figure dynamic transportation figure
"""

fig, ax = plt.subplots()
dot =[]
_dot =[]
color =['r','g','b','c','violet','lightsalmon','sage','skyblue','y','pink']
linestyle = ['o','o','o','o','o','o','o','o','o','o']
for i in range(m):
    for j in range(n):
        exec("dot%s%s,=ax.plot([], [],color = color[%s],marker = linestyle[%s], animated=False)" % (i,j,i,j))

def init():
    ax.set_xlim(0,15)
    ax.set_ylim(0,15)
    for i in range(m):
        for j in range (n):
            exec("_dot.append(dot%s%s)" % (i,j))
    return _dot

def generate_dot():
    # print("aaaa")
    for i in range(100):
        dot.clear()
        for k in range(m):
            for j in range(n):
                if (result[k][j] > 0):
                    x_frame = np.linspace(pi[k, 0], qj[j, 0], 100)
                    y_frame = np.linspace(pi[k, 1], qj[j, 1], 100)
                    newdot = [x_frame[i], y_frame[i]]
                    # print(newdot)
                    dot.append(newdot)
            # print(dot)
        yield dot

def update_dot(newd):
    _dot.clear()
    # print(len(newd))
    i = 0
    for k in range(m):
        for j in range(n):
            if (result[k][j] > 0):
                exec("dot%s%s.set_data(newd[i][0], newd[i][1])" % (k,j))
                exec("_dot.append(dot%s%s)" % (k,j))
                i += 1
        # print(_dot)
    return _dot

def draw_line(i):
    for j in range(n):
        if (result[i][j] > 0):
            x_line = np.linspace(pi[i, 0], qj[j, 0], 100)
            y_line = np.linspace(pi[i, 1], qj[j, 1], 100)
            ax.plot(x_line,y_line,color=color[i],linestyle='-.')

def draw_location():
    # print(pi[0, 0], qj[0, 0])
    # print(pi[0, 1], qj[0, 1])


    plt.scatter(pi[:, 0], pi[:, 1], marker='.', color='red',  alpha=0.5, label='factory')
    plt.scatter(qj[:, 0], qj[:, 1], marker='*', color='blue',  alpha=0.5, label='wareHouse')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper right')

def set_annotation(i):
    for j in range(n):
        if (result[i][j] > 0):
            plt.annotate("%s" % result[i][j],
                         xy=(((pi[i, 0]+qj[j,0])/2), ((pi[i, 1]+qj[j,1])/2)),
                         xytext=(((pi[i, 0]+qj[j,0])/2), ((pi[i, 1]+qj[j,1])/2)),
                         color="b",
                       )
draw_location()
for i in range(m):
    draw_line(i)
    set_annotation(i)
ani = FuncAnimation(fig, update_dot, frames=generate_dot, init_func=init, interval=5)
ani.save("transportation.gif")
"""
    # figure dynamic transportation figure End
"""

plt.show()
