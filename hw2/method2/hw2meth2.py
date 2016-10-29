#coding=utf-8
import numpy as np
from math import e
import sys
import matplotlib.pyplot as plt
import random as rd
data = np.genfromtxt(sys.argv[1], delimiter =',' ,usecols = range(0,59))



#ans 為標準答案0或1，總共有4001個
ans = data[:,58]


# x = y = 0

# for i in range(0,4000):
# 	if ans[i] == 0:
# 		x+=1
# 	elif ans[i] ==  1:
# 		y+=1

# print(x)
# print(y)

w = np.zeros(57)
#bias
b=1.0
#a0   是loss對w的偏微分，總共有57維
#b0   是loss對b的偏微分
a0 = np.zeros(57)
b0=0.0
#learning rate
learn = 0.00005
#iteration
k=0
# x 是4001*57個的矩陣
x = np.ones((4001,57))
# define z   and  	function
#z為一57維的陣列，每一維都是一個值z=sum(w*x+b)，有4001個不同的z值
z =np.zeros(4001)
#func要是一個在0-1之間的值，總共有4001個
func = np.zeros(4001)
#cross entropy
err = 0.0

#最後的誤差
error = np.zeros(50) 
#忽略除以0的錯誤
np.seterr(divide='ignore')
{'under':'ignore'}


#x[i] 是第i個資料，總共有57維, i have 4000個 
# 讓y是1或是-1
for i in range(0,4001):
	x[i,:]= data[i,1:58]
	ans[i] = (ans[i]-0.5)*2



#資料的標準化
data_avg = np.zeros(58)
data_std = np.zeros(58)
for i in range(0,57):
	data_std[i+1] = np.std(data[:,i+1])
	data_avg[i+1] = np.average(data[:,i])
	x[:,i] = data[:,i+1]-data_avg[i+1]
	x[:,i] = x[:,i]/data_std[i+1]

err_old = 0.0
w = np.append(w,b)

# x成為(x,1)的array
x = np.column_stack((x , np.ones((4001,1))))

for i in range(0,4001):
	if ans[i]*(sum(w*x[i])) < 0 :
		err_old += 1



index = 0

while k<20000:
	err_new = 0.0
		
	while ans[index]*sum(w*x[index]) >= 0 :
		# index會成為0到4000中隨機產生的數
		index = rd.randint(0,4000)
		#  w = w+yi*xi
	w = w + ans[index]*x[index]*10000
		#index += 1

	for j in range(0,4001):
		if ans[j]*((w*x[j]).sum()) < 0 :
	
			err_new += 1

	if   err_new > err_old:
		w = w - ans[index]*x[index]*10000
		
	elif err_new <= err_old:
		w = w
		err_old = err_new
		
	index = rd.randint(0,4000)
	k+=1

stand = np.linspace(0.35,0.55,50)

output = []
output = w

np.savetxt(sys.argv[2],output,delimiter = ',' ,fmt ='%f')





