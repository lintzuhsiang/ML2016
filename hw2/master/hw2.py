#coding=utf-8
import numpy as np
from math import e
import sys
import matplotlib.pyplot as plt


data = np.genfromtxt(sys.argv[1], delimiter =',' ,usecols = range(0,59))

#忽略除以0的錯誤
np.seterr(divide='ignore')
{'under':'ignore'}

#ans 為標準答案0或1，總共有4001個
ans = data[:,58]


w = np.zeros(57)
#bias
b=0.0
#a0   是loss對w的偏微分，總共有57維
#b0   是loss對b的偏微分
a0 = np.zeros(57)
b0=0.0
#learning rate
learn = 0.00005
#iteration
k=0
#x是4001*57個的矩陣
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



#x[i] 是第i個資料，總共有57維, i have 4000個 
for i in range(0,4001):
	x[i,:]= data[i,1:58]



#資料的標準化
data_avg = np.zeros(58)
data_std = np.zeros(58)
for i in range(0,57):
	data_avg[i+1] = np.average(data[:,i+1])
	data_std[i+1] = np.std(data[:,i+1])
	x[:,i] = data[:,i+1]-data_avg[i+1]
	x[:,i] = x[:,i]/data_std[i+1]

while k<600:
	err=0.0
	a0=np.zeros(57)
	b0=0.0

	for i in range(0,4001):
		z[i] = sum(w*x[i,:])+b
		func[i] = 1/(1+e**(-z[i]))
		if -np.log(1-func[i]) == np.inf:
			err = err
		elif ans[i] == 1:
			err -= ans[i]*np.log(func[i])
		elif ans[i] == 0 :
			err -= (1-ans[i])*np.log(1-func[i])
			

#計算a0跟b0
#a0[j]為第j個的loss的偏微分	
#b0是loss對b的偏微分值
	for j in range(0,57):
		for i in range(0,4001):
			a0[j] += (ans[i]-func[i])*x[i][j]
			b0 += (ans[i]-func[i])*1
		w[j] = w[j] + learn*a0[j]
		b = b + learn*b0/57

	k+=1

	print('k',k)
	#print('err',err)


stand = np.linspace(0.35,0.55,50)
#求最小值的位置
ele = 0
for j in range (0,50):
	for i in range(0,4001):
		if func[i] < stand[j]:
			func[i] =0
		else :
			func[i] = 1
		error[j] += abs(ans[i]-func[i])
		z[i] = sum(w*x[i,:])+b
		func[i] = 1/(1+e**(-z[i]))	

#求出error最小的stand值，ele為其最小的位置
# error = np.append(error[0],error)
# error = map(lambda x: (x), error)
ele = np.argmin(error)

error = 0
for i in range(0,4001):
	if func[i] < stand[ele]:
		func[i] =0
	else :
		func[i] = 1
	error += abs(ans[i]-func[i])
	z[i] = sum(w*x[i,:])+b
	func[i] = 1/(1+e**(-z[i]))



output = []
output = np.append(w,b)
output = np.append(output,stand[ele])

np.savetxt(sys.argv[2],output,delimiter = ',' ,fmt ='%f')








