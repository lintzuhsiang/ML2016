#coding=utf-8
import numpy as np
from math import e
import sys

#testing data
#xx 是測試data的矩陣
#忽略除以0的錯誤
np.seterr(divide='ignore')
{'under':'ignore'}
ww = []
ww = np.genfromtxt(sys.argv[1],usecols = range(0,1))
test = np.genfromtxt(sys.argv[2],delimiter = ',',usecols = range(0,58))

w = ww[0:57]
b=0

xx = np.zeros((600,57))
zz = np.zeros(600)
ffunc = np.zeros(600)
output = np.array([[1,ffunc[0]]])

test_avg = np.zeros(58)
test_std = np.zeros(58)
#資料的標準化
for i in range(0,57):
	test_avg[i+1] = np.average(test[:,i+1])
	test_std[i+1] = np.std(test[:,i+1])
	xx[:,i] = test[:,i+1]-test_avg[i+1]
	xx[:,i] = xx[:,i]/test_std[i+1]

for i in range(1,600):
	zz[i] = (w*xx[i,:]).sum()+b
	ffunc[i] = 1/(1+e**(-zz[i]))
	print(i)
	if ffunc[i] < 0.5:
		ffunc[i] =0
	else :
		ffunc[i] = 1
	output = np.append(output,[[i+1,ffunc[i]]],axis = 0)
np.savetxt(sys.argv[3],output,delimiter = ',' ,fmt ='%i',comments='',header ="id,label")

