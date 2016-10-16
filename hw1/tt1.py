import numpy as np
import scipy as sc
import csv

arr=np.genfromtxt("train.csv",delimiter = ',',skip_header = 1,usecols = range(3,27))
brr=arr[9::18]
test=np.genfromtxt("test_X.csv",delimiter = ',',usecols = range(2,11))
trr=test[9::18]


k=1
w0 = np.zeros(9)
w1 = np.zeros(9)
x0=np.ones(9)
a0=np.zeros(9)
a1=np.zeros(9)
b0=0.0
y0=0.0
err=0.0
bias = 22.0
learn = 0.000000000001
lamda1=100
lamda2=100
delta=np.zeros(9)
a=0.0
crr=brr[0,:]

for i in range(1,240):
	crr=np.append(crr,brr[i,:])

while k<1000 :
	err=0.0
	a0 =np.zeros(9)
	b0 =0.0
	for i in range(0,len(crr)-10):
		x0 = crr[i+0:i+9]	
		if (i%10) == 9:	
			y0 = crr[i+9]
		else :	
			a0 -= 2*(y0-(bias+sum(w0*x0)+sum(w1*x0**2)))*x0
			a1 -= 2*(y0-(bias+sum(w0*x0)+sum(w1*x0**2)))*2*x0
			b0 -= 2*(y0-(bias+sum(w0*x0)+sum(w1*x0**2)))
		err += abs((y0-(bias+sum(w0*x0)+sum(w1*x0**2))))/5750
				
	delta += a0**2	
	for j in range(0,9):
		a0[j]=a0[j]+2*lamda1*w0[j]
		a1[j]=a1[j]+2*lamda2*w1[j]
		w0[j] = w0[j]-learn*a0[j]/np.sqrt(delta[j])
		w1[j] = w1[j]-learn*a1[j]
		bias = bias - learn*b0
	k+=1
	print(k,"err",err)


#testing data
output = np.array([["id","value"]])
print("id_i","value")
for i in range(0,240):
	xx0 = trr[i,:]
	y = bias+sum(w0*xx0)+sum(w1*xx0**2)
	print(y)
	output = np.append(output,[["id_"+str(i),y]], axis=0 )
np.savetxt( "linear_regression.csv" ,output,delimiter =',',fmt = "%s")


