#usage: python -i linearReg.py


import numpy as np
import matplotlib.pyplot as plt
def linearReg(x,y,a):
    plt.figure()
    k=0
    b=0
    diff=[0,0]
    T=0
    m=len(x)
    while T<10:
        
        diff[0]=(-1/m)*sum((list(map(lambda y,x:(y-(k*x+b))*x,y,x))))
        diff[1]=(-1/m)*sum((list(map(lambda y,x:(y-(k*x+b)),y,x))))
        k=k-a*diff[0]
        b=b-a*diff[1] 
        dif=(1/(2*m))*sum((list(map(lambda y,x:(y-(k*x+b))**2,y,x))))
        
        print(dif,k,b)
        T+=1
   
    i=np.linspace(0,10,100)
    plt.plot(i,k*i+b,'-',label='Linear Regression')
    plt.plot(x,y,'ro')
    plt.legend(loc='upper left', shadow=True, fontsize='medium')
    plt.show(block=False)
#alpha relation in gradient descent

x=[1,2,3,4,5]
y=[1,3,4,4,6]
a=1
linearReg(x,y,0.01)

def gradientDescent():
    plt.figure()
    x=np.linspace(-20,20,100)
    plt.plot(x,x**2,'-',label='Gradient Descent')
    T=0
    x0=10
    while T<30:        
        des=2*x0
        x0-=0.1*des
        T+=1
        plt.plot(x0,x0**2,'ro')
    plt.legend(loc='upper left', shadow=True, fontsize='medium')
    plt.show(block=False)

gradientDescent()    

#linear regression in stochasitc gradient descent
def sgdescent(x,y,a):
    plt.figure()
    m=len(x)
    k=0
    b=0    
    T=0
    diff=[0,0]
    while T<10:
       for i in range(m):
           diff[0]=(y[i]-(k*x[i]+b))*(-x[i])
           diff[1]=(y[i]-(k*x[i]+b))*(-1)
           k=k-a*diff[0]
           b=b-a*diff[1]
       dif=(1/(2*m))*sum((list(map(lambda y,x:(y-(k*x+b))**2,y,x))))
       print(dif)
       T+=1
    i=np.linspace(0,10,100)
    plt.plot(i,k*i+b,'-',label='stochastic gradient descent')
    plt.legend(loc='upper left', shadow=True, fontsize='medium')
    plt.plot(x,y,'ro')    
    plt.show(block=False)

x=[1,2,3,4,5]
y=[1,3,4,4,6]
sgdescent(x,y,0.1)
    

#Normal equation
def equboundary(x,y):
    parameters=((x.T).dot(x)).I.dot((x.T).dot(y))
    return parameters


x=np.mat([[1,1],[1,2],[1,3],[1,4],[1,5]])
y=np.mat([[1],[3],[4],[4],[6]])
parameters=equboundary(x,y)
x1=[1,2,3,4,5]
y1=[1,3,4,4,6]
i=np.linspace(0,10,100)
data=np.array(parameters)
plt.figure()
plt.plot(i,data[1][0]*i+data[0][0],'-',label='Normal equations')
plt.plot(x1,y1,'ro')
plt.legend(loc='upper left', shadow=True, fontsize='medium')
plt.show(block=False)

