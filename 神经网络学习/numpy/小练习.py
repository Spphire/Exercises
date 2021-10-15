import numpy as np
#1
print('#1')
x=np.arange(0,9,1)
print(x.reshape(3,3))
#2
print('#2')
print(np.random.randn(3,3)*2+1)
#3
print('#3')
print(x[-1::-1])
#4
print('#4')
x=np.ones([10,10])
x[1:-1:1,1:-1:1]=0
print(x)
#5
print('#5')

a=np.random.randn(2,10,64,64)
b=np.zeros([2,10,66,66])
b[:,:,1:-1:1,1:-1:1]+=a
print(b)

#6
print('#6')
a=np.random.randn(3,3)
b=np.random.randn(3,3)
print(a)
print(b)
print(np.where(a>b,a,b))

#7
print('#7')
a=np.random.randn(2,2)
lenth=np.sum(a*a)
print(a/lenth)

#8
print('#8')
a=np.random.randn(3,3)
b=np.zeros([3,3])
print(a)
print(b)
print(np.where(a>b,a,b))

#9
print('#9')
a=np.random.uniform(-10,10,[3,3])
a_=np.floor(a)
_a=np.ceil(a)
b=np.zeros([3,3])
print(a)
print(np.where(a>b,a_,_a))

#10
print('#10')
a=np.random.randn(3,5)
print(a)
b=np.random.randn(5,10)
print(b)
print(a.dot(b))

#11
print('#11')
a=np.zeros([10,10])
a[0::2,1::2]=1
a[1::2,0::2]=1
print(a)

#12
print('#12')
a=np.ones([10,10])
b=np.where(a==1)
print(b[0].reshape(10,10))
print(b[1].reshape(10,10))
