import cv2
import numpy as np
origimg=cv2.imread("1.jpg")

deepimg=cv2.imread("1_dpt.tiff",-1)
H=origimg.shape[0]
W=origimg.shape[1]
C_matrix=np.array([1.1141804900000000e+03,0.                    , 1.0742415300000000e+03,
                   0.                    ,1.1134568400000001e+03, 6.0864778799999999e+02,
                   0.                    ,0.                    , 1.                     ])
D_matrix=np.array([9.1675828000000004e-01 , 1.8388357100000000e-01 ,3.5459989800000002e-01 , 1.6049487700000000e+02,
                   -2.0549181599999999e-01, 9.7836655400000005e-01 ,2.3916500500000000e-02 , 5.4823469099999997e+01,
                   -3.4253082899999998e-01, -9.4793027000000002e-02,9.3471220899999996e-01 , -8.4464335099999996e-02,
                   0.                     , 0.                     ,0.                     , 1.                      ])
C_matrix=C_matrix.reshape([3,3])
D_matrix=D_matrix.reshape([4,4])

a=np.ones([H,W])
Z=np.ones([H,W]).reshape(1,H*W)    #[[1,1,1,...,1,1,1]]
b=np.where(a==1)
#print(H)
#print(b[0])
pos=np.concatenate((b[0].reshape([1,H*W]),b[1].reshape([1,H*W]),Z),axis=0)   #照片坐标 (x,y,1)*n
print(pos)
pos_1=(np.linalg.inv(C_matrix)@pos)
pos_1=deepimg.reshape([1,H*W])*pos_1        #相机内参
#print(np.linalg.inv(C_matrix)@pos)
#print(pos_1)
#pos_1=deepimg.reshape([1,H*W])*pos
pos_real1=np.concatenate((pos_1,Z),axis=0)              #(x,y,z,1)
pos_real2=D_matrix@pos_real1                                 #参照系变换
pos_x=pos_real2[0,:]
pos_y=pos_real2[1,:]
pos_z=pos_real2[2,:]
pos_em=np.concatenate((    (pos_x/pos_z).reshape(1,H*W),(pos_y/pos_z).reshape(1,H*W),(pos_z/pos_z).reshape(1,H*W)   ),axis=0) #(x,y,1)
#print()
#print(pos_real2)
#print(pos_em)
pos_m=(C_matrix)@pos_em                  #相机内参

mask1=(0<=pos_m[0]).astype('int64')
print(pos_m[0])
print(mask1)
mask2=(H-0.5>pos_m[0]).astype('int64')
print(pos_m[0])
print(mask2*mask1)
mask3=(0<=pos_m[1]).astype('int64')
mask4=(W-0.5>pos_m[1]).astype('int64')
mask=(mask2*mask1*mask4*mask3).astype('bool')
print(mask)

output=np.zeros([H,W,3]).astype('uint8')


x=pos_m[0][mask].round().astype('int64')
y=pos_m[1][mask].round().astype('int64')


output[x,y]=origimg[b[0][mask].astype('int64'),b[1][mask].astype('int64')]

cv2.imshow("1",output)
cv2.waitKey(0)

