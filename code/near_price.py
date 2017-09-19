#coding=utf-8
import numpy as np
import math
import heapq
def torad(d):
    return d*math.pi/180.0
def transfer(a1,b1,a2,b2):
    a1=torad(a1)
    b1=torad(b1)
    a2=torad(a2)
    b2=torad(b2)
    s=2.0*math.asin(math.sqrt((math.sin((b2-b1)/2.0)**2)+math.cos(b1)*math.cos(b2)*(math.sin((a2-a1)/2.0)**2)))*6378.137
    return s

def think_near(magnitude,latitude,num,price):
    price_all=np.zeros((len(magnitude),num+4))
    for i in range(len(magnitude)):
        dis=[]
        if i==4:
            print("debug")
        for j in range(len(magnitude)):
            if i==j:
                continue
            else:
                tmp_dis=transfer(magnitude[j],latitude[j],magnitude[i],latitude[i])
                dis.append(tmp_dis)
        index=heapq.nsmallest(num, range(len(dis)), dis.__getitem__)
        price_tmp=[]
        for k in range(len(index)):
            price_tmp.append(price[index[k]])
        tmp=np.array(price_tmp)
        price_tmp.append(np.max(tmp))
        price_tmp.append(np.min(tmp))
        price_tmp.append(np.mean(tmp))
        price_tmp.append(np.median(tmp))
        price_all[i]=price_tmp
    return  price_all

def write_file(name,price_all,num,price):
    out=open(name,"w")
    for i in range(num):
        out.write(str(i+1)+',')
    out.write('max,min,mean,median,price\n')
    for i in range(np.shape(price_all)[0]):
        for j in range(np.shape(price_all)[1]):
            out.write(str(price_all[i][j])+',')
        out.write(str(price[i])+'\n')
    out.close()


def comnpute_error(price_all,price,num):
    max_error = 0
    min_error = 0
    mean_error = 0
    median_error = 0
    for i in range(len(price)):
        max_error += np.abs(price_all[i][num] - price[i])
        min_error += np.abs(price_all[i][num+1] - price[i])
        mean_error += np.abs(price_all[i][num+2] - price[i])
        median_error += np.abs(price_all[i][num+3] - price[i])
    print("max_error is \t"+str(max_error))
    print("min_error is \t"+str(min_error))
    print("mean_error is \t"+str(mean_error))
    print("median_error is \t"+str(median_error))

data1=np.genfromtxt("/home/scs4450/PycharmProjects/数学建模/3-reset_lat.csv",dtype=float,delimiter=',')
data2=np.genfromtxt("/home/scs4450/PycharmProjects/数学建模/output2.csv",dtype=float,delimiter=',')
magnitude=data1[1:,2]
latitude=data1[1:,1]
price=data1[1:,3]
price_all_ten=think_near(magnitude,latitude,10,price)
write_file("/home/scs4450/PycharmProjects/数学建模/mean2.csv",price_all_ten,10,price)
comnpute_error(price_all_ten,price,10)