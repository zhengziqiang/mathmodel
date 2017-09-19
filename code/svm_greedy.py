#coding=utf-8
from sklearn import svm
import numpy as np
import math
all_data_train = np.genfromtxt('/home/scs4450/PycharmProjects/数学建模/reset-svm.csv',dtype=float, delimiter=',')
all_train_data_x=all_data_train[:,:np.shape(all_data_train)[1]-1]
all_train_data_label=all_data_train[:,-1]
clf = svm.SVC()
clf.fit(all_train_data_x,all_train_data_label)
step=0.001
thresh=0.05
all_power=[]
have_done=np.zeros((1,13))
cnt=0
for i in range(np.shape(all_data_train)[0]):
    if all_data_train[i,-1]==1.0:
        if cnt==0:
            have_done=all_data_train[i,:]
            have_done=np.reshape(have_done,[1,-1])

        else:
            tmp=all_data_train[i, :]
            tmp=np.reshape(tmp,[1,-1])
            have_done = np.concatenate([have_done, tmp], axis=0)
        cnt+=1
    else:
        continue
        # print("debug")
print ("the cnt of have done is"+str(cnt))
cnt=0

print (np.sum(have_done[:,-2])/np.shape(have_done)[0])

not_have_done=np.zeros((1,13))
for i in range(np.shape(all_data_train)[0]):
    if all_data_train[i, -1] == 0.0:
        if cnt == 0:
            not_have_done = all_data_train[i, :]
            not_have_done = np.reshape(not_have_done, [1, -1])

        else:
            tmp = all_data_train[i, :]
            tmp = np.reshape(tmp, [1, -1])
            not_have_done = np.concatenate([not_have_done, tmp], axis=0)
        cnt += 1
    else:
        continue
print ("the cnt of not have done is"+str(cnt))
print(np.shape(not_have_done))
print(np.shape(have_done))
for i in range(np.shape(have_done)[0]):
    for j in range(50):
        price_tmp=have_done[i,-2]-j*step
        old_price=have_done[i,-2]
        have_done[i,-2]=price_tmp
        tmp_x=have_done[i,:12]
        tmp_x=np.reshape(tmp_x,[1,-1])
        predict_x = clf.predict(tmp_x)
        if predict_x==0.0:
            have_done[i,-2]-=(j-1)*step
            break
        else:
            have_done[i, -2]=old_price
            continue

another_have_done=have_done
add_dict={}

for i in range(np.shape(not_have_done)[0]):
    for j in range(100):
        price_tmp=not_have_done[i,-2]+j*step
        old_price = not_have_done[i, -2]
        not_have_done[i,-2]=price_tmp
        tmp_x=not_have_done[i,:12]

        tmp_x=np.reshape(tmp_x,[1,-1])
        predict_x = clf.predict(tmp_x)
        if predict_x==1.0:
            not_have_done[i, -2] += (j - 1) * step
            not_have_done[i,-1]=1.0
            add_dict[i]=not_have_done[i, -2]
            break
        else:
            not_have_done[i,-2]=old_price
            continue
# sorted(dict.items(), lambda x, y: cmp(x[1], y[1]))
# for i in range(len(add_dict.keys())):
#     if (np.sum(have_done[:, -2]) + tmp_add[-2]) / (np.shape(have_done)[0] + 1) < (np.sum(have_done[:, -2]) / np.shape(have_done)[0]):
#

list_remove=[]
for i in range(np.shape(not_have_done)[0]):
    for j in range(100):
        price_tmp=not_have_done[i,-2]+j*step
        old_price = not_have_done[i, -2]
        not_have_done[i,-2]=price_tmp
        tmp_x=not_have_done[i,:12]

        tmp_x=np.reshape(tmp_x,[1,-1])
        predict_x = clf.predict(tmp_x)
        if predict_x==1.0:
            not_have_done[i, -2] += (j - 1) * step
            not_have_done[i,-1]=1.0
            break
        else:
            not_have_done[i,-2]=old_price
            continue

    tmp_add=not_have_done[i,:]
    if (np.sum(have_done[:,-2])+tmp_add[-2])/(np.shape(have_done)[0]+1) < (np.sum(have_done[:,-2])/np.shape(have_done)[0]):
        tmp_add=np.reshape(tmp_add,[1,-1])
        have_done=np.concatenate([have_done,tmp_add],axis=0)
        list_remove.append(i)

all_set=set(range(0,313,1))
remove_set=set(list_remove)
other=list(all_set-remove_set)
other_not_have_done=np.zeros((len(other),13))
for i in range(len(other)):
    other_not_have_done[i,:]=not_have_done[other[i],:]
print(np.shape(have_done))
print(np.shape(other_not_have_done))

new_data=np.concatenate([other_not_have_done,have_done],axis=0)
with open("svm_greedy.csv","w") as foo:
    np.savetxt(foo,new_data[:,:np.shape(new_data)[1]-1],delimiter=',')
print(np.sum(have_done[:,-2])/np.shape(have_done)[0])




