import re
import numpy as np
import scipy.io as sio
import pickle
x = np.zeros((7062048, 1), dtype=float)
#
#
# # with open('hidden_all_stack.txt') as f:
# #      x.append(re.findall('\d*?\.\d+', f.read()))
#
#
i = 0
fn = open('hidden_all_stack_20190807.txt','r')
fn = fn.read()
fn = fn.replace('Columns 0 to 9','') #替换所有的空格为,
fn = fn.replace('Columns 10 to 19','') #替换所有的空格为,
fn = fn.replace('Columns 20 to 29','') #替换所有的空格为,
fn = fn.replace('Columns 30 to 31','') #替换所有的空格为,
fn = fn.replace('1.00000e-03 *','') #替换所有的空格为,
fn = fn.replace("[torch.cuda.FloatTensor of size 1x32 (GPU 0)]","") #替换所有的空格为,
fn = fn.replace('[Variable containing:','') #替换所有的空格为,
fn = fn.replace('  ',' ') #替换所有的空格为,

fn = fn.replace("      ",' ') #替换所有的空格为,
fn = fn.replace('\n',' ') #替换所有的空格为,
fn = fn.replace(']','') #替换所有的空格为,
fn = fn.replace(', Variable containing:','') #替换所有的空格为,
print(fn)
s = open('hidden_all_stack_20190807_2.txt','a+')
s = s.write(fn)
#
# fn = open('hidden_all_stack2.txt','r')
# fn = fn.read()
# fn = fn.replace('\n',' ') #替换所有的空格为,
# s = open('hidden_all_stack2.txt','a+')
# s = s.write(fn)
#
#
# fn = open('hidden_all_stack2.txt','r')
# fn = fn.read()
# fn = fn.replace('  ',' ') #替换所有的空格为,
# s = open('hidden_all_stack2.txt','a+')
# s = s.write(fn)
#
# fn = open('hidden_all_stack2.txt','r')
# fn = fn.read()
# fn = fn.replace('\]','') #替换所有的空格为,
# s = open('hidden_all_stack2.txt','a+')
# s = s.write(fn)
#
with open('hidden_all_stack_20190807_2.txt') as f:
    for line in f:
        for num in line.split:
            print(num)
            print(i)
            x[i] = float(num)
            i = i + 1
# #with open('hidden_all', 'rb') as f:
# #    x = pickle.load(f)
#
print(np.shape(x))
# x = x[0]
#
#

# # y = []
#
# # for i in range(0, 7062048):
# #     y.append(float(x[i]))
# #
# #
# # y = np.array(y)
# # y = np.reshape(y, (32*791, -1))
# # print(np.shape(y))
# # print(y[:,275])# 0 11 22 33 44 55 66 77 88 99 110 121 132 143 154 165 176 187 198 209 220  231 242 253 264
# # y = np.delete(y, 275, axis=1)
# # y = np.delete(y, 264, axis=1)
# # y = np.delete(y, 253, axis=1)
# # y = np.delete(y, 242, axis=1)
# # y = np.delete(y, 231, axis=1)
# # y = np.delete(y, 220, axis=1)
# # y = np.delete(y, 209, axis=1)
# # y = np.delete(y, 198, axis=1)
# # y = np.delete(y, 187, axis=1)
# # y = np.delete(y, 176, axis=1)
# # y = np.delete(y, 165, axis=1)
# # y = np.delete(y, 154, axis=1)
# # y = np.delete(y, 143, axis=1)
# # y = np.delete(y, 132, axis=1)
# # y = np.delete(y, 121, axis=1)
# # y = np.delete(y, 110, axis=1)
# # y = np.delete(y, 99, axis=1)
# # y = np.delete(y, 88, axis=1)
# # y = np.delete(y, 77, axis=1)
# # y = np.delete(y, 66, axis=1)
# # y = np.delete(y, 55, axis=1)
# # y = np.delete(y, 44, axis=1)
# # y = np.delete(y, 33, axis=1)
# # y = np.delete(y, 22, axis=1)
# # y = np.delete(y, 11, axis=1)
# # y = np.delete(y, 0, axis=1)
#
y= np.reshape(x, (32,791,253))
y = np.swapaxes(y,0,2)
# ### zscore y#####
import matplotlib.pyplot as plt
from scipy import stats
from numpy import *

# print(y)
# y = np.reshape(y, (791 * 253, 1, 32))
t = range(0, 253)
# plt.plot(t, y[253 * 0: 253 * 1, 0, 1])
# y_norm = np.zeros((253, 791, 32), dtype=float)
# img_step = 252
# cnt = 0
# for num2 in range(np.shape(y)[0]):
#     cnt += 1
#     if cnt == (img_step + 1):
#         cnt = 0
#         y_norm[num2 - img_step: num2 + 1, :] = stats.zscore(y[num2 - img_step: num2 + 1, :])
#
# where_are_NaNs = isnan(y_norm)
# y_norm[where_are_NaNs] = 0
# plt.plot(t, y_norm[253 * 0: 253 * 1, 0, 1])



##### fit networks by ElasticNet ####

from sklearn.linear_model import ElasticNet
import scipy.io as sio
import numpy as np

# sub_data = np.memmap('sub_Gambling.mymemmap', dtype='float32', mode='r+', shape=(253*791,59421))

y1 = np.squeeze(y_norm)

clf = ElasticNet(alpha=0.7, l1_ratio=0.005)
components_img = np.zeros((791, 32, 59421), dtype=float)
sub_data = np.memmap('/home/qing/PycharmProjects/RAE/sub_Gambling.mymemmap', dtype='float32', mode='r+', shape=(253*791,59421))

for i in range(0, 791):
    a = y1[i * 253: (i + 1) * 253, :]
    b = sub_data[i * 253: (i + 1) * 253, :]
    clf.fit(a, b)
    components_img[i, :, :] = np.transpose(clf.coef_)

components_img_avg = np.zeros((32,59421),dtype = float)
for i in range(0,59421):
    components_img_avg[:,i] = np.mean(components_img[:,:,i],axis = 0)

### zscore y#####
import matplotlib.pyplot as plt
from scipy import stats
from numpy import *

components_img_avg_norm = np.zeros((32, 59421), dtype=float)
# print(components_img_GLM.shape)
for num2 in range(0, np.shape(components_img_avg)[0]):
    components_img_avg_norm[num2, :] = stats.zscore(components_img_avg[num2, :])

where_are_NaNs = isnan(components_img_avg_norm)
components_img_avg_norm[where_are_NaNs] = 0

#### save patterns####
# import nibabel as nib
# import pandas as pd
# from nibabel import cifti2 as ci
#
#
# image = nib.load('/home/qing/Documents/tfMRI_Emotion_preproc/100307_3T_tfMRI_EMOTION_preproc/tfMRI_EMOTION_LR_Atlas_MSMAll1.dtseries.nii')
# header = image.header
# print(header)
# print(np.shape(image.get_data()))
# #print(image.nifti_header.get_data_shape())
# #image.update_headers()
#
# image_to_write = image.get_data()
# image_to_write[:, :] = 0
# image_to_write[:32, :59421] = components_img_avg_norm
#
# print(np.shape(image_to_write))
# #print(image.nifti_header.get_data_shape())
# #print(header)
# write_img = ci.Cifti2Image(image_to_write, image.header, image.nifti_header)
# nib.save(write_img, '791sub_253back_norm_32.dtseries.nii')
