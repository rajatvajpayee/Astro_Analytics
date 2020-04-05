import time
s0 = time.time()
print('>>> Loading Libraries')    
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm,tqdm_notebook

import matplotlib.pyplot as plt
import csv
import os
import numpy as np
from sklearn.decomposition import PCA

_DIR_ = './Dataset/'
output_dir = './Exploration/'

# Loading Training
print('>>> Loading Training Data ....')
train_data = pd.read_csv(_DIR_+'train_techsoc.csv')
# train_data.head()

# Loading Test Data
test_data = pd.read_csv('test_techsoc.csv')
# test_data.head()
print('>>> Data Loaded ...')

cols = ['id','sat_id','epoch','x_sim','y_sim','z_sim','Vx_sim','Vy_sim','Vz_sim']
combined = pd.concat([train_data[cols],test_data],axis=0)
combined.head()

x_train = train_data[['sat_id','x_sim','y_sim','z_sim']]
x_act = train_data[['sat_id','x','y','z']]
Vx_act = train_data[['sat_id','Vx','Vy','Vz']]
pca_trans_position= {}
n = len(train_data.sat_id.unique())
# n = 1
print('>>> Processing PCA for positions ...')    

for k in tqdm(range(n)):
    tt = combined[combined.sat_id == k]
    ttt = tt[['x_sim','y_sim','z_sim']]
    temp_train = ttt
    temp_act = x_act[x_act.sat_id == int(k)]
    n = len(temp_act)
    scalar_train = StandardScaler() 
    scalar_train.fit(ttt)
    scaled_data_train = scalar_train.transform(ttt)
    
    scalar_act = StandardScaler() 
    scalar_act.fit(temp_act[temp_act.columns[1:]])
    scaled_data_act = scalar_act.transform(temp_act[temp_act.columns[1:]])
    
    pca_train = PCA(n_components = 2) 
    pca_act = PCA(n_components = 2) 
    
    pca_train.fit(scaled_data_train)
    x_pca_train = pca_train.transform(scaled_data_train)     

    pca_act.fit(scaled_data_act)
    x_pca_act = pca_act.transform(scaled_data_act)     
    
    pca_trans_position[str(k)] = {'pca_train':pca_train
                     ,'pca_act':pca_act
                     ,'scalar_train':scalar_train
                     ,'scalar_act':scalar_act
                     ,'x_pca_train':x_pca_train[:n]
                     ,'x_pca_act':x_pca_act}
    
print('>>> Done processing')    

pca_trans_velocity = {}
n = len(train_data.sat_id.unique())
# n = 1
print('>>> Processing PCA for Velocities ...')
for k in tqdm(range(n)):
    tt = combined[combined.sat_id == k]
    ttt = tt[['Vx_sim','Vy_sim','Vz_sim']]
    temp_train = ttt
    temp_act = Vx_act[x_act.sat_id == int(k)]
    n = len(temp_act)
    scalar_train = StandardScaler() 
    scalar_train.fit(ttt)
    scaled_data_train = scalar_train.transform(ttt)
    
    scalar_act = StandardScaler() 
    scalar_act.fit(temp_act[['Vx','Vy','Vz']])
    scaled_data_act = scalar_act.transform(temp_act[temp_act.columns[1:]])
    
    pca_train = PCA(n_components = 2) 
    pca_act = PCA(n_components = 2) 
    
    pca_train.fit(scaled_data_train)
    x_pca_train = pca_train.transform(scaled_data_train)     

    pca_act.fit(scaled_data_act)
    x_pca_act = pca_act.transform(scaled_data_act)     
    
    pca_trans_velocity[str(k)] = {'pca_train':pca_train
                     ,'pca_act':pca_act
                     ,'scalar_train':scalar_train
                     ,'scalar_act':scalar_act
                     ,'x_pca_train':x_pca_train[:n]
                     ,'x_pca_act':x_pca_act}
print('>>> Done processing')    

print('>>> Now, Preparing combined data frame ...')

datafram_ = {}
import time
start = time.time()
for i in tqdm(range(600)):
    temp = pca_trans_position[str(i)]
    _act_ = temp['x_pca_act']
    _train_ = temp['x_pca_train']
    _x_train = {'x':[x[0] for x in _train_],'y':[x[1] for x in _train_],'act_x':[x[0] for x in _act_]}
    _y_train = {'x':[x[0] for x in _train_],'y':[x[1] for x in _train_],'act_y':[x[1] for x in _act_]}

    Vtemp = pca_trans_velocity[str(i)]
    V_act_ = Vtemp['x_pca_act']
    V_train_ = Vtemp['x_pca_train']
    V_x_train = {'x':[x[0] for x in V_train_],'y':[x[1] for x in V_train_],'act_x':[x[0] for x in V_act_]}
    V_y_train = {'x':[x[0] for x in V_train_],'y':[x[1] for x in V_train_],'act_y':[x[1] for x in V_act_]}

    datafram_[str(i)] = {'sat_id':[i]*len(temp['x_pca_act']),
                            'x':_x_train['act_x'],'x_sim':_x_train['x'],
                            'y':_y_train['act_y'],'y_sim':_y_train['y'],
                            'Vx':V_x_train['act_x'],'Vx_sim':V_x_train['x'],
                            'Vy':V_y_train['act_y'],'Vy_sim':V_y_train['y'],
                        }

print('>>> Completed in {:.2f} seconds'.format(time.time()-start))

import pickle
with open('./PCA/position.pickle', 'wb') as f:
    pickle.dump(pca_trans_position, f)
print('>>> Saving PCA for position at {}'.format('./PCA/position.pickle'))
with open('./PCA/velocity.pickle', 'wb') as f:
    pickle.dump(pca_trans_velocity, f)
print('>>> Saving PCA for velocity at {}'.format('./PCA/velocity.pickle'))

with open('./combined_data/combined_data_after_pca.pickle', 'wb') as f:
    pickle.dump(datafram_, f)
print('>>> Saving combined data after PCA at {}'.format('./combined_data/combined_data_after_pca.pickle'))

data_dir = './combined_data/'
pca_dir = './PCA/'

print('>>> Combining the data ...')
combi_data = pd.DataFrame({'sat_id':[],'x':[],'y':[],'x_sim':[],'y_sim':[],'Vx':[],'Vy':[],'Vx_sim':[],'Vy_sim':[]})
for i in tqdm(range(600)):
    temp = datafram_[str(i)]
    combi_data = pd.concat([combi_data,pd.DataFrame({'sat_id':[int(x) for x in temp['sat_id']]
                        ,'x':temp['x'],'y':temp['y'],'x_sim':temp['x_sim'],'y_sim':temp['y_sim'],
                        'Vx':temp['Vx'],'Vy':temp['Vy'],'Vx_sim':temp['Vx_sim'],'Vy_sim':temp['Vy_sim']})],axis = 0,ignore_index=True)

print('>>> Writing combined data into csv ... ')
combi_data.to_csv('./combined_data/combi_data.csv')
print('>>> CSV saved in combined_data folder.')
print('>>> Total processing done in {:.2f} seconds'.format(time.time()-s0))

