print('>>> Loading Libraries')    
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm,tqdm_notebook
import pickle
import time
s0 = time.time()
import matplotlib.pyplot as plt
import csv
import os
import numpy as np
from sklearn.decomposition import PCA

data_dir = './combined_data/'
pca_dir = './PCA/'
test_data = pd.read_csv('./Dataset/test_techsoc.csv')
print('>>> Loaded Test Dataset')

name= input('Enter the model name : ')
with open('./models/{}.pickle'.format(name),'rb') as f:
    models = pickle.load(f)
print('>>> Done Loading Models')

with open('./PCA/velocity.pickle','rb') as f:
    velocity_pca = pickle.load(f)
print('>>> Done Loading PCA model for Velocity')

with open('./PCA/position.pickle','rb') as f:
    position_pca = pickle.load(f)
print('>>> Done Loading PCA model for Position')

test_sats = test_data['sat_id'].unique()

pca_transform_test = {}
# for x in tqdm(test_sats):
#     pca_model = position_pca[str(x)][]
#     y_pred = model
predicted_ = {}
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def inv_sigmoid(y):
    return(np.log(y/(1-y)))
def retrei(n_sat,pred,pca):
    temp =  np.dot(pred,pca[str(n_sat)]['pca_act'].components_) 
    return((pca[str(n_sat)]['scalar_act'].inverse_transform(temp)))

Final_soln = pd.DataFrame({'id':[],'x':[],'y':[],'z':[],'Vx':[],'Vy':[],'Vz':[]})

for i in tqdm(test_sats):
    tsd = test_data[test_data['sat_id'] == int(i)]
    n = len(tsd)
    sub_id = list(tsd.id)
    sc_data = position_pca[str(i)]['scalar_train'].transform(tsd[['x_sim','y_sim','z_sim']])
    pcs_sc_data =  position_pca[str(i)]['pca_train'].transform(sc_data)
    pcs_sc_data_1 = [x[0] for x in pcs_sc_data]
    pcs_sc_data_2 = [x[1] for x in pcs_sc_data]


    vsc_data = velocity_pca[str(i)]['scalar_train'].transform(tsd[['Vx_sim','Vy_sim','Vz_sim']])
    vpcs_sc_data =  velocity_pca[str(i)]['pca_train'].transform(vsc_data)
    vpcs_sc_data_1 = [x[0] for x in vpcs_sc_data]
    vpcs_sc_data_2 = [x[1] for x in vpcs_sc_data]

    training_data = pd.DataFrame({'1':pcs_sc_data_1,'2':pcs_sc_data_2,'3':vpcs_sc_data_1,'4':vpcs_sc_data_2})

    xsc_model = models[str(i)]['x']
    pred_xsc = xsc_model.predict(sigmoid(training_data))
    pred_xsc = inv_sigmoid(pred_xsc)
    ysc_model = models[str(i)]['y']
    pred_ysc = ysc_model.predict(sigmoid(training_data))
    pred_ysc = inv_sigmoid(pred_ysc)
    Vxsc_model = models[str(i)]['Vx']
    pred_Vxsc = Vxsc_model.predict(sigmoid(training_data))
    pred_Vxsc = inv_sigmoid(pred_Vxsc)
    Vysc_model = models[str(i)]['Vy']
    pred_Vysc = Vysc_model.predict(sigmoid(training_data))
    pred_Vysc  = inv_sigmoid(pred_Vysc)

    set_pos = np.array([[pred_xsc[i],pred_ysc[i]] for i in range(len(pred_xsc))])
    set_vel = np.array([[pred_Vxsc[i],pred_Vysc[i]] for i in range(len(pred_Vxsc))])

    pos_z = retrei(str(i),set_pos,pca = position_pca)
    vel_z = retrei(str(i),set_vel,pca = velocity_pca)
    df_pos = pd.DataFrame(pos_z,columns=['x','y','z'])
    df_vel = pd.DataFrame(vel_z,columns=['Vx','Vy','Vz'])

    final = pd.concat([df_pos,df_vel],axis = 1)
    final['id'] = sub_id
    final = final[['id','x','y','z','Vx','Vy','Vz']]
    Final_soln = pd.concat([Final_soln,final],axis = 0)


output_name = input('Enter the output file name: ')
Final_soln.to_csv('./Submissions/{}.csv'.format(output_name))
print('>>> Total Time Taken is {:.2f} seconds'.format(time.time()-s0))
print(len(test_data))
print(len(Final_soln))