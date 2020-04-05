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
data = pd.read_csv('./combined_data/combi_data.csv')
print('>>> Loading Completed')
model = {}
from sklearn.linear_model import LinearRegression,Lasso,ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.linear_model import Ridge
mse_ = []
output_file_name = input('Enter output File name txt : ')
 
print('>>> Training Model ...')
import math

def sigmoid(x):
  return 1 / (1 + np.exp(-x))
file1 = open("./models/{}.txt".format(output_file_name),"w")
st = 'distance'
for i in tqdm(range(600)):
    
    file1.write('-------------------------------------------------------------------------------------------------------\n')
    file1.write('>>> Training for satalite {}\n'.format(i))
    temp = data[data.sat_id == i]
    file1.write('>>> Length {}\n'.format(len(temp)))
    xx_train,xx_test,xy_train,xy_test = train_test_split(temp[['x_sim','y_sim','Vx_sim','Vy_sim']],temp['x'], test_size=0.05, random_state=42)
    yx_train,yx_test,yy_train,yy_test = train_test_split(temp[['x_sim','y_sim','Vx_sim','Vy_sim']],temp['y'], test_size=0.05, random_state=42)    
    Vxx_train,Vxx_test,Vxy_train,Vxy_test = train_test_split(temp[['x_sim','y_sim','Vx_sim','Vy_sim']],temp['Vx'], test_size=0.05, random_state=42)    
    Vyx_train,Vyx_test,Vyy_train,Vyy_test = train_test_split(temp[['x_sim','y_sim','Vx_sim','Vy_sim']],temp['Vy'], test_size=0.05, random_state=42)
    
    model_x = KNeighborsRegressor(n_neighbors=2,weights=st) ; model_y =  KNeighborsRegressor(n_neighbors=2,weights=st)
    model_Vx =  KNeighborsRegressor(n_neighbors=2,weights=st) ; model_Vy =  KNeighborsRegressor(n_neighbors=2,weights=st)

    file1.write('>>> Fitting Model for predicting position ...\n')
    # xx_train['x_sim'] = sigmoid(np.array(xx_train['x_sim'].tolist() ))
    # xx_train['y_sim'] = sigmoid(np.array(xx_train['y_sim'].tolist() ))
    # xx_train['Vx_sim'] = sigmoid(np.array(xx_train['Vx_sim'].tolist() ))
    # xx_train['Vy_sim'] = sigmoid(np.array(xx_train['Vy_sim'].tolist() ))
    # xy_train = sigmoid(np.array(xy_train))

    # yx_train['x_sim'] = sigmoid(np.array(yx_train['x_sim'].tolist() ))
    # xx_train['y_sim'] = sigmoid(np.array(xx_train['y_sim'].tolist() ))
    # xx_train['Vx_sim'] = sigmoid(np.array(xx_train['Vx_sim'].tolist() ))
    # xx_train['Vy_sim'] = sigmoid(np.array(xx_train['Vy_sim'].tolist() ))
    # xy_train = sigmoid(np.array(xy_train))


    model_x.fit(sigmoid(xx_train),sigmoid(xy_train))
    y_pred = model_x.predict(sigmoid(xx_test))
    file1.write('>>> MSE for predicting x coordinate is {:.4f}\n'.format(mean_squared_error(sigmoid(xy_test),model_x.predict(sigmoid(xx_test)))))
    model_y.fit(sigmoid(yx_train),sigmoid(yy_train))
    y_pred = model_y.predict(sigmoid(yx_test))
    file1.write('>>> MSE for predicting y coordinate is {:.4f}\n'.format(mean_squared_error(sigmoid(yy_test),model_y.predict(sigmoid(yx_test)))))
    
    file1.write('>>> Fitting Model for predicting Velocity ...\n')
    model_Vx.fit(sigmoid(Vxx_train),sigmoid(Vxy_train))
    y_pred = model_Vx.predict(sigmoid(Vxx_test))
    file1.write('>>> MSE for predicting x coordinate is {:.4f}\n'.format(mean_squared_error(sigmoid(Vxy_test),model_Vx.predict(sigmoid(Vxx_test)))))
    model_Vy.fit(sigmoid(Vyx_train),sigmoid(Vyy_train))
    y_pred = model_Vy.predict(sigmoid(Vyx_test))
    file1.write('>>> MSE for predicting y coordinate is {:.4f}\n'.format(mean_squared_error(sigmoid(Vyy_test),model_Vy.predict(sigmoid(Vyx_test)))))
    model[str(i)] = {'x':model_x,
                    'y':model_y,
                    'Vx':model_Vx,
                    'Vy':model_Vy}
    mse_.append((mean_squared_error(sigmoid(xy_test),model_x.predict(sigmoid(xx_test)))+mean_squared_error(sigmoid(yy_test),model_y.predict(sigmoid(yx_test)))+
                        mean_squared_error(sigmoid(Vxy_test),model_Vx.predict(sigmoid(Vxx_test)))+ mean_squared_error(sigmoid(Vyy_test),model_Vy.predict(sigmoid(Vyx_test))))/4)
print('Average MSE : {}'.format(mse_[-1]))

print('>>> Completed Training ')
print('>>> Saving models')
model_name = input('Enter model name : ')
with open('./models/{}.pickle'.format(model_name), 'wb') as f:
    pickle.dump(model, f)
print('>>> Saved model')
print('>>> Output for trained model stored')
plt.figure(figsize = (10,8))
print('Pipleline Completed in {:.2f}'.format(time.time()-s0))
