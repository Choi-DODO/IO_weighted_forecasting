# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 22:46:24 2022

@author: CHOI, Dohyeong.
"""

#%% Packages required
#Preparing Data
import csv
import numpy as np
import pandas as pd
import geopandas as gpd #geojson파일 불러올 때 사용해야함.

#Something else
import libpysal
from libpysal.examples import load_example
from libpysal import cg as geometry

#Packages for Visualization
import matplotlib.pyplot as plt
import seaborn
import contextily as ctx

#Creating or Loading Weight Matrix
import scipy.sparse
from libpysal import weights
from libpysal.weights import lat2SW, WSP, WSP2W
from libpysal.weights import Queen

#Running Regression Model
from spreg import OLS
from spreg import ML_Error_Regimes
from spreg import ML_Error
from spreg import GM_Combo
from spreg import ML_Lag

from statsmodels.formula.api import ols
import statsmodels.api as sm

from mgwr.gwr import GWR
from random import uniform
import random


#%% Import DataSet
FD_GRDP_final = pd.read_csv("FD_GRDP_input.csv", sep = ',', header = 0, index_col = 0, encoding = 'euc-kr') #'euc-kr' for files in Korean

"Final Demand Data"
cond1 = ~(FD_GRDP_final['FD'].isna())
FD_DATA = FD_GRDP_final[cond1]; FD_DATA.reset_index(inplace=True, drop = True)

"GRDP Data"
GRDP_DATA = FD_GRDP_final

"Mask for the Final Demand"
cond = np.hstack([np.array([1, 4, 7, 8, 9, 11, 18, 24, 25, 26]), np.array([1, 4, 7, 8, 9, 11, 18, 24, 25, 26])+30*1,
                 np.array([1, 4, 7, 8, 9, 11, 18, 24, 25, 26])+30*2, np.array([1, 4, 7, 8, 9, 11, 18, 24, 25, 26])+30*3,
                 np.array([1, 4, 7, 8, 9, 11, 18, 24, 25, 26])+30*4, np.array([1, 4, 7, 8, 9, 11, 18, 24, 25, 26])+30*5,
                 np.array([1, 4, 7, 8, 9, 11, 18, 24, 25, 26])+30*6, np.array([1, 4, 7, 8, 9, 11, 18, 24, 25, 26])+30*7,
                 np.array([1, 4, 7, 8, 9, 11, 18, 24, 25, 26])+30*8, np.array([1, 4, 7, 8, 9, 11, 18, 24, 25, 26])+30*9,
                 np.array([1, 4, 7, 8, 9, 11, 18, 24, 25, 26])+30*10, np.array([1, 4, 7, 8, 9, 11, 18, 24, 25, 26])+30*11,
                 np.array([1, 4, 7, 8, 9, 11, 18, 24, 25, 26])+30*12, np.array([1, 4, 7, 8, 9, 11, 18, 24, 25, 26])+30*13,
                 np.array([1, 4, 7, 8, 9, 11, 18, 24, 25, 26])+30*14, np.array([1, 4, 7, 8, 9, 11, 18, 24, 25, 26])+30*15])


"Setting variables"
FD_Y_name = 'ln_FD_rate'
GRDP_Y_name = 'ln_GRDP_rate'
FD_X_names = ['WYD_rate', 'ln_pop', 'ln_UEM', 'ln_inc']
GRDP_X_names = ['WYS_rate', 'ln_pop', 'ln_UEM']


"Dictionary for Results"
FD_Y_actual = dict(); GRDP_Y_actual = dict();

for i in range(2000, 2018, 1):
    FD_Y_actual[i] = FD_DATA.loc[FD_DATA['year'] == i, FD_Y_name]

for i in range(2000, 2018, 1):
    GRDP_Y_actual[i] = GRDP_DATA.loc[GRDP_DATA['year'] == i, GRDP_Y_name]
    

#%% Input-output weighted spatio-time series model
"GWR Model with random noise weights + time lag term"

class GWR_prediction:
    
    def __init__(self, start_year, end_year, step):
        self.start_year = start_year
        self.end_year = end_year
        self.step = step
              
        self.IO = dict();
        self.FD_Y_pred = dict(); self.GRDP_Y_pred = dict()
        self.FD_result = dict(); self.GRDP_result = dict()
        self.FD_params = dict(); self.GRDP_params = dict()
        
        
        self.FD_residual = dict(); self.GRDP_residual = dict()
        self.FD_RMSE = dict(); self.GRDP_RMSE = dict()
        self.FD_APE = dict(); self.GRDP_APE = dict()
        self.FD_MAPE = dict(); self.GRDP_MAPE = dict()
        
    def cal_pred(self, i, IO, FD_Y_pre, FD_Y_base, GRDP_Y_pre, GRDP_Y_base):
               
        """####################### first FD Calibration #######################"""
        
        "Weights Matrix"
        FD_W0 = IO[0]  #수요 기반 투입계수
        FD_W0 = np.delete(FD_W0, (cond), axis = 0) 
        FD_W0 = np.delete(FD_W0, (cond), axis = 1)

        FD_W1 = FD_W0 + np.eye(len(FD_W0))
        
        "Dependent Var."
        FD_Y_BASE = FD_Y_base[i]
        FD_Y_BASE.reset_index(inplace=True, drop = True)
        
        "Independepnt Var"
        FD_X_base = FD_DATA.loc[FD_DATA['year'] == i, FD_X_names]
        FD_X_base.reset_index(inplace=True, drop = True)
        
        "time-lag term"
        FD_Y_PRE = FD_Y_pre[i - 1]
        FD_Y_PRE.reset_index(inplace=True, drop = True)
        
        "Independepnt Var with time-lag term"
        FD_X_BASE = pd.concat([FD_Y_PRE, FD_X_base], axis = 1)
        FD_X_BASE.reset_index(inplace=True, drop = True)
        
        "Pseudo-Coordinates"
        coords = list(zip(range(len(FD_X_BASE)), range(len(FD_X_BASE))))             
        
        "Model"
        FD_model_base = GWR_W(coords, np.array(FD_Y_BASE).reshape(320, -1), np.array(FD_X_BASE).reshape(320, -1), bw = 400, W = np.transpose(FD_W0), constant = True)
        FD_result = FD_model_base.fit()
        FD_params = FD_result.params
        
        
        """####################### first GRDP Calibration #######################"""
        
        "Weights Matrix"
        GRDP_W0 = IO[1]  #공급 기반 산출계수
        GRDP_W1 = GRDP_W0 + np.eye(len(GRDP_W0))

        for n in range(len(GRDP_W0)):
            for m in range(len(GRDP_W0)):
                random.seed(n*1000 + m)
                GRDP_W0[n][m] = GRDP_W0[n][m] + uniform(0, 0.00000001)

        "Dependent Var."
        GRDP_Y_BASE = GRDP_Y_base[i]
        GRDP_Y_BASE.reset_index(inplace=True, drop = True)
        
        "Independent Var."      
        GRDP_X_base = GRDP_DATA.loc[GRDP_DATA['year'] == i, GRDP_X_names]
        GRDP_X_base.reset_index(inplace=True, drop = True)
        
        "time-lag term"
        GRDP_Y_PRE = GRDP_Y_pre[i - 1]
        GRDP_Y_PRE.reset_index(inplace=True, drop = True)
        
        "Independepnt Var with time-lag term"
        GRDP_X_BASE = pd.concat([GRDP_Y_PRE, GRDP_X_base], axis = 1)
        GRDP_X_BASE.reset_index(inplace=True, drop = True)
        
        "Pseudo-Coordinates"
        coords = list(zip(range(len(GRDP_X_BASE)), range(len(GRDP_X_BASE))))
        
        "Model"
        GRDP_model_base = GWR_W(coords, np.array(GRDP_Y_BASE).reshape(480, -1), np.array(GRDP_X_BASE).reshape(480, -1), bw = 400, W = GRDP_W0, constant = True)
        GRDP_result = GRDP_model_base.fit()
        GRDP_params = GRDP_result.params
        
        """####################### first Prediction #######################"""
        
        "Independepnt Var"
        FD_X_next = FD_DATA.loc[FD_DATA['year'] == i + 1, FD_X_names]
        FD_X_next.reset_index(inplace=True, drop = True)
        
        "Independepnt Var with time-lag term"
        FD_X_next = pd.concat([FD_Y_BASE, FD_X_next], axis = 1)
        FD_X_next.reset_index(inplace=True, drop = True)
        
        "Independepnt Var with time-lag term and constant term"
        FD_X_NEXT = sm.add_constant(FD_X_next, has_constant = 'add')

        "Forcasting Final Demand of the NEXT YEAR using parameters of BASE YEAR"
        FD_pred_next = []
        for j in range(len(FD_X_next)):
                FD_pred_next.append(np.dot(FD_params[j], FD_X_NEXT.iloc[j, :]))
                
        FD_Y_pred = pd.Series(FD_pred_next)


        "Independepnt Var"
        GRDP_X_next = GRDP_DATA.loc[GRDP_DATA['year'] == i + 1, GRDP_X_names]
        GRDP_X_next.reset_index(inplace=True, drop = True)
        
        "Independepnt Var with time-lag term"
        GRDP_X_next = pd.concat([GRDP_Y_BASE, GRDP_X_next], axis = 1)
        GRDP_X_next.reset_index(inplace=True, drop = True)
        
        "Independepnt Var with time-lag term and constant term"
        GRDP_X_NEXT = sm.add_constant(GRDP_X_next, has_constant = 'add')

        "Forcasting GRDP of the NEXT YEAR using parameters of BASE YEAR"
        GRDP_pred_next = []
        for j in range(len(GRDP_X_next)):
                GRDP_pred_next.append(np.dot(GRDP_params[j], GRDP_X_NEXT.iloc[j, :]))
                
        GRDP_Y_pred = pd.Series(GRDP_pred_next)
        
        return FD_Y_pred, GRDP_Y_pred, FD_result, GRDP_result, FD_params, GRDP_params
                      

    def IO_update(self, IO, FD_Y_base, FD_Y_next, GRDP_Y_base, GRDP_Y_next):
        
        """####################### IO matrix update #######################"""
        
        FD_Y_BASE = np.ones((480,)); FD_Y_BASE[cond] = np.nan
        FD_Y_BASE[~np.isnan(FD_Y_BASE)] = np.exp(FD_Y_base)         #!!!!!!!!!!!!!! only if Y is log transformed !!!!!!!!!!!!!!#
        FD_Y_BASE = pd.Series(FD_Y_BASE)

        FD_Y_NEXT = np.ones((480,)); FD_Y_NEXT[cond] = np.nan
        FD_Y_NEXT[~np.isnan(FD_Y_NEXT)] = np.exp(FD_Y_next)
        FD_Y_NEXT = pd.Series(FD_Y_NEXT)

        GRDP_Y_BASE = np.exp(GRDP_Y_base); GRDP_Y_NEXT = np.exp(GRDP_Y_next)        #!!!!!!!!!!!!!! only if Y is log transformed !!!!!!!!!!!!!!#

        WM_next = UPDATE(FD_Y_BASE, FD_Y_NEXT, GRDP_Y_BASE, GRDP_Y_NEXT, COEF = IO)
        WM_next.run()
        
        IO_NEXT = WM_next.new_result
        
        return IO_NEXT
        
    def run(self):
        
        for year in range(self.start_year, self.end_year + 1, 1):
            
            FD_Y_pred = dict(); GRDP_Y_pred = dict()
            FD_result = dict(); GRDP_result = dict()
            FD_params = dict(); GRDP_params = dict()
                       
            for parameter in range(1, self.step + 1, 1):
                
                if parameter == 1:
                    i = year - self.step
                    IO = IO_mat[i]
                    FD_Y_pre = FD_Y_actual
                    FD_Y_base = FD_Y_actual
                    GRDP_Y_pre = GRDP_Y_actual
                    GRDP_Y_base = GRDP_Y_actual
                    
                    FD_Y_pred[i+1], GRDP_Y_pred[i+1], FD_result[i], GRDP_result[i], FD_params[i], GRDP_params[i] = self.cal_pred(i, IO, FD_Y_pre, FD_Y_base, GRDP_Y_pre, GRDP_Y_base)
                    
                    IO_NEXT = self.IO_update(IO, FD_Y_base[i], FD_Y_pred[i+1], GRDP_Y_base[i], GRDP_Y_pred[i+1])
                    
                    
                elif parameter == 2:
                    i = year - self.step + 1
                    IO = IO_NEXT
                    FD_Y_pre = FD_Y_actual
                    FD_Y_base = FD_Y_pred
                    GRDP_Y_pre = GRDP_Y_actual
                    GRDP_Y_base = GRDP_Y_pred
                    
                    FD_Y_pred[i+1], GRDP_Y_pred[i+1], FD_result[i], GRDP_result[i], FD_params[i], GRDP_params[i] = self.cal_pred(i, IO, FD_Y_pre, FD_Y_base, GRDP_Y_pre, GRDP_Y_base)
                    
                    IO_NEXT = self.IO_update(IO, FD_Y_base[i], FD_Y_pred[i+1], GRDP_Y_base[i], GRDP_Y_pred[i+1])
                    

                else:
                    i = year - self.step + (parameter - 1)
                    IO = IO_NEXT
                    FD_Y_pre = FD_Y_pred
                    FD_Y_base = FD_Y_pred
                    GRDP_Y_pre = GRDP_Y_pred
                    GRDP_Y_base = GRDP_Y_pred
                    
                    FD_Y_pred[i+1], GRDP_Y_pred[i+1], FD_result[i], GRDP_result[i], FD_params[i], GRDP_params[i] = self.cal_pred(i, IO, FD_Y_pre, FD_Y_base, GRDP_Y_pre, GRDP_Y_base)
                    
                    IO_NEXT = self.IO_update(IO, FD_Y_base[i], FD_Y_pred[i+1], GRDP_Y_base[i], GRDP_Y_pred[i+1])
                    
            
            self.FD_Y_pred[year] = FD_Y_pred[year]; self.GRDP_Y_pred[year] = GRDP_Y_pred[year]
            self.FD_result[year-1] = FD_result[year-1]; self.GRDP_result[year-1] = GRDP_result[year-1]
            self.FD_params[year-1] = FD_params[year-1]; self.GRDP_params[year-1] = GRDP_params[year-1]
            self.IO[year] = IO_NEXT
            
            """############################# Forcasting Score #############################"""

            self.FD_residual[year] = np.exp(np.array(self.FD_Y_pred[year])) - np.exp(np.array(FD_Y_actual[year]))        
            self.GRDP_residual[year] = np.exp(np.array(self.GRDP_Y_pred[year])) - np.exp(np.array(GRDP_Y_actual[year]))
               
            
            self.FD_RMSE[year] = np.sqrt(np.mean(np.square(self.FD_residual[year])))            
            self.GRDP_RMSE[year] = np.sqrt(np.mean(np.square(self.GRDP_residual[year])))

            
            self.FD_APE[year] = np.abs(self.FD_residual[year] / np.exp(np.array(FD_Y_actual[year])))                
            self.GRDP_APE[year] = np.abs(self.GRDP_residual[year] / np.exp(np.array(GRDP_Y_actual[year])))
                
            
            self.FD_MAPE[year] = np.mean(self.FD_APE[year])
            self.GRDP_MAPE[year] = np.mean(self.GRDP_APE[year])
            
        
        
        """############################# Output merging #############################"""
        
        self.res_FD_pred = self.FD_Y_pred[self.start_year]
        for i in range(self.start_year, self.end_year, 1):
            self.res_FD_pred = np.vstack([self.res_FD_pred, self.FD_Y_pred[i+1]])
        self.res_FD_pred = np.transpose(self.res_FD_pred)
        
        
        self.res_GRDP_pred = self.GRDP_Y_pred[self.start_year]
        for i in range(self.start_year, self.end_year, 1):
            self.res_GRDP_pred = np.vstack([self.res_GRDP_pred, self.GRDP_Y_pred[i+1]])
        self.res_GRDP_pred = np.transpose(self.res_GRDP_pred)
        
        
        self.res_FD_resid = self.FD_residual[self.start_year]
        for i in range(self.start_year, self.end_year, 1):
            self.res_FD_resid = np.vstack([self.res_FD_resid, self.FD_residual[i+1]])
        self.res_FD_resid = np.transpose(self.res_FD_resid)
        
        
        self.res_GRDP_resid = self.GRDP_residual[self.start_year]
        for i in range(self.start_year, self.end_year, 1):
            self.res_GRDP_resid = np.vstack([self.res_GRDP_resid, self.GRDP_residual[i+1]])
        self.res_GRDP_resid = np.transpose(self.res_GRDP_resid)
        
        
        self.res_FD_RMSE = self.FD_RMSE[self.start_year]
        for i in range(self.start_year, self.end_year, 1):
            self.res_FD_RMSE = np.vstack([self.res_FD_RMSE, self.FD_RMSE[i+1]])
        self.res_FD_RMSE = np.transpose(self.res_FD_RMSE)
        
        
        self.res_GRDP_RMSE = self.GRDP_RMSE[self.start_year]
        for i in range(self.start_year, self.end_year, 1):
            self.res_GRDP_RMSE = np.vstack([self.res_GRDP_RMSE, self.GRDP_RMSE[i+1]])
        self.res_GRDP_RMSE = np.transpose(self.res_GRDP_RMSE)
               
        
        self.res_FD_APE = self.FD_APE[self.start_year]
        for i in range(self.start_year, self.end_year, 1):
            self.res_FD_APE = np.vstack([self.res_FD_APE, self.FD_APE[i+1]])
        self.res_FD_APE = np.transpose(self.res_FD_APE)
        
        
        self.res_GRDP_APE = self.GRDP_APE[self.start_year]
        for i in range(self.start_year, self.end_year, 1):
            self.res_GRDP_APE = np.vstack([self.res_GRDP_APE, self.GRDP_APE[i+1]])
        self.res_GRDP_APE = np.transpose(self.res_GRDP_APE)
        
        
        self.res_FD_MAPE = self.FD_MAPE[self.start_year]
        for i in range(self.start_year, self.end_year, 1):
            self.res_FD_MAPE = np.vstack([self.res_FD_MAPE, self.FD_MAPE[i+1]])
        self.res_FD_MAPE = np.transpose(self.res_FD_MAPE)
        
        
        self.res_GRDP_MAPE = self.GRDP_MAPE[self.start_year]
        for i in range(self.start_year, self.end_year, 1):
            self.res_GRDP_MAPE = np.vstack([self.res_GRDP_MAPE, self.GRDP_MAPE[i+1]])
        self.res_GRDP_MAPE = np.transpose(self.res_GRDP_MAPE)      
            
          

#%% 1-step ahead
M1 = GWR_prediction(2011, 2017, 1)
M1.run()

M1_FD_Result = dict(); M1_GRDP_Result = dict()

M1_FD_Result['result'] = M1.FD_result
M1_FD_Result['params'] = M1.FD_params
M1_FD_Result['pred'] = M1.res_FD_pred
M1_FD_Result['resid'] = M1.res_FD_resid
M1_FD_Result['RMSE'] = M1.res_FD_RMSE
M1_FD_Result['APE'] = M1.res_FD_APE
M1_FD_Result['MAPE'] = M1.res_FD_MAPE

M1_GRDP_Result['result'] = M1.GRDP_result
M1_GRDP_Result['pred'] = M1.res_GRDP_pred
M1_GRDP_Result['params'] = M1.GRDP_params
M1_GRDP_Result['resid'] = M1.res_GRDP_resid
M1_GRDP_Result['RMSE'] = M1.res_GRDP_RMSE
M1_GRDP_Result['APE'] = M1.res_GRDP_APE
M1_GRDP_Result['MAPE'] = M1.res_GRDP_MAPE

#%% 2-step ahead
M2 = GWR_prediction(2011, 2017, 2)
M2.run()

M2_FD_Result = dict(); M2_GRDP_Result = dict()


M2_FD_Result['result'] = M2.FD_result
M2_FD_Result['params'] = M2.FD_params
M2_FD_Result['pred'] = M2.res_FD_pred
M2_FD_Result['resid'] = M2.res_FD_resid
M2_FD_Result['RMSE'] = M2.res_FD_RMSE
M2_FD_Result['APE'] = M2.res_FD_APE
M2_FD_Result['MAPE'] = M2.res_FD_MAPE

M2_GRDP_Result['result'] = M2.GRDP_result
M2_GRDP_Result['pred'] = M2.res_GRDP_pred
M2_GRDP_Result['params'] = M2.GRDP_params
M2_GRDP_Result['resid'] = M2.res_GRDP_resid
M2_GRDP_Result['RMSE'] = M2.res_GRDP_RMSE
M2_GRDP_Result['APE'] = M2.res_GRDP_APE
M2_GRDP_Result['MAPE'] = M2.res_GRDP_MAPE

#%% 3-step ahead
M3 = GWR_prediction(2011, 2017, 3)
M3.run()

M3_FD_Result = dict(); M3_GRDP_Result = dict(); 


M3_FD_Result['result'] = M3.FD_result
M3_FD_Result['params'] = M3.FD_params
M3_FD_Result['pred'] = M3.res_FD_pred
M3_FD_Result['resid'] = M3.res_FD_resid
M3_FD_Result['RMSE'] = M3.res_FD_RMSE
M3_FD_Result['APE'] = M3.res_FD_APE
M3_FD_Result['MAPE'] = M3.res_FD_MAPE

M3_GRDP_Result['result'] = M3.GRDP_result
M3_GRDP_Result['pred'] = M3.res_GRDP_pred
M3_GRDP_Result['params'] = M3.GRDP_params
M3_GRDP_Result['resid'] = M3.res_GRDP_resid
M3_GRDP_Result['RMSE'] = M3.res_GRDP_RMSE
M3_GRDP_Result['APE'] = M3.res_GRDP_APE
M3_GRDP_Result['MAPE'] = M3.res_GRDP_MAPE

#%% 4-step ahead
M4_FD_Result = dict(); M4_GRDP_Result = dict()

M4 = GWR_prediction(2011, 2017, 4)
M4.run()

M4_FD_Result['result'] = M4.FD_result
M4_FD_Result['params'] = M4.FD_params
M4_FD_Result['pred'] = M4.res_FD_pred
M4_FD_Result['resid'] = M4.res_FD_resid
M4_FD_Result['RMSE'] = M4.res_FD_RMSE
M4_FD_Result['APE'] = M4.res_FD_APE
M4_FD_Result['MAPE'] = M4.res_FD_MAPE

M4_GRDP_Result['result'] = M4.GRDP_result
M4_GRDP_Result['pred'] = M4.res_GRDP_pred
M4_GRDP_Result['params'] = M4.GRDP_params
M4_GRDP_Result['resid'] = M4.res_GRDP_resid
M4_GRDP_Result['RMSE'] = M4.res_GRDP_RMSE
M4_GRDP_Result['APE'] = M4.res_GRDP_APE
M4_GRDP_Result['MAPE'] = M4.res_GRDP_MAPE


#%%
for i in range(2010, 2017, 1):
    print(M1_FD_Result['result'][i].summary())
    
for i in range(2010, 2017, 1):
    print(M1_GRDP_Result['result'][i].summary())
    

#%%
io_1 = dict() ; io_2 = dict()
io_3 = dict() ; io_4 = dict()

for i in range(2011, 2018, 1):
    io_1[i] = M1.IO[i][0]
    io_2[i] = M1.IO[i][1]
    io_3[i] = M1.IO[i][2]
    io_4[i] = M1.IO[i][3]
      

#%% singular matrix into non-singular matrix: adding random noise to each component

from random import seed
from random import random

import copy
import matplotlib.pyplot as plt
import seaborn as sns #visualization

seed(1)

for i in range(10):
    value = random()
    print(value)

a = list(range(2))
mat1 = a
mat0 = a
mat0[0] = mat0[0]+1

#generate random noise 0~0.00001
mat_0 = copy.deepcopy(I.output[2008][0])

GRDP_W0 = copy.deepcopy(GRDP_W0)
np.mean(GRDP_W0) - np.std(GRDP_W0)
out =  list(range(100))
for k in list(range(100)):
    for i in range(len(GRDP_W0)):
        for j in range(len(GRDP_W0)):
            GRDP_W0[i][j] = GRDP_W0[i][j] + uniform(0, 0.00001)
    a = np.linalg.inv(GRDP_W0) - np.linalg.pinv(GRDP_W0)
    out[k] = a.sum()/480

sns.distplot(out, hist=False, kde=True) 
np.mean(out)
np.std(out)
plt.hist(out)

#%% Visual: FD_Coefficient - line graph

year = 2010
model = M1_FD_Result['params']


fig = plt.figure(figsize=(15,3))
fig.set_facecolor('white')

ax1 = fig.add_subplot()
ax1.set_xlabel('Sectors')
ax1.set_ylabel('Coefficient')

line1 = ax1.plot(np.transpose(model[year])[1], label = 'time-lag')
line2 = ax1.plot(np.transpose(model[year])[2], label = 'spatio temporal-lag')
line3 = ax1.plot(np.transpose(model[year])[3], label = 'ln_POP')
line4 = ax1.plot(np.transpose(model[year])[4], label = 'ln_UEM')
line5 = ax1.plot(np.transpose(model[year])[5], label = 'ln_DINC')

lines = line1 + line2 + line3 + line4 + line5
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, ncol = 5, loc='upper left', bbox_to_anchor = (0,-0.2))

ax1.axvspan(0,19, facecolor='gray', alpha=0.3)       
ax1.axvspan(0+20*2,19+20*2, facecolor='gray', alpha=0.3)
ax1.axvspan(0+20*4,19+20*4, facecolor='gray', alpha=0.3) 
ax1.axvspan(0+20*6,19+20*6, facecolor='gray', alpha=0.3) 
ax1.axvspan(0+20*8,19+20*8, facecolor='gray', alpha=0.3) 
ax1.axvspan(0+20*10,19+20*10, facecolor='gray', alpha=0.3) 
ax1.axvspan(0+20*12,19+20*12, facecolor='gray', alpha=0.3) 
ax1.axvspan(0+20*14,19+20*14, facecolor='gray', alpha=0.3) 

plt.title('Coefficient of FD model in 2010')
plt.show()


#%% Visual: GRDP_Coefficient - line graph

year = 2016
model = M1_GRDP_Result['params']

fig = plt.figure(figsize=(15,3))
fig.set_facecolor('white')

ax1 = fig.add_subplot()
ax1.set_xlabel('Sectors')
ax1.set_ylabel('Coefficient')

line1 = ax1.plot(np.transpose(model[year])[1], label = 'time-lag')
line2 = ax1.plot(np.transpose(model[year])[2], label = 'weighted-lag')
line3 = ax1.plot(np.transpose(model[year])[3], label = 'ln_POP')
line4 = ax1.plot(np.transpose(model[year])[4], label = 'ln_UEM')

lines = line1 + line2 + line3 + line4
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, ncol = 4, loc='upper left', bbox_to_anchor = (0,-0.1))

ax1.axvspan(0,29, facecolor='gray', alpha=0.3)       
ax1.axvspan(0+30*2,29+30*2, facecolor='gray', alpha=0.3)
ax1.axvspan(0+30*4,29+30*4, facecolor='gray', alpha=0.3) 
ax1.axvspan(0+30*6,29+30*6, facecolor='gray', alpha=0.3) 
ax1.axvspan(0+30*8,29+30*8, facecolor='gray', alpha=0.3) 
ax1.axvspan(0+30*10,29+30*10, facecolor='gray', alpha=0.3) 
ax1.axvspan(0+30*12,29+30*12, facecolor='gray', alpha=0.3) 
ax1.axvspan(0+30*14,29+30*14, facecolor='gray', alpha=0.3) 

plt.title('Coefficient of GRDP model in 2016')
plt.show()


#%% Visual: FD_Coefficient - heat map

year = 2016
model = M1_FD_Result['params']
variable = 5

coef = np.transpose(model[year])[variable]
data = coef.reshape(20, -1)

plt.figure(figsize = (10, 5))
plt.gray()
plt.pcolor(data, cmap = cm.coolwarm)
plt.colorbar()

plt.xlabel('region')
plt.ylabel('industry')
#plt.title('temporal lag Coefficient of FD model in 2010')
plt.show()

#%% Visual: GRDP_Coefficient - heat map
for year in range(2010, 2017, 1):
    model = M1_GRDP_Result['params']
    variable = 4

    coef = np.transpose(model[year])[variable]
    data = coef.reshape(30, -1)

    plt.figure(figsize = (10, 5))
    plt.gray()
    plt.pcolor(data, cmap = cm.coolwarm)
    plt.colorbar()

    plt.xlabel('region')
    plt.ylabel('industry')
    #plt.title('temporal lag Coefficient of GRDP model in 2010')
    plt.show()
    

#%% Visual: FD_dependent Variable - heat map


year = 2012

data = np.array(FD_Y_actual[year])


x = np.arange(0, 16, 1) 
y = np.arange(0, 20, 1)

temp = np.zeros((len(x), len(y)))
for i in range(len(x)):
    for j in range(len(y)):
        temp[i, j] = data[j+i*20]

plt.figure(figsize = (10, 4))
plt.gray()
plt.pcolor(temp, cmap = cm.coolwarm)
plt.colorbar()

plt.xlabel('Industry')
plt.ylabel('Region')
#plt.title('FD value in 2011')
plt.show()


#%% Visual: GRDP_dependent Variable - heat map


year = 2012

data = np.array(GRDP_Y_actual[year])


x = np.arange(0, 16, 1) 
y = np.arange(0, 30, 1)

temp = np.zeros((len(x), len(y)))
for i in range(len(x)):
    for j in range(len(y)):
        temp[i, j] = data[j+i*30]

plt.figure(figsize = (10, 4))
plt.gray()
plt.pcolor(temp, cmap = cm.coolwarm)
plt.colorbar()

plt.xlabel('Industry')
plt.ylabel('Region')
#plt.title('GRDP value in 2011')
plt.show()


#%% Visual: FD full-step ahead full year 2-D GRAPH
#initial unit = 10억원

model1 = M1_FD_Result['pred']
model2 = M2_FD_Result['pred']
model3 = M3_FD_Result['pred']
model4 = M4_FD_Result['pred']

fig = plt.figure(figsize=(10,2))
fig.set_facecolor('white')

ax1 = fig.add_subplot()
ax1.set_xlabel('Year')
ax1.set_ylabel('trillion USD')

year = [2011, 2012, 2013, 2014, 2015, 2016, 2017]

FD_actual = np.array(FD_Y_actual[2011])
for i in range(2012, 2018, 1):
    FD_actual = np.vstack((FD_actual, np.array(FD_Y_actual[i])))

line1 = ax1.plot(year, np.exp(FD_actual).sum(axis = 1)/1000000, color = 'deeppink', label = 'FD_actual')
line2 = ax1.plot(year, np.exp(model1).sum(axis = 0)/1000000, color = 'gray', linestyle = '-', label = '1-step ahead')
line3 = ax1.plot(year, np.exp(model2).sum(axis = 0)/1000000, color = 'gray', linestyle = '-.', label = '2-step ahead')
line4 = ax1.plot(year, np.exp(model3).sum(axis = 0)/1000000, color = 'gray', linestyle = '--', label = '3-step ahead')
line5 = ax1.plot(year, np.exp(model4).sum(axis = 0)/1000000, color = 'gray', linestyle = ':', label = '4-step ahead')

lines = line1 + line2 + line3 + line4 + line5
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, bbox_to_anchor = (1, 1), loc='upper left')

plt.title('FD Forecasting for each industrial sector')
plt.show()




#%% Visual: GRDP full-step ahead full year 2-D GRAPH
#initial unit = 10억원

model1 = M1_GRDP_Result['pred']
model2 = M2_GRDP_Result['pred']
model3 = M3_GRDP_Result['pred']
model4 = M4_GRDP_Result['pred']

fig = plt.figure(figsize=(10,2))
fig.set_facecolor('white')

ax1 = fig.add_subplot()
ax1.set_xlabel('Year')
ax1.set_ylabel('trillion USD')

year = [2011, 2012, 2013, 2014, 2015, 2016, 2017]

GRDP_actual = np.array(GRDP_Y_actual[2011])
for i in range(2012, 2018, 1):
    GRDP_actual = np.vstack((GRDP_actual, np.array(GRDP_Y_actual[i])))

line1 = ax1.plot(year, np.exp(GRDP_actual).sum(axis = 1)/1000000, color = 'orange', label = 'GRDP_actual')
line2 = ax1.plot(year, np.exp(model1).sum(axis = 0)/1000000, color = 'gray', linestyle = '-', label = '1-step ahead')
line3 = ax1.plot(year, np.exp(model2).sum(axis = 0)/1000000, color = 'gray', linestyle = '-.', label = '2-step ahead')
line4 = ax1.plot(year, np.exp(model3).sum(axis = 0)/1000000, color = 'gray', linestyle = '--', label = '3-step ahead')
line5 = ax1.plot(year, np.exp(model4).sum(axis = 0)/1000000, color = 'gray', linestyle = ':', label = '4-step ahead')

lines = line1 + line2 + line3 + line4 + line5
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, bbox_to_anchor = (1, 1), loc='upper left')

plt.title('GRDP Forecasting for each industrial sector')
plt.show()


#%% Visual: FD full-step ahead full year 2-D GRAPH
#initial unit = 10억원

model1 = M1_FD_Result['pred']
model2 = M2_FD_Result['pred']
model3 = M3_FD_Result['pred']
model4 = M4_FD_Result['pred']

fig = plt.figure(figsize=(10,2))
fig.set_facecolor('white')

ax1 = fig.add_subplot()
ax1.set_xlabel('Year')
ax1.set_ylabel('log Growth Rate')

year = [2011, 2012, 2013, 2014, 2015, 2016, 2017]

FD_actual = np.array(FD_Y_actual[2011])
for i in range(2012, 2018, 1):
    FD_actual = np.vstack((FD_actual, np.array(FD_Y_actual[i])))

line1 = ax1.plot(year, FD_actual.mean(axis = 1), color = 'deeppink', label = 'FD_actual')
line2 = ax1.plot(year, model1.mean(axis = 0), color = 'gray', linestyle = '-', label = '1-step ahead')
line3 = ax1.plot(year, model2.mean(axis = 0), color = 'gray', linestyle = '-.', label = '2-step ahead')
line4 = ax1.plot(year, model3.mean(axis = 0), color = 'gray', linestyle = '--', label = '3-step ahead')
line5 = ax1.plot(year, model4.mean(axis = 0), color = 'gray', linestyle = ':', label = '4-step ahead')

lines = line1 + line2 + line3 + line4 + line5
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, bbox_to_anchor = (1, 1), loc='upper left')

plt.ylim([-0.5, 0.5])
plt.title('FD Forecasting for each industrial sector')
plt.show()

#%% Visual: GRDP full-step ahead full year 2-D GRAPH
#initial unit = 10억원

model1 = M1_GRDP_Result['pred']
model2 = M2_GRDP_Result['pred']
model3 = M3_GRDP_Result['pred']
model4 = M4_GRDP_Result['pred']

fig = plt.figure(figsize=(10,2))
fig.set_facecolor('white')

ax1 = fig.add_subplot()
ax1.set_xlabel('Year')
ax1.set_ylabel('log Growth Rate')

year = [2011, 2012, 2013, 2014, 2015, 2016, 2017]

GRDP_actual = np.array(GRDP_Y_actual[2011])
for i in range(2012, 2018, 1):
    GRDP_actual = np.vstack((GRDP_actual, np.array(GRDP_Y_actual[i])))

line1 = ax1.plot(year, GRDP_actual.mean(axis = 1), color = 'orange', label = 'GRDP_actual')
line2 = ax1.plot(year, model1.mean(axis = 0), color = 'gray', linestyle = '-', label = '1-step ahead')
line3 = ax1.plot(year, model2.mean(axis = 0), color = 'gray', linestyle = '-.', label = '2-step ahead')
line4 = ax1.plot(year, model3.mean(axis = 0), color = 'gray', linestyle = '--', label = '3-step ahead')
line5 = ax1.plot(year, model4.mean(axis = 0), color = 'gray', linestyle = ':', label = '4-step ahead')

lines = line1 + line2 + line3 + line4 + line5
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, bbox_to_anchor = (1, 1), loc='upper left')

plt.ylim([-0.5, 0.5])

plt.title('GRDP Forecasting for each industrial sector')
plt.show()





#%%% Visual: FD APE 3-D GRAPH out of GWR model

model = M1_FD_Result['APE']
year = 2017

fig = plt.figure(figsize=(10, 5))
fig.set_facecolor('white')
cx = fig.add_subplot(111, projection = '3d')

#x축이 열이 된다.
x = np.arange(1, 21, 1) 
y = np.arange(1, 17, 1)

temp = np.zeros((len(y), len(x)))
for i in range(len(y)):
    for j in range(len(x)):
        temp[i, j] = model.loc[:,year][20*i+j]

        
xx, yy = np.meshgrid(x, y)

cx.plot_surface(xx, yy, temp, rstride = 1, cstride = 1, alpha = 0.3, color = 'deeppink', edgecolor = 'black')

#1st parameter vertical degree (0:from side, 90:from sky)
#2nd parameter horizontal degree (+: clockwise, -:reverse)
cx.view_init(30, 90)
plt.title('FD APE 2017 from GWR model')
plt.show()
