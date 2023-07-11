# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 13:52:04 2023

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 10:29:11 2022

@author: CHOI, Dohyeong.
"""
#%%
import numpy as np
import pandas as pd

"""
Z(array-like): 기준연도 중간재 투입 행렬 from IO matrix (R*I) by (R*I)
X(array-like): 기준연도 총산출액 from IO matrix (R*I) by 1
FD(DataFrame with colnames specifying years): 분석기간 지역/산업별 민간최종지출  (R*I) by T
GP(DataFrame with colnames specifying years): 분석기간 지역/산업별 민간최종지출  (R*I) by T
where
R = the # of regions (16)
I = the # of industries (30)
T = the # of years (in this case, 1992~2016)
"""


"""
<Calculating Coefficients>

input : 투입산출표의 중간거래(Z), 총산출(X)
output : 4가지 계수 행렬
    투입계수 : tech_coef
    산출계수 : B_coef
    생산유발계수 : D_inv_coef
    산출분배계수 : S_inv_coef
"""
class IO_COEF:
    def __init__(self, Z, X):
        self.Z = Z
        self.X = X
    
    def run(self):
        I = np.identity(len(self.X))
        
        diag_X = np.diag(self.X)
        inv_X = np.linalg.pinv(diag_X)
        tech_coef = self.Z@inv_X
        B_coef = inv_X@self.Z
        
        D_inv_coef = np.linalg.pinv(I - tech_coef)
        S_inv_coef = np.linalg.pinv(I - B_coef)
        D_inv_coef[D_inv_coef<0] = 0
        S_inv_coef[S_inv_coef<0] = 0
        
        self.tech_coef = tech_coef
        self.B_coef = B_coef
        self.D_inv_coef = D_inv_coef
        self.S_inv_coef = S_inv_coef
        
        self.result = [tech_coef, B_coef, D_inv_coef, S_inv_coef]
        self.names = ['tech_coef', 'B_coef', 'D_inv_coef', 'S_inv_coef']

"""
<Updating Coefficients>

input :
    투입산출표의 중간거래(Z), 총산출(X) 혹은 4가지 Coefficients
    기준연도 FD, 대상연도 FD
    기준연도 GRDP, 대상연도 GRDP
    
output : 4가지 계수 행렬
    기준&대상연도 투입계수 : tech_coef
    기준&대상연도 산출계수 : B_coef
    기준&대상연도 생산유발계수 : D_inv_coef
    기준&대상연도 산출분배계수 : S_inv_coef
"""
class UPDATE:
    
    """투입산출표 혹은 IO_COEF 클래스의 RESULT 투입 가능"""
    def __init__(self, BASE_FD, NEW_FD, BASE_GRDP, NEW_GRDP, Z = None, X = None, COEF = None):
        self.Z = Z
        self.X = X
        self.COEF = COEF
        
        self.BASE_FD = np.array(BASE_FD).reshape(-1, 1)
        self.NEW_FD = np.array(NEW_FD).reshape(-1, 1)
        self.BASE_GRDP = np.array(BASE_GRDP).reshape(-1, 1)
        self.NEW_GRDP = np.array(NEW_GRDP).reshape(-1, 1)
    
    def run(self):
        """기준연도 계수 계산"""
        if self.COEF == None:
            base = IO_COEF(self.Z, self.X)
            base.run()
            
            base_tech_coef = base.result[0]
            base_B_coef = base.result[1]
            base_D_inv_coef = base.result[2]
            base_S_inv_coef = base.result[3]
        
        elif self.COEF != None:
            base_tech_coef = self.COEF[0]
            base_B_coef = self.COEF[1]
            base_D_inv_coef = self.COEF[2]
            base_S_inv_coef = self.COEF[3]
        

        """계수 갱신"""
        "투입계수 및 산출계수 계산"
        PFD_A = self.NEW_FD / self.BASE_FD
        PFD_A[np.isnan(PFD_A)] = 1

        PGP_A = self.NEW_GRDP / self.BASE_GRDP
        PGP_A[np.isnan(PGP_A)] = 1
        PGP_A[np.isin(PGP_A, [0])] = 1

        PFD_N = base_D_inv_coef@PFD_A
        PGP_N = np.transpose(PGP_A)@base_S_inv_coef


        R = np.diag(np.array(PFD_A/np.transpose(PGP_N)).reshape(480,))
        S = np.diag(np.array(PGP_A/PFD_N).reshape(480,))

        new_tech_coef = R@base_tech_coef@S
        new_B_coef = S@base_B_coef@R
        
        "역행렬 계수 계산"
        self.I = np.identity(len(self.BASE_FD))
        
        new_D_inv_coef = np.linalg.pinv(self.I - new_tech_coef)
        new_S_inv_coef = np.linalg.pinv(self.I - new_B_coef)

        new_D_inv_coef[new_D_inv_coef<0] = 0
        new_S_inv_coef[new_S_inv_coef<0] = 0
        
        self.base_result = [base_tech_coef, base_B_coef, base_D_inv_coef, base_S_inv_coef]    
        self.new_result = [new_tech_coef, new_B_coef, new_D_inv_coef, new_S_inv_coef]
        self.names = ['tech_coef', 'B_coef', 'D_inv_coef', 'S_inv_coef']


class ITER_CONT:
    
    """투입산출표 혹은 IO_COEF 클래스의 RESULT 투입 가능"""
    def __init__(self, FD, GRDP, BASE_YEAR, START_YEAR, END_YEAR, Z = None, X = None, COEF = None):
        self.Z = Z
        self.X = X
        self.COEF = COEF
        
        self.FD = FD
        self.GRDP = GRDP
        self.BASE_YEAR = BASE_YEAR
        self.START_YEAR = START_YEAR
        self.END_YEAR = END_YEAR
        self.names = ['tech_coef', 'B_coef', 'D_inv_coef', 'S_inv_coef']
    
    def run(self):
        if self.COEF == None:
            base = IO_COEF(self.Z, self.X)
            base.run()
            
            base_tech_coef = base.result[0]
            base_B_coef = base.result[1]
            base_D_inv_coef = base.result[2]
            base_S_inv_coef = base.result[3]
        
        elif self.COEF != None:
            base_tech_coef = self.COEF[0]
            base_B_coef = self.COEF[1]
            base_D_inv_coef = self.COEF[2]
            base_S_inv_coef = self.COEF[3]

        self.result = dict()
        self.result[self.BASE_YEAR] = [base_tech_coef, base_B_coef, base_D_inv_coef, base_S_inv_coef]
        
        #base_year부터 시작할 경우
        if self.BASE_YEAR < self.START_YEAR:
            #정순으로 가는 updating
            for i in range(self.START_YEAR, self.END_YEAR+1, 1):
                update = UPDATE(self.FD[i-1], self.FD[i], self.GRDP[i-1], self.GRDP[i], COEF = self.result[i-1])
                update.run()
                self.result[i] = update.new_result

        #base_year가 시작연도보다 늦을 경우 
        elif self.BASE_YEAR > self.START_YEAR:
            #역순으로 가는 updating
            for i in reversed(range(self.START_YEAR, self.BASE_YEAR, 1)):
                update = UPDATE(self.FD[i+1], self.FD[i], self.GRDP[i+1], self.GRDP[i], COEF = self.result[i+1])
                update.run()
                self.result[i] = update.new_result
            
            if self.BASE_YEAR < self.END_YEAR:
                #정순으로 가는 updating
                for i in range(self.BASE_YEAR+1, self.END_YEAR+1, 1):
                    update = UPDATE(self.FD[i-1], self.FD[i], self.GRDP[i-1], self.GRDP[i], COEF = self.result[i-1])
                    update.run()        
                    self.result[i] = update.new_result
            
class ITER_JUMP:
    
    """투입산출표 혹은 IO_COEF 클래스의 RESULT 투입 가능"""
    def __init__(self, FD, GRDP, BASE_YEAR, START_YEAR, END_YEAR, Z = None, X = None, COEF = None):
        self.Z = Z
        self.X = X
        self.COEF = COEF
        
        self.FD = FD
        self.GRDP = GRDP
        self.BASE_YEAR = BASE_YEAR
        self.START_YEAR = START_YEAR
        self.END_YEAR = END_YEAR
        self.names = ['tech_coef', 'B_coef', 'D_inv_coef', 'S_inv_coef']
    
    def run(self):
        if self.COEF == None:
            base = IO_COEF(self.Z, self.X)
            base.run()
            
            base_tech_coef = base.result[0]
            base_B_coef = base.result[1]
            base_D_inv_coef = base.result[2]
            base_S_inv_coef = base.result[3]
        
        elif self.COEF != None:
            base_tech_coef = self.COEF[0]
            base_B_coef = self.COEF[1]
            base_D_inv_coef = self.COEF[2]
            base_S_inv_coef = self.COEF[3]

        self.result = dict()
        self.result[self.BASE_YEAR] = [base_tech_coef, base_B_coef, base_D_inv_coef, base_S_inv_coef]
        
        #base_year부터 시작할 경우
        if self.BASE_YEAR < self.START_YEAR:
            #정순으로 가는 updating
            for i in range(self.START_YEAR, self.END_YEAR+1, 1):
                update = UPDATE(self.FD[self.BASE_YEAR], self.FD[i], self.GRDP[self.BASE_YEAR], self.GRDP[i], COEF = self.result[self.BASE_YEAR])
                update.run()
                self.result[i] = update.new_result

        #base_year가 시작연도보다 늦을 경우 
        elif self.BASE_YEAR > self.START_YEAR:
            #역순으로 가는 updating
            for i in reversed(range(self.START_YEAR, self.BASE_YEAR, 1)):
                update = UPDATE(self.FD[self.BASE_YEAR], self.FD[i], self.GRDP[self.BASE_YEAR], self.GRDP[i], COEF = self.result[self.BASE_YEAR])
                update.run()
                self.result[i] = update.new_result
            
            if self.BASE_YEAR < self.END_YEAR:
                #정순으로 가는 updating
                for i in range(self.BASE_YEAR+1, self.END_YEAR+1, 1):
                    update = UPDATE(self.FD[self.BASE_YEAR], self.FD[i], self.GRDP[self.BASE_YEAR], self.GRDP[i], COEF = self.result[self.BASE_YEAR])
                    update.run()        
                    self.result[i] = update.new_result
                    
class FLEX_IO:
    
    def __init__(self, FD, GRDP, BASE_YEAR, START_YEAR, END_YEAR, Z = None, X = None, COEF = None, TYPE = 'CONT'):
        self.Z = Z
        self.X = X
        self.COEF = COEF
        self.TYPE = TYPE
        self.FD = FD
        self.GRDP = GRDP
        self.BASE_YEAR = BASE_YEAR
        self.START_YEAR = START_YEAR
        self.END_YEAR = END_YEAR
        self.names = ['tech_coef', 'B_coef', 'D_inv_coef', 'S_inv_coef']
        
    def run(self):
        if self.TYPE == 'CONT':
            ITER = ITER_CONT(self.FD, self.GRDP, self.BASE_YEAR, self.START_YEAR, self.END_YEAR, self.Z, self.X, self.COEF)
            ITER.run()
            self.result = ITER.result
            
        elif self.TYPE == 'JUMP':
            ITER = ITER_JUMP(self.FD, self.GRDP, self.BASE_YEAR, self.START_YEAR, self.END_YEAR, self.Z, self.X, self.COEF)
            ITER.run()
            self.result = ITER.result
            
        else:
            print('wrong type')