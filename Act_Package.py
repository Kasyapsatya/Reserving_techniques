import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import warnings
from sklearn.metrics import r2_score
import copy
import matplotlib.pyplot as plt
import statsmodels.api as sm
from tabulate import tabulate
import seaborn as sns
'''
pip install statsmodels
pip install tabulate

'''

#for triangular data      DONE
class triangle_tri() :
  def get_triangle(self,Triangle,num_bootstrap_samples=1000,paid=0,reported=0):
    self.gr=1
    self.pivottables = {}
    if paid==1:
      pivot_table = Triangle
      self.pivottables[self.gr] = pivot_table
      #print("Upper Triangle:", self.pivottables[self.gr] )
      return self.pivottables
    if reported==1:
      pivot_table = Triangle
      self.pivottables[self.gr] = pivot_table
      #print("Upper Triangle:", self.pivottables[self.gr] )
      return self.pivottables


  def init(self):
    for gr_code, pivot_table in self.pivottables.items():
      self.agefactors={}
      factors=[]
      for i in range(9):
        P = []
        for j in range(9):
          if (pivot_table.iloc[i,j] != 0):
            f = round(pivot_table.iloc[i,j+1]/pivot_table.iloc[i,j],4)
          elif (pivot_table.iloc[i,j] == 0 and pivot_table.iloc[i,j+1] == 0 ):
            f = 1
          else:
            f = None
          P.append(f)
        factors.append(P)
      Accident_Year = [1988+i for i in range(9)]
      col = [(12*(i+1),12*(i+2)) for i in range(9)]
      DF = pd.DataFrame.from_records(factors,columns = col, index = Accident_Year)
      self.agefactors[self.gr]=DF
      #print("Age-age factors:", self.agefactors[self.gr] )
    #def cap_ages(self):
    for gr_code, factors in self.agefactors.items():
      for i in range(9):
        his=[]
        for j in range(9):
          value=factors.iloc[j,i]
          try:
              del sum
          except :
              pass  #somewhere sum is defined as an integer
          if his:
            count=0
            for i in range(len(his)):
              count+=his[i]
            his_sum = count
            mean=his_sum/len(his)
            limit=mean+5
            if value>limit:
              factors.iloc[j,i]=mean
          his.append(factors.iloc[j,i])
      self.agefactors[self.gr]=factors
      #print("Age-Age factors:", self.agefactors[self.gr] )

    self.mean_age_factors={}
    for gr_code, factors in self.agefactors.items():
      mean=[]
      sum_=0
      num=0
      for i in range(9):
        for j in range(9):
          if not pd.isnull(factors.iloc[j, i]):
            sum_+=factors.iloc[j,i]
            num+=1
        mean.append(round((sum_/num),4))
        sum_=num=0
      mean.append(1)          #no extrapolation, assuming it is 1
      self.mean_age_factors[self.gr]=mean
      #print("Mean age factors:", self.mean_age_factors[self.gr] )


  def CDF(self):
    self.CDF={}
    for gr_code, dupl in self.mean_age_factors.items():
      mean=dupl.copy()
      for i in range(len(mean)):
        prod=1
        for j in range(i, len(mean)):
          prod*= mean[j]
        mean[i]=round(prod,4)
      self.CDF[self.gr]=mean
    #print("CDF:", self.CDF[self.gr] )
    return self.CDF
  def develop_triangle(self):
    for gr_code, pivot_table in self.pivottables.items() :
      #each pivot_table is a triangle, you have to fill the lower triangular values now
      #if devolopment lag+ AY >1998, then df[i,j]= df[i,j-1]*cdf[j]
      for i in range(pivot_table.shape[0]):
          for j in range(pivot_table.shape[1]):
            if i+j>9:
              pivot_table.iloc[i,j]=pivot_table.iloc[i,j-1]*self.mean_age_factors[self.gr][j-1]
    #print("Full Triangle:", self.pivottables[self.gr] )
    return self.pivottables

#self,df,gr,paid=0,reported=0

class method_triangle():
  def init(self,tri_paid):
    self.tri_paid=tri_paid
    self.gr=1

  def methods(self):
    Actuary_paid=triangle_tri()
    self.pivottables_paid=Actuary_paid.get_triangle(self.tri_paid,self.gr,paid=1)
    #undeveloped= copy.deepcopy(self.pivottables)
    Actuary_paid.init()    ##send gr
    self.pivottables_paid_devoloped= Actuary_paid.develop_triangle()
    self.CDF_paid=Actuary_paid.CDF()
    self.gr=1
    self.method={}
    self.UC={}

    for gr_code, pivot_table in self.pivottables_paid_devoloped.items():
      dev_c=[]
      dev_f=[]
      Age=[]
      for i in range(pivot_table.shape[0]):
        for j in range(pivot_table.shape[1]):
          if i+j==9:
            dev_c.append(pivot_table.iloc[i,j])
          if j==9:
            dev_f.append(pivot_table.iloc[i,j])
      for i in range(pivot_table.shape[1],0,-1):
        Age.append(12*i)
      ibnr=np.array(dev_f) - np.array(dev_c)
      dev_c=np.array(dev_c)
      dev_f=np.array(dev_f)
      cd_factors=np.array(self.CDF_paid[self.gr])
      DF = pd.DataFrame()
      AY= pivot_table.index.tolist()
      DF['AccidentYear']= AY
      DF['Age']=Age
      DF['Devolopment as of end of {}'.format(AY[0])]=dev_c
      DF['Chain Ladder']= dev_f
      DF['CDF']=cd_factors

      self.UC['Chain Ladder']= DF['Chain Ladder'].sum()

      columns=[ 'Chain Ladder']

      for i in range(1,len(columns)):
        plt.figure(figsize=(5, 3))
        plt.plot(DF['AccidentYear'], DF[columns[i]], label=columns[i])
        plt.xlabel('Accident Year')
        plt.ylabel('Projection')
        plt.title('Comparison: {}'.format(columns[i]) )
        plt.legend()
      self.method[self.gr]=DF
    return self.method, self.UC


class triangle() :    #df is the pd dataframe pointing to the input file

  def get_triangle(self,df,gr,paid=0,reported=0):
    self.df=df
    self.pivottables = {}
    self.gr=gr
    for gr_code, group in df.groupby('GRCODE'):
      if gr_code==self.gr:
        if paid==1:
          pivot_table = group.pivot_table(index='AccidentYear', columns='DevelopmentLag', values='CumPaidLoss')
          self.pivottables[self.gr] = pivot_table
          self.glm_pivottables= copy.deepcopy(self.pivottables)
          #print("Upper Triangle:", self.pivottables[self.gr] )
          print(tabulate(self.pivottables[self.gr], showindex=True , headers='keys', tablefmt = 'psql'))
          return self.pivottables
        if reported==1:
          pivot_table = group.pivot_table(index='AccidentYear', columns='DevelopmentLag', values='IncurLoss')
          self.pivottables[self.gr] = pivot_table
          self.glm_pivottables= copy.deepcopy(self.pivottables)
          #print("Upper Triangle:", self.pivottables[self.gr] )

          return self.pivottables

  def init(self):
    for gr_code, pivot_table in self.pivottables.items():
      self.agefactors={}
      factors=[]
      for i in range(9):
        P = []
        for j in range(9):
          if (pivot_table.iloc[i,j] != 0):
            f = round(pivot_table.iloc[i,j+1]/pivot_table.iloc[i,j],4)
          elif (pivot_table.iloc[i,j] == 0 and pivot_table.iloc[i,j+1] == 0 ):
            f = 1
          else:
            f = None
          P.append(f)
        factors.append(P)
      Accident_Year = [1988+i for i in range(9)]
      col = [(12*(i+1),12*(i+2)) for i in range(9)]
      DF = pd.DataFrame.from_records(factors,columns = col, index = Accident_Year)
      self.agefactors[self.gr]=DF
      #print("Age-age factors:", self.agefactors[self.gr] )
      print(tabulate(self.agefactors[self.gr], showindex=True , headers='keys', tablefmt = 'psql'))

    #def cap_ages(self):
    for gr_code, factors in self.agefactors.items():
      for i in range(9):
        his=[]
        for j in range(9):
          value=factors.iloc[j,i]
          try:
              del sum
          except :
              pass  #somewhere sum is defined as an integer
          if his:
            count=0
            for i in range(len(his)):
              count+=his[i]
            his_sum = count
            mean=his_sum/len(his)
            limit=mean+5
            if value>limit:
              factors.iloc[j,i]=mean
          his.append(factors.iloc[j,i])
      self.agefactors[self.gr]=factors
      #print("Age-Age factors:", self.agefactors[self.gr] )
      print(tabulate(self.agefactors[self.gr], showindex=True , headers='keys', tablefmt = 'psql'))


    self.mean_age_factors={}
    for gr_code, factors in self.agefactors.items():
      mean=[]
      sum_=0
      num=0
      for i in range(9):
        for j in range(9):
          if not pd.isnull(factors.iloc[j, i]):
            sum_+=factors.iloc[j,i]
            num+=1
        mean.append(round((sum_/num),4))
        sum_=num=0
      mean.append(1)          #no extrapolation, assuming it is 1
      self.mean_age_factors[self.gr]=mean
      print("Mean age factors:", self.mean_age_factors[self.gr] )


  def CDF(self):
    self.CDF={}
    for gr_code, dupl in self.mean_age_factors.items():
      mean=dupl.copy()
      for i in range(len(mean)):
        prod=1
        for j in range(i, len(mean)):
          prod*= mean[j]
        mean[i]=round(prod,4)
      self.CDF[self.gr]=mean
    print("CDF:", self.CDF[self.gr] )
    return self.CDF

  def glm_develop_triangle(self):
    for gr_code, pivot_table in self.glm_pivottables.items():
      for j in range(1,10):
        y = np.array(pivot_table.iloc[:10-j,j])
        x = np.array(pivot_table.iloc[:10-j, j-1])
        #print("x",x)
        #print("y",y)
        predict_x= np.array(pivot_table.iloc[-j:, j-1])
        model = sm.GLM(y, x, family=sm.families.Gaussian())
        result = model.fit(method = 'bfgs')      #method = 'bfgs' handles single values
        y_predict= np.array(result.predict(predict_x))
        #print("x_predict",predict_x)
        #print("y_predict",y_predict)
        index=-1
        for k in range(-1,-j-1,-1):
          pivot_table.iloc[k,j]=y_predict[index]
          index-=1
    return self.glm_pivottables

  def develop_triangle(self):
    for gr_code, pivot_table in self.pivottables.items() :
      #each pivot_table is a triangle, you have to fill the lower triangular values now
      #if devolopment lag+ AY >1998, then df[i,j]= df[i,j-1]*cdf[j]
      for i in range(pivot_table.shape[0]):
          for j in range(pivot_table.shape[1]):
            if i+j>9:
              pivot_table.iloc[i,j]=pivot_table.iloc[i,j-1]*self.mean_age_factors[self.gr][j-1]
    #print("Full Triangle:", self.pivottables[self.gr] )
    print(tabulate(self.pivottables[self.gr], showindex=True , headers='keys', tablefmt = 'psql'))

    return self.pivottables

  def premium(self):
    self.premium = {}
    grouped = self.df.groupby('GRCODE')
    for gr_code, group_df in grouped:
        table = group_df.groupby('AccidentYear')['EarnedPremNet'].mean().reset_index()
        #print(table)
        self.premium[self.gr] = table
        break
    #[1.338589096, 1.279143524, 1.20553075, 1.169688156, 1.136716145, 1.106310002, 1.077488464, 1.050814335, 1.017024053 ] got from https://www.bls.gov/data/inflation_calculator.htm
    for gr_code, table in self.premium.items():
      table['Onlevel factors']=np.array([1.338589096, 1.279143524, 1.20553075, 1.169688156, 1.136716145, 1.106310002, 1.077488464, 1.050814335, 1.017024053, 1 ])
      table['On level Premium']=table['EarnedPremNet']*table['Onlevel factors']
      self.premium[self.gr]=table
    #print("Premium:",self.premium[self.gr])

    return self.premium


class report():
  def reported(self,pivottables,gr,premium):  #needs reported triangle
    #need some information of CumRep
    self.premium=premium
    self.gr=gr
    self.reported={}
    self.capecod_ECR={}
    #print(pivottables[self.gr])
    for gr_code, pivot_table in pivottables.items():
      dev_c=[]
      dev_f=[]
      for i in range(pivot_table.shape[0]):
        for j in range(pivot_table.shape[1]):
          if i+j==9:
            dev_c.append(pivot_table.iloc[i,j])
          if j==9:
            dev_f.append(pivot_table.iloc[i,j])
      dev_c=np.array(dev_c)
      dev_f=np.array(dev_f)
      percentreported= dev_c/dev_f   #correct
      DF = pd.DataFrame()
      AY= pivot_table.index.tolist()
      DF['AccidentYear']= AY
      DF['On level premium']= self.premium[self.gr]['On level Premium']
      DF['cumrep']=dev_c     #cum reported as of current(1997/12/31)
      DF['pctrep'] =percentreported  #cum reported as of current(1997/12/31) using the devoloped triangle
      DF['Used-up Premium'] = self.premium[self.gr]['On level Premium'] *  DF['pctrep']
      self.reported[self.gr]=DF
      capecod_ecr= DF['cumrep'].sum() / DF['Used-up Premium'].sum()
      self.capecod_ECR[self.gr]=capecod_ecr
      #print(capecod_ecr)
      print(tabulate(self.reported[self.gr], showindex=True , headers='keys', tablefmt = 'psql'))
      return self.reported, self.capecod_ECR


#self,df,gr,paid=0,reported=0

class GLM():
  def create(self,df,gr): #df_filtered
    self.gr=gr
    self.pivottables_inc = {}
    for gr_code, group in df.groupby('GRCODE'):
      if gr_code==self.gr:
        pivot_table = group.pivot_table(index='AccidentYear', columns='DevelopmentLag', values='IncPaid')
        self.pivottables_inc[gr_code] = pivot_table
    for gr_code, pivot_table in self.pivottables_inc.items():
      if gr_code==self.gr:
        pivot_table.reset_index(drop=True, inplace=True)
        pivot_table.index += 1  # Adjust the index to start from 1
        pivot_table.reset_index(inplace=True)
        glm_df=pd.melt(pivot_table, id_vars='index', value_vars=None, var_name=None, value_name='value', col_level=None)
    glm_df.dropna(inplace=True)
    self.glm_df=glm_df
    self.glm_df2=copy.deepcopy(self.glm_df)

  def fit2(self):
    self.glm_df2['DevelopmentLag'] = self.glm_df2['DevelopmentLag'].astype(int)      ###################3
    self.glm_df2['index'] = self.glm_df2['index'].astype(int)
    self.glm_df2['value'] = self.glm_df2['value'].astype(int)
    glmdup_df = self.glm_df2[self.glm_df2['index'] != 1]
    x1= np.array(glmdup_df['DevelopmentLag'])
    x2= np.array(glmdup_df['index'])
    y = np.array(glmdup_df['value'])
    x1_2d = x1.reshape(-1, 1)
    x2_2d = x2.reshape(-1, 1)
    x3=[]
    x = np.concatenate((x1_2d, x2_2d), axis=1)
    for i in range(len(x)):
      specific_values=x[i]
      #print(specific_values)
      mat = self.glm_df2[(self.glm_df2['DevelopmentLag'] == specific_values[0]) &
                (self.glm_df2['index'] == specific_values[1]-1)]['value']
      x3.append(mat.values[0])
    x3=np.array(x3)
    x3_2d = x3.reshape(-1, 1)
    x = np.concatenate((x1_2d, x2_2d,x3_2d), axis=1)
    model = sm.GLM(y, x, family=sm.families.Gaussian())
    z=0 #for Gamma
    result = model.fit(method = 'bfgs')      #method = 'bfgs' handles single values
    self.result2=result
    print(result.summary())
  def finish2(self):
      predic_x=[]                           #############################
      predic_x_1=[]
      predic_x_0=[]
      start=10
      for k in range(2,11):
          temp=start
          for j in range(k-1):
            predic_y=[]
            predic_x=[]
            #predic_x_0.append(temp) #lag
            #predic_x_1.append(k)  #index
            mat = self.glm_df2[(self.glm_df2['DevelopmentLag'] == temp-1) &
                  (self.glm_df2['index'] ==k)]['value']
            predic_x= ([k,temp,mat.values[0]])
            #print(predic_x)
            predic_y =self.result2.predict(predic_x)
            #print(predic_y[0])
            #append k,temp,predic_y[0]
            #glm_df.loc[len(glm_df)]=np.array([k, temp,predic_y[0]])
            new_row={'index':k,'DevelopmentLag':temp, 'value':predic_y[0]}
            new_df = pd.DataFrame([new_row])
            self.glm_df2 = pd.concat([self.glm_df2, new_df], ignore_index=True)
            temp+=1
          print()
          start-=1
      self.glm_df2 = self.glm_df2.sort_values(by=['index','DevelopmentLag'])
  def predict2(self):
      num_rows, num_columns = self.glm_df2.shape     #################################
      self.glm_df2['CumPaid']=0
      j = 2  # column index of incremental loss
      for i in range(0, num_rows - 10+2, 10):
          self.glm_df2.iloc[i,3]=self.glm_df2.iloc[i,j]
          i+=1
          for k in range(9):
              self.glm_df2.iloc[i , 3] = self.glm_df2.iloc[i , j] + self.glm_df2.iloc[i -1, 3]
              #print("current",glm_df.iloc[i , j],"previous", glm_df.iloc[i -1, 3]," i:",i," j:",j , "curdata: ", glm_df.iloc[i , j],"Updata:",glm_df.iloc[i , j] + glm_df.iloc[i -1, 3])
              i=i+1
      self.glm_df2.head(40)
      #print(self.glm_df2)

      UC_values=[]
      for lag, cp in zip(self.glm_df2['DevelopmentLag'], self.glm_df2['CumPaid']):
        if lag==10:
          UC_values.append(cp)
      print(UC_values)

      UC=sum(UC_values)
      print("Ultimate cost using GLM22",UC)
      return UC_values

  def fit(self):
     y = np.array(self.glm_df['value'])
     self.glm_df['DevelopmentLag'] = self.glm_df['DevelopmentLag'].astype(int)
     x1= np.array(self.glm_df['DevelopmentLag'])
     x2= np.array(self.glm_df['index'])
     x1_2d = x1.reshape(-1, 1)
     x2_2d = x2.reshape(-1, 1)
     # Stack the predictor matrices horizontally
     x = np.concatenate((x1_2d, x2_2d), axis=1)
     model = sm.GLM(y, x, family=sm.families.Gaussian())
     self.z=0    #2 for Gamma  0 for Gaussian
     result = model.fit(method = 'bfgs')      #method = 'bfgs' handles single values
     print(result.summary())
     self.result=result
     self.glm_predict=copy.deepcopy(self.glm_df)
     self.glm_df['predicted']=result.predict(x)
     y_cap = np.array(self.glm_df['predicted'])
     self.residuals_pearson = (y-y_cap)/np.sqrt(np.power(y_cap, self.z))
  def finish(self):
     #finsih the rectangle
     predic_x=[]
     predic_x_1=[]
     predic_x_0=[]
     for i in range(2,11):
       temp=10
       for j in range(i-1):
         #print(i,temp)
         predic_x.append([i,temp])
         predic_x_0.append(temp) #lag
         predic_x_1.append(i)  #index
         temp-=1
     predic_x=np.array(predic_x)
     predic_y=self.result.predict(predic_x)
     predic_glm = pd.DataFrame()
     predic_glm['index']= np.array(predic_x_1)
     predic_glm['DevelopmentLag']= np.array(predic_x_0)
     predic_glm['predicted']= np.array(predic_y)
     #predic_glm.head(55)
     self.glm= pd.concat([self.glm_df, predic_glm], axis=0)

  def predict(self):
    predic_x=[]
    predic_x_1=[]
    predic_x_0=[]
    predic_y=[]
    for k in range(2,11):
        temp=10
        for j in range(k-1):
          #print(i,temp)
          predic_x.append([k,temp])
          predic_x_0.append(temp) #lag
          predic_x_1.append(k)  #index
          temp-=1
    predic_y =self.result.predict(predic_x)
    predic_glm = pd.DataFrame()
    predic_glm['index']= np.array(predic_x_1)
    predic_glm['DevelopmentLag']= np.array(predic_x_0)
    predic_glm['value']=np.array(predic_y)
    glm_rect= pd.concat([self.glm_predict, predic_glm], axis=0)
    glm_rect = glm_rect.sort_values(by=['index','DevelopmentLag'])
    #convert into cumm values
    num_rows, num_columns = glm_rect.shape
    glm_rect['CumPaid']=0
    j = 2  # column index of incurloss
    for i in range(0, num_rows - 10+2, 10):
        glm_rect.iloc[i,3]=glm_rect.iloc[i,j]
        i+=1
        for k in range(9):
            glm_rect.iloc[i , 3] = glm_rect.iloc[i , j] + glm_rect.iloc[i -1, 3]
            #print("current",glm_rect.iloc[i , j],"previous", glm_rect.iloc[i -1, 3]," i:",i," j:",j , "curdata: ", glm_rect.iloc[i , j],"Updata:",glm_rect.iloc[i , j] + glm_rect.iloc[i -1, 3])
            i=i+1
    UC_values=[]
    for lag, cp in zip(glm_rect['DevelopmentLag'], glm_rect['CumPaid']):
      if lag==10:
        UC_values.append(cp)
    return UC_values

  def bootstrap(self):
     num_bootstrap=50
     bootstrap_ult_cost=[]
     #print(residuals_pearson)
     for i in range(num_bootstrap):
       bootstrapped_values = np.random.choice(self.residuals_pearson, size=100, replace=True)
       #print(bootstrapped_values)
       self.glm['adjusted predicted']=self.glm['predicted']+(bootstrapped_values* np.sqrt(np.power(self.glm['predicted'], self.z)))
       self.glm = self.glm.sort_values(by=['index','DevelopmentLag'])
       #print(glm)
       #I want to cummulate these values across each accident year
       num_rows, num_columns =self.glm.shape
       self.glm['CumPaid']=0  #5th column
       j = 4  # column index of adjusted predicted
       for i in range(0, num_rows - 10+2, 10):
           self.glm.iloc[i,5]=self.glm.iloc[i,j]
           i+=1
           for k in range(9):
              self.glm.iloc[i , 5] = self.glm.iloc[i , j] + self.glm.iloc[i -1, 5]
              i=i+1
       UC=[]
       for index, row in self.glm.iterrows():
         if row['DevelopmentLag'] == 10:
             UC.append(row['CumPaid'])
       UC=np.array(UC)
       ult_cost=UC.sum()
       bootstrap_ult_cost.append(ult_cost)
     # Calculate the mean
     bootstrap_ult_cost=np.array(bootstrap_ult_cost)
     mean_value = np.mean(bootstrap_ult_cost)
     # Calculate the standard deviation
     std_deviation = np.std(bootstrap_ult_cost)
     # Print or use the mean and standard deviation
     print("Mean:", mean_value)
     print("Standard Deviation:", std_deviation)
     summary = np.percentile(bootstrap_ult_cost, [25, 50, 75,95])
     print("Summary (25th, 50th, 75th and 95th percentiles):", summary)
     return bootstrap_ult_cost

  def bootstrap_cl(self):
    num_bootstrap=50
    bootstrap_results=[]
    predic_x=[]
    predic_x_1=[]
    predic_x_0=[]
    #print(residuals_pearson)
    for i in range(num_bootstrap):
      bootstrapped_values = np.random.choice(self.residuals_pearson, size=55, replace=True)
      bootstrap_df=pd.DataFrame()
      bootstrap_df['index']= self.glm_df['index']
      bootstrap_df['DevelopmentLag']= self.glm_df['DevelopmentLag']
      bootstrap_df['value']=self.glm_df['predicted']+(bootstrapped_values* np.sqrt(np.power(self.glm_df['predicted'], self.z)))
      #got the triangle
      for k in range(2,11):
        temp=10
        for j in range(k-1):
          #print(i,temp)
          predic_x.append([k,temp])
          predic_x_0.append(temp) #lag
          predic_x_1.append(k)  #index
          temp-=1
      predic_glm = pd.DataFrame()
      predic_glm['index']= np.array(predic_x_1)
      predic_glm['DevelopmentLag']= np.array(predic_x_0)
      #predic_glm.head(55)
      bootstrap_df= pd.concat([bootstrap_df, predic_glm], axis=0)
      bootstrap_df= bootstrap_df.sort_values(by=['index','DevelopmentLag'])
      num_rows, num_columns = bootstrap_df.shape
      bootstrap_df['Cum']=0
      j = 2  # column index of inc loss
      for i in range(0, num_rows - 10+2, 10):
          bootstrap_df.iloc[i,3]=bootstrap_df.iloc[i,j]
          i+=1
          for k in range(9):
              bootstrap_df.iloc[i , 3] = bootstrap_df.iloc[i , j] + bootstrap_df.iloc[i -1, 3]
              #print("current",bootstrap_df.iloc[i , j],"previous", bootstrap_df.iloc[i -1, 3]," i:",i," j:",j , "curdata: ", bootstrap_df.iloc[i , j],"Updata:",bootstrap_df.iloc[i , j] - bootstrap_df.iloc[i -1, 3])
              i=i+1
      predic_x=[]
      predic_x_1=[]
      predic_x_0=[]
      pivot_table = pd.pivot_table(bootstrap_df, index='index', columns='DevelopmentLag', values='Cum')
      Actuary=method_triangle()
      triangle_array = pivot_table.values.copy()
      Actuary.init(pivot_table)
      results=[Actuary.methods()]
      bootstrap_results.append(results[0][1]['Chain Ladder'])
      print(bootstrap_results)            ###############check once
    summary = np.percentile(bootstrap_results, [25, 50, 75,95])
    print("Summary (25th, 50th, 75th and 95th percentiles):", summary)
    return bootstrap_results



class method():
  def methods(self,df,gr,flag=0):

    Actuary_validate=triangle()
    self.pivottables_verify=Actuary_validate.get_triangle(df,gr,paid=1) #for verification

    df = df[(df['AccidentYear'] + df['DevelopmentLag']) <= 1998]  #upper triangle
    #df.info()
    Actuary_paid=triangle()
    self.pivottables_paid=Actuary_paid.get_triangle(df,gr,paid=1)
    #bootstrapping here?

    Actuary_paid.init()
    self.pivottables_paid_devoloped= Actuary_paid.develop_triangle()
    self.premium=Actuary_paid.premium()
    self.CDF_paid=Actuary_paid.CDF()
    self.glm_pivottables_paid= Actuary_paid.glm_develop_triangle()

    Actuary_reported=triangle()
    self.pivottables_reported=Actuary_reported.get_triangle(df,gr,reported=1)
    Actuary_reported.init()
    self.pivottables_reported_devoloped= Actuary_reported.develop_triangle()
    self.CDF_reported=Actuary_reported.CDF()

    Actuary_report=report()
    self.reported, self.capecod_ECR =Actuary_report.reported(self.pivottables_reported,gr,self.premium)

    self.gr=gr
    self.method={}
    self.accuracy={}
    self.UC={}
    Actuary_glm=GLM()
    Actuary_glm.create(df,gr)
    Actuary_glm.fit()
    Actuary_glm.fit2()
    Actuary_glm.finish()
    Actuary_glm.finish2()
    self.GLM_UC=Actuary_glm.predict()
    self.GLM2_UC=Actuary_glm.predict2()
    self.bootstrap_ult_cost=Actuary_glm.bootstrap()  #use this for graphs
    self.bootstrap_ult_cost_cl=Actuary_glm.bootstrap_cl()


    for gr_code, pivot_table in self.pivottables_paid_devoloped.items():
      actual_reported_claims=[]
      dev_f_glm=[]
      dev_c=[]
      dev_f=[]
      Age=[]
      Premium=self.premium[self.gr]['EarnedPremNet']
      Premium=np.array(Premium)
      onlevelPremium=self.premium[self.gr]['On level Premium']
      onlevelPremium=np.array(onlevelPremium)
      for i in range(pivot_table.shape[0]):
        for j in range(pivot_table.shape[1]):
          if i+j==9:
            dev_c.append(pivot_table.iloc[i,j])
            actual_reported_claims.append(self.pivottables_reported[self.gr].iloc[i,j])
          if j==9:
            dev_f.append(pivot_table.iloc[i,j])
            dev_f_glm.append(self.glm_pivottables_paid[gr_code].iloc[i,j])
      for i in range(pivot_table.shape[1],0,-1):
        Age.append(12*i)
      ibnr=np.array(dev_f) - np.array(dev_c)
      actual_reported_claims=np.array(actual_reported_claims)
      dev_c=np.array(dev_c)
      dev_f=np.array(dev_f)
      with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in divide")
        ecratio = np.array(dev_f) / Premium
      ecr_1=[]
      ecr_2=[]
      for i in range(pivot_table.shape[0]//2):
        ecr_1.append(ecratio[i])
      ecr_1mean=sum(ecr_1)/len(ecr_1)
      for i in range(pivot_table.shape[0]//2,pivot_table.shape[0]):
        ecr_2.append(ecratio[i])
      ecr_2mean=sum(ecr_2)/len(ecr_2)
      selected_ecr1 = [ecr_1mean] *(pivot_table.shape[0]//2)
      selected_ecr2 = [ecr_2mean] *(pivot_table.shape[0]//2)
      selected_ecr= selected_ecr1+ selected_ecr2
      selected_ecr=np.array(selected_ecr)
      Exp_Ult= Premium * selected_ecr
      cd_factors=np.array(self.CDF_paid[self.gr])
      reported_pct=dev_c/dev_f           ###done
      #print("Reported %",reported_pct)
      #print(Exp_Ult)
      DF = pd.DataFrame()
      AY= pivot_table.index.tolist()
      DF['AccidentYear']= AY
      DF['Age']=Age
      DF['Devolopment as of end of {}'.format(AY[0])]=dev_c
      DF['Chain Ladder']= dev_f
      #DF['IBNR paid chainladder']=ibnr
      DF['Premium']=Premium
      DF['EC ratio']= ecratio
      DF['Selected EC ratio'] =selected_ecr
      DF['Expected Ult Loss']= Exp_Ult
      DF['CDF']=cd_factors
      DF['BF']= (dev_c) + ((1-reported_pct)* Exp_Ult)
      #print(capecod_ECR[gr_code])
      DF['On level premium']= onlevelPremium
      DF['capecodECR']=self.capecod_ECR[self.gr]
      #DF['ECR expected paid claims']= ((1-reported_pct)* Exp_Ult)
      expected_unreported_claims= np.array((onlevelPremium* self.capecod_ECR[self.gr])*(1-self.reported[self.gr]['pctrep']))
      DF['Cape-cod Ult paid claims']=   actual_reported_claims+  expected_unreported_claims
      DF['GLM']= dev_f_glm
      DF['GLM2']=np.array(self.GLM_UC)
      DF['GLM2(2)']=np.array(self.GLM2_UC)
      DF['Original UC']= np.array(self.pivottables_verify[self.gr].iloc[:,-1])
      #DF['Cape-cod Ult paid check']=   (dev_c) + ((capecod_ECR[gr_code])* Exp_Ult)
      #DF['IBNR capecod']= DF['UC capecod']*(1-reported[gr_code]['pctrep'])
      statements=[
       "r2_1 = r2_score( np.array(DF['Original UC']), np.array(DF['Chain Ladder']))",
       "r2_3 = r2_score( np.array(DF['Original UC']), np.array(DF['Expected Ult Loss']))",
       "r2_4 = r2_score( np.array(DF['Original UC']), np.array(DF['BF']))",
       "r2_5= r2_score( np.array(DF['Original UC']), np.array(DF['Cape-cod Ult paid claims']))",
       "r2_6= r2_score( np.array(DF['Original UC']), np.array(DF['GLM']))",
       "r2_7= r2_score( np.array(DF['Original UC']), np.array(DF['GLM2']))",
       "r2_8= r2_score( np.array(DF['Original UC']), np.array(DF['GLM2(2)']))",
       "self.accuracy['Chain Ladder']=r2_1",
       "self.accuracy['Expected Ult Loss']=r2_3",
       "self.accuracy['BF']=r2_4",
       "self.accuracy['Cape-cod Ult paid claims']=r2_5",
       "self.accuracy['GLM']=r2_6",
       "self.accuracy['GLM2']=r2_7",
       "self.accuracy['GLM2(2)']=r2_8",
       "print(self.accuracy)",

       "self.UC['Chain Ladder']= DF['Chain Ladder'].sum()",
       "self.UC['Expected Ult Loss']=DF['Expected Ult Loss'].sum()",
       "self.UC['BF']=DF['BF'].sum()",
       "self.UC['Cape-cod Ult paid claims']=DF['Cape-cod Ult paid claims'].sum()",
       "self.UC['GLM']=DF['GLM'].sum()",
       "self.UC['GLM2']=DF['GLM2'].sum()",
       "self.UC['Original UC']=DF['Original UC'].sum()",
       "self.UC['GLM Bootstrap']=np.mean(self.bootstrap_ult_cost)",
       "self.UC['GLM Bootstrap CL']=np.mean(self.bootstrap_ult_cost_cl)",
       "self.UC['GLM2(2)']=DF['GLM2(2)'].sum()",

       "original_uc_value = self.UC['Original UC']",
       "differences = {key: value- original_uc_value  for key, value in self.UC.items() if key != 'Original UC'}",
       "plt.figure(figsize=(10, 6))",
       "print('Differences:   ',differences)",
       "plt.bar(differences.keys(), differences.values(), color='red')",
       "plt.axhline(0, color='gray', linewidth=0.5)",
       "plt.xlabel('Models')",
       "plt.ylabel('Difference from Original UC')",
       "plt.title('Comparison with Original UC')",
       "plt.xticks(rotation=45, ha='right')" ,
       "plt.tight_layout()",
       "plt.show()"]

      for statement in statements:
       try:
         exec(statement)
       except :
         pass

      columns=['Original UC', 'GLM', 'GLM2','GLM2(2)', 'Cape-cod Ult paid claims', 'BF', 'Expected Ult Loss', 'Chain Ladder']
      for i in range(1,len(columns)):
        plt.figure(figsize=(5, 3))
        try:
         plt.plot(DF['AccidentYear'], DF[columns[i]], label=columns[i])
        except:
         pass
        plt.plot(DF['AccidentYear'], DF[columns[0]], label=columns[0])
        plt.xlabel('Accident Year')
        plt.ylabel('Projection')
        try:
           plt.title('Comparison: {}. R2={}'.format(columns[i],self.accuracy[columns[i]]))
        except :
         pass
        plt.legend()
        plt.savefig('{}.png'.format(columns[i]))
        #print(DF['Cape-cod Ult paid claims'].sum())
        #print(DF['Original UC'].sum())
      self.method[self.gr]=DF


      plt.figure(figsize=(5, 3))
      for i in range(1,len(columns)):
        try:
         if columns[i] in ['Original UC', 'Cape-cod Ult paid claims']:  # Highlight specific columns
            plt.plot(DF['AccidentYear'], DF[columns[i]], label=columns[i], linewidth=4)
         else:
            plt.plot(DF['AccidentYear'], DF[columns[i]], label=columns[i], alpha=0.2)
        except:
         pass
      plt.plot(DF['AccidentYear'], DF[columns[0]], label=columns[0])
      plt.xlabel('Accident Year')
      plt.ylabel('Projection')
      plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


    return self.method, self.accuracy, self.UC