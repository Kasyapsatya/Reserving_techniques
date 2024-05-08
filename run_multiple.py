import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
from tabulate import tabulate
from Act_Package import triangle_tri, method_triangle, triangle, report, GLM, method
import pickle

random_elements=[
    [337,353,388,620, 715],
    [669,7854,32514,33049,33111],
    [460,620,671,683,715],
    [43,266,353,388,460],
    [78,86,353,388,620],
    [86,337,353,388,671]
]

column_names = ['LOB','gr', 'Chain Ladder', 'Expected Ult Loss', 'BF','Cape-cod Ult paid claims', 'GLM','GLM2','GLM2(2)', 'Original UC', 'GLM Bootstrap', 'GLM Bootstrap CL',
                  'Error in Chian Ladder', 'Error in Expected Ult Loss','Error in BF', 'Error in Cape-cod Ult paid claims', 'Error in GLM','Error in GLM2','Error in GLM2(2)','Error in GLM Bootstrap','Error in GLM Bootstrap CL','Best R2', 'Best in terms of error in UC']
#print(len(column_names))
info = pd.DataFrame(columns=column_names)
for i in range(1,7):
  df = pd.read_csv('merged_data.csv')
  string='Line_of_Business_LOB{}'.format(i)
  #print(string)
  df=df[df[string]==1]
  for gr in random_elements[i-1]:
    Actuary=method()
    results=[Actuary.methods(df,gr)]
    #print(results)
    #print(results[0][2])
    #print(results[0][1])
    errors={}
    for key, valaue in results[0][2].items():
        if key!='Original UC':
          errors[key]=results[0][2][key]-results[0][2]['Original UC']


    max_key = max(results[0][1], key=results[0][1].get)      #max R2
    min_key=min(errors, key=lambda k: abs(errors[k]))        #min UC error
    input=[list(results[0][2].values()),list(errors.values()),max_key, min_key]
    input_row = [element for sublist in input for element in sublist if type(element) != str]
    input_row.append(input[-2])
    input_row.append(input[-1])
    input_row.insert(0,gr)
    input_row.insert(0,i)
    #print(input_row)
    #print(len(input_row))
    #append input_row to info
    info.loc[len(info)]=input_row
    #info = info.append(input_row, ignore_index=True)
#print(info)

with open('info.pickle', 'wb') as f:
    pickle.dump(info, f)
    
'''
writer= pd.ExcelWriter('output.xlsx', engine='xlsxwriter')
info.to_excel(writer, index=False, sheet_name='results', startrow=1)

# Access the xlsxwriter workbook and worksheet objects
workbook  = writer.book
worksheet = writer.sheets['results']

# Set the width of all columns to 15
for col_num, value in enumerate(info.columns.values):
    worksheet.set_column(col_num, col_num, width=28)
# Save the Excel file
writer.save()
'''