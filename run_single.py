import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
from tabulate import tabulate
from Act_Package import triangle_tri, method_triangle, triangle, report, GLM, method

#LOB1->commercial auto
#LOB2->medical malpractices
#LOB3->other liabilities
#LOB4->private passengers
#LOB5->product liabilities
#LOB6->worker's compensation

df = pd.read_csv('merged_data.csv')
df = df[df['Line_of_Business_LOB2'] == 1]  ###
df_filtered= df[(df['AccidentYear'] + df['DevelopmentLag']) <= 1998] #upper trianlge
#take a company code
gr= 669 #edit these as you wish
Actuary=method()
results=[Actuary.methods(df,gr)]
print(results)
print(tabulate(results[0][0][gr], showindex=True , headers='keys', tablefmt = 'psql'))
#results[0][2]['Chain Ladder'] is our chianladder UC reserve estimate