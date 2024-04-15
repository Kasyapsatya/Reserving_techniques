import pickle
import pandas as pd

# Unpickle the DataFrame
with open('info.pickle', 'rb') as f:
    info = pickle.load(f)

# Now you can use the DataFrame
writer= pd.ExcelWriter('output.xlsx', engine='xlsxwriter')
info.to_excel(writer, index=False, sheet_name='results', startrow=1)

# Access the xlsxwriter workbook and worksheet objects
workbook  = writer.book
worksheet = writer.sheets['results']

# Set the width of all columns to 15
for col_num, value in enumerate(info.columns.values):
    worksheet.set_column(col_num, col_num, width=28)
# Save the Excel file
writer._save()
