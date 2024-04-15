import pickle

# Unpickle the DataFrame
with open('df.pickle', 'rb') as f:
    df = pickle.load(f)

# Now you can use the DataFrame
print(df)
