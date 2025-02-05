#  plot data of all columns
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df_train = pd.read_csv("data/train.csv")
df_test = pd.read_csv("data/test.csv")

df_train.head()
df_test.head()

df_train.shape
df_test.shape

df_train.isnull().sum()
df_test.isnull().sum()

df_train.info()
df_test.info()

# create new dataframe, drop column Survived, and concatenate df_train and df_test
df = df_train.drop(columns=["Survived"])
df = pd.concat([df, df_test], ignore_index=True)

df.head()
df.shape

df.isnull().sum()

# save the dataframe to a csv file
df.to_csv("data/titanic.csv", index=False)

# read the dataframe from a csv file
df = pd.read_csv("data/titanic.csv")

df.head()
df.shape
