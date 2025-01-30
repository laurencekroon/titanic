import pandas as pd

df1 = pd.read_csv("data/accidents_2017.csv")
df2 = pd.read_csv("data/health_nutrition_population_statistics.csv")
df3 = pd.read_csv("data/winequality-red.csv")

df1.head()
df2.head()
df3.head()

df1.shape
df2.shape
df3.shape

# check if the dataframes have missing values
df1.isnull().sum()
df2.isnull().sum()
df3.isnull().sum()
