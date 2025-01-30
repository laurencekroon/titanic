import pandas as pd

df_gender = pd.read_csv("data/titanic/gender_submission.csv")
df_train = pd.read_csv("data/titanic/train.csv")
df_test = pd.read_csv("data/titanic/test.csv")

df_gender.head()
df_train.head()
df_test.head()

df_gender.shape
df_train.shape
df_test.shape

# make sure to merge the dataframes
df_test = df_test.merge(df_gender, on="PassengerId")

df_test.head()
df_train.head()

# concatenate the dataframes
df = pd.concat([df_train, df_test], ignore_index=True)
df.shape

# save the dataframe to a csv file
df.to_csv("data/titanic/titanic.csv", index=False)

###
# read the dataframe from a csv file
df = pd.read_csv("data/titanic/titanic.csv")

# drop the columns we don't need
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

df.head()
