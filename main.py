# Pandas
# Package that offers various data structures and operations for manipulating numerical data and time series.
# It is fast and it has high-performance & productivity for users.
# It is an open-source library that is made mainly for working with relational or
# labeled data both easily and intuitively.
# Installing pandas by pip install pandas
# Next import pandas as pd
# Pandas generally provide two data structures for manipulating data, They are:
# 1. Series
# 2. DataFrame

import pandas as pd
import numpy as np


# ----------------------DataFrame--------------------------
# It  is a 2D size-mutable, potentially heterogeneous tabular data structure with labeled axes (rows and columns).
# Pandas DataFrame consists of three principal components, the data, rows, and columns.

d = pd.DataFrame()
print("DataFrame-")
print(d)
dt = {"name":['Dhairya', 'Divyansh'],
      "lastname":["Saraswat", 'Kaushik'],
      "age": [21, 20]}
d = pd.DataFrame(dt)
print(d)
print("Selecting only two columns - ")
# Select only two columns
print(d[['name', 'age']])

# print("Row Selection -")
# data = pd.read_csv("DataBase.csv", index_col="Name")
# f = data.loc['C']
# Selecting a single row
# row2 = data.iloc[3]
# print(row2)
# print(f)

# Working with missing data
# Missing Data can occur when no information is provided for one or more items or for a whole unit.
# Missing Data can also refer to as NA(Not Available) values in pandas.
# isnull()
# notnull()
print("Checking missing data - ")
md = {
      "Maths":[100, 99, np.nan],
      "Science":[98, np.nan, 99],
       "English":[np.nan, 98, 100]
      }
df = pd.DataFrame(md)
print("Checking is null -\n", df.isnull())
print("Checking notnull -\n", df.notnull())

# Filling missing values using fillna(), replace() and interpolate()
# 1. __.fillna(_)
print("Filling the missing values - ")
print(df.fillna('*'))

print("Dropping the missing values - ")
# dropna() function this function drop Rows/Columns of datasets with Null values in different ways.
print(df.dropna())

print("Iteration - ")
# In order to iterate over rows, we can use three function iteritems(), iterrows(), itertuples()
# 1. iterrows()
for i, j in df.iterrows():
    print(i, j)
    print()

# 2. Iterating over columns
print("Value of a particular column - ")
columns = list(df)
for i in columns:
    print(df[i][2])


# -------------------Series-------------------------------
# Series is a one dimensional labeled array capable of holding data of any type.
# labeled axis are called indexes.

s = pd.Series(dtype='float')  # Creating a series
# s = pd.Series()  # Creating a series
# print(s)
print("Series-")
n = np.array(['D', 'H', 'A', 'I', 'R', 'Y', 'A'])
s = pd.Series(n)
print(s)

# Accessing elements from the series -
# 1. Accessing Elements from the series with Position
# 2. Accessing Elements using label(Index)

print("Access Elements -")
print("1. By slice operation -")
print(s[4:])
print("2. By using label(index) -")
l = pd.Series(n, index=["*", "+", "-", "/", "&", "%", "!"])
print(l["%"])
print(l["!"])
# print("If any incorrect index is given, then it will show error.")
# print(l["dd"])
print(l.head(2))
print(l.tail(5))

# Binary Operation on Series

print("Performing Operations -")
s1 = pd.Series([3, 4, 5.0])
s2 = pd.Series([5, 4, 3])
print(s1, "\n", s2)
print("")

print("Adding -")
print(s1.add(s2))

print("Subtraction -")
print(s1.sub(s2))

print("Multiplication -")
print(s1.mul(s2))

print("Division -")
print(s1.div(s2))

print("Modulus -")
print(s1.mod(s2))

# Using range in series
print("Range -")
r = pd.Series(range(10, 15))
print(r)

# Pandas head() method is used to return top n (5 by default) rows of a data frame or series.
# Syntax: Dataframe.head(n=5)

print("Using head and tail -")
top = r.head(2)
print("Head elements -\n", top)

bottom = r.tail(2)
print("Tail Elements -\n", bottom)

print("----------Describe-----------------")
# Pandas describe() is used to view some basic statistical details like percentile, mean, std etc. of a data frame
# or a series of numeric values. When this method is applied to a series of string, it returns a different output.
# Syntax: DataFrame.describe(percentiles=None, include=None, exclude=None)
print("Describe -")
des = np.array([45, 78, 67, 89, 84])
sd = pd.Series(des)
print(sd.describe(include=int))

print("Work done on 6 December 2022")
