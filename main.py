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

"""
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
# It gives count, mean , std, min, 25%, 50%, 75% and max with data type - float64
print("Describe -")
des = np.array([45, 78, 67, 89, 84])
sd = pd.Series(des)
print(sd.describe())


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


# Column Addition-
print("Adding a new column in a dataframe")
d = {"name":['Dhairya', 'Divyansh'],
     "age":[21, 20]}
di = pd.DataFrame(d)
print("Before adding a new column -\n", di)
did = [1, 2]
di['Id'] = did
print("After adding a new column -\n", di)

# Column Deletion -
# drop() method - Columns are deleted by dropping columns with column names.
print("Dropping column -")
di.drop(['Id'], axis=1, inplace=True)
print(di)

# Row Selection
# _____.loc[] method is used to retrieve rows from Pandas DataFrame
# Rows can also be selected by passing integer location to an iloc[] function.
print("Row Selection -")
database = pd.read_csv("nba.csv", index_col="Name")
d1 = database.loc["Jordan Mickey"]
print(d1)

# Adding a new row -
print("Adding a new row -")
print(di.head())
new_row = pd.DataFrame({"name":"Drishti", "age":19}, index=[2])
di = pd.concat([new_row, di]).reset_index(drop=True)
print("After adding a new row -")
print(di.head())

# Deleting a row
# Using a drop method to remove a row
print("Dropping a row -")
print("Before Dropping -")
print(di)
print("After Dropping -")
di.drop([0], inplace=True)
print(di)

# Extracting a row by loc
print("Getting a single data")
rw = di.loc[1]
print(rw)
print(type(rw))

# ---------------------------iloc-------------------------
# Dataframe.iloc[] method is used when the index label of a data frame is
# something other than numeric series of 0, 1, 2, 3….n or in case the user doesn’t know the index label.

print("iloc")
nd = pd.read_csv('table.csv')
row1 = nd.loc[3]
row2 = nd.iloc[3]
print(row1)
print("----------")
print(row2)
# print(row1==row2)
row3 = nd.loc[[3, 4, 5, 6]]
row4 = nd.iloc[3:7]  # Getting data is simple through iloc
print(row3)
print("----------")
print(row4)


# Indexing and selecting data with pandas
# Indexing in pandas means simply selecting particular rows and columns of data from a DataFrame.
# It could mean selecting all the rows and some of the columns, some of the rows and all of the columns,
# or some of each of the rows and columns. Indexing can also be known as Subset Selection.
# Pandas support four types of Multi-axes indexing they are:
# Dataframe.[ ] ; This function also known as indexing operator
# Dataframe.loc[ ] : This function is used for labels.
# Dataframe.iloc[ ] : This function is used for positions or integer based
# Dataframe.ix[] : This function is used for both label and integer based
# Collectively, they are called the indexers.


# Selecting a single column -
# Showing data by giving name of a column
print("Selecting a single column ")
col = pd.read_csv('table.csv', index_col='FIRST_NAME')
col_data = col['SALARY']
print(col_data)

# Selecting multiple columns -
print("Selecting multiple columns")
multicolumn = pd.read_csv('table.csv', index_col='EMPLOYEE_ID')
multicolumn_data = multicolumn[['FIRST_NAME', 'SALARY']]
print(multicolumn_data)

# Selecting a single row by loc -
print("Selecting a single row")
rw = pd.read_csv('table.csv', index_col='FIRST_NAME')
r1 = rw.loc['Himanshu']
print(r1)

# Selecting multiple rows by loc-
print("Showing multiple rows")
mrw = pd.read_csv('table.csv', index_col='EMPLOYEE_ID')
r2 = mrw.loc[[105, 107]]
print(r2)

# Selecting two rows and three columns
print("-----------------")
rnc = pd.read_csv("table.csv", index_col='EMPLOYEE_ID')
rc = rnc.loc[[103, 109], ['FIRST_NAME', 'SALARY']]
print(rc)

print("Selecting a single row by iloc -")
idt = pd.read_csv('table.csv', index_col="FIRST_NAME")
r3 = idt.iloc[5]
print(r3)

print("Selecting a single row by iloc -")
idt1 = pd.read_csv('table.csv', index_col="FIRST_NAME")
r4 = idt1.iloc[[5, 7, 9]]
print(r4)

# Selecting two rows and three columns
print("-----------------")
rnc1 = pd.read_csv("table.csv", index_col='EMPLOYEE_ID')
rc1 = rnc1.iloc[[3, 8], [0, 3]]
print(rc1)

# Selecting a single row using .ix[] as .loc[]
# In order to select a single row, we put a single row label in a .ix function.
# This function act similar as .loc[] if we pass a row label as a argument of a function.
# print("-----------------")
# ri = pd.read_csv("table.csv", index_col='EMPLOYEE_ID')
# ri1 = ri.loc[107]
# print(ri1)
# This is removed from python pandas.
print("-----------------------------------------------")


# Boolean Indexing -
# In boolean indexing, we will select subsets of data based on the actual values of
# the data in the DataFrame and not on their row/column labels or integer locations.

dict = {'name': ["A", "B", "C", "D"],
        'score': [90, 85, 80, 98]
        }
df = pd.DataFrame(dict, index=[True, False, True, False])

print(df)

# Accessing a Dataframe with a boolean index using .loc[]

# In order to access a dataframe with a boolean index using .loc[],
# simply pass a boolean value (True or False) in a .loc[] function.

print("Boolean index by loc -")
print(df.loc[True])

# Accessing a Dataframe with a boolean index using .iloc[]
# In order to access a dataframe using .iloc[], we have to pass a boolean value (True or False)
# but iloc[] function accepts only integer as an argument so it will throw an error
# Example -

# print("Boolean index by iloc -")
# print(df.iloc[True])
# print("It will show TypeError as only integers value can be passed.")

print("Correct method by iloc is by passing integer -")
print(df.iloc[0])
print(df.iloc[1])
print(df.iloc[2])
print(df.iloc[3])

# Applying boolean mask to a dataframe -

print("----------------------------------------")
tb = pd.read_csv('table.csv')

dt = pd.DataFrame(tb, index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

print(dt[[True, False, True, False, True, False, True, False, True, False]])

# Masking data based on column value
# In a dataframe we can filter a data based on a column value.
# In order to filter data, we can apply certain conditions on the dataframe using different operators
# like ==, >, <, <=, >=. When we apply these operators to the dataframe then it produces a Series of True and False.
print("---------------------------")
print(df['name'] == 'C')
print("Ex-2")
data = pd.read_csv("nba.csv", index_col="Name")
print(data['Age'] > 25)
print(data['Age'] == 29)


# ------------------------------Group By ---------------------------------
# creating a grouping of categories and apply a function to the categories.
# Splitting : It is a process in which we split data into group by applying some conditions on datasets.
# Applying : It is a process in which we apply a function to each group independently
# Combining : It is a process in which different datasets are combined after applying groupby

# Splitting data into groups -
# Grouping data with one key

data1 = {'Name':['Jai', 'Anuj', 'Jai', 'Prince',
                 'Gaurav', 'Anuj', 'Prince', 'Abhi'],
        'Age':[27, 24, 22, 32,
               33, 36, 27, 32],
        'Address':['Nagpur', 'Kanpur', 'Allahabad', 'Kannuaj',
                   'Jaunpur', 'Kanpur', 'Allahabad', 'Aligarh'],
        'Qualification':['Msc', 'MA', 'MCA', 'Phd',
                         'B.Tech', 'B.com', 'Msc', 'MA'],
         'Score': [23, 34, 35, 45, 47, 50, 52, 53]}

df = pd.DataFrame(data1)
# --
gn = df.groupby('Name')
print(df.groupby('Name').groups)  # Giving index
print(gn.first())

# Grouping data with multiple keys -
print("\nGrouping by multiple keys -")
asq = df.groupby(['Name', 'Qualification'])
print(asq.groups)
print(asq.first())

# Grouping data by sorting keys
# Group keys are sorted by default using the groupby operation.
# print("----------------------")
# print(df.groupby(['Name']).sum())
# print(df.groupby(['Name'], sort=False).sum())

# -------------------Iterating through groups---------------------------
print("Iteration -\n")
g = df.groupby('Name')
for name, group in g:
        print(name)
        print(group)
        print("")

# -------------------Selecting a group----------------------------------
# get_group('__') - Selecting a particular group among many groups
print("Selecting a group -")
g1 = df.groupby('Name')
print(g1.get_group('Prince'))
g2 = df.groupby(['Name', 'Qualification'])
print(g2.get_group(('Anuj', 'MA')))


# -------------------Applying function-----------------------------------
# Aggregation -
print("\nApplying functions- ")
grp = df.groupby('Name')
nm = grp.aggregate(np.sum)
print(nm)
# Multiple Functions -
print(grp['Age'].agg([np.sum, np.mean]))
print("\n", grp.agg({'Age':'sum', 'Score':'mean'}))


# Pandas Merging, Joining and Concatenating
#  --1--Concatenating DataFrame using .concat()
# concat() function this function concat a dataframe and returns a new dataframe.
d1 = {"Class":["1", "2"],
      "Section":["A", "B"]}
d2 = {"Class":["3", "4"],
      "Section":["C", "D"]}
df1 = pd.DataFrame(d1, index=[0, 1])
df2 = pd.DataFrame(d2, index=[2, 3])
print("Dataframe 1\n", df1, "\nDataFrame 2\n", df2)
print("After concatenation-")
print(pd.concat([df1, df2]))

# Concatenating DataFrame with group keys
print("Group keys -")
print(pd.concat([df1, df2], keys=['K1', 'K2']))

# Concatenating DataFrame by setting logic on axes
df1 = pd.DataFrame(d1, index=[0, 1])
df2 = pd.DataFrame(d2, index=[1, 2])
print("Dataframe 1\n", df1, "\nDataFrame 2\n", df2)
print("After concatenation by logic axes-")
print(pd.concat([df1, df2], axis=0, join='inner'))
print("----")
print(pd.concat([df1, df2], axis=1, join='inner'))
print("----")
# print(pd.concat([df1, df2], axis=1, sort=False))
print(pd.concat([df1, df2], axis=1, sort=False))

# Concatenating DataFrame using .append()
print("Appending -")
print(df1.append(df2))

# Concatenating DataFrame by ignoring indexes :
df3 = pd.DataFrame(d1, index=[0, 1])
df4 = pd.DataFrame(d2, index=[0, 1])
print("Result shown by ignoring indexes-")
print(pd.concat([df3, df4], ignore_index=True))

#------------------

data1 = {'key': ['K0', 'K1', 'K2', 'K3'],
         'Name':['Jai', 'Princi', 'Gaurav', 'Anuj'], 
        'Age':[27, 24, 22, 32],} 
    
data2 = {'key': ['K0', 'K1', 'K2', 'K3'],
         'Address':['Nagpur', 'Kanpur', 'Allahabad', 'Kannuaj'], 
        'Qualification':['Btech', 'B.A', 'Bcom', 'B.hons']} 
 
df = pd.DataFrame(data1)
  
df1 = pd.DataFrame(data2) 
  
print(df, "\n", df1) 

print(pd.merge(df, df1, on='key'))
"""










