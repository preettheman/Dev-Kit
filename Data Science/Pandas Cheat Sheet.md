# 🚀 Pandas Cheat Sheet!

Ready to supercharge your data science journey with Python? This cheat sheet is your trusty sidekick for mastering Pandas and making your data sing! Let's dive in!

---

## 🚀 Get Started: Installing Pandas!

Before you can work your data magic, you need to install Pandas. It's super easy with `pip`:

```bash
pip install pandas
````

Once installed, you'll typically import it like this in your scripts:

```python
import pandas as pd
import numpy as np # Often used alongside Pandas for numerical operations
```

-----

## 🐼 Pandas Fundamentals: Series & DataFrames

Pandas introduces two core data structures that are the backbone of almost everything you'll do:

### 🌟 **Series: Your Super-Powered 1D Array**

A **Series** is like a single column in a spreadsheet or a NumPy array with an index. It can hold any data type.

```python
# Create a Series from a list
s = pd.Series([10, 20, 30, 40, 50], name="My_Numbers")
# print(s) # Uncomment to see output

# Access elements by index
s[0] # Returns 10
s[2:4] # Returns a slice

# Series with custom index
s_indexed = pd.Series([1.5, 2.3, 4.1], index=['a', 'b', 'c'])
# print(s_indexed) # Uncomment to see output
```

### 🖼️ **DataFrame: Your Tabular Data Powerhouse**

A **DataFrame** is a 2-dimensional labeled data structure with columns that can be of different types. Think of it as a spreadsheet or a SQL table. It's essentially a collection of Series objects sharing the same index.

-----

## 🏗️ DataFrame Creation & Loading Data: Filling Your Sheets\!

### 🧱 **Creating DataFrames**

```python
# From a dictionary (most common way!)
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 28],
    'City': ['New York', 'London', 'Paris', 'New York']
}
df = pd.DataFrame(data)
# print(df) # Uncomment to see output

# From a list of lists (specify columns)
list_data = [[1, 'A'], [2, 'B'], [3, 'C']]
df_from_list = pd.DataFrame(list_data, columns=['ID', 'Label'])
# print(df_from_list) # Uncomment to see output

# From a NumPy array (specify columns)
arr_data = np.array([[100, 10.5], [101, 12.3]])
df_from_array = pd.DataFrame(arr_data, columns=['ProductID', 'Price'])
# print(df_from_array) # Uncomment to see output
```

### 📥 **Loading Data from Files**

```python
# From CSV (Comma Separated Values) – your daily bread and butter!
# df_csv = pd.read_csv('your_data.csv')

# From Excel files
# df_excel = pd.read_excel('your_data.xlsx', sheet_name='Sheet1')

# From SQL databases (requires SQLAlchemy and database-specific drivers)
# from sqlalchemy import create_engine
# engine = create_engine('postgresql://user:pass@host:port/db_name')
# df_sql = pd.read_sql('SELECT * FROM your_table', engine)

# From JSON files
# df_json = pd.read_json('your_data.json')
```

-----

## 🧐 Inspection & Summary: Getting to Know Your Data\!

Once your data is loaded, these commands help you get a quick overview.

### 👓 **Quick Peeks**

```python
# Show the first 5 rows (default)
df.head()

# Show the last 5 rows (default)
df.tail()

# Show first N rows
df.head(10)

# Show a random sample of N rows
df.sample(3)
```

### 📊 **Summary Statistics**

```python
# Concise summary: index, dtype, non-null values, memory usage
df.info()

# Descriptive statistics for numerical columns (count, mean, std, min, max, quartiles)
df.describe()

# Get unique values and their counts for a specific column
df['City'].value_counts()

# Get the number of unique values in a column
df['City'].nunique()

# Get unique values in a column
df['City'].unique()

# Get column data types
df.dtypes
```

### 📏 **Shape & Size**

```python
# Get (rows, columns) tuple
df.shape

# Get a list of column names
df.columns

# Get the index (row labels)
df.index
```

-----

## 🎯 Selection & Filtering: Finding What Matters\!

These are fundamental for targeting specific parts of your DataFrame.

### 🏷️ **Column Selection**

```python
# Select a single column (returns a Series)
df['Name']

# Select multiple columns (returns a DataFrame)
df[['Name', 'Age']]
```

### 📍 **Row Selection (by label & position)**

```python
# Select row(s) by label(s) (using the index)
df.loc[0] # Selects the first row by its label 0
df.loc[0:2, ['Name', 'Age']] # Slice rows 0 to 2, and specific columns

# Select row(s) by integer position
df.iloc[0] # Selects the first row by its integer position 0
df.iloc[0:2, 0:2] # Slice rows 0 to 2, and columns 0 to 2 (exclusive)
```

### 🔎 **Filtering Data (Conditional Selection)**

```python
# Filter rows where Age is greater than 30
df[df['Age'] > 30]

# Filter with multiple conditions (use & for AND, | for OR)
df[(df['Age'] > 28) & (df['City'] == 'New York')]

# Filter using .isin() for multiple values
df[df['City'].isin(['London', 'Paris'])]

# Filter for rows where a string column contains a substring
df[df['Name'].str.contains('a', case=False)] # case=False for case-insensitive
```

-----

## 🧹 Data Cleaning: Making Your Data Shine\!

Dealing with missing values and duplicates is a critical step in data preparation.

### ❓ **Missing Values (`NaN`)**

```python
# Create a DataFrame with missing values for demonstration
df_missing = pd.DataFrame({
    'Col1': [1, 2, np.nan, 4],
    'Col2': ['A', 'B', 'C', np.nan],
    'Col3': [True, np.nan, False, True]
})

# Check for missing values (returns boolean DataFrame)
df_missing.isnull()
df_missing.isna() # Alias for isnull()

# Count missing values per column
df_missing.isnull().sum()

# Drop rows with any missing values
df_missing.dropna()

# Drop columns with any missing values
df_missing.dropna(axis=1)

# Fill missing values with a specific value
df_missing.fillna(0) # Fills all NaN with 0

# Fill missing numerical values with the mean of the column
df_missing['Col1'].fillna(df_missing['Col1'].mean())

# Fill missing categorical values with the mode
df_missing['Col2'].fillna(df_missing['Col2'].mode()[0])

# Forward fill (propagates last valid observation forward)
df_missing.fillna(method='ffill')

# Backward fill (propagates next valid observation backward)
df_missing.fillna(method='bfill')
```

### 👯 **Duplicates**

```python
# Create a DataFrame with duplicates for demonstration
df_duplicates = pd.DataFrame({
    'A': [1, 2, 2, 3, 4],
    'B': ['x', 'y', 'y', 'z', 'x']
})

# Check for duplicate rows (returns boolean Series)
df_duplicates.duplicated()

# Drop duplicate rows (keeps the first occurrence by default)
df_duplicates.drop_duplicates()

# Drop duplicates based on specific columns
df_duplicates.drop_duplicates(subset=['A'])

# Keep the last occurrence of duplicates
df_duplicates.drop_duplicates(keep='last')
```

-----

## ⚙️ Data Manipulation: Transforming Your Data\!

These functions allow you to reshape, group, and combine your data.

### ➕ **Adding/Modifying Columns**

```python
# Add a new column based on existing ones
df['Age_Plus_5'] = df['Age'] + 5

# Create a new column with a constant value
df['Status'] = 'Active'

# Use .apply() for complex column creation
df['Name_Length'] = df['Name'].apply(len)

# Use np.where for conditional column creation
df['Senior'] = np.where(df['Age'] >= 30, 'Yes', 'No')
```

### 🔄 **Renaming Columns**

```python
# Rename a single column
df.rename(columns={'Name': 'Full_Name'})

# Rename multiple columns
df.rename(columns={'Name': 'Full_Name', 'Age': 'Years_Old'})
```

### 🔀 **Sorting Data**

```python
# Sort by a single column
df.sort_values(by='Age')

# Sort by multiple columns
df.sort_values(by=['City', 'Age'], ascending=[True, False]) # City ascending, Age descending
```

### 🤝 **Merging & Joining DataFrames**

```python
# Create another DataFrame for merging
df_orders = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Alice', 'Charlie'],
    'Order_ID': [101, 102, 103, 104]
})

# Merge DataFrames (like SQL JOIN)
# inner join (default, only matching rows)
df_merged = pd.merge(df, df_orders, on='Name', how='inner')
# print(df_merged) # Uncomment to see output

# left join (keep all rows from left, add matching from right)
df_left_join = pd.merge(df, df_orders, on='Name', how='left')
# print(df_left_join) # Uncomment to see output

# right join
# full outer join
```

### 📏 **Concatenating DataFrames**

```python
# Stack DataFrames vertically (add rows)
df_more_people = pd.DataFrame({'Name': ['Eve'], 'Age': [29], 'City': ['Berlin']})
df_combined_rows = pd.concat([df, df_more_people], ignore_index=True)
# print(df_combined_rows) # Uncomment to see output

# Combine DataFrames horizontally (add columns) - be careful with indices!
df_contact = pd.DataFrame({'Email': ['a@example.com', 'b@example.com'], 'Phone': ['111', '222']})
# Assuming indices align for df and df_contact
# df_combined_cols = pd.concat([df, df_contact], axis=1)
```

### 👨‍👩‍👧‍👦 **Group By: Summarizing Data**

```python
# Group by 'City' and calculate the mean age
df.groupby('City')['Age'].mean()

# Group by multiple columns and get multiple aggregations
df.groupby('City').agg(
    Avg_Age=('Age', 'mean'),
    Min_Age=('Age', 'min'),
    Total_People=('Name', 'count')
)

# Apply a custom function after grouping
def custom_range(x):
    return x.max() - x.min()

df.groupby('City')['Age'].apply(custom_range)
```

-----

## ⏰ Time Series: Working with Dates & Times\!

Pandas excels at handling time-series data.

```python
# Convert a column to datetime objects
df_dates = pd.DataFrame({'Date': ['2023-01-01', '2023-01-02', '2023-01-03'], 'Value': [10, 15, 20]})
df_dates['Date'] = pd.to_datetime(df_dates['Date'])

# Set Date column as index
df_dates = df_dates.set_index('Date')

# Resample data (e.g., daily to weekly sum)
# df_dates.resample('W').sum()

# Access date components
# df_dates.index.year
# df_dates.index.month
# df_dates.index.day_name()
```

-----

## 💾 Input/Output: Saving Your Work\!

Once you've cleaned and manipulated your data, you'll want to save it\!

```python
# Save to CSV
df.to_csv('cleaned_data.csv', index=False) # index=False prevents writing the DataFrame index as a column

# Save to Excel
df.to_excel('cleaned_data.xlsx', index=False, sheet_name='Processed_Data')

# Save to JSON
df.to_json('cleaned_data.json', orient='records', indent=4)
```
