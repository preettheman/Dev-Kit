# Python Data Science Power-Up Cheat Sheet!

Ready to supercharge your data science journey with Python? This cheat sheet is your trusty sidekick for navigating the essential libraries and making data sing! Let's dive in!

---

## 💡 Python Data Types: Your Building Blocks

Before we get to the cool libraries, let's quickly recap the fundamental data types that are the very fabric of Python. Understanding these will make everything else click!

* **Integers (`int`)**: 🔢 Whole numbers without decimals. Perfect for counting things, like the number of students or rows in your dataset.
    * *Example:* `num_students = 25`, `user_id = 1001`
* **Floats (`float`)**: 🌊 Numbers with a decimal point. Ideal for measurements, temperatures, or any continuous values.
    * *Example:* `temperature = 23.5`, `price = 99.99`
* **Strings (`str`)**: 📝 Sequences of characters. This is how Python handles text – names, addresses, descriptions, you name it!
    * *Example:* `name = "Alice"`, `city = "New York"`
* **Booleans (`bool`)**: ✅ Simply `True` or `False`. These are your decision-makers, great for checking conditions or flags.
    * *Example:* `is_active = True`, `data_cleaned = False`
* **Lists (`list`)**: \[ ] Ordered, **changeable** collections. Think of them as dynamic shopping lists where you can add, remove, or reorder items. Super versatile!
    * *Example:* `numbers = [1, 2, 3, 4]`, `my_hobbies = ["reading", "coding", "hiking"]`
* **Dictionaries (`dict`)**: { } Unordered collections of **key-value pairs**. Like a real-world dictionary, you look up a "word" (key) to get its "definition" (value). Perfect for structured data.
    * *Example:* `person = {"name": "Bob", "age": 30, "city": "London"}`, `product = {"id": "P001", "price": 49.99}`
* **Tuples (`tuple`)**: ( ) Ordered, **unchangeable** collections. Once you create a tuple, its elements are set. Great for things that shouldn't change, like coordinates.
    * *Example:* `coordinates = (latitude, longitude)`, `rgb_color = (255, 0, 0)`
* **Sets (`set`)**: { } Unordered collections of **unique** items. If you need to find all distinct values or check for membership efficiently, sets are your go-to!
    * *Example:* `unique_tags = {"python", "data", "science", "python"}` (will be `{"python", "data", "science"}`)

---

## 🐼 Pandas: Your Data's Best Friend!

Pandas is the rockstar of data manipulation and analysis, making it a breeze to work with tabular data using its awesome **DataFrames**!

### 📊 **DataFrame Creation & Inspection: Get to Know Your Data!**

```python
import pandas as pd
import numpy as np # Used for np.nan in later examples

# Create a DataFrame from a dictionary – super common!
data = {'feature_A': [10, 20, 30], 'category_B': ['X', 'Y', 'Z'], 'value_C': [1.1, 2.2, 3.3]}
df = pd.DataFrame(data)

# Load data from a CSV file – your daily routine!
# df_from_csv = pd.read_csv('your_awesome_data.csv')

# Peek at the first few rows (default 5) – a quick glance!
df.head()

# Glance at the last few rows – checking for late entries!
df.tail()

# Get a concise summary: types, non-nulls – your data's health report!
df.info()

# Statistical summary for numerical columns – quick insights!
df.describe()

# See all your column names – know what you're working with!
df.columns

# Check the shape (rows, columns) – how big is this beast?!
df.shape

```

⚙️ Selection & Filtering: Finding What Matters!

```python
# Select a single column – grab that specific data stream!
df['feature_A']

# Select multiple columns – multitasking like a pro!
df[['feature_A', 'category_B']]

# Select by row label (index) – pinpointing a specific record!
df.loc[0]

# Select by integer position – when you know exactly where it is!
df.iloc[1]

# Filter rows based on a condition – sifting for gold!
df[df['feature_A'] > 15]

# Filter with multiple conditions – getting super specific!
df[(df['feature_A'] > 15) & (df['category_B'] == 'Y')]
```
🔄 Data Cleaning & Transformation: Sculpting Your Data!
```python
# Create a DataFrame with some missing values for demonstration
df_messy = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': ['apple', np.nan, 'orange', 'grape'],
    'C': [True, False, True, np.nan]
})

# Drop rows with any missing values – cleaning house!
df_messy.dropna()

# Fill missing values – giving your data a second chance!
df_messy.fillna(value=0)

# Drop a column – when a feature isn't pulling its weight!
df_messy.drop('C', axis=1)

# Group by a column and calculate statistics – aggregate for insights!
# Using our original df for this
df.groupby('category_B')['feature_A'].mean()

# Apply a function to a column – customize your features!
df['feature_A_doubled'] = df['feature_A'].apply(lambda x: x * 2)
# The DataFrame 'df' now includes 'feature_A_doubled'
```
🔢 NumPy: The Powerhouse for Numbers!
NumPy is the bedrock of numerical computing in Python, especially when you're dealing with large arrays and matrices. It's super fast!

📝 Array Creation & Basics: Your Numerical Canvas
```python
import numpy as np

# Create a 1D array – your simple list, but faster!
arr1d = np.array([1, 2, 3, 4, 5])

# Create a 2D array (matrix) – for when your data gets multidimensional!
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Create an array of zeros – for placeholders or initial states!
zeros_array = np.zeros((3, 4)) # 3 rows, 4 columns

# Create an array of ones – similar to zeros, but with ones!
ones_array = np.ones((2, 2)) # 2 rows, 2 columns

# Create an array with a range of values – handy for sequences!
arange_array = np.arange(0, 10, 2) # start, stop (exclusive), step

# Get array shape – how many dimensions and elements?
arr2d.shape

# Get array data type – what kind of numbers are these?
arr1d.dtype
```

🧮 Array Operations: Crunching Numbers Like a Boss!
```python
# Element-wise addition – add arrays item by item!
arr_a = np.array([1, 2])
arr_b = np.array([3, 4])
sum_element_wise = arr_a + arr_b # result: [4, 6]

# Matrix multiplication (dot product) – essential for linear algebra!
mat_x = np.array([[1, 2], [3, 4]])
mat_y = np.array([[5, 6], [7, 8]])
dot_product = np.dot(mat_x, mat_y)

# Transpose an array – flip rows and columns!
transposed_mat_x = mat_x.T

# Calculate mean, sum, max, min – quick stats on your array!
np.mean(arr1d)
np.sum(arr1d)
np.max(arr1d)
np.min(arr1d)
```
📈 Matplotlib: Paint Your Data Story!
Matplotlib is your artistic tool for creating stunning visualizations. From simple lines to complex plots, it helps you tell your data's story visually!

🖼️ Basic Plotting: Your First Masterpiece!
```python
import matplotlib.pyplot as plt
import numpy as np

# Sample data – let's make some waves!
x_values = np.linspace(0, 10, 100)
y_sine = np.sin(x_values)
y_cosine = np.cos(x_values)

# Create a simple line plot – elegant and clear!
plt.figure(figsize=(8, 4)) # Make it look good!
plt.plot(x_values, y_sine, label='Sine Wave', color='teal', linewidth=2)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.title('Beautiful Sine Wave', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Create a scatter plot – showing individual data points!
plt.figure(figsize=(8, 4))
plt.scatter(x_values, y_cosine, label='Cosine Scatter', color='purple', marker='o', s=20, alpha=0.7)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.title('Scatter of Cosine Values', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.5)
plt.show()
```
📊 Advanced Plot Types: Beyond the Basics!
```python
# Histogram – see the distribution of your data!
random_data = np.random.randn(1000) # 1000 random numbers
plt.figure(figsize=(8, 4))
plt.hist(random_data, bins=30, alpha=0.8, color='skyblue', edgecolor='black')
plt.title('Histogram of Random Data Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Bar plot – comparing categories head-to-head!
categories = ['Apples', 'Bananas', 'Oranges']
fruit_counts = [10, 25, 15]
plt.figure(figsize=(7, 5))
plt.bar(categories, fruit_counts, color=['red', 'yellow', 'orange'])
plt.title('Fruit Count Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Fruit Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()

# Subplots – multiple plots in one figure, telling a bigger story!
fig, axes = plt.subplots(1, 2, figsize=(12, 5)) # 1 row, 2 columns

# Plot 1: Line plot
axes[0].plot(x_values, y_sine, color='green')
axes[0].set_title('Sine Wave', fontsize=12)
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].grid(True, linestyle=':', alpha=0.6)

# Plot 2: Scatter plot
axes[1].scatter(x_values, y_cosine, color='darkblue', alpha=0.7)
axes[1].set_title('Cosine Scatter', fontsize=12)
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
axes[1].grid(True, linestyle=':', alpha=0.6)

plt.tight_layout() # Makes sure plots don't overlap – crucial!
plt.suptitle('Side-by-Side Data Views!', fontsize=16, fontweight='bold', y=1.02) # Overall title
plt.show()
```
🤖 Scikit-learn: Your Machine Learning Engine!
Scikit-learn is the go-to library for building powerful predictive models. It's built on NumPy, SciPy, and Matplotlib, making it super integrated and efficient for machine learning tasks!

Preprocessing 🧹: Getting Your Data ML-Ready!
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Sample data – our little dataset for practice!
X_features = np.array([[10, 2.5, 0], [20, 3.1, 1], [30, 2.8, 0], [40, 3.9, 1], [50, 3.5, 0]])
y_target = np.array([0, 1, 0, 1, 0]) # Some labels for classification

# Categorical data for encoding example
df_categorical = pd.DataFrame({'color': ['red', 'blue', 'green', 'red'], 'size': ['S', 'M', 'L', 'S']})

# Splitting data into training and testing sets – essential for evaluating your model fairly!
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=42)

# Standardization (mean=0, std=1) – for features with different scales!
scaler_std = StandardScaler()
X_scaled_std = scaler_std.fit_transform(X_features)

# Normalization (min=0, max=1) – another way to scale features!
scaler_minmax = MinMaxScaler()
X_scaled_minmax = scaler_minmax.fit_transform(X_features)

# One-Hot Encoding for categorical features – turning text into numbers for ML!
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_encoded = encoder.fit_transform(df_categorical[['color']])
encoder.get_feature_names_out(['color']) # Get the names of the encoded features
```
🧠 Model Training & Prediction: Teaching Your Computer to Learn!
```python
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification, make_regression
import numpy as np

# Generate some synthetic data for demonstration
X_clf, y_clf = make_classification(n_samples=100, n_features=4, n_informative=2, n_redundant=0, random_state=42)
X_reg, y_reg = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# Split data for training (using 80% for train, 20% for test)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)


# Linear Regression (for predicting continuous numbers) – drawing a straight line through your data!
model_lr = LinearRegression()
model_lr.fit(X_train_reg, y_train_reg)
predictions_lr = model_lr.predict(X_test_reg)

# Logistic Regression (for binary classification: Yes/No, 0/1) – classifying with confidence!
model_logreg = LogisticRegression(random_state=42)
model_logreg.fit(X_train_clf, y_train_clf)
predictions_logreg = model_logreg.predict(X_test_clf)

# Decision Tree Classifier (making decisions like a flowchart!)
model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train_clf, y_train_clf)
predictions_dt = model_dt.predict(X_test_clf)

# Support Vector Classifier (finding the best boundary to separate classes!)
model_svc = SVC(random_state=42)
model_svc.fit(X_train_clf, y_train_clf)
predictions_svc = model_svc.predict(X_test_clf)

# K-Means Clustering (unsupervised: finding hidden groups in your data!)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10) # n_init is important for stability
clusters = kmeans.fit_predict(X_clf) # No y_target needed, it's unsupervised!
```
✅ Model Evaluation: How Good is Your Model?
```python
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
import numpy as np

# Let's use some dummy data for evaluation examples
true_reg_values = np.array([10, 12, 11, 13, 9])
pred_reg_values = np.array([10.5, 11.8, 10.9, 13.2, 9.1])

true_clf_labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
pred_clf_labels = np.array([0, 1, 0, 0, 0, 1, 0, 1])


# For Regression: Mean Squared Error (MSE) – how far off are your predictions on average?
mse = mean_squared_error(true_reg_values, pred_reg_values)

# For Classification: Accuracy Score – how many did your model get right?
accuracy = accuracy_score(true_clf_labels, pred_clf_labels)

# For Classification: Classification Report – a detailed breakdown of performance (precision, recall, f1-score)!
report = classification_report(true_clf_labels, pred_clf_labels)

# For Classification: Confusion Matrix – seeing where your model got confused!
conf_matrix = confusion_matrix(true_clf_labels, pred_clf_labels)
```
