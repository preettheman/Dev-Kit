# 🔢 NumPy Cheat Sheet!

Ready to level up your numerical computing in Python? This NumPy cheat sheet is your essential guide to lightning-fast array operations and powerful mathematical functions! Let's get started!

---

## 🚀 Get Started: Installing NumPy!

First things first, let's get NumPy installed. It's a breeze with `pip`:

```bash
pip install numpy
```
Once installed, you'll typically import it with the widely adopted alias np:

```Python
import numpy as np
```
## 💥 NumPy Arrays (ndarray): The Core!
At the heart of NumPy is the ndarray (N-dimensional array) object. This is a powerful, memory-efficient data structure that provides fast numerical operations compared to Python lists.

## ✨ Why ndarray?

Speed: NumPy operations are implemented in C, making them significantly faster than Python's built-in list operations.

Memory Efficiency: ndarrays store data in a contiguous block of memory, using less space and enabling faster access.

Functionality: Offers a vast collection of high-level mathematical functions to operate on these arrays.

## 🏗️ Array Creation: Your Numerical Building Blocks

```Python
# From a Python list or tuple
arr1d = np.array([1, 2, 3, 4, 5])         # 1-dimensional array
arr2d = np.array([[1, 2], [3, 4]])        # 2-dimensional array

# Arrays filled with zeros or ones
zeros_arr = np.zeros((3, 4))              # 3 rows, 4 columns of zeros
ones_arr = np.ones((2, 3))                # 2 rows, 3 columns of ones

# Empty array (values are typically random/garbage)
empty_arr = np.empty((2, 2))

# Arrays with a range of values
range_arr = np.arange(0, 10, 2)           # start (inclusive), stop (exclusive), step -> [0, 2, 4, 6, 8]
linear_space = np.linspace(0, 1, 5)       # start, stop (inclusive), num_elements -> [0.0, 0.25, 0.5, 0.75, 1.0]

# Arrays with a single, specified value
full_arr = np.full((2, 2), 7)             # 2x2 array filled with 7s

# Identity matrix (square 2D array with ones on the diagonal, zeros elsewhere)
identity_matrix = np.eye(3)               # 3x3 identity matrix

# Random arrays
random_uniform = np.random.rand(2, 2)     # 2x2 array with random floats between 0 and 1
random_integers = np.random.randint(0, 10, size=(3, 3)) # 3x3 array with random integers between 0 and 9
```
## 🧐 Array Attributes: Understanding Your Array's DNA

```Python
arr = np.array([[1, 2, 3], [4, 5, 6]])

arr.shape    # (2, 3) -> 2 rows, 3 columns
arr.ndim     # 2      -> 2 dimensions
arr.size     # 6      -> total number of elements
arr.dtype    # dtype('int64') -> data type of elements
arr.itemsize # 8      -> size in bytes of each element (e.g., 8 bytes for int64)
```
## 🎯 Indexing & Slicing: Pinpointing Your Data!
Accessing elements in NumPy arrays is powerful and flexible.

## 📍 Basic Indexing (1D & 2D)

```Python
arr1 = np.array([10, 20, 30, 40, 50])
arr2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

arr1[0]        # 10 (first element)
arr1[-1]       # 50 (last element)

arr2[0, 0]     # 1 (row 0, column 0)
arr2[1, 2]     # 6 (row 1, column 2)
arr2[2, -1]    # 9 (row 2, last column)
```
## ✂️ Slicing (Like Python lists, but for N-dimensions!)

```Python
arr1 = np.array([10, 20, 30, 40, 50])
arr2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

arr1[1:4]      # [20, 30, 40] (elements from index 1 up to, but not including, 4)
arr1[2:]       # [30, 40, 50] (elements from index 2 to the end)
arr1[:3]       # [10, 20, 30] (elements from start up to, but not including, 3)
arr1[::2]      # [10, 30, 50] (every second element)

arr2[0:2, 1:3] # Sub-array of rows 0-1, columns 1-2: [[2, 3], [5, 6]]
arr2[:, 0]     # First column of all rows: [1, 4, 7]
arr2[1, :]     # Second row of all columns: [4, 5, 6]
```
## ✅ Boolean Indexing: Filtering with Conditions!

```Python
data = np.array([10, 15, 20, 25, 30, 35])

# Select elements greater than 20
data[data > 20] # [25, 30, 35]

# Select elements that are even
data[data % 2 == 0] # [10, 20, 30]

# Combine conditions
data[(data > 15) & (data < 30)] # [20, 25]
```
## 🎩 Fancy Indexing: Selecting Specific Elements

```Python
arr = np.array([10, 20, 30, 40, 50, 60])

# Select elements at specific indices
arr[[0, 3, 5]] # [10, 40, 60]

# For 2D arrays: selecting specific rows and columns based on lists of indices
arr2d_fancy = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2d_fancy[[0, 2], [0, 1]] # Selects (0,0) -> 1, and (2,1) -> 8. Returns [1, 8]
```
## 🔀 Array Manipulation: Reshaping & Restructuring!
Transform your arrays to fit your needs.

## 📐 Reshaping

```Python
arr = np.arange(12) # [0, 1, 2, ..., 11]

arr.reshape(3, 4)     # 3x4 array: [[0,1,2,3], [4,5,6,7], [8,9,10,11]]
arr.reshape(2, 2, 3)  # 2x2x3 array (3D)

arr.ravel()           # Flatten the array into 1D (returns a view)
arr.flatten()         # Flatten the array into 1D (returns a copy)
```
## 🔄 Transposing

```Python
matrix = np.array([[1, 2], [3, 4]])

matrix.T         # [[1, 3], [2, 4]]
np.transpose(matrix) # Same as above
```
## 🧩 Joining Arrays

```Python
a = np.array([1, 2])
b = np.array([3, 4])
c = np.array([[5, 6]])

np.concatenate((a, b)) # [1, 2, 3, 4] (default axis=0 for 1D arrays)

np.vstack((a, b))      # Stack vertically: [[1, 2], [3, 4]]
np.hstack((a, b))      # Stack horizontally: [1, 2, 3, 4]

np.vstack((a, c))      # Can stack 1D and 2D if dimensions match: [[1, 2], [5, 6]]
np.hstack((c.T, c.T))  # Requires matching dimensions along stacking axis
```
## 🔪 Splitting Arrays

```Python
arr = np.arange(9) # [0, 1, 2, 3, 4, 5, 6, 7, 8]

np.split(arr, 3)     # Split into 3 equal arrays: [array([0,1,2]), array([3,4,5]), array([6,7,8])]
np.hsplit(arr.reshape(3, 3), 3) # Split columns horizontally
np.vsplit(arr.reshape(3, 3), 3) # Split rows vertically
```
➕ Array Operations: Math Made Easy!
NumPy excels at performing element-wise and aggregate operations rapidly.

## 🧮 Element-wise Arithmetic

```Python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

a + b  # [5, 7, 9]
a - b  # [-3, -3, -3]
a * b  # [4, 10, 18]
a / b  # [0.25, 0.4, 0.5]
a ** 2 # [1, 4, 9]

np.sqrt(a) # [1.0, 1.414, 1.732]
np.exp(a)  # [2.718, 7.389, 20.085]
np.log(a)  # [0.0, 0.693, 1.098]
```
COMPARISONS (Element-wise)

```Python
a = np.array([1, 2, 3])
b = np.array([3, 2, 1])

a == b # [False, True, False]
a < b  # [True, False, False]
a > b  # [False, False, True]
```
## 📦 Aggregate Functions: Summarizing Your Data!

```Python
arr = np.array([[1, 2, 3], [4, 5, 6]])

np.sum(arr)          # 21 (sum of all elements)
np.mean(arr)         # 3.5 (mean of all elements)
np.max(arr)          # 6 (maximum element)
np.min(arr)          # 1 (minimum element)
np.std(arr)          # 1.707 (standard deviation)

# Operations along an axis
np.sum(arr, axis=0)  # [5, 7, 9] (sum down the columns)
np.sum(arr, axis=1)  # [6, 15] (sum across the rows)

np.argmax(arr, axis=1) # [2, 2] (index of max value for each row)
np.argmin(arr, axis=0) # [0, 0, 0] (index of min value for each column)
```
## ✖️ Matrix Multiplication (Dot Product)

```Python
mat1 = np.array([[1, 2], [3, 4]])
mat2 = np.array([[5, 6], [7, 8]])

np.dot(mat1, mat2) # Standard matrix multiplication
mat1 @ mat2        # Same as np.dot() (Python 3.5+ operator)
```
## 📡 Broadcasting: Smart Operations Across Different Shapes!
Broadcasting is a powerful mechanism that allows NumPy to perform operations on arrays with different shapes. It implicitly "stretches" the smaller array across the larger array so they have compatible shapes for element-wise operations.

```Python
a = np.array([[1, 2, 3], [4, 5, 6]]) # Shape (2, 3)
b = np.array([10, 20, 30])           # Shape (3,)

result = a + b
# 'b' is effectively stretched to [[10, 20, 30], [10, 20, 30]]
# result: [[11, 22, 33], [14, 25, 36]]

c = np.array([[100], [200]])         # Shape (2, 1)

result_c = a + c
# 'c' is effectively stretched to [[100,100,100], [200,200,200]]
# result_c: [[101, 102, 103], [204, 205, 206]]
```
## 💾 Input/Output: Saving & Loading Your Arrays!
Don't let your hard work disappear! Save your NumPy arrays and load them back when needed.

## 📁 Binary Files (.npy, .npz)

These are the standard, most efficient ways to save NumPy arrays.

```Python
data_to_save = np.arange(10).reshape(2, 5)

# Save a single array to a .npy file
np.save('my_array.npy', data_to_save)

# Load it back
loaded_array = np.load('my_array.npy')

# Save multiple arrays to a single .npz file (compressed archive)
np.savez('multiple_arrays.npz', array1=data_to_save, random_data=np.random.rand(2, 2))

# Load multiple arrays (returns a NpzFile object, access by key)
loaded_npz = np.load('multiple_arrays.npz')
array_from_npz = loaded_npz['array1']
random_data_from_npz = loaded_npz['random_data']
```
## 📝 Text Files (.txt)

Useful for sharing data with other software that might not read .npy format, though less efficient for large numerical data.

```Python
data_to_save_txt = np.array([[1.1, 2.2], [3.3, 4.4]])

# Save to a text file
np.savetxt('my_data.txt', data_to_save_txt, delimiter=',', fmt='%.2f') # delimiter and format options

# Load from a text file
loaded_txt_data = np.loadtxt('my_data.txt', delimiter=',')
```
