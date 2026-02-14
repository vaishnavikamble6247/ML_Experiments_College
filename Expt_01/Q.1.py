# Q. To familiarize students with Python programming essentials and foundational libraries like NumPy and Pandas,
# which are critical for machine learning.

# Basic Python Program

# Variables and Data Types
a = 10
b = 20

print("Addition:", a + b)
print("Multiplication:", a * b)

# Conditional Statement
if a < b:
    print("a is smaller than b")
else:
    print("a is greater than or equal to b")

# Loop
print("Numbers from 1 to 5:")
for i in range(1, 6):
    print(i)

# Function
def square(n):
    return n * n

print("Square of 5:", square(5))

print("------------------------------------------")

import numpy as np

# Create NumPy Array
arr = np.array([10, 20, 30, 40, 50])

print("Array:", arr)

# Basic Operations
print("Mean:", np.mean(arr))
print("Sum:", np.sum(arr))
print("Maximum:", np.max(arr))

# 2D Array
matrix = np.array([[1, 2], [3, 4]])
print("2D Array:\n", matrix)

print("--------------------------------------------")

import pandas as pd

# Create DataFrame
data = {
    "Name": ["Amit", "Sneha", "Rahul"],
    "Marks": [85, 92, 78],
    "Age": [20, 21, 19]
}

df = pd.DataFrame(data)

print("DataFrame:")
print(df)

# Basic Operations
print("Average Marks:", df["Marks"].mean())
print("Maximum Age:", df["Age"].max())

print("----------------------------------------")

import matplotlib.pyplot as plt

students = ["Amit", "Sneha", "Rahul"]
marks = [85, 92, 78]

plt.bar(students, marks)
plt.title("Student Marks")
plt.xlabel("Students")
plt.ylabel("Marks")

plt.show()
