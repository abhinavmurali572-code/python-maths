# Assignment 1: Linear Algebra and Probability - Wine Quality Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ================= TASK 1: Load dataset and handle missing data (2 Marks) =================
# Load the dataset
df = pd.read_csv('winequality.csv')  # replace with your file path
# If red and white separate, use: pd.read_csv('winequality-red.csv', sep=';')

print(f"Shape: {df.shape}")  # Should be 1599 x 12
print(df.head())
print(df.info())
print("Null values before:\n", df.isnull().sum())

# Handle missing data by replacing null with mean of respective column
# For numeric columns only
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    mean_val = df[col].mean()
    df[col].fillna(mean_val, inplace=True)

print("Null values after:\n", df.isnull().sum())

# ================= TASK 2: Extract columns as vectors (2 Marks) =================
# Extract alcohol, citric acid as vectors
alcohol_vector = df['alcohol'].values  # Vector 1
citric_acid_vector = df['citric acid'].values  # Vector 2  (note: might be 'citric_acid' or 'citric acid' - check your csv)

print(f"Alcohol vector shape: {alcohol_vector.shape}")
print(f"Citric acid vector shape: {citric_acid_vector.shape}")
print(f"First 5 values of alcohol: {alcohol_vector[:5]}")
print(f"First 5 values of citric acid: {citric_acid_vector[:5]}")

# Vector operations example (Linear Algebra)
dot_product = np.dot(alcohol_vector, citric_acid_vector)
magnitude_alcohol = np.linalg.norm(alcohol_vector)
magnitude_citric = np.linalg.norm(citric_acid_vector)

print(f"Dot Product: {dot_product}")
print(f"Magnitude of alcohol vector: {magnitude_alcohol}")
print(f"Magnitude of citric acid vector: {magnitude_citric}")

# ================= TASK 3: Covariance Matrix (2 Marks) =================
# Select two features e.g., alcohol and density
# Check your column names - might be 'density' or 'alcohol'
feature_1 = 'alcohol'
feature_2 = 'density'  # you can also use 'pH' or 'sulphates'

X = df[[feature_1, feature_2]].values  # Feature matrix (1599 x 2)
print(f"Feature matrix X shape: {X.shape}")

# Calculate covariance matrix using np.cov(X.T)
cov_matrix = np.cov(X.T)
print("Covariance Matrix:\n", cov_matrix)
print(f"Cov(X) shape: {cov_matrix.shape}")

# Interpretation: diagonal = variance, off-diagonal = covariance

# ================= TASK 4: Eigen Decomposition (2 Marks) =================
# Perform eigen decomposition on covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("Eigenvalues:\n", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# Identify top 2 eigenvalues and corresponding eigenvectors
# Sort eigenvalues in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

print(f"\nTop 2 Eigenvalues:")
print(f"1st Eigenvalue: {sorted_eigenvalues[0]}")
print(f"2nd Eigenvalue: {sorted_eigenvalues[1]}")

print(f"\nCorresponding Eigenvectors:")
print(f"Eigenvector for 1st eigenvalue: {sorted_eigenvectors[:, 0]}")
print(f"Eigenvector for 2nd eigenvalue: {sorted_eigenvectors[:, 1]}")

# Interpretation
print("""
INTERPRETATION:
- Eigenvalues represent variance explained by each principal component.
- Larger eigenvalue = more variance captured in that direction.
- Eigenvectors represent the direction of maximum variance.
- Eigenvector 1 shows the primary axis where data varies most (mostly along alcohol if larger).
- This is the basis of PCA - used for dimensionality reduction.
""")

# Optional visualization of eigen decomposition
plt.figure()
plt.scatter(X[:,0], X[:,1], alpha=0.3, label='Data points')
# Plot eigenvectors scaled by eigenvalues
origin = np.mean(X, axis=0)
for i in range(2):
    vec = sorted_eigenvectors[:, i] * np.sqrt(sorted_eigenvalues[i]) * 3
    plt.arrow(origin[0], origin[1], vec[0], vec[1], color='r', width=0.02, head_width=0.1)
plt.xlabel(feature_1)
plt.ylabel(feature_2)
plt.title(f'Eigen Decomposition - {feature_1} vs {feature_2}')
plt.legend()
plt.show()

# ================= TASK 5: Most common wine quality (2 Marks) =================
quality_counts = df['quality'].value_counts().sort_index()
most_common_quality = df['quality'].mode()[0]
print(quality_counts)
print(f"Most common wine quality: {most_common_quality}")

# Visualization
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
sns.countplot(x='quality', data=df, palette='viridis')
plt.title('Distribution of Wine Quality Scores')
plt.xlabel('Quality Score')
plt.ylabel('Count')

plt.subplot(1,2,2)
quality_counts.plot(kind='bar', color='coral', edgecolor='black')
plt.title('Wine Quality Frequency')
plt.xlabel('Quality')
plt.ylabel('Frequency')
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()

# Probability interpretation
total = len(df)
prob_distribution = quality_counts / total
print("\nProbability Distribution of Quality:")
print(prob_distribution)

print(f"""
INTERPRETATION:
- Most common quality is {most_common_quality} with {quality_counts.max()} wines.
- Distribution is approximately Normal / Bell-shaped, centered around 5-6.
- Very few wines have quality 3 or 8-9, indicating extreme qualities are rare.
- P(Quality = {most_common_quality}) = {prob_distribution.max():.3f} or {prob_distribution.max()*100:.1f}%
- This suggests majority wines are average quality, winemakers produce mostly mid-range wines.
""")
