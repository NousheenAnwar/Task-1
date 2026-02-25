import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: Make plots look better
sns.set_theme(style="whitegrid")

# Load dataset
iris = sns.load_dataset("iris")

# Display basic information about the dataset
print("Shape of dataset:", iris.shape)
#print First 10 rows of dataset
print(iris.head(10))
print("Columns:", iris.columns)

# Summary statistics
print(iris.describe())

#Scatter Plot (Relationship Between Sepal Length and Petal Length)
plt.figure(figsize=(8,6))
sns.scatterplot(data=iris, 
                x="sepal_length", 
                y="petal_length", 
                hue="species")

plt.title("Sepal Length vs Petal Length")
plt.show()

# Plot shows clear separation between species based on sepal and petal lengths. 
# Setosa is distinct, while Versicolor and Virginica overlap more. 
# This suggests that sepal and petal lengths are good features for distinguishing between species, especially for Setosa.

#Histogram (Distribution of Sepal Length)
plt.figure(figsize=(8,6))
sns.histplot(iris["sepal_length"], bins=20, kde=True)

plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Frequency")
plt.show()

# The histogram of sepal length shows a bimodal distribution, with peaks around 5.0 and 6.5. 
# This suggests that there are two groups of sepal lengths in the dataset, likely corresponding to different species. 
# The presence of a peak around 5.0 may be associated with Setosa, while the peak around 6.5 may correspond to Versicolor and Virginica.

#Box Plot (Detect Outliers & Spread)
plt.figure(figsize=(8,6))
sns.boxplot(data=iris, 
            x="species", 
            y="petal_length")

plt.title("Petal Length by Species")
plt.show()

# The box plot of petal length by species shows that Setosa has a much smaller range of petal lengths compared to Versicolor and Virginica. 
# Setosa's petal lengths are tightly clustered around 1.5, while Versicolor and Virginica have wider ranges, with Virginica having the longest petal lengths. 
# There are no significant outliers in the petal length for any of the species, but the spread of petal lengths is much greater for Versicolor and Virginica compared to Setosa.

#pair Plot (Relationships Between All Features)
sns.pairplot(iris, hue="species")
plt.show()
# The pair plot shows that Setosa is clearly separated from the other two species across all feature combinations, while Versicolor and Virginica show more overlap, particularly in sepal length and width. 
# However, petal length and petal width provide better separation between Versicolor and Virginica, with Virginica generally having longer petals. 
# Overall, the pair plot indicates that petal measurements are more effective for distinguishing between species than sepal measurements.
