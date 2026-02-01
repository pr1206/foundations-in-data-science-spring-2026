import sklearn
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
data = {"weight": [4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14, 4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89,
                   4.32, 4.69, 6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26], "group": ["ctrl"] * 10 + ["trt1"] * 10 + ["trt2"] * 10}
PlantGrowth = pd.DataFrame(data)

# Question 1a
iris_df.info()
plt.hist(iris_df['sepal width (cm)'], bins=15)
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.title("Sepal Width Distribution")
plt.show()

# Question 1b
# The mean should be higher than the median. This histogram is slightly right-skewed,
# with the tail extending out into values between 4.0 cm and 4.5 cm. These outlier values
# will raise the overall mean and therefore create the expectation of a higher mean relative
# to the median.

# Question 1c
print('Mean: ' + str(iris_df['sepal width (cm)'].mean()))
print('Median: ' + str(iris_df['sepal width (cm)'].median()))

# Question 1d
np.percentile(iris_df['sepal width (cm)'], 73)

# Question 1e
sns.pairplot(iris_df, corner=True)
plt.show()

# Question 1f
# Petal Length vs Petal Width seems to have the strongest relationship, as there is a
# very strong positive correlation prevalent in the scatter plot that depicts their relationship.
# Meanwhile, Sepal Width vs Sepal Length seems to have the weakest relationship, as there is no
# correlation that can be deduced based on the scatterplot shown above. I also created a
# correlation matrix using seaborn to confirm my hypothesis - as shown below, Petal Length vs
# Petal Width has the highest correlation with a value of 0.96 and Sepal Width vs Sepal Length
# has the lowest correlation with a value of -0.12.
corr = iris_df.corr()
plt.figure(figsize=(8, 6))
plt.title("Correlation Heatmap of Iris Variables")
plt.show()

# Question 2a
bins = np.arange(3.3, PlantGrowth.weight.max() + 0.3, 0.3)
plt.hist(PlantGrowth.weight, bins=bins)
plt.xlabel("Weight")
plt.ylabel("Frequency")
plt.title("Plant Weight Distribution")
plt.show()

# Question 2b
sns.boxplot(x="group", y="weight", data=PlantGrowth)
plt.title("Plant Weight by Group")
plt.show()

# Question 2c
# The minimum trt2 weight is 4.92. Two outlier values that are above 5.5 are clearly visible 
# in the box plot for trt1, and the remaining values look like they are under 4.92. Since we know 
# that there are 10 values in the trt1 group, it looks like approximately 8/10 or 80% of the trt1
# weights are below 4.92.
min_trt2 = PlantGrowth[PlantGrowth.group == "trt2"].weight.min()
print("Minimum trt2 Weight: " + str(min_trt2))

# Question 2d
trt1_weights = PlantGrowth[PlantGrowth['group'] == 'trt1']['weight']
percentage_below = (trt1_weights < min_trt2).mean() * 100
print("Percentage of trt1 Weights that Fall Below Minimum trt2 Weight: " +
      str(percentage_below))

# Question 2e
heavy_plants = PlantGrowth[PlantGrowth['weight'] > 5.5]
sns.countplot(x='group', data=heavy_plants, palette='Set2')
plt.title('Distribution of Heavy Plants (Weight > 5.5)')
plt.show()
