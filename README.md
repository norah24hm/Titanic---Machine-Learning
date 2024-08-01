# Titanic-Machine-Learning
Predict survival on the Titanic

-------------------------------------------------------------------------------------------------------------------------------------
### Download the dataset:
It is available on kaggle: https://www.kaggle.com/competitions/titanic
To download it :
```
kaggle competitions download -c titanic
```
### Dataset Description
This dataset contains information about the passengers on the Titanic, including their demographics, travel details, and survival status. It can be used in data analysis, machine learning, and predictive modeling.

----------------------------------------------------------------------------------------------------------------------------------------
Columns:
* PassengerId: unique identifier for each passenger
* Survived: survival status (0 = died, 1 = survived)
* Pclass: passenger class (1 = 1st class, 2 = 2nd class, 3 = 3rd class)
* Name: passenger name
* Sex: passenger sex (male/female)
* Age: passenger age
* SibSp: number of siblings/spouses aboard
* Parch: number of parents/children aboard
* Ticket: ticket number
* Fare: passenger fare
* Cabin: cabin number
* Embarked: port of embarkation (C = Cherbourg, S = Southampton, Q = Queenstown)

----------------------------------------------------------------------------------------------------------------------------------------

## ML-steps:
### 1. Importing the Dependencies:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```
### 2. Reading the data:

   We will first use pd.read_csv to load the data from csv file to Pandas DataFrame using upload method:
```
import pandas as pd
from google.colab import files

# Upload the CSV file
uploaded = files.upload()
```
```
# Read the CSV file
data = pd.read_csv('train.csv')
```
After reading the data, we will now review the data to ensure it has been read correctly by using the command head:
```
#this will print first 5 rows in the dataset
data.head()
```
The output:
![data head](https://github.com/user-attachments/assets/31418b65-5062-4064-a732-182bd6bfa385)
```
# number of rows and columns
data.shape
```
No. of rows and columns is : (891, 12)

### 3. Data Preprocessing
Now we will use the (data.info) command to learn more about the data, such as the number of rows and columns, data types, and the number of missing values.
The output:
![data info](https://github.com/user-attachments/assets/f0333998-9d7f-4655-a18f-b167c928ad82)

From that we can have some observations :

There are missing values in the Age column (177 rows) and the Cabin column (687 rows). We need to handle these missing values using a 
 strategy that can handle missing data.
 
#### 3.1  Handling Missing Data
 
 ```
 # to view the Missing valuse in each column:
data.isnull().sum()
```
The output:

![missing data](https://github.com/user-attachments/assets/d487fb25-5633-49be-9d5d-f3633595e3aa)

There are three columns contains Missing values: Age, Cabin, Embarked.

We have three options to fix this:

1. Delete rows that contains missing valuse
2. Delete the whole column that contains missing values
3. Replace missing values with some value (Mean, Median, Mode, constant)

In the Age column, we will fill the missing values with the mean since it is a simple and quick method to handle missing data and helps maintain the overall distribution of the dataset.
```
# Calculate the mean of the Age column
mean_age = data['Age'].mean()

# Fill the missing values in Age with the mean
data['Age'] = data['Age'].fillna(mean_age)
```
Run this command to make sure that there is no missing values:
```
data['Age'].isnull().sum()
```
There are a large number of missing values in the Cabin column = (687) so we will drop this column from the dataset.
```
data = data.drop(['Cabin'], axis=1)
```
View the data to make sure of droping this column using the command (data.head()).

In the Embarked column, there are only two missing values. Let's see what the categories in this column are.
![em-cat](https://github.com/user-attachments/assets/ef91f598-2e1c-4b2b-b5b9-6bb4f6d8919b)

As for the Embarked column, since it only has 2 missing values, we will impute them with the most frequent value (mode).

![embarked](https://github.com/user-attachments/assets/9eb900ec-f6f9-48d2-ab02-2f511bc5c504)


#### 3.2 Drop useless columns
The PassengerId and Name of the Passenger do not affect the probability of survival. and ticket column does not have a clear relationship to the survival of passengers, so they shuold be dropped:
Drop the PassengerId, Name, and Ticket columns
```
data = data.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
```


#### 3.3 Encode Categorical Columns
Sex and Embarked columns values are text, we can't give this text directly to the machine learning model, so we need to replace this text values to meaningful numerical values.

In Age column we will replace all male values with 0 and all the female values with 1.
and we will do the same in Embarked column: S=> 0 , C=> 1, Q => 2
```
data.replace({'Sex':{'male':0,'female':1},'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)
```
#### 3.4 Dealing with Duplicates
Now let's look for duplicates in the dataset:
```
# Check for duplicate rows
duplicates = data.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")
```
The output: Number of duplicate rows: 107, we will drop all duplicates using the following command:
```
# Drop duplicate rows
data = data.drop_duplicates()
```
#### 3.5 Data Analysis
Now we will explore the data and the relationships between features using statistical analysis and visualization techniques. This will help us understand the underlying patterns and correlations in the dataset, providing valuable insights for model building.

describe() provides summary statistics for numerical columns, including count, mean, standard deviation, min, max, and quartiles. This function helps us understand the distribution and central tendencies of the data. However, in our Titanic dataset, while useful, it may not be the primary focus since many insights come from categorical features and their relationships with survival, which are better explored through other means.


![descrip](https://github.com/user-attachments/assets/49cdfd07-ecdd-4e3e-ad11-4347e53617dd)


Look for Correlations:

Now to understand the relations between the features we can use the correlation matrix which shows the correlation coefficients between different features in a dataset. Each cell in the matrix represents the correlation between two features. The correlation coefficient ranges from -1 to 1, where:

* 1 indicates a perfect positive correlation: as one feature increases, the other feature increases proportionally.

* -1 indicates a perfect negative correlation: as one feature increases, the other feature decreases proportionally.

* 0 indicates no correlation: the features do not show any linear relationship.
  ```
  data.corr()['Survived']
  ```
The correlation values provide insights into how different features relate to the survival outcome in the Titanic dataset:

* Pclass: Negative correlation (-0.332). Higher classes (lower number) are more likely to survive.
* Sex: Positive correlation (0.515). Females are more likely to survive.
* Age: Slight negative correlation (-0.080). Older passengers have a marginally lower chance of survival.
* SibSp: Slight negative correlation (-0.036). Having more siblings/spouses aboard slightly decreases survival chances.
* Parch: Slight positive correlation (0.070). Having more parents/children aboard slightly increases survival chances.
* Fare: Positive correlation (0.246). Passengers who paid higher fares are more likely to survive.
* Embarked: Slight positive correlation (0.073). The port of embarkation has a minor effect on survival.
* 
These correlations help identify which features may be important for predicting survival.

To understand more about data lets find the number of people survived and not survived
```
data['Survived'].value_counts()
```

The output we got: 


![sur-co](https://github.com/user-attachments/assets/9e600666-938c-40a9-9690-644aabc85fa4)


The output shows that:

* 461 passengers did not survive (Survived=0)
* 323 passengers survived (Survived=1)
-----------------------------------------------------------------------------------------------------------------------------------------------------

Count Plots

Next, we create count plots to visualize the distribution of the Survived and Sex columns:
```
# making a count plot for 'Survived' column
sns.countplot(x='Survived', data=data)
```
![sur-plot](https://github.com/user-attachments/assets/90d7a1c5-2f21-4592-a160-6f03e66f0726)

```
# making a count plot for 'Sex' column
sns.countplot(x='Sex', data=data)
```
![sex-plot](https://github.com/user-attachments/assets/e9ab0884-5b33-4bba-b785-390b91651405)

Now lets compare the number of survived beasd on the gender:
```
sns.countplot(x='Sex', hue='Survived', data=data)
```
![compare](https://github.com/user-attachments/assets/8598dbf2-7501-466a-bf51-7867eed737a7)

As we can see, even we have more number of male in our dataset, the number of fmale who have survived is more. this is one of the very important insight that we can get from this data.
So lets compare the number of survived with another feature: Pclass.
```
sns.countplot(x='Pclass', hue='Survived', data=data)
```
![places](https://github.com/user-attachments/assets/0cf6d62a-89e2-43ae-89ca-d7f5d753bc67)

We can compare with other features to have more insights

---------------------------------------------------------------------------------------------------------------------------------------------------
#### Model Building
1. Separation

Separating features and target so that we can prepare the data for training machine learning models. In the Titanic dataset, the Survived column is the target variable, and the other columns are the features.
```
x = data.drop(columns = ['Survived'], axis=1)
y = data['Survived']
```

2. Splitting the data into training data & Testing data

To build and evaluate a machine learning model effectively, it's essential to split the dataset into training and testing sets. The training set is used to train the model, allowing it to learn patterns and relationships within the data. The testing set, on the other hand, is used to evaluate the model's performance on unseen data, ensuring it can generalize well to new instances. This split helps prevent overfitting and provides a reliable estimate of the model's predictive accuracy.

We can split our dataset using the following command:
```
from sklearn.model_selection import train_test_split

# Split the data into training data & Testing data using train_test_split function :
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```
In this code:

* x_train and y_train represent the training data (features and target, respectively)
* x_test and y_test represent the testing data (features and target, respectively)
* test_size=0.2 specifies that 20% of the data should be used for testing, and the remaining 80% for training
* random_state=42 sets the random seed for reproducibility
  
By splitting the data in this way, we can train a model on the training set and evaluate its performance on the testing set, which helps prevent overfitting and provides a reliable estimate of the model's predictive accuracy.

Now that we have our data split into training and testing sets, we're ready to start building and evaluating our machine learning model.

3. Model Training

Model training is a crucial step in the machine learning where the algorithm learns from the training data to make predictions. Logistic Regression is a commonly used algorithm for binary classification tasks, such as predicting whether a passenger survived in the Titanic dataset. By training the model on our training data, we aim to find the best-fit parameters that minimize prediction errors. Once trained, this model can be used to predict outcomes on new, unseen data. 


Let's create a Logistic Regression model and train it on our training data using the following code:
```
from sklearn.linear_model import LogisticRegression

# Create a Logistic Regression model and Train it on the training data:

# Create a Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)

# Train the model on the training data
log_reg.fit(x_train, y_train)

```
![log-reg](https://github.com/user-attachments/assets/dc9e2f85-a027-4c19-a770-287e76b28016)


4. Model Evaluation
Model evaluation is crucial in machine learning to assess the performance of a trained model on testing data. The accuracy score, a common evaluation metric, measures the proportion of correct predictions out of all predictions. This helps to gauge the model's effectiveness, ensure it generalizes well to new data, and guide further improvements.

Model Evaluation using Accuracy Score:
```
from sklearn.metrics import accuracy_score

# Make predictions on the testing data
y_pred = log_reg.predict(x_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print("Accuracy:", accuracy)
```
Accuracy: 0.796 ≈ 0.80

The accuracy score can give us an idea of how well our model is performing on unseen data. A higher accuracy score indicates that the model is doing a good job of making correct predictions.

Interpreting the Accuracy Score:
The accuracy score ranges from 0 to 1, where:
* 1 represents perfect accuracy (all predictions are correct)
* 0 represents complete randomness (no better than chance)
In general, an accuracy score above 0.8 is considered good, while an accuracy score above 0.9 is considered excellent.


Our model has achieved an accuracy of approximately 80% on the testing data.
Model Performance Analysis: 
With an accuracy score of 0.80, our model is doing a good job of making correct predictions. However, there's still room for improvement.


Let's take a closer look at the confusion matrix to gain more insights into our model's performance:
```
from sklearn.metrics import confusion_matrix

# Create a confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)

# Print the confusion matrix
print(conf_mat)
```
The confusion matrix is as follows:

[74 14]

 [18 51]

###### Confusion Matrix Analysis
Let's break down the confusion matrix:

True Positives (TP): 74 (correctly predicted positive class).

False Positives (FP): 14 (incorrectly predicted positive class).

True Negatives (TN): 51 (correctly predicted negative class).

False Negatives (FN): 18 (incorrectly predicted negative class). 


###### Performance Metrics
We can calculate some additional performance metrics from the confusion matrix:

Precision: TP / (TP + FP) = 74 / (74 + 14) = 0.841. 

Recall: TP / (TP + FN) = 74 / (74 + 18) = 0.804.

F1-score: 2 * (Precision * Recall) / (Precision + Recall) = 2 * (0.841 * 0.804) / (0.841 + 0.804) = 0.822.


###### Insights

From the confusion matrix and performance metrics, we can see that:

* Our model is doing a good job of predicting the positive class (74 correct predictions out of 92 total positive instances).
* Our model is also doing a good job of predicting the negative class (51 correct predictions out of 65 total negative instances).
* However, our model is making some mistakes, particularly in predicting the positive class (14 false positives and 18 false negatives).
 
----------------------------------------------------------------------------------------------------------------------------
###### Future work

Here are some possible steps to improve the model:

1. Hyperparameter Tuning: We can try tuning the model's hyperparameters, such as the regularization strength (C), the maximum number of iterations (max_iter), or the solver algorithm (solver), to see if we can improve the model's performance.
2. Feature Engineering: We can explore additional features that might be relevant to the problem, such as extracting more information from the existing features or incorporating new data sources.
3. Model Selection: We can try using a different machine learning algorithm, such as a decision tree, random forest, or support vector machine, to see if it performs better on our dataset.
4. Data Preprocessing: We can revisit our data preprocessing steps to ensure that we're handling missing values, outliers, and feature scaling correctly.


