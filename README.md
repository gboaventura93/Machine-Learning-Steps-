# Machine Learning Steps

This is my guide and perspective on creating Machine Learning models~~for beginners~~ in Python.  

## Python Libraries
### NumPy
  NumPy is a fundamental package for scientific computing in Python, providing support for multidimensional arrays and matrices.  
  `import numpy as np`
### Pandas
  Pandas is a powerful data manipulation and analysis library that provides data structures such as dataframes to work with structured data. Essential in data preprocessing, cleaning, and exploration.  
  `import pandas as pd`
### Matplotlib
  Matplotlib is a plotting library that allows for the creation of various types of visualizations, such as histograms, scatter plots, and line plots.  
  `import matplotlib.pyplot as plt`
### Seaborn
Seaborn is ease of use and specialized functions for statistical data visualization. It's a great choice for many common data visualization tasks.  
  `import seaborn as sns`
### Scikit-learn
  Scikit-learn is a machine learning library that provides a wide range of algorithms and tools for building predictive models.  
  `from sklearn.MODEL import MODEL`
  
## Data Preprocessing
Data preprocessing is extremely important to clean and transform the data you are working with.  

### Verify if the data has null or NaN values
`df.isna().sum()` This command will allow you to check if there are any null values in each column of your dataset.  
Some options to fill those values include:  
#### Median  
I like to use the median to fill null data in case the columns contain `integers` or `floats`.  
`df['Column A'] = dfdf['Column A'].fillna(df['Column A'].median())`  
#### Mode
Using the mode allows you to fill NaN spaces with the most common value. I usually use it for `string` values.  
`df['Column B'].fillna(df['Column B'].mode()[0], inplace=True)`  

### get_dummies  
get_dummies is an extremely useful function to organize the data. You can create new columns with the strings contained in a column. The result will be a boolean (1 and 0).  
`df = pd.get_dummies(df, columns=['Column B'])`  

## Exploratory Data Analysis (EDA)
Now it's time to visualize your data and gain insights through visualization. Tools like Matplotlib and Seaborn are useful here.    
To visualize data with *Matplotlib*:  
`plt.figure(figsize=(10, 6))`  
`plt.plot(df['time'], df['value'])`  
`plt.xlabel('Time')`  
`plt.ylabel('Value')`  
`plt.title('Time vs Value')`  
`plt.show()`     

To visualize data with *Seaborn*:  
`sns.set_theme(style="whitegrid")`  
`sns.lineplot(x='time', y='value', data=df)`  
`plt.title('Time vs Value')`  
`plt.show()`  

## Model Selection
You have to choose a machine learning algorithm appropriate for your problem. The most common ones are:  
- Classification: Logistic Regression, Decision Trees, Random Forest, SVC, SVM, KNN.  
  `from sklearn.linear_model import LogisticRegression`  
  `from sklearn.tree import DecisionTreeClassifier`  
  `from sklearn.ensemble import RandomForestClassifier`  
  `from sklearn.svm import SVC`  
  `from sklearn import svm`  
  `from sklearn.neighbors import KNeighborsClassifier`  

- Regression: Linear Regression.  
  `from sklearn.linear_model import LinearRegression`  
  
- Clustering: K-means.  
    `from sklearn.cluster import KMeans`  
  
It's really interesting to try every model and then rank them by their accuracy. This allows you to decide which one is more appropriate for your work. 
  
Scikit-learn is a huge library with a big active community. So I highly recommend checking more models and details on their [webpage](https://scikit-learn.org/stable/supervised_learning.html)

## Model Training - Spliting data  
Here you should split your data into training and testing sets, then train your model on the training data.    
**X**: X will be all the values in your dataset except the target variable you want to predict.    
**y**: y will be your target variable.    
`X = df.drop(['Goal'], axis=1)`  
`y = train_df['Goal']`  
After that, you can split them using train-test-split:    
`from sklearn.model_selection import train_test_split`  
`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)`  
`test_size=0.2` means that you are using 20% of the data for testing. This means  `0<test_size<1`. You can change and try it however you want.    
*Note*: If test data is already available, maybe you don't need to split your data. You can use this test data as `X_test = test_df`, where you would predict `y_test`.  

## Model Training
Now you can train your model and finally predict:    
`model_random_forest = RandomForestClassifier()`  
`model.fit(X_train, y_train)`  
`y_pred = model.predict(X_test)`  

It's important to note that every model, such as Random Forest, has many parameters that you can change to get a better model.  
By default, Random Forest has these parameters:    
`RandomForestClassifier(n_estimators=100, *, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)`  

But the question is, are they the best parameters for our data?  
To get this answer, you can use a process called `Hyperparameter Tuning`. The most common methods are `GridSearch` and `Randomized search`.  

To learn more about them, I really recommend this article from [KDnuggets](https://www.kdnuggets.com/hyperparameter-tuning-gridsearchcv-and-randomizedsearchcv-explained), and also the Scikit-learn modules for [GridSearch](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html), and [RandomizedSearch](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html).  

## Model Evaluation
To evaluate your model, you can score it using `accuracy_score from scikit-learn`. After that, you can rank your models and choose the one with the highest accuracy.  
`from sklearn.metrics import accuracy_score`  
`accuracy_model = round(model.score(X_train, y_train) * 100, 2)` *0-100%*  
`score = pd.DataFrame({'Model': ['Model_1', 'Model_2', ...],
                       'Score': [accuracy_model_1, accuracy_model_2, ...]})`
`score.sort_values(by='Score', ascending=False)`

## Output in CSV
To save your results in a CSV file, you can create a DataFrame with an ID (e.g.) and your predictions:  
`output = pd.DataFrame({
        "Id": test_df['Id'],
        "Predicted": y_pred
    })`  
`output.to_csv('output.csv', index=False)`

I will update this repository as I learn and study more about it.  
Furthermore, you can check some machine learning examples in this repository, where I'm adding some projects I've done.  

Finally, I hope this can be useful to someone. Feel free to suggest, comment, or correct anything.  
![](https://i.pinimg.com/originals/e7/92/45/e792455c05cd8199d903432c24020cf2.gif)
 



