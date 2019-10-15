import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize']=(10,6)

#read data from house prices
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(train.SalePrice.describe())

#visualize for normalization of data/check for skewness
print(train.SalePrice.skew())
plt.hist(train.SalePrice)
plt.show()

#apply log transformation of data to normalize
target = np.log(train.SalePrice)
print('skew is', target.skew())
plt.hist(target)

#find correlation of features
numeric_features = train.select_dtypes(include=[np.number])
corr = numeric_features.corr()
print(corr['SalePrice'].sort_values(ascending=False)[:5],'\n')
print(corr['SalePrice'].sort_values(ascending=False)[-5:])

#handling null values
nulls = pd.DataFrame(train.isull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'

##handling missing value
data = train.select_dtypes(include=[np.number]).interpolate().dropna()

##Build a linear model
y = np.log(train.SalePrice)
X = data.drop(['SalePrice', 'Id'], axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, random_state=42, test_size=.33)
#call model and fit to your data
from sklearn import linear_model
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
#Evaluate the performance with Rsquared (value closer to 1 means better fit)
print (model.score(X_test, y_test))
predictions = model.predict(X_test)
#evaluate using RMSE, difference between actual value and predicted value (want close to 0)
from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, predictions))

