import numpy as np
import seaborn as sns
import pandas as pd
from numpy import absolute

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt

# for scaling
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# for for one hot encoder
from sklearn.preprocessing import OneHotEncoder

# For Imputing Missing Values
from sklearn.impute import SimpleImputer

# for correlation plots
from pandas.plotting import scatter_matrix

# for train-test set
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

# for LS linear regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# for LS linear regression
from sklearn.linear_model import SGDRegressor

# for MLP regression
from tensorflow import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

# for 10 fold cross validation
from sklearn.model_selection import cross_val_score

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# for pandas print
desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)

# Load the data import pandas as pd
housing = pd.read_csv("housing.csv")

# Explore the data
# print(housing.head(10))

# housing.info()

# print(housing.describe())


# Οπτικοποίηση Δεδομένων
# 1
housing.hist(bins=50, figsize=(20, 15))
plt.show()

# Given that .botplot() and .hist() only handle numerical features. We cannot forget ocean_proximity, which is object
# type (no need to change to string).
# print(housing['ocean_proximity'].value_counts())
op_count = housing['ocean_proximity'].value_counts()
plt.figure(figsize=(10, 5))
sns.barplot(x=op_count.index, y=op_count.values, alpha=0.7)
plt.title('Ocean Proximity Summary')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Ocean Proximity', fontsize=12)
plt.show()

# 2
# Exploring high density areas
# Correlation plots with 2 features (longitude with latitude)
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
plt.show()

# Lets look at housing prices with circle representing district population and color representing price
# Correlation plots with 3 features (median_house_value with longitude,latitude)
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"] / 100, label="population", figsize=(10, 7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True, sharex=False)

plt.show()

# Correlation plots with 4 features (median_house_value with median_income,total_rooms,housing_median_age)
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.show()

# train-test set definition
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
'''Here, we used random sampling to create train and test datasets. Usually, median income of any neighborhood is 
great indicator of wealth distribution in that area. So, we want to make sure that test datasets is representative of 
various categories of income which is actually numeric variable. This means we have to convert it into categorical 
variables and create different levels of income and use stratified sampling instead of random sampling. '''
# housing["median_income"].hist()

# Checking for the right number of bins for the response variable
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
# housing["income_cat"].hist()

# Startified sampling based on income_cat to make the datasets more random and representative
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Check if the strata worked for entire datasets
# print(housing["income_cat"].value_counts() / len(housing))

# Now, lets see if the same proportion has been applied in the test sets.
# print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

'''We see the same distribution of income category variable in the test sets as in the entire datasets. Now, 
lets get the data back to original state by dropping income category variable. '''
# Removing income _cat from the dataset so data goes back to original state
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# Προ-επεξεργασια δεδομενων

# Here first we will create a copy and separate the target variable as we do not want to do the same transformation

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# 1
housing_num = housing.drop("ocean_proximity", axis=1)
housing_numeric_features = list(housing_num)

housing_cat = housing[['ocean_proximity']]
housing_categorical_features = ['ocean_proximity']

# print(housing_num)
# print(housing_numeric_features)
# print(housing_cat)
# print(housing_categorical_features)


# 3
'''
cat_encoder = OneHotEncoder(sparse=False)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
# print(housing_cat_1hot)
# print(cat_encoder.categories_)
'''

# 4
'''
imputer = SimpleImputer(strategy="median")
# Now, lets impute missing values
imputer.fit(housing_num)  # this computes median for each attributes and store the result in statistics_ variable

# print(imputer.statistics_)  # same result as housing_num.median().values
# print(housing_num.median().values)

# see attributes with missing values
# housing_num.info()

x = imputer.transform(housing_num)  # this is a Numpy array
housing_tr = pd.DataFrame(x, columns=housing_num.columns)  # change a Numpy array to a DataFrame

# housing_tr.info()  # no missing values
'''

# 2
# Transforming Pipelines and scaling numeric features except for target variable
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])
# housing_num_tr = num_pipeline.fit_transform(housing_num)
# X = pd.DataFrame(housing_num_tr)
# print(X.head(10))
# print(X.info())


# Transforming Pipelines for categorical feature(ocean proximity) and combine with numeric
# features transformation(full_pipeline)
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, housing_numeric_features),
    ("cat", OneHotEncoder(), housing_categorical_features),
])

X_train = full_pipeline.fit_transform(housing)
y_train = housing_labels
X_test = strat_test_set.drop("median_house_value", axis=1)
X_test = full_pipeline.transform(X_test)
y_test = strat_test_set["median_house_value"].copy()


# print(X_train.shape)
# X_test = pd.DataFrame(X_train)
# print(X_test)
# print(X_test.info())
def cross_val(model, x, y):
    scores_mse = cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv=10)
    scores_mse = absolute(scores_mse)
    print("Scores(metric:'MSE'):", scores_mse)
    print("MSE:", scores_mse.mean())

    print()

    scores_mae = cross_val_score(model, x, y, scoring="neg_mean_absolute_error", cv=10)
    scores_mae = absolute(scores_mae)
    print("Scores(metric:'MAE'):", scores_mae)
    print("MAE:", scores_mae.mean())


# Παλινδρόμηση Δεδομένων
# ερωτημα 1
print('\n------------------------Least Mean Squares Linear Regression_____________________________________')
sgd_reg = SGDRegressor(alpha=0.0001, epsilon=0.01, eta0=0.1, penalty='elasticnet')
sgd_reg.fit(X_train, y_train)

# Without using cross validation on linear regression
print('\nTrain set evaluation WITHOUT 10 FOLD CROSS VALIDATION:\n_____________________________________')
train_housing_predictions = sgd_reg.predict(X_train)
lin_mse = mean_squared_error(y_train, train_housing_predictions)
lin_mae = mean_absolute_error(y_train, train_housing_predictions)
print("MSE", lin_mse)
print("MAE", lin_mae)

print('\nTest set evaluation WITHOUT 10 FOLD CROSS VALIDATION:\n_____________________________________')
test_housing_predictions = sgd_reg.predict(X_test)
lin_mse = mean_squared_error(y_test, test_housing_predictions)
lin_mae = mean_absolute_error(y_test, test_housing_predictions)
print("MSE", lin_mse)
print("MAE", lin_mae)

# Using cross validation
print('\nTrain set evaluation WITH 10 FOLD CROSS VALIDATION:\n_____________________________________')
cross_val(sgd_reg, X_train, y_train)

print('\nTest set evaluation WITH 10 FOLD CROSS VALIDATION:\n_____________________________________')
cross_val(sgd_reg, X_test, y_test)

# ερωτημα 2
print('\n------------------------Least Squares Linear Regression_____________________________________')
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Without using cross validation on linear regression
print('\nTrain set evaluation WITHOUT 10 FOLD CROSS VALIDATION:\n_____________________________________')
train_housing_predictions = lin_reg.predict(X_train)
lin_mse = mean_squared_error(y_train, train_housing_predictions)
lin_mae = mean_absolute_error(y_train, train_housing_predictions)
print("MSE", lin_mse)
print("MAE", lin_mae)

print('\nTest set evaluation WITHOUT 10 FOLD CROSS VALIDATION:\n_____________________________________')
test_housing_predictions = lin_reg.predict(X_test)
lin_mse = mean_squared_error(y_test, test_housing_predictions)
lin_mae = mean_absolute_error(y_test, test_housing_predictions)
print("MSE", lin_mse)
print("MAE", lin_mae)

# Using cross validation
print('\nTrain set evaluation WITH 10 FOLD CROSS VALIDATION:\n_____________________________________')
cross_val(lin_reg, X_train, y_train)

print('\nTest set evaluation WITH 10 FOLD CROSS VALIDATION:\n_____________________________________')
cross_val(lin_reg, X_test, y_test)

# ερωτημα 3
print('\n------------------------Multilayer Neural Network(MLP) Regression_____________________________________')

model = Sequential([
    Dense(14, activation="relu"), Dense(14, activation="relu"),
    Dense(1)
])
model.compile(loss="mean_squared_error", optimizer=optimizers.SGD(learning_rate=1e-3))
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
# model.evaluate(X_test, Y_test)

'''
# Using cross validation
print('\nTrain set evaluation WITH 10 FOLD CROSS VALIDATION:\n_____________________________________')
cross_val(model, X_train, y_train)

print('\nTest set evaluation WITH 10 FOLD CROSS VALIDATION:\n_____________________________________')
cross_val(model, X_test, y_test)
'''


