# SERVO PREDICTION USING LINEAR REGRESSION 
IMPORT LIBRARY

[1]
0s
import pandas as pd
[2]
0s
import numpy as np
IMPORT CSV AS DATAFRAME

[3]
1s
df = pd.read_csv(r'https://github.com/YBI-Foundation/Dataset/raw/main/Servo%20Mechanism.csv')
GET THE FIRST 5 ROWS


[4]
0s
df.head()

Next steps:
GET INFO OF DATAFRAME


[5]
0s
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 167 entries, 0 to 166
Data columns (total 5 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   Motor   167 non-null    object
 1   Screw   167 non-null    object
 2   Pgain   167 non-null    int64 
 3   Vgain   167 non-null    int64 
 4   Class   167 non-null    int64 
dtypes: int64(3), object(2)
memory usage: 6.6+ KB
GET SUMMARY STATISTICS


[6]
0s
df.describe()

GET COLUMN NAMES


[7]
0s
df.columns
Index(['Motor', 'Screw', 'Pgain', 'Vgain', 'Class'], dtype='object')
GET SHAPE OF DATAFRAME


[8]
0s
df.shape
(167, 5)
GET CATEGORIES AND COUNTS OF CATEGORICAL VARIABLES


[9]
0s
df[['Motor']].value_counts()


[10]
0s
df[['Screw']].value_counts()

GET ENCODING OF CATEGORICAL FEATURES


[11]
0s
df.replace({'Motor':{'A':0,'B':1,'C':2,'D':3,'E':4}},inplace=True)

[12]
0s
df.replace({'Screw':{'A':0,'B':1,'C':2,'D':3,'E':4}},inplace=True)
DEFINE y(DEPENDENT VARIABLE) AND X(INDEPENDENT VARIABLE)


[13]
0s
y = df['Class']

[14]
0s
y.shape
(167,)

[15]
0s
y


[16]
0s
X = df[['Motor','Screw','Pgain','Vgain']]

[17]
0s
X.shape
(167, 4)

[18]
0s
X

Next steps:
GET TRAIN TEST SPLIT


[19]
3s
from sklearn.model_selection import train_test_split

[20]
0s
X_train, X_test, y_train , y_test = train_test_split(X,y, test_size=0.3, random_state=2529)

[21]
0s
X_train.shape, X_test.shape, y_train.shape , y_test.shape
((116, 4), (51, 4), (116,), (51,))
GET MODEL TRAIN


[22]
0s
from sklearn.linear_model import LinearRegression

[23]
0s
lr = LinearRegression()

[24]
0s
lr.fit(X_train,y_train)

GET MODEL PREDICTION


[25]
0s
y_pred = lr.predict(X_test)

[26]
0s
y_pred.shape
(51,)

[27]
0s
y_pred
array([24.55945258, 30.98765106, 18.54485477, 25.51524243, 38.56082023,
       23.52007775, 11.61947065, 20.03335614, 40.60404401, 41.7009556 ,
       13.66269443, 26.01242807, 16.50163099, 16.54663453, 21.92598051,
       22.52570646, -5.46449561, 30.68912392, 32.7323477 ,  1.41282941,
       33.97718702, 31.63543611, 33.52806048, 30.04133887, 19.38557109,
        6.49364826, 28.5528375 , 17.04382017, 25.06611589,  3.50411229,
       30.59606128, 23.67067716, 35.72188367, 32.08456265, 12.46018697,
        3.6547117 , 23.47201865, 33.03087484, 17.49294672, 37.61450804,
       27.54898855, 22.07657992, 11.51387478,  9.470651  , 30.53852451,
       28.64590014, 33.67865989,  4.60102388, 24.1198037 , 21.13026773,
       25.71390094])
GET MODEL EVALUATION


[28]
0s
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

[29]
0s
mean_squared_error(y_test,y_pred)
66.03589175595563

[30]
0s
mean_absolute_error(y_test,y_pred)
7.190539677251235

[31]
0s
r2_score(y_test,y_pred)
0.6807245170563927
GET VISUALIZATION OF ACTUAL VS PREDICTED RESULTS


[32]
0s
import matplotlib.pyplot as plt
plt.scatter(y_test,y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual Vs Predicted")
plt.show()

GET FUTURE PREDICTIONS


[33]
0s
X_new = df.sample(1)

[34]
0s
X_new


[35]
0s
X_new.shape
(1, 5)

[36]
0s
X_new = X_new.drop('Class',axis=1)

[37]
0s
X_new


[38]
0s
X_new.shape
(1, 4)

[39]
0s
y_pred_new = lr.predict(X_new)

[40]
0s
y_pred_new
EXPLANATION 
Predicting server behavior with a linear equation involves relating input variables (like user count) to an output (like response time). The general equation is:

\[
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_n X_n
\]

- \(Y\): Predicted output (e.g., response time).
- \(X_1, X_2, \dots\): Inputs (e.g., number of users, time).
- \(\beta_0\): Intercept, baseline value.
- \(\beta_1, \beta_2, \dots\): Coefficients showing the impact of each input.

Use historical data to estimate coefficients. Then, plug in new input values to predict server behavior.
