import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pylab as pylab
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,accuracy_score,classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.svm import SVC,SVR
from xgboost import XGBClassifier,XGBRegressor
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
plt.ion() 


df= pd.read_csv('dataset_diamonds.csv')
print(df.head())

print(df.info())

print(df.describe())

# Check for missing values
print(df.isnull().sum())

df.shape
## DATA PREPROCESSING 
df.info()


#The first column seems to be just index
df = df.drop(["Unnamed: 0"], axis=1)
print([df.describe()])

# Min value of "x", "y", "z" are zero this indicates that there are faulty values in data that represents dimensionless or 2-dimensional diamonds. So we need to filter out those as it clearly faulty data points.
df=df.drop(df[(df.x==0) | (df.y==0) | (df.z==0)].index)
print(df.shape)

# We can clearly spot outliers in these attributes. Next up, we will remove these data points.
df=df[(df["depth"]<75) & (df["depth"]>45)]
df=df[(df["table"]<80) & (df["table"]>40)]
df=df[(df["x"]<30)]
df=df[(df["y"]<30)]
df=df[(df["z"]<30)&(df["z"]>2)]
print(df.shape)


ax=sns.pairplot(df, hue= "cut",palette="husl", diag_kind='kde')
plt.show()

s=(df.dtypes=="object")
object_cols=list(s[s].index)
print(object_cols)
print("Categorical variables: ",object_cols)

sns.violinplot(x='cut', y='price', data=df, palette="muted")
plt.title("Violin Plot of Price by Cut")
plt.show()

Label_data=df.copy()
label_encoder=LabelEncoder()
for col in object_cols:
    Label_data[col]=label_encoder.fit_transform(Label_data[col])
Label_data.head()

#correlation matrix
cmap=sns.diverging_palette(70,20,s=50,l=40,n=6,as_cmap=True)
corrmat=Label_data.corr()
f,ax=plt.subplots(figsize=(12,10))
sns.heatmap(corrmat,cmap=cmap,annot=True)
plt.show()

x=Label_data.drop(['price'],axis=1)
y=Label_data['price']
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=42)
y

pipeline_lr=Pipeline([("scalar1",StandardScaler()),
                     ("lr_classifier",LinearRegression())])

pipeline_dt=Pipeline([("scalar2",StandardScaler()),
                     ("dt_classifier",DecisionTreeRegressor())])

pipeline_rf=Pipeline([("scalar3",StandardScaler()),
                     ("rf_classifier",RandomForestRegressor())])


pipeline_kn=Pipeline([("scalar4",StandardScaler()),
                     ("rf_classifier",KNeighborsRegressor())])


pipeline_xgb=Pipeline([("scalar5",StandardScaler()),
                     ("rf_classifier",XGBRegressor())])

# List of all the pipelines
pipelines = [pipeline_lr, pipeline_dt, pipeline_rf, pipeline_kn, pipeline_xgb]

# Dictionary of pipelines and model types for ease of reference
pipe_dict = {0: "LinearRegression", 1: "DecisionTree", 2: "RandomForest",3: "KNeighbors", 4: "XGBRegressor"}

# Fit the pipelines
for pipe in pipelines:
    pipe.fit(X_train, Y_train)


cv_results_rms = []
for i, model in enumerate(pipelines):
    cv_score = cross_val_score(model, X_train,Y_train,scoring="neg_root_mean_squared_error", cv=10)
    cv_results_rms.append(cv_score)
    print("%s: %f " % (pipe_dict[i], cv_score.mean()))

pred = pipeline_xgb.predict(X_test)

print("R^2:",metrics.r2_score(Y_test, pred))
print("Adjusted R^2:",1 - (1-metrics.r2_score(Y_test, pred))*(len(Y_test)-1)/(len(Y_test)-X_test.shape[1]-1))
print("MAE:",metrics.mean_absolute_error(Y_test, pred))
print("MSE:",metrics.mean_squared_error(Y_test, pred))
print("RMSE:",np.sqrt(metrics.mean_squared_error(Y_test, pred)))    