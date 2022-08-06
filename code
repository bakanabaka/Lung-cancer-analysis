import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import precision_score,recall_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import zero_one_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold,StratifiedKFold,LeaveOneOut
import statistics as st
df = pd.read_csv(r'C:\Users\shrav\Dropbox\My PC (DESKTOP-689769P)\Downloads\survey lung cancer.csv')
print(df.head())
df_copy = df.copy(deep = True)

#statistics
print(df.describe())
print(df.info)
print(df.shape)
print(df.dtypes)

#cleaning and filtering
print(df_copy.fillna(0))
print(df_copy.dropna)
print(df_copy.replace('No',''))
print(df_copy.isnull().sum())

#counting
print(df.values)
print(df.columns)

df.columns = df.columns.str.title()# all capital letters in column name to only first capital letter
print(df.head())
#gender vs lung cancer
print(df["Gender"].unique())#shows only unique values , like for this male or female
gender_count = df['Gender'].value_counts(sort=True)
print(gender_count)

# plt.figure(figsize=(6,8))
# values = gender_count.values
# labels = gender_count.index
# plt.pie(values,labels=labels)
# plt.title('Gender distribution')
# plt.legend(["Male","Female"])
# plt.show()


print(df.groupby(["Gender","Lung_Cancer"]).agg(total_cancer = ("Lung_Cancer","count"),
    minimum_age=("Age","min"),max_age = ("Age","max")))

plt.figure(figsize=(6,6))
sns.countplot(data=df,x="Gender",hue="Lung_Cancer")
plt.title("Gender Vs. Lung Cancer")
plt.show()

print(df["Age"].max())
print(df["Age"].min())

plt.figure(figsize=(8,6))
sns.countplot(x="Age",hue="Gender",data=df)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

#alchoholic consumption
print(df["Alcohol Consuming"].unique())
alcohol_count = df["Alcohol Consuming"].value_counts(sort=True)
print(alcohol_count)

labels = alcohol_count.index
values = alcohol_count.values
plt.figure(figsize=(6,6))
plt.pie(values,labels=labels)
plt.title("alcohol consumption")
plt.legend(["Alcohol","non Alcohol"])
plt.show()


#machine learning
df["Gender"]=df["Gender"].replace(["Male","Female"],[1,0])
df["Lung_Cancer"]=df["Lung_Cancer"].replace(["YES","NO"],[1,0])

x = df.iloc[:,:-1]
y = df.iloc[:,-1]
xtrain,xtest,ytrain,ytest =  train_test_split(x,y,test_size=0.2)

def gen_cls_matrix(y_test,ypred):
    cm = confusion_matrix(y_test,ypred)
    cl = classification_report(y_test,ypred)
    print(cm)
    print(cl)
def score(model):
    print('training score',model.score(xtrain,ytrain))
    print('testing score',model.score(xtest,ytest))

print(ytest)
m1 = LogisticRegression()
m1.fit(xtrain,ytrain)
predicted_value = m1.predict(xtest)
print(predicted_value)

comparation = pd.DataFrame({"Actual value":ytest,"Predicted value":predicted_value})
print(comparation.sample(10))#prints only sample 10 values

av = comparation["Actual value"]
pv = comparation["Predicted Value"]
acc = accuracy_score(av,pv)
pre = precision_score(av,pv)
rec = recall_score(av,pv)
print("accuracy score",av)
print("Precision score: ",pre)
print("Recall score: ",rec)
