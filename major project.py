import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.metrics import classification_report , ConfusionMatrixDisplay
df = pd.read_csv('train.csv')
df
df.info()
df.duplicated().sum()
df.drop_duplicates(inplace=True)
df.duplicated().sum()
df.describe()
sns.countplot(x= df['quality'])
plt.xlabel('Wine quality')
plt.ylabel('Frequancy')
plt.title('Wine quality classes');
#Relation between volatile acidity vs quality
plot = plt.figure(figsize= (5,5))
sns.barplot(x='quality' , y= 'volatile acidity' , data= df )
plt.title('volatile acidity vs quali;ty')
plot = plt.figure(figsize= (5,5))
sns.barplot(x='quality' , y= 'citric acid' , data= df )
plt.title('citric acid vs quality');
plt.figure(figsize=(10,10))
corr = df.corr()
sns.heatmap(corr ,annot=True , cmap= 'Blues');
X = df.drop(columns= ['quality'])
X
y = (df['quality'] > 6).astype(int)
y
print("X shape:", X.shape)
print("y shape:", y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
acc_baseline = y.value_counts(normalize= True).max()
print("Baseline Accuracy:", round(acc_baseline, 4))
k_neighbors = range(2 , 11 , 2)
Training_acc = []
Testing_acc = []
for i in k_neighbors:
    
    model = KNeighborsClassifier(n_neighbors= i)
    model.fit(X_train , y_train)
    Training_acc.append(model.score(X_train , y_train))
    Testing_acc.append(model.score(X_test , y_test))   
print(f'Training accuracy: f{Training_acc}')
print(f'Testing accuracy: f{Testing_acc}')
plt.plot(k_neighbors , Training_acc , label= 'Training')
plt.plot(k_neighbors , Testing_acc , label= 'Testing')
plt.xlabel('K-neighbors')
plt.ylabel('Accuracy_score')
plt.legend();
final_model = KNeighborsClassifier(n_neighbors= 4)
final_model.fit(X_train , y_train)
print(f'Training accuracy: f{final_model.score(X_train , y_train)}')
print(f'Testing accuracy: f{final_model.score(X_test , y_test)}')