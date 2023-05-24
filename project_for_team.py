import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score 
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score 
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

class Project:
    def __init__(self):
        self.Data=pd.read_csv('.archhealthcare-dataset-stroke-data.csv')
        
    def explore_data(self):
        print(f'The Data :\n{self.Data}') 
        print(f'\nData size={self.Data.shape}\n')
        print(f'\nthe columns of data: \n{self.Data.columns}')
        print(f'\nFirst 5 Rows in from the Data:\n {self.Data.head()}')
        print(f'The Data Info : {self.Data.info()}')
        print(f'\nNumber of duplicated row={self.Data.duplicated().sum()}')
        print(f'The empty cell in each columns before cleaning:\n{self.Data.isnull().sum()}')
        print(f'The Describe of Data :\n {self.Data.describe()}')
        print(f'The Element of Gender column without Repeating:\n{self.Data.gender.unique()}')
        

        
    def preprocess_data(self):
        bmi_mean=self.Data['bmi'].mean()
        self.Data['bmi']=self.Data['bmi'].fillna(bmi_mean)
        self.Data['gender']=self.Data['gender'].replace({'Other': 'Male'})
        print(f'The empty cell in each columns after cleaning:\n{self.Data.isnull().sum()}')
        
    def visualize_data(self):
        plt.figure(figsize=(10, 5))
        sns.histplot(self.Data["age"], bins=10, kde=True)
        plt.title("Age Distribution")
        plt.xlabel("Age")
        plt.ylabel("Count")
        plt.show()

        plt.figure(figsize=(10, 5))
        sns.countplot(x="gender",data=self.Data)
        plt.title("Gender Count")
        plt.xlabel("Gender")
        plt.ylabel("Count")
        plt.show()

        plt.figure(figsize=(10, 5))
        sns.boxplot(x="smoking_status",y="bmi",data=self.Data)
        plt.title("BMI Distribution by Smoking Status")
        plt.xlabel("Smoking Status")
        plt.ylabel("BMI")
        plt.show()

        plt.figure(figsize=(10, 5))
        sns.countplot(x="stroke",data=self.Data)
        mean_stroke = self.Data["stroke"].mean()
        plt.axhline(mean_stroke, color='red',linestyle='--',label='Mean')
        plt.show()

        plt.figure(figsize=(8, 8))
        self.Data["smoking_status"].value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title("Smoking Status Distribution")
        plt.show()

        plt.figure(figsize=(10, 5))
        sns.scatterplot(x="age", y="avg_glucose_level",data=self.Data)
        plt.title("Age vs. Average Glucose Level")
        plt.xlabel("Age")
        plt.ylabel("Average Glucose Level")
        plt.show()
        
        
        
    def encode_categorical(self):
        Data_types = self.Data.dtypes
        for i in range(self.Data.shape[1]):
            if Data_types[i] == "O":
                pr_data = preprocessing.LabelEncoder()
                self.Data[self.Data.columns[i]]=pr_data.fit_transform(self.Data[self.Data.columns[i]])
            
                
    def correlation_heatmap(self):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.Data.corr(),annot=True)
        plt.title("Correlation Heatmap")
        plt.show()
        
        plt.figure(figsize=(10, 5))
        sns.boxplot(x="ever_married", y="age",data=self.Data)
        plt.title("Age Distribution by Marital Status")
        plt.xlabel("Marital Status")
        plt.ylabel("Age")
        plt.show()
        
    def drop_columns(self):
        self.Data = self.Data.drop(['work_type','id'], axis=1)
        print(f'the columns of data:\n{self.Data.columns}')
        print(f'\nSize of data after droping={self.Data.shape}')
        
class DataPreprocessing:
    def __init__(self, data):
        self.Data=data
    
    def scaling(self):
        self.features=self.Data.iloc[:, :-1]
        self.scaler=preprocessing.MinMaxScaler()
        self.scaled_data=self.scaler.fit_transform(self.features)
        self.scaled_data=pd.DataFrame(self.scaled_data, columns=self.features.columns)
        print(f'Data after scaling :\n{self.scaled_data} ')
        
class ModelEvaluation:
    def __init__(self, X, y):
        self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(X, y,test_size=0.2)
        print(f'X_train=\n{self.X_train}')
        print(f'X_test=\n{self.X_test}')
        print(f'Y_train=\n{self.y_train}')
        print(f'Y_test=\n{self.y_test}')
        
        
        
    def evaluate_logistic_regression(self):
        model = LogisticRegression()
        model.fit(self.X_train, self.y_train)
        y_pred=model.predict(self.X_test)
        
        con = confusion_matrix(self.y_test, y_pred)
        print('Confusion Matrix for Logistic Regression:\n',con)
        sns.heatmap(con,annot=True,cmap='coolwarm')
        plt.title("Correlation Heatmap of Confusion Matrix for Logistic Regression")
        plt.show()
        
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f'Accuracy for Logistic Regression:{accuracy}')
        
        f1 = f1_score(self.y_test, y_pred,average='micro')
        print(f'F1 Score for Logistic Regression: {f1}')
        
        recall = recall_score(self.y_test,y_pred,average='micro')
        print(f'Recall Score for Logistic Regression:{recall}')
        
      
        
    def evaluate_svc(self):
        clf=SVC(kernel='linear')
        clf.fit(self.X_train,self.y_train)
        y_pred_svc=clf.predict(self.X_test)
        
        con_svc=confusion_matrix(self.y_test, y_pred_svc)
        print('Confusion Matrix for SVC:\n', con_svc)
        sns.heatmap(con_svc, annot=True, cmap='coolwarm')
        plt.title("Correlation Heatmap of Confusion Matrix for SVC")
        plt.show()
        
        accuracy_svc=accuracy_score(self.y_test, y_pred_svc)
        print(f'Accuracy for SVC: {accuracy_svc}')
        
        f1_svc=f1_score(self.y_test, y_pred_svc, average='micro')
        print(f'F1 Score for SVC: {f1_svc}')
        
        recall_svc=recall_score(self.y_test, y_pred_svc, average='micro')
        print(f'Recall Score for SVC: {recall_svc}')
    
    
    

    
      

Project_data=Project()
Project_data.explore_data()

Project_data.preprocess_data()

Project_data.visualize_data()

Project_data.encode_categorical()

Project_data.correlation_heatmap()

Project_data.drop_columns()

data_preprocessing=DataPreprocessing(Project_data.Data)
data_preprocessing.scaling()

X=data_preprocessing.scaled_data
Y=Project_data.Data.iloc[:, -1]

model_evaluation=ModelEvaluation(X,Y)
model_evaluation.evaluate_logistic_regression()
model_evaluation.evaluate_svc()

