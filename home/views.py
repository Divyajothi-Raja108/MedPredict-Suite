from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# Create your views here.
def index(request):
    return render(request, 'index.html')
def heart1(request):
    return render(request, 'heart1.html')
def diabetes(request):
    return render(request, 'diabetes.html')
def liver(request):
    return render(request, 'liver.html')
def result(request):
    df = pd.read_csv(r'C:\Users\divyajothi\Desktop\Multiple Disease Prediction System Final\Multiple_Disease_Prediction\Heart_data1.csv')
    df2 = df.copy()
    def remove_outliers_iqr_column(column):
     Q1 = np.percentile(column, 25)
     Q3 = np.percentile(column, 75)
     IQR = Q3 - Q1 
     lower_bound = Q1 - 1.5 * IQR
     upper_bound = Q3 + 1.5 * IQR
     outliers_indices = np.where((column < lower_bound) | (column > upper_bound))
     cleaned_column = column[~((column < lower_bound) | (column > upper_bound))]
     return cleaned_column, outliers_indices
    for column in df.columns:
      cleaned_column, outliers_indices = remove_outliers_iqr_column(df[column])
      df2[column] = cleaned_column
    df1 = df2.fillna(df.mean())
    df1.drop(columns='id', axis=1, inplace=True)
    X = df1.drop(columns='cardio', axis=1)
    Y= df1['cardio']
    scaler = StandardScaler()
    scaler.fit(X)
    st_data = scaler.transform(X)
    X = st_data
    Y= df1['cardio']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    model = LogisticRegression()
    model.fit(X_train, Y_train)

    var1 = int(request.GET['age'])
    var2 = int(request.GET['gender'])
    var3 = int(request.GET['height'])
    var4 = float(request.GET['weight'])
    var5 = int(request.GET['systole'])
    var6 = int(request.GET['diastole']) 
    var7 = int(request.GET['chol'])
    var8 = int(request.GET['glu'])
    var9 = int(request.GET['smoke'])  
    var10 = int(request.GET['alc'])
    var11 = int(request.GET['act'])

    input_data = (var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    std_data = scaler.transform(input_data_reshaped)
    prediction = model.predict(std_data)
    result1 = ""
    if(prediction[0] == 0):
      result1 = "The Person is predicted not to have a Heart Disease"
    else:
      result1 = "The Person is predicted to have a Heart Disease"



    return render(request, 'heart1.html', {"result":result1})

def result2(request):
   df3 = pd.read_csv(r'C:\Users\divyajothi\Desktop\Multiple Disease Prediction System Final\Multiple_Disease_Prediction\diabetes_prediction_dataset.csv')
   df4 = df3.copy()
   label_encoder = LabelEncoder()
   df4['gender'] = label_encoder.fit_transform(df4['gender'])
   df4['smoking_history'] = label_encoder.fit_transform(df4['smoking_history'])
   X = df4.drop(columns ='diabetes', axis=1)
   Y = df4['diabetes']
   scaler = StandardScaler()
   scaler.fit(X)
   st_data = scaler.transform(X)
   X = st_data
   X_train, X_test, Y_train,Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
   smote = SMOTE(random_state=2)
   X_train_res, Y_train_res = smote.fit_resample(X_train, Y_train)
   model = RandomForestClassifier(random_state=2)
   model.fit(X_train_res, Y_train_res)

   var1 = int(request.GET['gender'])
   var2 = float(request.GET['age'])
   var3 = int(request.GET['hypertension'])
   var4 = int(request.GET['heart_disease'])
   var5 = int(request.GET['smoking_history'])
   var6 = float(request.GET['bmi']) 
   var7 = float(request.GET['hba1c_level'])
   var8 = int(request.GET['blood_glucose_level'])
  


   input_data = (var1, var2, var3, var4, var5, var6, var7, var8)
   input_data_as_numpy_array = np.asarray(input_data)
   input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
   std_data = scaler.transform(input_data_reshaped)
   prediction = model.predict(std_data)
   predict=""
   if(prediction[0] == 0):
     predict="The Person is predicted not to have Diabetes"
   else:
     predict="The Person is predicted to have Diabetes"
   return render(request,'diabetes.html',{"result2":predict})
   
def result3(request):
   df = pd.read_csv(r'C:\Users\divyajothi\Desktop\Multiple Disease Prediction System Final\Multiple_Disease_Prediction\Liver Patient Dataset (LPD)_train.csv', encoding='ISO-8859-1')
   df1 = df.copy()
   df1['Gender of the patient'].fillna('Female', inplace=True)
   label_encoder = LabelEncoder()
   df1['Gender of the patient'] = label_encoder.fit_transform(df1['Gender of the patient'])

   new_names = {
    'Age of the patient': 'Age',
    'Gender of the patient': 'Gender',
    'Total Bilirubin': 'Total_Bilirubin',
    'Direct Bilirubin': 'Direct_Bilirubin',
    '\xa0Alkphos Alkaline Phosphotase': 'Alkphos',
    '\xa0Sgpt Alamine Aminotransferase': 'Alam',
    'Sgot Aspartate Aminotransferase': 'Aspartate',
    'Total Protiens': 'Proteins',
    '\xa0ALB Albumin': 'Albumin',
    'A/G Ratio Albumin and Globulin Ratio': 'Ratio'
}
   df1 = df1.rename(columns=new_names)

   df1['Age'].fillna(df1['Age'].mean(), inplace=True)
   df1['Total_Bilirubin'].fillna(df1['Total_Bilirubin'].mean(), inplace=True)
   df1['Direct_Bilirubin'].fillna(df1['Direct_Bilirubin'].mean(), inplace=True)
   df1['Alkphos'].fillna(df1['Alkphos'].mean(), inplace=True)
   df1['Alam'].fillna(df1['Alam'].mean(), inplace=True)
   df1['Aspartate'].fillna(df1['Aspartate'].mean(), inplace=True)
   df1['Proteins'].fillna(df1['Proteins'].mean(), inplace=True)
   df1['Albumin'].fillna(df1['Albumin'].mean(), inplace=True)
   df1['Ratio'].fillna(df1['Ratio'].mean(), inplace=True)
   
   X = df1.drop('Result', axis=1)
   Y = df1['Result']
   scaler = StandardScaler()
   scaler.fit(X)
   st_data = scaler.transform(X)
   X = st_data
   X_train, X_test, Y_train,Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
   smote = SMOTE(random_state=2)
   X_train_res, Y_train_res = smote.fit_resample(X_train, Y_train)
   model = RandomForestClassifier(random_state=2)
   model.fit(X_train_res, Y_train_res)

   var1 = float(request.GET['age'])
   var2 = int(request.GET['gender'])
   var3 = float(request.GET['bilirubin'])
   var4 = float(request.GET['direct'])
   var5 = float(request.GET['alkaline'])
   var6 = float(request.GET['alamine']) 
   var7 = float(request.GET['aspartate'])
   var8 = float(request.GET['totalprotiens'])
   var9 = float(request.GET['albumin'])  
   var10 = float(request.GET['albuminandglobulinratio'])



   input_data = (var1, var2, var3, var4, var5, var6, var7, var8, var9, var10)
   input_data_as_numpy_array = np.asarray(input_data)
   input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
   std_data = scaler.transform(input_data_reshaped)
   prediction = model.predict(std_data)
   predict1=""
   if(prediction[0] == 1):
     predict1="The Person is predicted  to have Liver Disease"
   else:
     predict1="The Person is predicted  not to have Liver Disease"
   return render(request, 'liver.html', {"result3":predict1})