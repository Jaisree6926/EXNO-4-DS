# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
       import pandas as pd
       import numpy as np
       import seaborn as sns
       
       from sklearn.model_selection import train_test_split
       from sklearn.neighbors import KNeighborsClassifier
       from sklearn.metrics import accuracy_score, confusion_matrix
       
       data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
       data
![322338017-b544c435-1cc1-4bc6-83c9-de2945348808](https://github.com/user-attachments/assets/5700970e-5571-45fd-bafd-e3b01bbce381)

       python
       data.isnull().sum()

![322338037-40b1ab98-5a1a-41a1-b943-102b7c4cabed](https://github.com/user-attachments/assets/12ccdcfe-bb62-433d-bd09-baf429be3aa2)

       
       missing=data[data.isnull().any(axis=1)]
       missing

![322338066-a5fe88ab-c993-4c97-b249-cffea5a21a54](https://github.com/user-attachments/assets/601122b2-69e1-42d6-bd8f-ea1c478c271d)


       data2=data.dropna(axis=0)
       data2
![322338086-40a10680-63a6-4f18-87ae-517ceda76ca9](https://github.com/user-attachments/assets/bc3cdc5a-8292-48fd-af51-c9e8a026a62e)

       sal=data["SalStat"]
       
       data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
       print(data2['SalStat'])
![322338114-e59ce957-1bdc-4455-97a5-15d66108b864](https://github.com/user-attachments/assets/2b93f963-eb8f-46cd-b113-87d6293a056a)

              sal2=data2['SalStat']
              
              dfs=pd.concat([sal,sal2],axis=1)
              dfs
![322338170-f8435063-835b-4eba-af2e-c46c67ea55e9](https://github.com/user-attachments/assets/f300cce2-5100-40c4-9a8a-1455e4b0928c)

       data2
![322338187-c034e83a-8e21-400e-bc40-103e3da86d0e](https://github.com/user-attachments/assets/f4b86319-8a87-410b-800e-c37bda76ed75)


       new_data=pd.get_dummies(data2, drop_first=True)
       new_data

![322338213-f21819e3-a5bd-47e6-b1b7-9bc08b64bed9](https://github.com/user-attachments/assets/ed35e44c-28ad-41eb-8e66-03d962336fcb)

       columns_list=list(new_data.columns)
       print(columns_list)

![322338249-8af6f5ce-4d99-4ed6-9371-730aeaa5a56b](https://github.com/user-attachments/assets/0ab67ca6-ee7e-4f3a-8d85-ef656244c707)
       
       features=list(set(columns_list)-set(['SalStat']))
       print(features)

![322338261-5f31a677-7d30-417a-8044-d5db741cafbf](https://github.com/user-attachments/assets/343e4786-fdf7-4d9e-9f6a-e1cc1ad3adf3)

       y=new_data['SalStat'].values
       print(y)
![322338286-f4c779af-4c87-449e-9daa-be5d8d275212](https://github.com/user-attachments/assets/5a5e5081-726a-43b7-bb6f-6da5d1020423)


       x=new_data[features].values
       print(x)
       
![322338321-4154db03-4c87-4b98-a13b-964f19bee9b0](https://github.com/user-attachments/assets/01454170-6809-43c9-956f-3091bc5728c6)


              train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
              
              KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
              
              KNN_classifier.fit(train_x,train_y)
              
![322338343-e5e02520-eb39-436c-ac2e-e43048c1d672](https://github.com/user-attachments/assets/b697b78e-cd92-4e03-a6ed-76927acdea7d)

       prediction=KNN_classifier.predict(test_x)
       
       confusionMatrix=confusion_matrix(test_y, prediction)
       print(confusionMatrix)

![322338371-a6eedfe3-aedd-4500-958f-6faafd54f464](https://github.com/user-attachments/assets/eb295a92-1a24-42d5-85a1-f11d7615c6fd)

       accuracy_score=accuracy_score(test_y,prediction)
       print(accuracy_score)

![322338387-0e56ff41-2f35-4d01-b479-53547391567b](https://github.com/user-attachments/assets/936c8be4-6126-4d23-a4dc-03138963adb1)

       print("Misclassified Samples : %d" % (test_y !=prediction).sum())
![322338405-4af5ed3f-362a-40c6-a438-c89f31584e51](https://github.com/user-attachments/assets/198b5d0a-0288-4ab3-ac5a-bc7e76785b01)

       data.shape
![322338420-1986f990-26e6-4b42-acfc-b2a6e52f8042](https://github.com/user-attachments/assets/1104a0f6-834c-4d04-b0ab-b1a79c4c3f0f)


       import pandas as pd
       from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
       data={
           'Feature1': [1,2,3,4,5],
           'Feature2': ['A','B','C','A','B'],
           'Feature3': [0,1,1,0,1],
           'Target'  : [0,1,1,0,1]
       }
       
       df=pd.DataFrame(data)
       x=df[['Feature1','Feature3']]
       y=df[['Target']]
       
       selector=SelectKBest(score_func=mutual_info_classif,k=1)
       x_new=selector.fit_transform(x,y)
       
       selected_feature_indices=selector.get_support(indices=True)
       
       selected_features=x.columns[selected_feature_indices]
       print("Selected Features:")
       print(selected_features)

![322338454-20777b0d-3cdb-4ae9-80e4-1f76ed093191](https://github.com/user-attachments/assets/02f3c04c-b665-4883-b468-97cf155b70d6)


       import pandas as pd
       import numpy as np
       from scipy.stats import chi2_contingency
       
       import seaborn as sns
       tips=sns.load_dataset('tips')
       tips.head()
![322338479-6d6f7ff2-b1da-4568-9cd1-cb6fa9553cd6](https://github.com/user-attachments/assets/d3cb44dd-4886-4f9a-a553-3b95c0de9fcd)


       tips.time.unique()

![322338497-f77bc757-8a31-4a5d-be15-5a447e6549c6](https://github.com/user-attachments/assets/53674748-882a-4bce-aeae-a3cc7773032e)

       contingency_table=pd.crosstab(tips['sex'],tips['time'])
       print(contingency_table)

![322338518-06365e9f-f51b-4cf6-ab04-8a136726a025](https://github.com/user-attachments/assets/9dc5c2a0-b39e-4a30-b5f8-177c62d1d9e8)
       
       chi2,p,_,_=chi2_contingency(contingency_table)
       print(f"Chi-Square Statistics: {chi2}")
       print(f"P-Value: {p}")
![322338543-6adc4da7-421c-458f-9ec6-f6158aa6f731](https://github.com/user-attachments/assets/5ec374ab-3bbe-41d7-93c0-4d7de2fe10dd)

# RESULT:
      Thus, Feature selection and Feature scaling has been used on thegiven dataset.
