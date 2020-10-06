import math
import numpy as np
import pandas as pd
from seaborn import countplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure,show
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def MyTitanicLogistic():
    titanic_data = pd.read_csv("TitanicDataset.csv")

    print("First 5 entries: ")
    print(titanic_data.head())

    print("Total no of passangers are: ",len(titanic_data))


    print("Visualization: Survived and Non-Survived passangers")
    figure()
    target = "Survived"
    countplot(data = titanic_data, x = target).set_title("Survived and Non-Survived passangers")
    show()

    print("Visualization: Survived and Non-Survived passangers based on Gender")
    figure()
    target = "Survived"
    countplot(data = titanic_data, x = target, hue = "Sex").set_title("Survived and Non-Survived passangers based on Gender")
    show()

    print("Visualization: Survived and Non-Survived passangers based on Passanger Class")
    figure()
    target = "Survived"
    countplot(data = titanic_data, x = target, hue = "Pclass").set_title("Survived and Non-Survived passangers based on Passanger Class")
    show()


    AgeGrp = []
    FareGrp =[]
    for i in range(len(titanic_data)):
        if titanic_data["Age"][i]<18:
            AgeGrp.append("Children")
        elif titanic_data["Age"][i]>18 and titanic_data["Age"][i]<65:
            AgeGrp.append("Adult")
        else:
            AgeGrp.append("Elder")

        if titanic_data['Fare'][i]<50:
            FareGrp.append("Low")
        elif titanic_data['Fare'][i]>50 and titanic_data['Fare'][i]<100:
            FareGrp.append("Medium")
        else:
            FareGrp.append("High")

    titanic_data1 = titanic_data.assign(AgeGroup = AgeGrp)
    titanic_data1 = titanic_data1.assign(FareGroup = FareGrp)
    #print(titanic_data1.head(10))
    print("Visualization: Survived and Non-Survived passangers based on Age")
    figure()
    target = "Survived"
    countplot(data = titanic_data1, x = target, hue = "AgeGroup").set_title("Survived and Non-Survived passangers based on Passanger Age")
    show()

    print("Visualization: Survived and Non-Survived passangers based on Fare")
    figure()
    target = "Survived"
    countplot(data = titanic_data1, x = target, hue = "FareGroup").set_title("Survived and Non-Survived passangers based on Passanger Fare")
    show()

    titanic_data.drop("zero",axis = 1, inplace=True)
    print("First 5 entries after removing zero column")
    print(titanic_data.head())

    print("Values of Sex Column")
    print(pd.get_dummies(titanic_data["Sex"]).head())

    print("Values of Sex column after removing one field")
    Sex = pd.get_dummies(titanic_data["Sex"], drop_first=True)
    print(Sex.head())

    print("Values of Pclass column after removing one field")
    Pclass = pd.get_dummies(titanic_data["Pclass"], drop_first=True)
    print(Pclass.head())

    print("Values of Dataset after concatinating new columns")
    titanic_data = pd.concat([titanic_data, Sex, Pclass], axis = 1)
    print(titanic_data.head())

    print("Values of Dataset after removing irrelevant columns")
    titanic_data.drop(["Sex","sibsp","Parch","Embarked"],axis=1,inplace=True)
    print(titanic_data.head())

    x = titanic_data.drop("Survived",axis=1)
    y = titanic_data["Survived"]

    xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size = 0.5)
    logmodel = LogisticRegression()
    logmodel.fit(xtrain,ytrain)

    prediction = logmodel.predict(xtest)

    print("Classification Report is: ")
    print(classification_report(ytest,prediction))

    print("Confusion matrix is: ")
    print(confusion_matrix(ytest, prediction))

    print("Accuracy is: ")
    print(accuracy_score(ytest,prediction))



MyTitanicLogistic()
