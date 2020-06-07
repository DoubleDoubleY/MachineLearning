#DMSL PMA 1951478
import pandas as pd
import numpy as np
import re #正则表达式
import time
import datetime
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC as SVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier as DTC

#read data
#df = pd.read_csv('/Users/mmy/Desktop/PMA_blockbuster_movies_test.csv')
#clean data
def data_clean(df):
    df=df.drop(columns=['poster_url','title','release_date'])
    #check the data type of each columns, and I find the majority of the data type are object
    #So the next step is to transfrom data
    print('Blow is the data type of all columns')
    print(str(df.dtypes))

    #data transformation
    #transform "2015_inflation"
    for i in range(len(df)):
        #use the Regular expression to transfrom the data from object to float
        a=re.findall(r"\d+\.?\d*",df['2015_inflation'][i])
        df['2015_inflation'][i]=float(a[0])
        if df['2015_inflation'][i]== 0.26:  #Whether it is negative
            df['2015_inflation'][i]= -0.26
    df['2015_inflation']=df['2015_inflation'].astype('float64')
    print(str(df.dtypes))
    print(str(df['2015_inflation']))
    #transform "adjusted" and "worldwide_gross"
    df['adjusted'] = df['adjusted'].replace('[\$,]', '', regex=True).astype(float)
    #Reference: https://stackoverflow.com/questions/32464280/converting-currency-with-to-numbers-in-python-pandas
    df['worldwide_gross'] = df['worldwide_gross'].replace('[\$,]', '', regex=True).astype(float)
    print(str(df.dtypes))
    #transfrom the genre
    #For instance, if the movies has three genres, the number of the genre is "3"
    for i in range(len(df)):
        if str(df['Genre_2'][i])=='nan':
            df['genres'][i]=int(1)
        elif str(df['Genre_3'][i])=='nan':
            df['genres'][i]=int(2)
        else:
            df['genres'][i]=int(3)
    df['genres']=df['genres'].astype(int)
    df=df.drop(columns=['Genre_1','Genre_2','Genre_3'])
    #Trans "rating" column to dummies
    df = pd.get_dummies(df, columns=["rating"], prefix=["rating_type"])
    #Transfrom studio
    #if the '/' in the string, it means the movie has more than two studios 
    pattern1 = "/"
    for i in range(len(df)):
        if (pattern1 in str(df['studio'][i])):
            df['studio'][i]=1 #This means the movie has more than two studios
        else:
            df['studio'][i]=0 #This means the movie is completed by only one studio
    df['studio']=df['studio'].astype(int)
    print(df.dtypes)
    #delete the duplicate data
    df=df.drop_duplicates()
    #Bacause the Y in this programme is 'adjusted', and the model is classifier question, so the data needs to be transfrom to discrete data
    mean=df['adjusted'].mean()
    for i in range(len(df)):
        if df['adjusted'][i] >= mean:
            #print('adjusted is:'+str(df['adjusted'][i])+'  And mean is:'+str(df['adjusted'].mean()))
            df['adjusted'][i]=1
        else:
            #print('adjusted is:'+str(df['adjusted'][i])+'  And mean is:'+str(df['adjusted'].mean()))
            df['adjusted'][i]=0
    print('The mean isssssssssssss'+str(df['adjusted'].mean()))
    print(df['adjusted'])
    #Data normalization
    max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
    df=df.apply(max_min_scaler)
    #reset index
    df=df.reset_index(drop=True)
    #df = pd.get_dummies(df, columns=['SupportedLanguages'],)
    df.to_csv('/Users/mmy/Desktop/NEWWWWWW.csv')
    return df



def SVM(df):
    #Separate target Y and X
    X_train, X_test, Y_train, Y_test = train_test_split(df.iloc[:,1:], df['adjusted'], 	test_size = 0.2, random_state=5)
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)
    #run model on the training data
    svc=SVC()
    print(X_train)
    print(Y_train)
    # fit the model using some training data
    svc_fit = svc.fit(X_train, Y_train)
    # generate a mean accuracy score for the predicted data
    train_score = svc.score(X_train, Y_train)
    # print the mean accuracy of testing predictions
    print("Accuracy score = " + str(round(train_score, 4)))
    #Grid search
    tuned_parameters = [{'C': [1.0,5.0,10.0],
                     'kernel': ['rbf'],
                     'degree': [3,5,7]}]
    scores = ['accuracy', 'f1']
    #Optimise hyperparameters
    for score in scores:
        print("# Tuning hyperparameters for %s" % score)
        print("\n")
        clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                           scoring= score)
        clf.fit(X_train, Y_train)
        print("Best parameters set found on the training set:")
        print(clf.best_params_)
        print("\n")
    #Check the accuracy metrics again
    svc = SVC(C=10.0, degree=3, kernel= 'rbf')
    # fit the model using some training data
    svc_fit = svc.fit(X_train, Y_train)
    # generate a mean accuracy score for the predicted data
    train_score = svc.score(X_train, Y_train)
    # print the mean accuracy of testing predictions
    print("Training predictions Accuracy score  = " + str(round(train_score, 4)))
    ################predict
    # predict the test data
    predicted = svc.predict(X_test)
    print(predicted)
    # generate a mean accuracy score for the predicted data
    test_score = svc.score(X_test, Y_test)
    # print the mean accuracy of testing predictions
    print("Testing Accuracy score = " + str(round(test_score, 4)))
    return predicted


def LR(df):
    X_train, X_test, Y_train, Y_test = train_test_split(df.iloc[:,1:], df['adjusted'], test_size = 0.2, random_state=5) 
    # print the shapes to check everything is OK
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    '''
    params={'penalty':['l1','l2'],'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
    model = LogisticRegression()
    grid = GridSearchCV(model,params)
    grid_result=grid.fit(X_train,Y_train)
    print(grid_result.best_score_)
    print(grid_result.best_params_)
    '''
    modelLR=LogisticRegression(C=1, penalty='l2')
    #train model
    modelLR.fit(X_train,Y_train)
    a=modelLR.score(X_test,Y_test,)
    print(str(round(a,4)))
    predicted = modelLR.predict(X_test)
    print(predicted)
    return predicted


def DTC(df):
    from sklearn.tree import DecisionTreeClassifier as DTC
    X_train, X_test, Y_train, Y_test = train_test_split(df.iloc[:,1:], df['adjusted'], test_size = 0.2, random_state=5)  
    # print the shapes to check everything is OK
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)
    #params={'criterion':['gini','entropy'],'splitter':['best','random'],'max_depth': [1,5,7,9] }
    dtc=DTC()
    print(X_train)
    print(Y_train)
    # fit the model using some training data
    dtc_fit = dtc.fit(X_train, Y_train)
    # generate a mean accuracy score for the predicted data
    train_score = dtc.score(X_train, Y_train)
    # print the mean accuracy of testing predictions
    print("Accuracy score = " + str(round(train_score, 4)))

    tuned_parameters = [{'criterion':['gini','entropy'],'splitter':['best','random'],'max_depth': [1,5,7,9] }]
    scores = ['accuracy', 'f1']
    #Optimise hyperparameters
    for score in scores:
        print("# Tuning hyperparameters for %s" % score)
        print("\n")
        clf = GridSearchCV(DTC(), tuned_parameters, cv=5,
                           scoring= score)
        clf.fit(X_train, Y_train)
        print("Best parameters set found on the training set:")
        print(clf.best_params_)
        print("\n")
    
    #Check the accuracy metrics again
    dtc = DTC(criterion='gini', max_depth=7, splitter='random')
    # fit the model using some training data
    dtc_fit = dtc.fit(X_train, Y_train)
    # generate a mean accuracy score for the predicted data
    train_score = dtc.score(X_train, Y_train)
    # print the mean accuracy of testing predictions
    print("Training predictions Accuracy score  = " + str(round(train_score, 4)))
    ################predict
    # predict the test data
    predicted = dtc.predict(X_test)
    print(predicted)
    # generate a mean accuracy score for the predicted data
    test_score = dtc.score(X_test, Y_test)
    # print the mean accuracy of testing predictions
    print("Testing Accuracy score = " + str(round(test_score, 4)))
    
    return predicted


def main():
    address=input('*********Please enter your document address: *********')
    df = pd.read_csv(address)
    #df = pd.read_csv('/Users/mmy/Desktop/PMA_blockbuster_movies_test.csv')
    ensemble=input('***********Please enter your choice of algorithm(ensemble or single):')
    if ensemble=='ensemble':
        print('**********please enter your weight of three algorithm***********\n ********the sum of the three weight should be 1******')
        a=float(input())
        b=float(input())
        c=float(input())
        if a+b+c!=1:
            main()
        #if the sum of the three weight is not one,then run the main function again
        df=data_clean(df)
        LRpredicted=LR(df)
        SVMpredicted=SVM(df)
        DTCpredicted=DTC(df)
        
        #Voting
        result=DTCpredicted
        for i in range(len(LRpredicted)):
            weight=round(LRpredicted[i]*a+SVMpredicted[i]*b+DTCpredicted[i]*c)
            if weight>0.5:
                result[i]=1
            else:
                result[i]=0
            print(weight)
        #Output result
        print(result)
    elif ensemble=='single':
        df=data_clean(df)
        LRpredicted=LR(df)
    else:
        print('It seems your enter encoutered some mistake')
        main()
    return df

main()
