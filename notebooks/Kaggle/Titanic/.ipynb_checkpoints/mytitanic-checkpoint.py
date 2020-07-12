import numpy as np
import pandas as pd
from scipy import stats
from copy import deepcopy


train_df = pd.read_csv("titanic_train.csv")
test_df  = pd.read_csv("titanic_test.csv") 


"----------------FILL NA----------------"
for str_column in ['Embarked']:
    [df[str_column].fillna(stats.mode(df.Embarked).mode[0], inplace=True) for df in [train_df, test_df]]

for float_column in ['Age', 'Fare']:
    [df[float_column].fillna(df[float_column].median(), inplace=True) for df in [train_df, test_df]]
    
    
"--------------REMOVE QUANTILES----------"
Ntrain, Ntest = len(train_df), len(test_df)

percent = 0.01
for column in ['Parch', 'SibSp']:
    vcounts = train_df[column].value_counts()
    alive_indexes = vcounts.index[vcounts >= percent * len(train_df)]
    print(f'removed {vcounts.index[vcounts <= percent * len(train_df)]} for column {column}')
    train_df = train_df[train_df[column].isin(alive_indexes).values]
    

"-------------ONE HOTE ENCODING ----------"

train_df = pd.concat([train_df, 
                      pd.get_dummies(train_df.Embarked,prefix='Embarked'),
                      pd.get_dummies(train_df.Pclass,  prefix='Pclass'),
                      pd.get_dummies(train_df.Parch,   prefix='Parch'),
                      pd.get_dummies(train_df.Sex,     prefix='Sex'),
                      pd.get_dummies(train_df.SibSp,   prefix='SibSp'),
                     ], axis=1)

test_df = pd.concat([test_df, 
                      pd.get_dummies(test_df.Embarked,prefix='Embarked'),
                      pd.get_dummies(test_df.Pclass,  prefix='Pclass'),
                      pd.get_dummies(test_df.Parch,   prefix='Parch'),
                      pd.get_dummies(test_df.Sex,     prefix='Sex'),
                      pd.get_dummies(test_df.SibSp,   prefix='SibSp'),
                     ], axis=1)


[df.drop(['Pclass', 'Name', 'Sex', 'SibSp','Parch', 'Ticket', 'Cabin', 'Embarked', 'PassengerId'], axis=1, inplace=True) for df in [train_df, test_df]]

X_train, y_train = train_df.drop('Survived', axis=1), train_df['Survived']


"-------------ADOPT TEST DF TO TRAIN DF -----------"
untrained_columns = set(test_df.columns) - set(train_df.columns)
test_df.drop(untrained_columns, axis=1, inplace=True)