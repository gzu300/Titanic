#!/usr/bin/env python3
'''
23/1/2020
data cleaning chunks done. 
to do:
feature engineering; feature selection; model selection; ensembl
'''
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier

def preprocess(df):
    train = df.copy()
    train['cabin_cat'] = train.Cabin.str[0]
    train.cabin_cat.fillna('no_cabin', inplace=True)
    train.Cabin.fillna('no_cabin', inplace=True)
    # assign all cabins to 'has_cabin'
    train.cabin_cat.mask(train.cabin_cat!='no_cabin', 'has_cabin', inplace=True)

    # merge 'Sibsp' and 'Parch'
    train['family_size'] = train.SibSp + train.Parch
    #categorize family_size to 3 categories
    small_size = (train.family_size >0)|(train.family_size < 4)
    alone = train.family_size < 1
    large_size = train.family_size > 3 
    train['family_cat'] = train.family_size.copy()
    train.family_cat.mask(small_size, 'small', inplace=True)
    train.family_cat.mask(alone, 'alone', inplace=True)
    train.family_cat.mask(large_size, 'large', inplace=True)
    
    #'Age' can be inferred by the title in names
    train['Initial'] = train.Name.str.extract(r'(\w+\.)')
    
    #mop titles
    train.Initial.replace(['Mrs.','Miss.'], 'Ms.', inplace=True)
    train.Initial.replace(['Dr.', 'Col.', 'Major.', 'Jonkheer.', 'Capt.', 'Sir.', 'Don.', 'Rev.', 'Mlle.', 'Lady.', 'Mme.', 'Countess.', 'Dona.', 'Mlle.'], 'Noble.', inplace=True)
    train['Age'] = train.groupby('Initial')['Age'].transform(lambda x: x.fillna(x.mean())) #fill missing ages with mean value of their group by 'Initial'

    #assign a new group in 'Sex': 'Children'
    train.loc[train.Initial == 'Master.', 'Sex'] = 'Children'
    train.loc[(train.Initial == 'Ms.')&(train.Age < 18), 'Sex'] = 'Children'

    # transform 'Fare' into 3 categorical groups
    train['Fare'] = train.groupby('Pclass')['Fare'].transform(lambda x:x.fillna(x.mean()))
    train['fare_range'] = pd.qcut(train.Fare, 3)
    train['fare_cat'] = train['fare_range'].map(dict(zip(train['fare_range'].cat.categories,['poor', 'mid', 'rich'])))

    train.Embarked.fillna('S', inplace=True) #fill with the most freq
   
    return train

def stratify(df):

    validation = train_raw.sample(frac=0.2, random_state=0)
    train = train_raw.loc[~train_raw.index.isin(validation.index), :]

    return train.drop('Survived', axis=1), train.Survived.values, validation.drop('Survived', axis=1), validation.Survived.values


def clean_data():
    train_raw = pd.read_csv('input/train.csv')
    test_raw = pd.read_csv('input/test.csv')

    train = preprocess(train_raw)
    x_test = preprocess(test_raw)

    x_train, x_val, y_train, y_val = stratify(train)
    
    return x_train, x_val, y_train, y_val, x_test

def pipeline():

    x_train, x_val, y_train, y_val, x_test = clean_data()

    onehot_coding = []
    ordinal_coding = []

    STEPS = [(''), ()]
    PARAMS = [{},{}]

    pipeline = Pipeline(steps=steps)
    voting_ests = VotingClassifier(pipeline)
    grid_cv = GridSearchCV(voting_ests, para_grid=params)


if __name__ == '__main__':
    pipeline()