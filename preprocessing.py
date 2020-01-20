import pandas as pd
from sklearn.preprocessing import FunctionTransformer

def impute_age(data):
    return

Age_Imputer = FunctionTransformer(impute_age, validate=True)

if __name__ == '__main__':
    train_df = pd.read_csv('input/train.csv')
    test_df = pd.read_csv('input/test.csv')

    print(train_df.info())
    print(test_df.info())

