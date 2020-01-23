import pandas as pd

def preprocess():
    train_df = pd.read_csv('input/train.csv')
    test_df = pd.read_csv('input/test.csv')

    print(train_df.info())
    print(test_df.info())

def clean_data():
    pass
def pipeline():
    pass



if __name__ == '__main__':

