import numpy as np

def get_predict(df):
    lr_coef = np.load('coef.npy')
    SANHOK_columns = []
    for i in range(3):
        SANHOK_columns.append(f'SANHOK_bid{i}_price')
        SANHOK_columns.append(f'SANHOK_ask{i}_price')
    y_pred = np.matmul(df[SANHOK_columns], lr_coef)
    return y_pred
