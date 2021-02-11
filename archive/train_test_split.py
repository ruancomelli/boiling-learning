from sklearn.model_selection import train_test_split
import numpy as np
import modin.pandas as pd

train_key = 'train'
test_key = 'test'
n_samples = 5
indices = np.arange(n_samples, dtype=int)
print(indices)
train_indices, test_indices = train_test_split(indices, train_size=2, test_size=None)
print(train_indices, test_indices)

df = pd.DataFrame({
    'x': [4, 10, 15, 11, 95],
    'y': [25, 0, -20, 3, -np.pi]
})
# df['z'] = [9, 10, 2]
print(df)
# new_series = pd.Series(data=[train_key] * len(train_indices) + [test_key] * len(test_indices), index=np.concatenate([train_indices, test_indices]), dtype='category')
# new_series = pd.concat((
#     pd.Series(data=train_key, index=train_indices, dtype='category'),
#     pd.Series(data='val', index=[], dtype='category'),
#     pd.Series(data=test_key, index=test_indices, dtype='category')
# )) 

new_series = pd.Series(pd.concat((
    pd.Series(data=train_key, index=train_indices),
    pd.Series(data='val', index=[]),
    pd.Series(data=test_key, index=test_indices)
)), dtype='category')

print(new_series)
df['set'] = new_series
df.dtypes