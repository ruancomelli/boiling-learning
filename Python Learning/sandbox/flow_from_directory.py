# from tensorflow.keras.image import ImageDataGenerator

# https://stackoverflow.com/questions/59266506/pandas-json-normalize-and-its-behavior

import json
from pandas.io.json import json_normalize

sample_data = '''
[
    {
        "productid": 123,
        "name": "produkt 1",
        "shortname": "p1",
        "owner": [{
            "name": "Peewee Herman",
            "orgId": "ACME Inc.",
            "shortname": "ph"
        }]
    },
    {
        "productid": 456,
        "name": "produkt 2",
        "shortname": "p2",
        "owner": [{
            "name": "Darth Vader",
            "orgId": "The Empire Inc.",
            "shortname": "rt"
        }
    }
]
'''

import pandas as pd

data = json.loads(sample_data)
print(
    json_normalize(
        data,
        'productid',
        meta=['name', ['owner', 'name']]
    )
)

# print(data)
# as_dict_of_lists = pd.DataFrame(data) #.to_dict('list')
# print(as_dict_of_lists)
# as_dict_of_lists['owner.name'] = as_dict_of_lists['owner'].apply(lambda x: x['name'])
# print(as_dict_of_lists)

# regular_data = {}
# for d in data:
#     for key, value in d.items():
#         if key not in regular_data:
#             regular_data[key] = []
#         if key == 'owner':
#             regular_data[key].append(value['name'])
#         else:
#             regular_data[key].append(value)
# print(regular_data)
