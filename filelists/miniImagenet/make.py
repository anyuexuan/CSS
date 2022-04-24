import numpy as np
import os
import json
import pandas as pd
from tqdm import tqdm

cwd = os.getcwd()
data_path = os.path.join(cwd, 'miniImagenet')

all = {}
all['label_names'] = []
all['image_names'] = []
all['image_labels'] = []

trains = np.array(pd.read_csv(os.path.join(cwd, 'train.csv')))
base = {}
base['label_names'] = []
base['image_names'] = []
base['image_labels'] = []
for i in tqdm(range(trains.shape[0] // 600)):
    all['label_names'].append(trains[600 * i, 1])
    base['label_names'].append(trains[600 * i, 1])
    names = os.listdir(os.path.join(data_path, trains[600 * i, 1]))
    for name in names:
        all['image_names'].append(os.path.join(data_path, trains[600 * i, 1], name))
        all['image_labels'].append(i)
        base['image_names'].append(os.path.join(data_path, trains[600 * i, 1], name))
        base['image_labels'].append(i)

vals = np.array(pd.read_csv(os.path.join(cwd, 'val.csv')))
val = {}
val['label_names'] = []
val['image_names'] = []
val['image_labels'] = []
for i in tqdm(range(vals.shape[0] // 600)):
    all['label_names'].append(vals[600 * i, 1])
    val['label_names'].append(vals[600 * i, 1])
    names = os.listdir(os.path.join(data_path, vals[600 * i, 1]))
    for name in names:
        all['image_names'].append(os.path.join(data_path, vals[600 * i, 1], name))
        all['image_labels'].append(i + trains.shape[0] // 600)
        val['image_names'].append(os.path.join(data_path, vals[600 * i, 1], name))
        val['image_labels'].append(i + trains.shape[0] // 600)

tests = np.array(pd.read_csv(os.path.join(cwd, 'test.csv')))
test = {}
test['label_names'] = []
test['image_names'] = []
test['image_labels'] = []
for i in tqdm(range(tests.shape[0] // 600)):
    all['label_names'].append(tests[600 * i, 1])
    test['label_names'].append(tests[600 * i, 1])
    names = os.listdir(os.path.join(data_path, tests[600 * i, 1]))
    for name in names:
        all['image_names'].append(os.path.join(data_path, tests[600 * i, 1], name))
        all['image_labels'].append(i + (trains.shape[0] + vals.shape[0]) // 600)
        test['image_names'].append(os.path.join(data_path, tests[600 * i, 1], name))
        test['image_labels'].append(i + (trains.shape[0] + vals.shape[0]) // 600)

json.dump(base, open('base.json', 'w'))
json.dump(val, open('val.json', 'w'))
json.dump(test, open('novel.json', 'w'))
json.dump(all, open('all.json', 'w'))

data = json.load(open('all.json'))
print(data.keys())
print(len(data['label_names']))
print(len(data['image_names']))
print(len(data['image_labels']), np.min(data['image_labels']), np.max(data['image_labels']))

data = json.load(open('base.json'))
print(data.keys())
print(len(data['label_names']))
print(len(data['image_names']))
print(len(data['image_labels']), np.min(data['image_labels']), np.max(data['image_labels']))

data = json.load(open('val.json'))
print(data.keys())
print(len(data['label_names']))
print(len(data['image_names']))
print(len(data['image_labels']), np.min(data['image_labels']), np.max(data['image_labels']))

data = json.load(open('novel.json'))
print(data.keys())
print(len(data['label_names']))
print(len(data['image_names']))
print(len(data['image_labels']), np.min(data['image_labels']), np.max(data['image_labels']))
