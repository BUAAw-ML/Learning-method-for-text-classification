import numpy as np
import pandas as pd

df0 = pd.read_csv('./data/APIs.csv')

domain = pd.read_csv('./data/tagnet.csv')

print(df0.columns)

tagset = domain['source'].drop_duplicates().values


apis = list()
taglist = list()

for idx, row in df0.iterrows():
	tags = ''
	t = row['tags']
	for t0 in t.split('###'):
		t0 = t0.strip()
		if t0 in tagset:
			tags = tags+t0+'###'
	if tags != '':
		apis.append(row['APIName'])
		taglist.append(tags)

df1 = df0[df0['APIName'].isin(apis)]
print(df1.shape)
df1['tags2'] = taglist
df2 = df1.drop(['tags'],axis=1)
print(df2.columns)

df2.to_csv('./data/newAPIs2.csv')


# df1 = df0.groupby(by = ['primary_category']).count()
# print(df1[df1.index =='Map'])
# cnt = 0
# for idx, row in df0.iterrows():
# 	t = row['tags']
# 	for t0 in t.split('###'):
# 		t0 = t0.strip()
# 		if t0 == 'Map':
# 			cnt += 1
# print(cnt)