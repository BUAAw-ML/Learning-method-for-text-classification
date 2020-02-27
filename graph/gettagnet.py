import numpy as np
import pandas as pd
import math
from tqdm import tqdm

def getmat():
    df2 = pd.read_csv('./data/apiset.csv')
    df3 = df2[['APIName', 'tags','primary_category','descr']]

    tag2api = dict()
    api2tag = dict()

    apilist = []
    ttlist = []
    simlist = []

    for ind, row in df3.iterrows():
        api = row['APIName']
        tags = row['tags']
        pri = row['primary_category'].strip()
        des  = row['descr'].strip()
        taglist = []
        for t in tags.split('###'):
            taglist.append(t.strip())
        api2tag[api] = taglist
        
        simset = []
        for tag in taglist:
            if tag in tag2api:
                tag2api[tag].append(api)
            else:
                tag2api[tag] = [api]


            apilist.append(api)
            ttlist.append(tag)
            sim = 0.45 * api.count(tag) + 0.35 * pri.count(tag) + 0.2 * des.count(tag) + 1
            simset.append(sim)



        for sim in simset:
            simi = sim / sum(simset)
            simlist.append(simi)


    dependset = pd.DataFrame({'tag': ttlist, 'api': apilist, 'dep': simlist})

    # print(len(tag2api))
    # print(len(api2tag))
    # print(dependset)

    return tag2api, api2tag, dependset


def graph(t2a,a2t,dep,apinet):
    
    sourcelist = []
    targetlist = []
    weightlist = []

    apiset = apinet['from'].drop_duplicates().values


    k = list(t2a.keys())
    # print(len(k))
    # exit()

    cnt = 0
    for i in range(473):
        t1 = k[i]
        api1 = t2a[t1]
        ts1 = dep[dep['tag'] == t1]
        api1 =  list(set(api1).intersection(set(apiset)))

        for j in range(i+1 ,473):
            t2 = k[j]
            ts2 = dep[dep['tag'] == t2]
            api2 = t2a[t2]
            api2 =  list(set(api2).intersection(set(apiset)))

            if t1 != t2:
                sim  = 0
                for a1 in api1:
                    asim = apinet[apinet['from'] == a1]
                    for a2 in api2:
                        a2a = asim[asim['to'] == a2]
                        if (len(a2a) != 0):
                            ag = a2a['similar'].values[0]   
                            tst1 = ts1[ts1['api'] == a1]
                            tsv1 = tst1['dep'].values[0]
                            tst2 = ts2[ts2['api'] == a2]
                            tsv2 = tst2['dep'].values[0]
                            sim += ag * tsv1 * tsv2
                        else:
                            sim += 0
                # print(sim)
                sourcelist.append(t1)
                targetlist.append(t2)
                weightlist.append(sim)



    tagnet = pd.DataFrame({'source': sourcelist, 'target': targetlist, 'weight': weightlist})
    tagnet.to_csv('./data/tagnet.csv',index=0)

def main():
    t2a, a2t, dep =getmat()
    apinet = pd.read_csv('./data/apinet.csv')
    graph(t2a,a2t,dep,apinet)


if __name__ == '__main__':
    main()