import numpy as np
import pandas as pd
import math
from tqdm import tqdm


def get_matrix():
    ### read data -- mashups
    df0 = pd.read_csv('./data/Mashups.csv')
    df1 = df0[['MashupName', 'MemberAPIs']]

    apidict = dict()
    mashdegree = dict()
    for index, row in df1.iterrows():
        li = row['MemberAPIs']
        li = li.strip()
        val = row['MashupName']
        cnt = 0
        for k in li.split('@@@'):
            ### get api to mashup dict
            k = k.strip()
            if k not in apidict:
                apidict[k] = [val]
            else:
                apidict[k].append(val)
            ### get mashup degree
            cnt += 1
        mashdegree[val] = cnt

    ### read data -- apis

    df2 = pd.read_csv('./data/apiset.csv')
    df3 = df2[['APIName', 'primary_category']]

    do2api = dict()
    api2do = dict()

    for ind, row in df3.iterrows():
        api = row['APIName']
        if api in apidict:
            domain = row['primary_category']
            api2do[api] = domain
            if domain in do2api:
                do2api[domain].append(api)
            else:
                do2api[domain] = [api]

    return apidict, mashdegree, do2api, api2do


def similar1(a, b, apidict, mashdegree):
    masha = apidict[a]
    mashb = apidict[b]
    inter = list(set(masha).intersection(set(mashb)))
    if len(inter):
        de_sum = 0.0
        for mash in inter:
            de_sum += 1 / mashdegree[mash]
        call_sqrt = math.sqrt(len(masha) * len(mashb))
        sim = de_sum / call_sqrt
        return sim
    return 0


def similar2(a, b, d2a, apinet):
    apisA = d2a[a]
    apisB = d2a[b]


    cerFromA = lambda row: row['from'] in apisA
    cerFromB = lambda row: row['from'] in apisB
    cerToA = lambda row: row['to'] in apisA
    cerToB = lambda row: row['to'] in apisB
    fromA = apinet[apinet.apply(cerFromA, axis=1)]
    fAtA = fromA[fromA.apply(cerToA, axis=1)]
    fAtB = fromA[fromA.apply(cerToB, axis=1)]
    fromB = apinet[apinet.apply(cerFromB, axis=1)]
    fBtB = fromB[fromB.apply(cerToB, axis=1)]



    numerator = sum(fAtB['similar'])

    if numerator == 0:
        return 0
    else:
        denominator = sum(fromA['similar']) + sum(fromB['similar']) - sum(fAtA['similar']) - sum(fAtB['similar']) - sum(
            fBtB['similar'])
        return numerator / denominator

# def getDegree(apinet, d2a, do):
#     apisA = d2a[do]
#     cerFromA = lambda row: row['from'] in apisA
#     cerToA = lambda row: row['to'] in apisA
#     fromA = apinet[apinet.apply(cerFromA, axis=1)]
#     fAtA = fromA[fromA.apply(cerToA, axis=1)]



def save_graph():
    ad, md, d2a, a2d = get_matrix()
    apis = ad.keys()


    print(len(ad))
    print(len(md))
    print(len(d2a))
    print(len(a2d))

    exit()




    ### get apinet
    from_array = list()
    to_array = list()
    sim_array = list()

    for apiA in apis:
        for apiB in apis:
            sim = similar1(apiA, apiB, ad, md)
            if sim:
                from_array.append(apiA)
                to_array.append(apiB)
                sim_array.append(sim)

    apinet = pd.DataFrame({'from': from_array, 'to': to_array, 'similar': sim_array})
    apinet.to_csv('data/apinet.csv', index=0)

    ### get domain net

    domains = d2a.keys()

    from_array = list()
    to_array = list()
    sim_array = list()

    domain_degree = {}

    # for do in domains:
    #     domain_degree[do] = getDegree(apinet,d2a, do)

    cnt = 0
    print(len(domains))
    for doA in tqdm(domains):
        for doB in domains:
            if doA != doB:
                from_array.append(doA)
                to_array.append(doB)
                sim = similar2(doA, doB, d2a, apinet)
                sim_array.append(sim)

    domainnet = pd.DataFrame({'from': from_array, 'to': to_array, 'similar': sim_array})
    domainnet.to_csv('data/domainnet.csv', index=0)


def main():
    save_graph()
    # get_matrix()


if __name__ == '__main__':
    main()
