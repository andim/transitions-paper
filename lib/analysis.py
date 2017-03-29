import numpy as np
import pandas as pd

def collapsemax(group, collapsecolumn='Lambda'):
    """Collapse to maximum along column""" 
    index = group[collapsecolumn].idxmax()
    return pd.Series({column : group[column].loc[index] for column in group.columns})

def printunique(df, nunique=1, returnnonunique=False):
    if returnnonunique:
        nonuniquecolumns = []
    for key in sorted(df.columns):
        uniques = df[key].unique()
        if len(uniques) == 1:
            print('{0}: {1}'.format(key, uniques[0]))
        elif len(uniques) <= nunique:
            print('{0}: {1}'.format(key, '; '.join(str(s) for s in uniques)))
        elif returnnonunique:
            nonuniquecolumns.append(key)
    if returnnonunique:
        return nonuniquecolumns

def intelligent_describe(df, **kwargs): 
    ukwargs = dict(returnnonunique=kwargs.get('returnnonunique', True),
                   nunique=kwargs.get('nunique', 1))
    print('-----------------------------------------------------')
    print('values of columns with no more than {0} unique entries'.format(ukwargs['nunique']))
    print('')
    nonuniquecolumns = printunique(df, **ukwargs)
    print('-----------------------------------------------------')
    print('summary statistics of other columns')
    print('')
    dkwargs = dict(include=kwargs.get('include', None),
                   exclude=kwargs.get('exclude', None),
                   percentiles=kwargs.get('percentiles', None))
    print(df[nonuniquecolumns].describe(**dkwargs))
    print('-----------------------------------------------------')

def flatten(l):
    return [item for sublist in l for item in sublist]

def loadnpz(filename):
    """Load a npz file as a pandas DataFrame"""
    f = np.load(filename)
    df = pd.DataFrame(f['data'], columns=f['columns'])
    df = df.apply(pd.to_numeric, errors='ignore')
    df.sort_values(list(df.columns), inplace=True)
    return df
