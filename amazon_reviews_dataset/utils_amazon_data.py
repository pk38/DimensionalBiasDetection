import json
import gzip
import pandas as pd

def convertGzipToDf(location):
    ### load the meta data

    data = []
    with gzip.open(location) as f:
        for l in f:
            data.append(json.loads(l.strip()))
        
    # total length of list, this number equals total number of products
    print("total length of list, this number equals total number of products: ", len(data))

    # first row of the list
    print("First row of the list: ", data[0])

    # convert list into pandas dataframe
    df = pd.DataFrame.from_dict(data)
    print("Length of the DataFrame: ", len(df))
    return df

def removeCols(df, cols):
    return df.drop[cols]