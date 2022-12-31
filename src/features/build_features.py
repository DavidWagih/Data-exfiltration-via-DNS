
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from scipy.stats import entropy


def entropy1(labels, base=None):
  value,counts = np.unique(labels, return_counts=True)
  return entropy(counts, base=base)


def generate_attribute(url):
    df_record={}
    df_generated =pd.DataFrame(columns=['FQDN_count','subdomain_length','upper','lower','numeric','entropy','special','labels','labels_max','labels_average','longest_word','sld','len','subdomain'])
    
    df_record["FQDN_count"]=len(url)
    df_record["subdomain_length"]=(len(url.split('.')[-3]) if len(url.split('.'))>2 else 0)
    df_record["upper"]=len(list(filter(lambda x: x.isupper()==True, list(url))))
    df_record["lower"]=len(list(filter(lambda x: x.islower()==True, list(url))))
    df_record["numeric"]=len(list(filter(lambda x: x.isnumeric()==True, list(url))))
    df_record["entropy"]=entropy1(list(url), base=2)
    df_record["special"]=sum([not c.isalnum() for c in url])
    df_record["labels"]=len(url.split('.'))
    df_record["labels_max"]=max([len(lbl) for lbl in url.split('.')])
    df_record["labels_average"]=len(url)/len(url.split('.'))
    df_record["longest_word"]=(len(url.split('.')[-2]) if len(url.split('.'))>1 else 0)
    df_record["sld"]=(url.split('.')[-2] if len(url.split('.'))>1 else 0)
    df_record["len"]=(len(url.split('.')[-3]) if len(url.split('.'))>2 else 0) + (len(url.split('.')[-2]) if len(url.split('.'))>1 else 0)
    df_record["subdomain"]=1 if len(url.split('.'))>2 else 0

    df_generated=df_generated.append(df_record,ignore_index=True)
    return df_generated

def reformat_record(record):
    ret_record_columns=['FQDN_count','subdomain_length','upper','lower','numeric','entropy','special','labels','labels_max','labels_average','len','subdomain'] # no object columns
    return record[ret_record_columns]