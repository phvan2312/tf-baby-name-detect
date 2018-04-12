import sys,os
sys.path.append('..')

import pandas as pd
from Components.Data.data_component import Data_component
from shutil import rmtree
from math import ceil
import json

all_data_path = './all.csv'
k = 5 # k > 0

def print_to_file(sents, labels, poss, save_path):
    #order ['label','pos','token']
    new_labels, new_poss, new_tokens = [],[],[]

    for label, pos, sent in zip(labels, poss, sents):
        new_labels   += ['O'] + label + ['O']
        new_poss     += ['N'] + pos + ['N']
        new_tokens   += ['['] + sent + [']']

    df = pd.DataFrame({'label':new_labels,'pos':new_poss,'token':new_tokens})
    df.to_csv(save_path,index=False,columns=['label','pos','token'],encoding='utf-8')

if __name__ == '__main__':
    data_component = Data_component()

    sents, labels, poss = data_component.load_file(path_to_file=all_data_path) # both of them are type list
    n_samples = len(sents)

    n_samples_per_fold  = int(ceil(int(n_samples) / float(k)))
    print ('n_samples_per_fold: ', n_samples_per_fold)

    kfold_ids = [(s,min(s + n_samples_per_fold, n_samples)) for s in range(0,n_samples,n_samples_per_fold)]
    print (json.dumps(kfold_ids))

    for i,(s,e) in enumerate(kfold_ids):
        dir_path = './fold_%d' % i
        if os.path.isdir(dir_path): rmtree(dir_path)
        os.mkdir(dir_path)

        train_sents  = sents[0:s] + sents[e:n_samples]
        train_labels = labels[0:s] + labels[e:n_samples]
        train_poss   = poss[0:s] + poss[e:n_samples]

        test_sents, test_labels, test_poss = sents[s:e], labels[s:e], poss[s:e]

        print_to_file(sents=train_sents,labels=train_labels,poss=train_poss,save_path=os.path.join(dir_path,'train.csv'))
        print_to_file(sents=test_sents, labels=test_labels, poss=test_poss,save_path=os.path.join(dir_path,'test.csv'))


    pass