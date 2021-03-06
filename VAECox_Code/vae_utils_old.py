import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from sklearn.metrics import explained_variance_score, r2_score
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.metrics import mean_absolute_error, median_absolute_error
from lifelines.utils import concordance_index as cindex
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import *
import pickle
import os
import multiprocessing as mp
import sys
import random
np.set_printoptions(threshold=sys.maxsize)

#RACHEL: Define the dictionary of cancer TCGA data (currently dummy data). Defines the dataset names according to TCGA codes
cancer_list_dict = {
    'ching': ['BLCA', 'BRCA', 'HNSC', 'KIRC', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'STAD'],
    'wang': ['ACC', 'BLCA', 'BRCA', 'CESC', 'UVM', 'CHOL', 'ESCA', 'HNSC', 'KIRC', 'KIRP',
             'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'PAAD', 'SARC', 'SKCM', 'STAD', 'UCEC', 'UCS'],
    'all': ['ACC', 'BLCA', 'BRCA', 'CESC', 'UVM', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM',
            'HNSC', 'KICH', 'KIPAN', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC',
            'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 'STAD', 'STES',
            'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS'],
    'test': ['CESC','CHOL','COAD']
}

#RACHEL: Pull out the values for the requested dataset
def cancer_list(key):
    return cancer_list_dict[key]

#RACHEL: Access the dataset
def get_dataset_811(config):
    #RACHEL:initialize data dictionary for training, validation, testing, and coo
    data_dict = {
        'train': dict(),
        'valid': dict(),
        'test': dict(),
        'coo': dict()
    }

    #RACHEL: PUT THIS IN A LOOP SO CAN LOOP THROUGH ALL OF THE CANCERS (ADD THE NAME TO THE FRONT OF BOTH FILES)
    #NEED TO CHANGE THE FOR LOOP TO APPENDING TO THE LOOP INSTEAD RECREATING THE INDEX.

    file_names = cancer_list('test')

    # for cancer_name in cancer_list



    WHAT_OMICS = "_".join(config.omic_list) #RACHEL: omic_list is option in config
    #RACHEL:original file contains left column of id, top row gene name< following rows normalized gene data
    # ORIGINAL_FILE = "./data/{0}_811_{1}.tsv".format(config.vae_data,WHAT_OMICS) #RACHEL: vae_data is option in config - default='ember_libfm_190507', type=str

    ORIGINAL_FILE = "./data/{0}_811_{1}.csv".format(config.vae_data,WHAT_OMICS) #RACHEL: vae_data is option in config - default='ember_libfm_190507', type=str
    #RACHEL: this file contains left column patient id, top row genes< following rows whether or not gene is included
    # MASKING_FILE = "./data/{0}_{1}_binary.csv".format(config.vae_data,WHAT_OMICS)

    MASKING_FILE = "./data/{0}_{1}_binary.csv".format(config.vae_data,WHAT_OMICS)

    #RACHEL: develop the pickle file oath name based on run_time system information
    PICKLE_PATH = "./data/{0}_811_{1}_{2}_{3}_{4}_{5}.pickle".format(config.vae_data, config.gcn_mode, config.feature_scaling, config.feature_selection, config.sub_graph, WHAT_OMICS)
        ##RACHEL:**may just eb able to change to match pickle name
    #RACHEL: if the pickle file has not been made yet, make one
    if not os.path.isfile(PICKLE_PATH):
        print("Making new pickle file...")
        # Missing Value Handling
        #RACHEL: read the original file*****
        df = pd.read_csv(ORIGINAL_FILE, sep=",", header=0, index_col=0) #RACHEL: edited to work with csv (Used to be "\t")
        #RACHEL: read the masking file********
        mf = pd.read_csv(MASKING_FILE, sep=",", header=0, index_col=0)
        #RACHEL: re-index labels so match order
        mf = mf.reindex(df.index)
        print("printing index**********************************:")
        #RACHEL:this prints the patient ID information (row names) **comment out when run
        print(df.index)
        print("printing columns***************************")
        print(df.columns)
        print(mf.index)
        # mf = mf.replace(0,np.nan)
        # mf = mf.dropna(how='all',axis=0)
        # mf = mf.dropna(how='all',axis=1)

        # df = df.dropna(subset=['Cli@Days2Death', 'Cli@Days2FollowUp'], how='all')
        df.fillna(0.0, axis=1, inplace=True)

        # Dataset Split Train, Valid and Test
        temp =df['Fold@811']#RACHEL" Before 'Fold@811'
        # print("printing Fold@811")
        # print(temp)
        df_train = df.loc[df['Fold@811'] == 0]
        print("training index:")
        print(df_train.index)
        print("training columns:")
        print(df_train.columns)
        df_valid = df.loc[df['Fold@811'] == 1]
        print("valid index:")
        print(df_valid.index)
        df_test = df.loc[df['Fold@811'] == 2]
        print("test index:")
        print(df_test.index)

        # print(df_train.columns)
        #RACHEL: get the data and labels???
        for omic in config.omic_list:
            print("omic:*************************************")
            print(omic)
            # print(type(data_dict['train'][omic]))
            #RACHEL: pull out gene information (mRNA expression)
            data_dict['train'][omic] = df_train[[x for x in df_train.columns if omic in x]] #RACHEL: eliminated get_values()
            data_dict['valid'][omic] = df_valid[[x for x in df_valid.columns if omic in x]]
            data_dict['test'][omic] = df_test[[x for x in df_test.columns if omic in x]]
            #RACHEL:pull out mask (whether or not to include gene)
            data_dict['train'][omic + '_mask'] = mf.loc[df_train.index]
            data_dict['valid'][omic + '_mask'] = mf.loc[df_valid.index]
            data_dict['test'][omic + '_mask'] = mf.loc[df_test.index]

        # Dataset Feature Extraction
        if config.feature_selection is not None:
            data_dict = _feature_selection(config, data_dict)

        # Dataset 'Numpification'
        for omic in config.omic_list:
            data_dict['train'][omic] = np.array(data_dict['train'][omic].values).astype('float64')
            data_dict['valid'][omic] = np.array(data_dict['valid'][omic].values).astype('float64')
            data_dict['test'][omic] = np.array(data_dict['test'][omic].values).astype('float64')
            data_dict['train'][omic + '_mask'] = np.array(data_dict['train'][omic + '_mask'].values).astype('float64')
            data_dict['valid'][omic + '_mask'] = np.array(data_dict['valid'][omic + '_mask'].values).astype('float64')
            data_dict['test'][omic + '_mask'] = np.array(data_dict['test'][omic + '_mask'].values).astype('float64')

        # Dataset Feature Scaling - RACHEL: does nothing if set to NONE
        data_dict = _feature_scaling(config, data_dict)

        with open(PICKLE_PATH, "wb") as handle:
            #RACHEL: write to file level
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(PICKLE_PATH, "rb") as handle:
        #RACHEL: read from file
        data_dict = pickle.load(handle)

    #RACHEL: for each dataset print number of samples in train, validate, and test
    for o in config.omic_list:
        print('Train', o, data_dict['train'][o].shape)
        print('Valid', o, data_dict['valid'][o].shape)
        print('Test', o, data_dict['test'][o].shape)
        # _data_stats(data_dict, o)

    return data_dict

##RACHEL: perform feature selection????
def _feature_selection(config, data_dict):
    if 'None' not in config.feature_selection:
        selected_genes = []
        for omic in config.omic_list:
            old_genes = data_dict['train'][omic].columns.get_values() #RACHEL*****MAY NEED TO DELETE GET_VALUES******
            temp_genes = [g.split("|")[0] for g in old_genes]
            print(temp_genes)
            with open('./data/{}_genes.txt'.format(config.feature_selection), "r") as fr:
                for gene in fr.readlines():
                    selected_genes.append(omic + gene.split("\n")[0])
                print(selected_genes)
                with open("./minji_vae_genes.pickle", "wb") as handle:
                    pickle.dump(selected_genes, handle, protocol=pickle.HIGHEST_PROTOCOL)
                for mode in ['train', 'valid', 'test']:
                    data_dict[mode][omic].columns = temp_genes
                    data_dict[mode][omic] = data_dict[mode][omic][selected_genes]
                    data_dict[mode][omic + '_mask'].columns = temp_genes
                    data_dict[mode][omic + '_mask'] = data_dict[mode][omic + '_mask'][selected_genes]
    return data_dict

#RACHEL:perform feature scaling ***I MAY NOT WANT TO INCLUDE THIS*********
def _feature_scaling(config, data_dict):
    if 'None' not in config.feature_scaling:
        for key in data_dict['train'].keys():
            if 'Cli@' not in key and '_mask' not in key:
                scaler = StandardScaler()
                if config.feature_scaling == 'z':
                    scaler = StandardScaler()
                elif config.feature_scaling == 'minmax':
                    scaler = MinMaxScaler()
                else:
                    scaler = None
                data_dict['train'][key] = scaler.fit_transform(data_dict['train'][key])
                try:
                    data_dict['valid'][key] = scaler.transform(data_dict['valid'][key])
                except Exception as e:
                    pass
                data_dict['test'][key] = scaler.transform(data_dict['test'][key])
            assert not np.isnan(data_dict['train'][key]).any()
            assert not np.isnan(data_dict['test'][key]).any()
    return data_dict


######################################################################################################################

def get_optimizer(optimizer_name):
    opti_dict = {
        'Adam': optim.Adam,
        'SGD': optim.SGD
    }
    return opti_dict[optimizer_name]

class Torch_Dataset:
    def __init__(self, X, y, c, m):
        self.X, self.y, self.c, self.m = X, y, c, m
        self.coo = None
        self.num_samples = self.X.size()[0]

    def split_train_valid(self, kfold_indices, cancer_list):
        train_x, train_y, train_c, train_m = dict(), dict(), dict(), dict()
        valid_x, valid_y, valid_c, valid_m = dict(), dict(), dict(), dict()
        for c in cancer_list:
            train_index, valid_index = kfold_indices[c]
            train_x[c] = self.X[c][train_index]
            train_y[c] = self.y[c][train_index]
            train_c[c] = self.c[c][train_index]
            train_m[c] = self.m[c][train_index]
            valid_x[c] = self.X[c][valid_index]
            valid_y[c] = self.y[c][valid_index]
            valid_c[c] = self.c[c][valid_index]
            valid_m[c] = self.m[c][valid_index]
        train_dataset = Torch_Dataset(train_x, train_y, train_c, train_m)
        valid_dataset = Torch_Dataset(valid_x, valid_y, valid_c, valid_m)
        return train_dataset, valid_dataset

def torchify_vaeserin(config):
    #RACHEL: get the type of genetic data (mRNA data was used in paper and will be used in this project)
    omic_type = config.omic_list[0]
    device = torch.device(config.device_type)
    dataset = get_dataset_811(config)

    num_cols = dataset['train'][omic_type].shape[1]
    x = torch.tensor(dataset['train'][omic_type], dtype=torch.float32).to(device)
    m = torch.tensor(dataset['train'][omic_type + '_mask'], dtype=torch.float32).to(device)
    xx = torch.tensor(dataset['valid'][omic_type], dtype=torch.float32).to(device)
    mm = torch.tensor(dataset['valid'][omic_type + '_mask'], dtype=torch.float32).to(device)
    xxx = torch.tensor(dataset['test'][omic_type], dtype=torch.float32).to(device)
    mmm = torch.tensor(dataset['test'][omic_type + '_mask'], dtype=torch.float32).to(device)

    train_dataset = Torch_Dataset(x, None, None, m)
    valid_dataset = Torch_Dataset(xx, None, None, mm)
    test_dataset = Torch_Dataset(xxx, None, None, mmm)

    return train_dataset, valid_dataset, test_dataset, num_cols

def get_mse_loss_masked(recon_x, x, m):
    square = (recon_x - x)**2
    sum_squares = torch.sum(square, dim=1) / torch.sum(m, dim=1)
    mse = torch.mean(sum_squares)
    return mse

def get_mse_kld_loss_masked(recon_x, x, mu, logvar, m):
    MSE = get_mse_loss_masked(recon_x, x, m)
    # KLD = 0.0
    # KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / torch.sum(m, dim=1))
    KLD = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum() / (mu.size(0) * mu.size(1))
    # KLD = nn.KLDivLoss(recon_x, x, reduction='mean')
    return MSE + KLD

def get_mse_loss(recon_x, x):
    MSE = F.mse_loss(recon_x, x)
    return MSE

def get_mse_kld_loss(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x)
    # KLD = 0.0
    KLD = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum() / (mu.size(0) * mu.size(1))
    return MSE + KLD

def get_evaluations(recon_x, x):
    pred, true = recon_x.cpu().data.numpy(), x.cpu().data.numpy()
    evs = explained_variance_score(pred, true)
    r2 = r2_score(pred, true)
    mse = mean_squared_error(pred, true)
    mae = mean_absolute_error(pred, true)
    float_list = [evs, r2, mse, mae]
    return ["%.3f"%item for item in float_list]

#RACHEL: NOT USED FOR THIS PROJECT (survival prediction)**********************************
def get_cindex(y, y_pred, c):
    try:
        return cindex(y, y_pred, c)
    except Exception as e:
        print(e)
        print(y)
        print(y_pred)
        print(c)
        return 0.0
