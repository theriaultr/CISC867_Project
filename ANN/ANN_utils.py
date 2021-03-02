import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
#MAIN CODE IS FROM VAE - just edited to match purposes of this

def develop_dataset(config):
    #RACHEL:initialize data dictionary for training, validation, testing, and coo
    data_dict = {
        'train': dict(),
        'valid': dict(),
        'test': dict(),
    }
    WHAT_OMICS = "_".join(config.omic_list) #RACHEL: omic_list is option in config

    #****CHANGE THIS SO ONLY USING ORIGINAL FILE AND LABEL IS GIVEN AS A COLUMN

    #original file contains left column of id, top row gene name< following rows normalized gene data
    ORIGINAL_FILE = "./data/{0}_811_{1}.csv".format(config.model_type,WHAT_OMICS)

    #develop the pickle file path name (Ex. lasso_811_mrna_formatted.pickle)
    PICKLE_PATH = "./data/{0}_811_{1}_labels.pickle".format(config.model_type, WHAT_OMICS)
        ##RACHEL:**may just eb able to change to match pickle name
    #RACHEL: if the pickle file has not been made yet, make one
    if not os.path.isfile(PICKLE_PATH):
        print("Making new pickle file...")
        # Missing Value Handling
        #read the original file (CSV)****
        df = pd.read_csv(ORIGINAL_FILE, sep=",", header=0, index_col=0)

        #RACHEL:this prints the patient ID information (row names) **comment out when run
        print(df.index)
        print("printing columns***************************")
        print(df.columns)
        # mf = mf.replace(0,np.nan)
        # mf = mf.dropna(how='all',axis=0)
        # mf = mf.dropna(how='all',axis=1)

        # df = df.dropna(subset=['Cli@Days2Death', 'Cli@Days2FollowUp'], how='all')
        df.fillna(0.0, axis=1, inplace=True)

        # Dataset Split Train, Valid and Test
        temp =df['Fold@811']
        df_train = df.loc[df['Fold@811'] == 0] #pull out rows where
        df_valid = df.loc[df['Fold@811'] == 1]
        df_test = df.loc[df['Fold@811'] == 2]

        # print(df_train.columns)
        #RACHEL: get the data and labels???
        for omic in config.omic_list:
            #RACHEL: pull out gene information (mRNA expression)
            data_dict['train'][omic] = df_train[[x for x in df_train.columns if omic in x]] #RACHEL: eliminated get_values()
            data_dict['valid'][omic] = df_valid[[x for x in df_valid.columns if omic in x]]
            data_dict['test'][omic] = df_test[[x for x in df_test.columns if omic in x]]
            #RACHEL:pull out mask (whether or not to include gene)
            data_dict['train'][omic + '_label'] = df_train['Stage_Label']
            data_dict['valid'][omic + '_label'] = df_valid['Stage_Label']
            data_dict['test'][omic + '_label'] = df_test['Stage_Label']

        # Dataset 'Numpification'
        for omic in config.omic_list:
            data_dict['train'][omic] = np.array(data_dict['train'][omic].values).astype('float64')
            data_dict['valid'][omic] = np.array(data_dict['valid'][omic].values).astype('float64')
            data_dict['test'][omic] = np.array(data_dict['test'][omic].values).astype('float64')
            data_dict['train'][omic + '_label'] = np.array(data_dict['train'][omic + '_label'].values).astype('float64')
            data_dict['valid'][omic + '_label'] = np.array(data_dict['valid'][omic + '_label'].values).astype('float64')
            data_dict['test'][omic + '_label'] = np.array(data_dict['test'][omic + '_mask'].values).astype('float64')

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

class Torch_Dataset:
    def __init__(self, X, y):
        self.X, self.y = X, int(y)
        self.num_samples = self.X.size()[0]

    #UNCOMMENT AND USE IF DECIDE TO TRY K-FOLD VALIDATION
    # def split_train_valid(self, kfold_indices, cancer_list):
    #     train_x, train_y, train_c, train_m = dict(), dict(), dict(), dict()
    #     valid_x, valid_y, valid_c, valid_m = dict(), dict(), dict(), dict()
    #     for c in cancer_list:
    #         train_index, valid_index = kfold_indices[c]
    #         train_x[c] = self.X[c][train_index]
    #         train_y[c] = self.y[c][train_index]
    #         train_c[c] = self.c[c][train_index]
    #         train_m[c] = self.m[c][train_index]
    #         valid_x[c] = self.X[c][valid_index]
    #         valid_y[c] = self.y[c][valid_index]
    #         valid_c[c] = self.c[c][valid_index]
    #         valid_m[c] = self.m[c][valid_index]
    #     #RACHEL: WHAT IS X,Y,C,M????? --> MAY NEED TO RE-WRITE CODE ANYWAYS DUE TO DIFFERENCE IN DATA
    #     train_dataset = Torch_Dataset(train_x, train_y, train_c, train_m)
    #     valid_dataset = Torch_Dataset(valid_x, valid_y, valid_c, valid_m)
    #     return train_dataset, valid_dataset

def get_data(config):

    #omic type will be mrna for this project
    omic_type = config.omic_list[0]
    # device = torch.device(config.device_type)
    dataset = develop_dataset(config)

    #determine the number of features (genes)
    num_cols = dataset['train'][omic_type].shape[1]

    #define x/y for train, validation and test data
    x_train = torch.tensor(dataset['train'][omic_type], dtype=torch.float32)
    y_train = torch.tensor(dataset['train'][omic_type + '_label'], dtype=torch.float32)
    x_valid = torch.tensor(dataset['valid'][omic_type], dtype=torch.float32).to(device)
    y_valid = torch.tensor(dataset['valid'][omic_type + '_label'], dtype=torch.float32)
    x_test = torch.tensor(dataset['test'][omic_type], dtype=torch.float32).to(device)
    y_test = torch.tensor(dataset['test'][omic_type + '_label'], dtype=torch.float32)

    #combine the x and y data into a class Torch_Dataset for processing with the torch model
    train_dataset = Torch_Dataset(x_train, y_train) #the y data is class 0-4
    valid_dataset = Torch_Dataset(x_valid, y_valid) #the y data is class 0-4
    test_dataset = Torch_Dataset(x_test, y_test) #the y data is class 0-4

    return train_dataset, valid_dataset, test_dataset, num_cols
