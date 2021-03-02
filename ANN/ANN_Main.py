#NEED TO UPDATE FOR HANDLING MULTIPLE LOSSES -> COLUMNS FOR TARGET 1 or 2 --> https://discuss.pytorch.org/t/a-model-with-multiple-outputs/10440

from ANN_for_Lasso import *
from ANN_utils import *
import argparse

parser = argparse.ArgumentParser()

#set up the parameters that need to be set when running the model
parser.add_argument('--model_type', 'mt', type=str, default='lasso') #options lasso or vae
parser.add_argument('--nodes_hidden_1', 'nh1', type=int, default=512)
parser.add_argument('--nodes_hidden_2', 'nh2', type=int, default=128)
parser.add_argument('--nodes_hidden_3', 'nh3', type=int, default=32)
#out is always 5
parser.add_argument('--learning_rate', 'lr', type=float, default=0.001)
parser.add_argument('--max_epochs', '-mx', type=int, default=500)
parser.add_argument('--dropout_rate', '-dr', type=float, default=0.0)
parser.add_argument('--batch_size', '-bs', type=int, default=64)
parser.add_argument('--momentum', '-bs', type=float, default = 0.9)
parser.add_argument('--patience', '-p', type=int, default = 5)
parser.add_argument('--num_folds', 'nf', type=int, default = 10)

config = parser.parse_args()

#NOTE: file name: ORIGINAL_FILE = "./data/{0}_811_{1}.csv".format(config.model_type,WHAT_OMICS) #RACHEL: vae_data is option in config - default='ember_libfm_190507', type=str

def get_data(config):
    return 1
#train and test the model using 10/fold cross validation
def run_session(config):
    #split the data into 10 folds
    #when actually run, will load csv of the data, for now, using dummy dataset
    #access the dataset
    train, valid, test, num_genes = get_data(config) #assuming  has properties .X and .y
    if config.model_type == 'lasso':

        #NOTE: may want to switch to 10-fold cross validation (train and test 10 models and take average loss w/ no validation set)
        lasso_model = ANN_Lasso(config)
        test_loss = lasso_model.fit_predict(train, valid, test)
        #print the final RMSE values (currently just mse)
        print('RMSE Loss\t{0:0.4f}\t{1:0.4f}\t{2:0.4f}'.format(math.sqrt(lasso_model.global_train_loss), math.sqrt(lasso_model.global_valid_loss), math.sqrt(test_loss)))
        print('\n')
        print("TRAINING IS COMPLETE - MODEL HAS BEEN SAVED")
    else:
        return NotImplemented




if __name__ == "__main__":
    run_session(config)
