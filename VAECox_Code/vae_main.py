from vae_models import *
import argparse
import logging
import sys
import statistics

parser = argparse.ArgumentParser()

#Rachel: Add argument options for VAE
parser.add_argument('--cancer_list', '-cl', type=str, default='coxnnet')
parser.add_argument('--omic_list', '-ol', nargs='+', type=str)
parser.add_argument('--missing_impute', '-mi', type=str, default='mean')
parser.add_argument('--exclude_impute', '-xi', default=False, action='store_true')
parser.add_argument('--feature_scaling', '-fc', type=str, default='None')
parser.add_argument('--feature_selection', '-fs', type=str, default='None')
parser.add_argument('--gcn_mode', '-gcn', default=False, action='store_true')
parser.add_argument('--gcn_func', '-gcf', default='None', type=str)
parser.add_argument('--ipcw_mode', '-ipcw', default=False, action='store_true')
parser.add_argument('--device_type', '-dv', type=str, default='cuda')
parser.add_argument('--cuda_device', '-cd', type=str, default='0')
parser.add_argument('--hidden_nodes', '-hn', type=int, default=2048)
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
parser.add_argument('--weight_sparsity', '-ws', type=float, default=1e-6)
parser.add_argument('--weight_decay', '-wd', type=float, default=1e-6)
parser.add_argument('--max_epochs', '-mx', type=int, default=500)
parser.add_argument('--model_optimizer', '-mo', type=str, default='SGD')
parser.add_argument('--dropout_rate', '-dr', type=float, default=0.0)
parser.add_argument('--acti_func', '-af', default="ReLU", type=str)
# Graph Convolution
parser.add_argument('--sub_graph', '-sg', default=0, type=int)
parser.add_argument('--batch_size', '-bs', default=64, type=int)
parser.add_argument('--pool_func', '-pf', default='Single', type=str)
parser.add_argument('--topk_pooling', '-tp', default=0.5, type=float)
# Cox Regression
parser.add_argument('--multi_task', '-mu', default=False, action='store_true')
parser.add_argument('--mt_regularization', '-mr', default='None', type=str)
parser.add_argument('--num_clusters', '-nc', default=8, type=int)
parser.add_argument('--augment_autoencoder', '-aug', default='None', type=str)
parser.add_argument('--deseq2', '-deseq', default=False, action='store_true')
# parser.add_argument('--file_version', '-fv', type=str, default='15%')
parser.add_argument('--hp_search', '-hs', default=False, action='store_true')
parser.add_argument('--vae_data', '-vd', default='ember_libfm_190507', type=str)
parser.add_argument('--test_mode', '-tm', default=False, action='store_true')
parser.add_argument('--model_struct', '-ms', default='basic', type=str)
parser.add_argument('--model_type', '-mt', default='coxrgmt', type=str)
parser.add_argument('--save_mode', '-sm', default=False, action='store_true')
parser.add_argument('--checkpoint_dir', '-cp', default='./results/', type=str) #RACHEL:Created this file
parser.add_argument('--session_name', '-sn', default='test', type=str)
parser.add_argument('--pickle_save', '-ps', default=False, action='store_true')
config = parser.parse_args()

#RACHEL: set the environment for the code to be run in
os.environ["CUDA_VISIBLE_DEVICES"]=config.cuda_device

#RACHEL:object indicating where tensor will be stored
device = torch.device(config.device_type)

#RACHEL: create an instance of the logger so can perform logging of crictical events throughout code
LOGGER = logging.getLogger()

#RACHEL: initialize the logger
def init_logging(config):
    #RACHEL: specify lowest severity log message as INFO
    LOGGER.setLevel(logging.INFO)
    #RACHEL: specify the format the logger will display results in
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    #RACHEL: send messages to streams (file-like)
    console = logging.StreamHandler()
    #RACHEL: set the format to that defined earlier
    console.setFormatter(fmt)
    #RACHEL: set the handler object to console so will follow defined formats
    LOGGER.addHandler(console)
    # For logfile writing
    #RACHEL:save with file name containing the system arguments (command line calls)
    logfile = logging.FileHandler(
        config.checkpoint_dir + 'logs/' + ' '.join(sys.argv) + '.txt', 'w')
    #RACHEL: use format defined
    logfile.setFormatter(fmt)
    #RACHEL: add another handler object for the log file with the name given
    LOGGER.addHandler(logfile)

#RACHEL: set the seed (based on time) and log information of seed and process id
def init_seed(seed=None):
    if seed is None:
        seed = int(round(time.time() * 1000)) % 10000
    LOGGER.info("Using seed={}, pid={}".format(seed, os.getpid()))
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

#RACHEL: run a session of the VAE
def run_session(config):
    result_list = []
    #RACHEL: Running a session with the VAE
    if config.model_type == 'vae':
        #RACHEL: Retreive and split the data - function torchify_vaeserin in vae_utils
        train, valid, test, num_cols = torchify_vaeserin(config)
        #RACHEL: Create VAE class --> in vae_models.py
        maven = VAE(config=config, logger=LOGGER, num_features=num_cols)
        #RACHEL: initialize the layers from VAE class function
        maven.init_layers()
        #RACHEL: fit the VAE to the training and validation data, predict the testing data and report the test_loss
        test_loss = maven.fit_predict(train, valid, test)
        #RACHEL: Print the results from the logger (train, validation, and test RMSE)
        LOGGER.info('==============Final Results==============')
        LOGGER.info('Metric\tTraining\tValidation\tTesting')
        LOGGER.info('RMSE Loss\t{0:0.4f}\t{1:0.4f}\t{2:0.4f}'.format(math.sqrt(maven.global_train_loss), math.sqrt(maven.global_valid_loss), math.sqrt(test_loss)))
        print('\n')

    #RACHEL: running a session with just the AE
    elif config.model_type == 'ae':
        #RACHEL: Retreive and split the data - function torchify_vaeserin in vae_utils
        train, valid, test, num_cols = torchify_vaeserin(config)
        #RACHEL: Create AE class --> in vae_models.py
        maven = AE(config=config, logger=LOGGER, num_features=num_cols)
        #RACHEL: initialize the layers from AE class function
        maven.init_layers()
        #RACHEL: fit the AE to the training and validation data, predict the testing data and report the test_loss
        test_loss = maven.fit_predict(train, valid, test)
        #RACHEL: Print the results from the logger (train, validation, and test RMSE)
        LOGGER.info('==============Final Results==============')
        LOGGER.info('Metric\tTraining\tValidation\tTesting')
        LOGGER.info('RMSE Loss\t{0:0.4f}\t{1:0.4f}\t{2:0.4f}'.format(math.sqrt(maven.global_train_loss), math.sqrt(maven.global_valid_loss), math.sqrt(test_loss)))
        print('\n')

    #RACHEL: running a session with DAE
    elif config.model_type == 'dae':
        #RACHEL: Retreive and split the data - function torchify_vaeserin in vae_utils
        train, valid, test, num_cols = torchify_vaeserin(config)
        #RACHEL: Create DAE class --> in vae_models.py
        maven = DAE(config=config, logger=LOGGER, num_features=num_cols)
        #RACHEL: initialize the layers from DAE class function
        maven.init_layers()
        #RACHEL: fit the DAE to the training and validation data, predict the testing data and report the test_loss
        test_loss = maven.fit_predict(train, valid, test)
        #RACHEL: Print the results from the logger (train, validation, and test RMSE)
        LOGGER.info('==============Final Results==============')
        LOGGER.info('Metric\tTraining\tValidation\tTesting')
        LOGGER.info('RMSE Loss\t{0:0.4f}\t{1:0.4f}\t{2:0.4f}'.format(math.sqrt(maven.global_train_loss), math.sqrt(maven.global_valid_loss), math.sqrt(test_loss)))
        print('\n')
    #RACHEL: requested model type has not been implemented in this code (nothing run)
    else:
        return NotImplemented
#RACHEL: initialize the logger, run the session with requested model type
if __name__ == "__main__":
    init_logging(config)
    LOGGER.info('COMMAND: {}'.format(' '.join(sys.argv)))
    #train and test the model and display the results (tracked with Logger)
    run_session(config)
