import torch
from torch import nn
from torch.nn import functional as F

class ANN_Lasso(nn.module):
#This class is run to process input data that only includes features seleced from ANN_Lasso
#The network predicts cancer stage (control, stage i, stage ii, stage iii, stage iv)
    def __init__(self, config, num_features):
        #Define the features of the network --> Number of layers will be tuned during the final training process
        self.num_features = num_features
        self.nodes_hidden_1 = config.nodes_hidden_1 #start with 512
        self.nodes_hidden_2 = config.nodes_hidden_2 #move to 128
        self.nodes_hidden_3 = config.nodes_hidden_3 #move to 32
        self.nodes_output = config.num_outputs #5
        self.learning_rate = config.learning_rate
        self.dropout_rate = config.dropout_rate
        self.batch_size = config.batch_size

        #define parameters to keep track of during training
        self.global_train_loss = 0.0
        self.global_valid_loss = 0.0
        self.best_valid_loss = 9999
        self.num_since_best_valid_loss = 0.0
        self.patience = config.patience

        super(ANN_Genes self).__init__()

        #define the layers of the model (when onl considering genetoc data)
        self.network = nn.Sequential(
            nn.Linear(self.nodes_hidden_1),
            nn.Tanh,
            nn.Dropout(self.dropout_rate)
            nn.Linear(self.nodes_hidden_2),
            nn.Tanh,
            nn.Dropout(self.dropout_rate)
            nn.Linear(self.nodes_hidden_3),
            nn.Tanh,
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.nodes_output)
        )

    def init_layers(self):
        #all layers will be initialized using xavier_normal
        nn.init.xavier_normal(network)

    def forward(self, x):
        #This function performs one round of  feature reduction
        x = self.network





class ANN_VAE.Module(nn.module):
#This class is designed to use all gene features and the first 2 layers are transferred from vae
#The network predicts cancer stage (control, stage i, stage ii, stage iii, stage iv)
    def __init__(self, config, num_features):
        #Define the features of the network
        self.num_features = num_features
        self.nodes_hidden_1 = config.nodes_hidden_1 #start with 512
        self.nodes_hidden_2 = config.nodes_hidden_2 #move to 128
        self.nodes_hidden_3 = config.nodes_hidden_3 #move to 32
        self.nodes_output = config.num_outputs #5
        self.learning_rate = config.learning_rate
        self.dropout_rate = config.dropout_rate
        self.batch_size = config.batch_size

        #define parameters to keep track of during training
        self.global_train_loss = 0.0
        self.global_valid_loss = 0.0
        self.best_valid_loss = 9999
        self.num_since_best_valid_loss = 0.0
        self.patience = config.patience

        super(ANN_Genes self).__init__()

        #define the layers of the model (when onl considering genetoc data)
        self.fc1 = nn.Linear(self.nodes_hidden_1)
        self.fc2 = nn.Linear(self.nodes_hidden_2)
        self.fc3 = nn.Linear(self.nodes_hidden_3)
        self.out = nn.Linear(self.nodes_output)
