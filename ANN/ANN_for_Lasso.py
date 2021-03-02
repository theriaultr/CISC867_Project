import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

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
        self.momentum = config.momentum
        self.max_epochs = config.max_epochs

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
        #This funtion defines how forward propogration occurs throughout the network
        z = self.network(x)
        return z

    #Fit method is same as VAE except added in patiene for training and edited some portions
    def fit(self, trainset, validset=None):
        #MAY NEED TO EDIT WHAT TRAINSET IS AND THE FORMAT IT IS IN...

        train_RMSE = []
		valid_RMSE = []
		self.init_layers()
        print(self)
        optimizer = get_optimizer()(self.parameters(), lr=self.learning_rate, momentum = self.momentum)
        batch_num = int(trainset.num_samples / self.batch_size) if self.batch_size != 0 else 1
		batch_val = int(validset.num_samples / self.batch_size) if self.batch_size != 0 else 1

        #this variable is to keep track of how long it has been since the model
        #has improved (stopped after haven't improved in patience number of epochs)
        no_improvement = 0

		#RACHEL:set each epoch
		t = trange(self.max_epochs + 1, desc='Training...')
		for epoch in t:

            #check if the model has not improved in patience number of times
            if no_improvement == self.patience:
                break

			self.batch_flag = False
			#put the model in training mode
			self.train()

			#if using batches
			if self.batch_size != 0:
				# BATCH-WISE TRAINING PHASE
				self.global_train_loss, self.global_valid_loss = 0.0, 0.0
				#train one batch at a time
				for b in range(batch_num):
					#calculate indices for the batch
					i, j = (self.batch_size * b) % trainset.num_samples, (self.batch_size * (b+1)) % trainset.num_samples
					#make sure the model is in training mode
					self.train()
					#for the training data, calculate the loss
					loss = self(trainset.X[i:j,:])
                    #calculate the total
					self.global_train_loss += loss.item() * self.batch_size
                    #set the gradients to 0 so can backpropogate
					optimizer.zero_grad()
					#back propogate to comput the gradients of the model
					loss.backward()
					#update the model based on gradients
					optimizer.step()
                #end of batch
                # #Keep this section???********************************************
                # #determine the number of samples in each batch
				# lb = trainset.num_samples % self.batch_size
                # #calculate loss for all of the data
				# loss = self(trainset.X[:-b,:])
				# #compute gradients
				# loss.backward()
				# #update parameters
				# optimizer.step()
                # #make sure none of the loss values contain a NaN (if they do, model will fail)
				# assert torch.isnan(loss).sum().sum() != 1
                # #update the global training loss for the loss value (multiply by num samples in batch)
                # #using all items in 1 of the batches
				# self.global_train_loss += loss.item() * lb
                # #*************************************************************
            #end if statement

            #if the validation set is not none
            if validset is not None:
                #make sure not to use gradient during validation
				with torch.no_grad():
                    #make sure the model is in evaluation mode
					model.eval()
                    #perform a forward pass using the validation data
					vloss = self(validset.X)
                    #make sure there are not NaNs in the results
					assert torch.isnan(vloss).sum().sum() != 1
                    #add to the global validation loss
					self.global_valid_loss = vloss.item()

                    #average the training set loss for the number of samples (because of multiplication earlier)
					if self.batch_size != 0:
                        lb = trainset.num_samples % self.batch_size
						# self.global_train_loss /= trainset.num_samples
                        self.global_train_loss /= lb

					#save the model if the (same way as VAE)***********
					SAVE_PATH = '{}best_model'.format(self.save_path)
					if self.save_mode and (self.global_valid_loss < self.best_valid_loss):
						self.best_valid_loss = float(self.global_valid_loss)
						torch.save({'epoch': epoch,
									'model_state_dict': model.state_dict(),
									'optimizer_state_dict': optimizer.state_dict()}, SAVE_PATH)
						self.write_best_loss()
						self.best_valid_flag = True
                    else:
                        no_improvement +=1


            #Add training and validation loss to keep track (will plot at end)
			train_RMSE.append(float(math.sqrt(self.global_train_loss)))
			valid_RMSE.append(float(math.sqrt(self.global_valid_loss)))

        #end epochs (either because reched max epochs or reached patience)
        #plot the training and validation error on the same figure
        plt.figure
    	plt.plot([x for x in range(1,len(train_RMSE)+1)], train_RMSE)
    	plt.plot([x for x in range(1,len(train_RMSE)+1)], valid_RMSE)
    	plt.title("")
    	plt.xlabel("Epoch")
    	plt.ylabel("RMSE")
    	plt.legend(['Train', 'Validation'])
    	plt.show()

        return self
        #Todo: restore the model to its best weights (access saved weights)
    #end fit functions

    #Test the model using the predict function
	def predict(self, dataset): #VAE code has a model option here
        #set the initial loss
		loss = 0.0
		#determine how many batches needed
		model = self
		with torch.no_grad():
			#put the model into evaluation mode (turn off dropout etc.)
			model.eval()
			#calculate the loss of the model
			vloss = self(dataset.X)
			loss = vloss.item()
		#RACHEL:return average loss
		return loss

def fit_predict(self, trainset, validset, testset):
    '''
    The purpose of this function is to fit the model using training and VALIDATION
    data, then using the final model, predict on the testing data
    '''
    print("----------TRAINING-----------")
    model = self.fit(trainset, validset)
    print("----------TESTING------------")
    return self.predict(testset)
