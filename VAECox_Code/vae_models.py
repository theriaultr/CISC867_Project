import os
import gc
import sys
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from progress.bar import Bar
from tqdm import trange
import math
from sklearn.metrics import explained_variance_score, r2_score
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.metrics import mean_absolute_error, median_absolute_error
from sklearn.cluster import KMeans
from vae_utils import *
import logging
from multiprocessing import Pool

#RACHEL
import matplotlib.pyplot as plt

#RACHEL: Define the dictionary of cancer TCGA data (currently dummy data). Defines the dataset names according to TCGA codes
cancer_list_dict = {
	'ching': ['BLCA', 'BRCA', 'HNSC', 'KIRC', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'STAD'],
	'wang': ['ACC', 'BLCA', 'BRCA', 'CESC', 'UVM', 'CHOL', 'ESCA', 'HNSC', 'KIRC', 'KIRP',
			 'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'PAAD', 'SARC', 'SKCM', 'STAD', 'UCEC', 'UCS'],

	'all': ['ACC', 'BLCA', 'BRCA', 'CESC', 'UVM', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM',
			'HNSC', 'KICH', 'KIPAN', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC',
			'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 'STAD', 'STES',
			'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS']
}

#RACHEL: define a dictionary of activation functions for use in the networks
acti_func_dict = {
	'ReLU': nn.ReLU(),
	'Tanh': nn.Tanh(),
	'LeakyReLU': nn.LeakyReLU(negative_slope=0.001),
	'Tanhshrink': nn.Tanhshrink(),
	'Hardtanh': nn.Hardtanh()
}


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = torch.device("cpu")

#RACHEL: Define the AE class and all of the functions used within it
class AE(nn.Module):
	#RACHEL: intitialize the AE
	def __init__(self, config, logger, num_features):
		#RACHEL:define arguments for the class that are initialized when created
		self.LOGGER = logger
		self.num_features = num_features
		self.save_path = './results/{}/{}/'.format(config.model_type, config.session_name)
		#if not os.path.exists(self.save_path):
		os.makedirs(self.save_path, exist_ok=True)
		self.max_epochs = config.max_epochs
		self.learning_rate = config.learning_rate
		self.opti_name = config.model_optimizer
		self.weight_sparsity = config.weight_sparsity
		self.weight_decay = config.weight_decay
		self.dropout_rate = config.dropout_rate
		self.save_mode = config.save_mode
		self.device_type = config.device_type
		self.exclude_imp = config.exclude_impute
		self.evaluation = dict()
		self.batch_size = config.batch_size
		self.batch_flag = False
		self.batch_index = 0

		#RACHEL: Set parameters for keeping track of the metrics
		self.global_train_loss = 0.0
		self.global_valid_loss = 0.0
		self.best_valid_loss = 999.999
		self.best_valid_flag = False

		#RACHEL: create another instances of AE? Give acess to AE code?
		super(AE, self).__init__()

		#RACHEL:Create the encoding layers - 2 linear layers 1 of size defined, and the second of size 128 (reduced feature number)
		self.encode = nn.Sequential(
			nn.Linear(self.num_features, config.hidden_nodes),
			acti_func_dict[config.acti_func],
			nn.Dropout(self.dropout_rate),
			nn.Linear(config.hidden_nodes, 1024), #RACHEL changed from 128 to 1024
			acti_func_dict[config.acti_func],
			nn.Dropout(self.dropout_rate)
		)

		#RACHEL: Create a decoding layer - symmetric to encoding layer
		self.decode = nn.Sequential(
			nn.Linear(1024, config.hidden_nodes),  #RACHEL changed from 128 to 1024
			acti_func_dict[config.acti_func],
			nn.Dropout(self.dropout_rate),
			nn.Linear(config.hidden_nodes, self.num_features)
		)

		#RACHEL: Save the variables and information used
		self.hp = 'hn_{}_af_{}_ms_{}_mt_{}_vd_{}'.format(config.hidden_nodes,
			config.acti_func,
			config.model_struct,
			config.model_type,
			config.vae_data) #RACHEL:***is this meant to be vae_data?
		try:
			#RACHEL:Save the results to results/{model name}/{session_name}
			with open(self.save_path + 'best_loss.txt', "r") as fr:
				for line in fr.readlines():
					l = line.split('\t')
					self.best_valid_loss = float(l[1])
					print('BEST VALID LOSS: {}'.format(self.best_valid_loss))
		except IOError:
			with open(self.save_path + 'best_loss.txt', "w") as fw:
				fw.write(self.hp + '\t999.999')

    #RACHEL: write the best loss to saved file
	def write_best_loss(self):
		file_name = self.save_path + 'best_loss.txt'
		with open(file_name, "w") as fw:
			fw.write('{}\t{}'.format(self.hp, self.best_valid_loss))

	#RACHEL: initialize the weights for the layers using xavier_normal weight initialization (**may want to change this?)
	def init_layers(self):
		nn.init.xavier_normal_(self.encode[0].weight.data)
		nn.init.xavier_normal_(self.decode[0].weight.data)
		try:
			nn.init.xavier_normal_(self.encode[3].weight.data)
			nn.init.xavier_normal_(self.decode[3].weight.data)
		except:
			pass
		try:
			nn.init.xavier_normal_(self.encode[6].weight.data)
			nn.init.xavier_normal_(self.decode[6].weight.data)
		except:
			pass

    #RACHEL: Calculate the L1_norm loss from based on the parameters
	def _l1_norm(self, model):
		l1_loss = 0.0
		for param in model.parameters():
			l1_loss += torch.sum(torch.abs(param))
		return self.weight_sparsity * l1_loss

    #RACHEL: perform the dimensionality reduction (encode layers)
	def dimension_reduction(self, x):
		return self.encode(x)

    #RACHEL: perform a foward pass of the data with params x (data), m, and coo
	def forward(self, x, m=None, coo=None):
		z = self.encode(x)
		recon = self.decode(z)
		x = x * m if self.exclude_imp else x
		recon = recon * m if self.exclude_imp else recon

		#get the loss value
		if not self.exclude_imp:
			return get_mse_loss(recon, x)
		else:
			return get_mse_loss_masked(recon, x, m)

    #RACHEL:switch the device to CPU
	def _switch_device(self, a, b):
		cpu_device = torch.device("cpu")
		gpu_device = self.device_type
		a = a.to(cpu_device)
		b = b.to(gpu_device)
		return a, b

	#RACHEL: fit the training data (using validation set as well) for the AE
	def fit(self, trainset, validset=None):
		#RACHEL:initialize the layers
		train_RMSE = []
		valid_RMSE = []
		self.init_layers()
		#RACHEL:use the appropriate device type
		model = self.to(self.device_type)
		#RACHEL:print the model information
		print(model)
		#RACHEL:get the optimizer and set parameters based on those defined at the beginning of the class
		optimizer = get_optimizer(self.opti_name)(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.5)
		#RACHEL:print the optimizer information
		print(optimizer)
		#RACHEL:determine the number of batches needed for the training and validation data based on provided batch_size
		batch_num = int(trainset.num_samples / self.batch_size) if self.batch_size != 0 else 1
		batch_val = int(validset.num_samples / self.batch_size) if self.batch_size != 0 else 1

		#RACHEL:set each epoch
		t = trange(self.max_epochs + 1, desc='Training...')
		for epoch in t:
			self.batch_flag = False
			#RACHEL:put the model in training mode
			model.train()
			#RACHEL:if using batches
			if self.batch_size != 0:
				# BATCH-WISE TRAINING PHASE
				self.global_train_loss, self.global_valid_loss = 0.0, 0.0
				#RACHEL:train one batch at a time
				for b in range(batch_num):
					#RACHEL:calculate indices for the batch
					i, j = (self.batch_size * b) % trainset.num_samples, (self.batch_size * (b+1)) % trainset.num_samples
					#RACHEL:make sure the model is in training mode
					model.train()
					#RACHEL:Define a model with only btach data
					loss = model(trainset.X[i:j,:], trainset.m[i:j,:], trainset.coo)
					assert torch.isnan(loss).sum().sum() != 1
					#RACHEL:keep track of the global training loss
					self.global_train_loss += loss.item() * self.batch_size
					optimizer.zero_grad()
					loss += self._l1_norm(model)
					#RACHEL:compute gradients
					loss.backward()
					#RACHEL:update the parameters
					optimizer.step()
				self.batch_flag = False
				lb = trainset.num_samples % self.batch_size
				loss = model(trainset.X[:-b,:], trainset.m[:-b,:], None)
				loss += self._l1_norm(model)
				#RACHEL: compite gradients
				loss.backward()
				#RACHEL:update parameters
				optimizer.step()
				assert torch.isnan(loss).sum().sum() != 1
				self.global_train_loss += loss.item() * lb
			else: #RACHEL: same comments as above apply
				# FULL_BATCH TRAINING PHASE
				loss = model(trainset.X, trainset.m, None)
				assert torch.isnan(loss).sum().sum() != 1
				self.global_train_loss = loss.item()
				optimizer.zero_grad()
				loss += self._l1_norm(model)
				loss.backward()
				optimizer.step()
			self.batch_flag = False
			if validset is not None:
				with torch.no_grad():
					model.eval()
					if self.batch_size != 0:
						# BATCH-WISE VALIDATION PHASE
						for b in range(batch_val):
							i, j = (self.batch_size * b) % validset.num_samples, (self.batch_size * (b+1)) % validset.num_samples
							vloss = model(validset.X[i:j,:], validset.m[i:j,:], trainset.coo)
							assert torch.isnan(vloss).sum().sum() != 1
							self.global_valid_loss += vloss.item() * self.batch_size
						self.batch_flag = False
						lb = validset.num_samples % self.batch_size
						vloss = model(validset.X[-lb:,:], validset.m[-lb:,:], None)
						assert torch.isnan(vloss).sum().sum() != 1
						self.global_valid_loss += vloss.item() * lb
					else:
						# FULL_BATCH VALIDATION PHASE
						vloss = model(validset.X, validset.m, None)
						assert torch.isnan(vloss).sum().sum() != 1
						self.global_valid_loss = vloss.item()
					if self.batch_size != 0:
						self.global_train_loss /= trainset.num_samples
						self.global_valid_loss /= validset.num_samples

					# SAVE BEST MODEL***********
					SAVE_PATH = '{}best_model'.format(self.save_path)
					if self.save_mode and (self.global_valid_loss < self.best_valid_loss):
						self.best_valid_loss = float(self.global_valid_loss)
						torch.save({'epoch': epoch,
									'model_state_dict': model.state_dict(),
									'optimizer_state_dict': optimizer.state_dict()}, SAVE_PATH)
						self.write_best_loss()
						self.best_valid_flag = True
			t.set_description('(Training: %g)' % float(math.sqrt(self.global_train_loss)) + '(Validation: %g)' % float(math.sqrt(self.global_valid_loss)))
			#RACHEL: keep track of trainng and validation so can produce a plot at the end
			train_RMSE.append(float(math.sqrt(self.global_train_loss)))
			valid_RMSE.append(float(math.sqrt(self.global_valid_loss)))

		#RACHEL: added after all epochs completed create a predict_log_partial_hazard
		plt.figure
		plt.plot([x for x in range(1,len(train_RMSE)+1)], train_RMSE)
		plt.plot([x for x in range(1,len(train_RMSE)+1)], valid_RMSE)
		plt.title("")
		plt.xlabel("Epoch")
		plt.ylabel("RMSE")
		plt.legend(['Train', 'Validation'])
		plt.show()

		# plt.figure
		# plt.plot([x for x in range(1,len(valid_RMSE)+1)], valid_RMSE)
		# plt.title("Validation RMSE")
		# plt.xlabel("Epoch")
		# plt.ylabel("RMSE")
		# plt.show()



		return model

	#RACHEL:test the model (check encoding + decoding)
	def predict(self, dataset, model):
		self.batch_flag = False
		loss = 0.0
		#RACHEL:determine how many batches needed
		batch_val = int(dataset.num_samples / self.batch_size) if self.batch_size != 0 else dataset.num_samples
		#RACHEL:define model
		model = self.to(self.device) if model is None else model
		with torch.no_grad(): #RACHEL:test
			#RACHEL: put the model into evaluation mode (turn off dropout etc.)
			model.eval()
			#RACHEL: evaluate in batches or altogether
			if self.batch_size != 0:
				for b in range(batch_val):
					i, j = (self.batch_size * b) % dataset.num_samples, (self.batch_size * (b+1)) % dataset.num_samples
					model.eval()
					vloss = model(dataset.X[i:j,:], dataset.m[i:j,:], dataset.coo)
					loss += vloss.item() * self.batch_size
				self.batch_flag = False
				lb = dataset.num_samples % self.batch_size
				vloss = model(dataset.X[-lb:,:], dataset.m[-lb:,:], None)
				loss += vloss.item() * lb
			else:
				vloss = model(dataset.X, dataset.m, None)
				loss = vloss.item()
		#RACHEL:return average loss
		if self.batch_size != 0:
			return loss / dataset.num_samples
		else:
			return loss

	#RACHEL:perform both training and testing phases
	def fit_predict(self, trainset, validset, testset):
		print("--------TRAINING--------")
		model = self.fit(trainset, validset)
		self.LOGGER.info('Best Loss Updated: {}'.format(self.best_valid_flag))
		if self.save_mode:
			SAVE_PATH = self.save_path + 'final_model'
			self.LOGGER.info('Saving Model....')
			#RACHEL:save the final model as a dictionary
			torch.save({'model_state_dict': model.state_dict()}, SAVE_PATH)
		print("---------TESTING---------")
		#RACHEL:return the testing results
		return self.predict(testset, model)

#RACHEL: create the VAE class - called with AE as parameters
class VAE(AE):
	#RACHEL: initialize the VAE
	def __init__(self, config, logger, num_features=20531):
		super().__init__(config, logger, num_features)
		#RACHEL:create encoding layer(1 linear, with dropout)
		self.encode = nn.Sequential(
			nn.Linear(self.num_features, config.hidden_nodes),
			acti_func_dict[config.acti_func],
			nn.Dropout(self.dropout_rate)
		)
		#RACHEL: create encoding mu layer (1 linear with dropout)
		self.encode_mu = nn.Sequential(
			nn.Linear(config.hidden_nodes, 1024), #RACHEL:Changed from 128 to 1024
			acti_func_dict[config.acti_func],
			nn.Dropout(self.dropout_rate)
		)
		#RACHEL:create encoding si layer (1 linear with dropout)
		self.encode_si = nn.Sequential(
			nn.Linear(config.hidden_nodes, 1024),#RACHEL:Changed from 128 to 1024
			acti_func_dict[config.acti_func],
			nn.Dropout(self.dropout_rate)
		)
		#create the deconding layer (2 linear layers with dropout in between - symmetric to encoding layer)
		self.decode = nn.Sequential(
			nn.Linear(1024, config.hidden_nodes), #RACHEL:Changed from 128 to 1024
			acti_func_dict[config.acti_func],
			nn.Dropout(self.dropout_rate),
			nn.Linear(config.hidden_nodes, num_features)
		)

	#RACHEL:perform dimensionality reduction
	def dimension_reduction(self, x, coo):
		#RACHEL:coo means to perform topological conversion?????
		if coo is None:
			h = self.encode(x)
			mu = self.encode_mu(h)
			return mu
		else:
			x = self._topological_conv(x, coo)
			h = self.encode(x)
			mu = self.encode_mu(h)
			return mu

	#RACHEL:find std and eps given mu and log variaiton for VAE
	def _reparameterize(self, mu, logvar):
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)
		assert not torch.isnan(std).any() and not torch.isnan(eps).any()
		return eps.mul(std).add_(mu)

	#RACHEL:forward pass of the data
	def forward(self, x, m=None, coo=None):
		#RACHEL:encodedong portion
		h = self.encode(x)
		mu = self.encode_mu(h)
		logvar = self.encode_si(h)
		z = self._reparameterize(mu, logvar)
		#RACHEL:decoding portion
		recon = self.decode(mu)
		x = x * m if self.exclude_imp else x
		recon = recon if self.exclude_imp else recon
		#RACHEL:calculate k-fold loss? --> function in vae_utils
		if not self.exclude_imp:
			return get_mse_kld_loss(recon, x, mu, logvar)
		else:
			return get_mse_kld_loss_masked(recon, x, mu, logvar, m)

#RACHEL: NOT USING THIS ONE FOR PROJECT******************************************
class DAE(AE):
	def __init__(self, config, logger, num_features=20531):
		super().__init__(config, logger, num_features)
		self.encode = nn.Sequential(
			nn.Linear(self.num_features, config.hidden_nodes),
			acti_func_dict[config.acti_func],
			nn.Dropout(self.dropout_rate),
			nn.Linear(config.hidden_nodes, 1024),#RACHEL changed from 128 to 1024
			acti_func_dict[config.acti_func],
			nn.Dropout(self.dropout_rate)
		)
		self.decode = nn.Sequential(
			nn.Linear(1024, config.hidden_nodes),#RACHEL: changed from 128 to 1024
			acti_func_dict[config.acti_func],
			nn.Dropout(self.dropout_rate),
			nn.Linear(config.hidden_nodes, self.num_features)
		)

	def init_layers(self):
		try:
			nn.init.xavier_normal_(self.encode[0].weight.data)
			nn.init.xavier_normal_(self.decode[3].weight.data)
		except:
			pass
		nn.init.xavier_normal_(self.decode[0].weight.data)

	def dimension_reduction(self, x):
		return self.encode(x)

	def forward(self, x, m=None, coo=None):
		x = torch.randn(x.size()).to(self.device_type) * 0.01 + x
		z = self.encode(xx)
		recon = self.decode(z)
		x = x * m if self.exclude_imp else x
		recon = recon * m if self.exclude_imp else recon

		if not self.exclude_imp:
			return get_mse_loss(recon, x)
		else:
			return get_mse_loss_masked(recon, x, m)
