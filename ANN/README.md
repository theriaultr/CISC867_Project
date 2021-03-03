This section code is written by Rachel Theriault however,
some code is the same as the VAE portion and has just been edited to align
with training methods for this portion. This was a design choice to enable easier comparison between the methods, and for better flow of code.

Currently the Lasso ANN has been implemented and run on "dummy data" of 1000 genes and 10 patients. This was implemented because it was expected the method would do poorly and patience in training could be tested (which was successful)

The VAE model will be implemented after training of the Lasso model (as best training method will be implemented in the VAE model). The code will be nearly the same except contain addition layers for which the weights will be initialized the same way as in the VAE code.

Example for running Lasso: all default options are set so can just run "python ANN_Main.py"
