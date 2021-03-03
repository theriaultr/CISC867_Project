function [new_x_data, new_stages] = RemoveStages(x_data, stages, stagea, stageb)
%% Author: Rachel Theriault
%PURPOSE: The purpose of this function is to pull out the data from
%sepcific stages from the provided x_data and labels to enable pair-wise
%analysis
%INPUT:
%   x_data (numeric matrix) - the gene expression data in format of genes(rows) x samples
%   (columns)
%   stages(string array): the labels of the cancer stage for each sample
%   stagea (string): the first stage to extract
%   stageb(string): the second stage to extract
%OUTPUT
%   new_data(numeric matrix): x_data with only samples that are of stagea or stageb
%ENVIRONMENT:MATLAB_R2020b
%Log:
%   created February 17, 2021
%NOTES:
%   Developed for CISC 867 course project

idx = stages == stagea | stages == stageb;
new_x_data = x_data(:, idx);
new_stages = stages(idx);


end