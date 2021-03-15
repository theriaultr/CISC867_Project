function FormatDataForVAE(table_data, sample_names, file_name_data, file_name_binary, fold_decimals)
%% Author: Rachel Theriault
%PURPOSE: The prupose of this function is to reformat TCGA RMSE_normalized
%data into scaled data for entrance into a variational autoencoder
%developed by __.
%INPUT: 
%   table_data: expected format is table name is the TCGAID without dashes, and column 1
%   contains gene names
%   gene_names: the names of the genes
%   stage_label: the label of the cancer (control, stage i, stage ii, stage
%   iii or stage iv)
%   file_name_data: the name of the file (including extension) to save the
%   data to
%   file_name_binary: the name of the file to save binary mask data to
%   fold_decimals: the train, validation, and testing percentages as
%   decimals in vector format
%RESULTS SAVED:
%   The saved file will be in the format where column 1 is sample ID
%   labelled Samples, Columns will be the z-score normalized data with
%   label mRNA@gene_name
%ENVIRONMENT: MATLAB2020b
%NOTES:
% - after visualizing the data I decided to additionally log transform the
% data before performing feature-wise z-score to the data as it produces a
% distribution closer to what is followed in typical genetic studies

%% Code
%1. Extract the data as a matrix
%Still to implement
%Handling missing values

gene_expression_data = table2array(table_data(:,2:end));
gene_names = table2array(table_data(:,1));

%2. Perform feature-wise z-score normalization (for each gene, z-score
%normalize across the patients
lt_gene_expression_data = log2(replaceZeros(gene_expression_data, 'lowval'));
figure;
boxplot(lt_gene_expression_data);
%zscore along the rows (features/genes)
z_score_gene_expression_data = (zscore(lt_gene_expression_data,0, 2))';

%visualize the z-scsore data
figure;
%plot so each sample is an entry
boxplot(z_score_gene_expression_data');

%3. Re-name the features with @mRNA in the front
gene_names_extended = "mrna@" + gene_names; %change to mRNA*****
gene_names_extended = ["Samples"; gene_names_extended;"Fold@811"];
gene_names = ["Samples"; gene_names;"Fold@811"];

[M,N] = size(z_score_gene_expression_data); %M is num samples, N is num_genes

%4. Get Folds
%find number of genes in training fold:
num_train = round(M*fold_decimals(1));
num_valid = round(M*fold_decimals(2));
% num_test = M*fold_decimals(3);
start_train = 2;
end_train = 2+num_train;
start_valid = end_train+1;
end_valid = start_valid+num_valid;


%5. put the data into a cell array


% %want to create a cell array of size M+1 x N+1
cell_array_data = cell(M+1, N+2);
cell_array_data(1,:) = cellstr(gene_names_extended);
cell_array_data(2:end,1) = sample_names; %used to be cellstr
cell_array_data(2:end, 2:end-1) = num2cell(z_score_gene_expression_data);
cell_array_data(start_train:end_train, end) = {0};
cell_array_data(start_valid:end_valid, end) = {1};
cell_array_data(end_valid+1:end, end) = {2};


%6. export data to csv (too big for .xlsx)
writecell(cell_array_data, file_name_data); %TOO BIG

%Make binary file (1 if use data)
%same initial cell array except fill everything with 1
cell_array_binary = cell(M+1, N+2);
cell_array_binary(1,:) = cellstr(gene_names); %use gene names with no @mRNA
cell_array_binary(2:end,1) = sample_names; %used to be cellstr
cell_array_binary(2:end, 2:end-1) = {1};%place a value 1 in each entry
cell_array_binary(start_train:end_train, end) = {0};
cell_array_binary(start_valid:end_valid, end) = {1};
cell_array_binary(end_valid+1:end, end) = {2};

writecell(cell_array_binary, file_name_binary);

end