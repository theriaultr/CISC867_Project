function FormatDataForANN(gene_expression_data, gene_names, stage_label, file_name_data,fold_decimals)
%% Author: Rachel Theriault
%PURPOSE: The prupose of this function is to reformat TCGA RMSE_normalized
%data into scaled data for entrance into an ANN
%INPUT: 
%   gene_expression_data: the gene expression data from LIHC (saved
%   workspace)
%   gene_names: the names of the genes
%   stage_label: the label of the cancer (control, stage i, stage ii, stage
%   iii or stage iv)
%   file_name_name: the name of the file (including extension) to save the
%   data to
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

%% Code - currently producing "Dummy" code of first 10 sample and first 1000 genes
%1. Perform feature-wise z-score normalization (for each gene, z-score
%normalize across the patients
lt_gene_expression_data = log2(replaceZeros(gene_expression_data(1:1000,1:10), 'lowval'));
% figure;
% boxplot(lt_gene_expression_data);
%zscore along the rows (features/genes)
z_score_gene_expression_data = (zscore(lt_gene_expression_data,0, 2))';
% z_score_gene_expression_data(11:end, :) = []; %for dummy data --> take only 10 samples and 1000 genes
% z_score_gene_expression_data(:, 1001:end) = []; %for dummy data --> take only 10 samples and 1000 genes

% %visualize the z-scsore data
% figure;
% %plot so each sample is an entry
% boxplot(z_score_gene_expression_data');

%2. Re-name the features with @mRNA in the front
gene_names = gene_names(1:1000); %for dummy data
gene_names_extended = "mrna@" + gene_names; %change to mRNA*****
gene_names_extended = ["Samples"; gene_names_extended;"Fold@811"; "Stage_Label"];

[M,N] = size(z_score_gene_expression_data); %M is num samples, N is num_genes

%3. Get Folds
%find number of genes in training fold:
num_train = round(M*fold_decimals(1));
num_valid = round(M*fold_decimals(2));
% num_test = M*fold_decimals(3);
start_train = 2;
end_train = 2+num_train;
start_valid = end_train+1;
end_valid = start_valid+num_valid;


%4. put the data into a cell array
%turn the stage labels numeric
stage_labels_num = zeros(size(stage_label)); %control is class 0
idx_stagei = stage_label == 'stage_i';
idx_stageii = stage_label == 'stage_ii';
idx_stageiii = stage_label == 'stage_iii';
idx_stageiv = stage_label == 'stage_iv';
stage_labels_num(idx_stagei) = 1;
stage_labels_num(idx_stageii) = 2;
stage_labels_num(idx_stageiii) = 3;
stage_labels_num(idx_stageiv) = 4;

stage_labels_num(11:end) = []; %for dummy data

% %want to create a cell array of size M+1 x N+1
%for dummy data:
cell_array_data = cell(M+1, N+3);
cell_array_data(1,:) = cellstr(gene_names_extended);
cell_array_data(2:end,1) = {0};
cell_array_data(2:end, 2:end-2) = num2cell(z_score_gene_expression_data);
cell_array_data(2:9, end-1) = {0};
cell_array_data(10, end-1) = {1};
cell_array_data(11, end-1) = {2};
cell_array_data(2:end,end) = {stage_labels_num};

%5. export data to csv (too big for .xlsx)
writecell(cell_array_data, file_name_data); 

end