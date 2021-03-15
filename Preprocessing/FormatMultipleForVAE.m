function FormatMultipleForVAE(path_name, file_names)
%PURPOSE - the purpose of the function is to format multiple datafiles 
%INPUTS
%   PATH_NAME(string) - the name of the path all of the datafiles are
%   stored in
%   FILE_NAMES(string vector) - vector of file names (Mx1)

[~, num_files] = size(file_names);

%opening the file 
for file_idx=1:num_files
    disp(file_names(file_idx))
    %read in the csv
    all_data = readtable(path_name+file_names(file_idx)+".csv", 'VariableNamingRule', 'preserve');
    data = all_data(3:end,:);
    names = all_data.Properties.VariableNames;
    sample_names = names(2:end);
    [genes, ~] = size(data)
%     data = all_data(3:end, :);
    %produce the csv
    FormatDataForVAE(data, sample_names, file_names(file_idx)+"_VAE_811_mRNA@.csv", file_names(file_idx)+"_VAE_mRNA@.csv",[0.8, 0.1, 0.1]);
end




end