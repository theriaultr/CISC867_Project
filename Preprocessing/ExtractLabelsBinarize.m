function [reformatted_labels] = ExtractLabelsBinarize(y_labels, label_1)
%% Author: Rachel Theriault
%Purpose: To "binarize" labels as only 1 or 0
%INPUT
%   x_data - the x_data with rows as observations and columns as features
%   y_labels - the y_labels for the data as a row vector ("control, stage i, stage
%   ii, stage iii, stage iv"
%   label_1 - the label to make =1, all others will be 0
%Output:
%   extracted_data - x_data only corresponding to certain labels
%   extracted_labels - the labels for extracted_data
%ENVIRONMENT: MATLAB2020b
%% Code
%find indices of samples with requested labels
idx_label_1 = y_labels == label_1;

%extract data from those samples only
reformatted_labels = zeros(size(y_labels));
reformatted_labels(idx_label_1) = 1;

end