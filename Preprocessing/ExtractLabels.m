function [extracted_data, extracted_labels] = ExtractLabels(x_data, y_labels, label_0, label_1)
%% Author: Rachel Theriault
%Purpose: To extract data with 1 of 2 labels
%INPUT
%   x_data - the x_data with rows as observations and columns as features
%   y_labels - the y_labels for the data as a row vector ("control, stage i, stage
%   ii, stage iii, stage iv"
%   label_0 - the first label to extract
%   label_1 - the second label to extract
%Output:
%   extracted_data - x_data only corresponding to certain labels
%   extracted_labels - the labels for extracted_data
%ENVIRONMENT: MATLAB2020b

%% Code
%find indices of samples with requested labels
idx_label_0 = y_labels == label_0;
idx_label_1 = y_labels == label_1;

all_labels = find(idx_label_0 == 1 | idx_label_1 == 1);

%extract data from those samples only
extracted_data = x_data(all_labels, :);
extracted_labels_string = y_labels(all_labels);

extracted_idx_label_1 = extracted_labels_string == label_1;

extracted_labels = zeros(size(extracted_labels_string));
extracted_labels(extracted_idx_label_1) = 1;

end