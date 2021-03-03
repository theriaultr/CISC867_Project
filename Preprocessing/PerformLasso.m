function [selected_c1, selected_c2, selected_c3, selected_c4, selected_1, selected_2, selected_3, selected_4] =  PerformLasso(x_data, y_labels)
%% Author:Rachel Theriault
%PURPOSE: This function performs Lasso on provided data
%INPUTS
%   x_data - the x_data with rows as observations and columns as features
%   y_labels - the y_labels for the data as a row vector ("control, stage i, stage
%   ii, stage iii, stage iv"
%OUTPUT:
%   selected_cx - selected genes binary comparison of stage x to control
%   selected_x - selected genes binary comparison of stage x to all other
%   stages including control
%ENVIRONMENT: MATLAB2020b
%NOTES:
% - MSE value for selecting lambda is hardcoded and selected based on MSE
% versus number of feature plots (selecetd to be at the inflection point of
% the plot)

%% STAGE I********************************************************************
[c1_data, c1_labels] = ExtractLabels(x_data, y_labels, "control", "stage i");
[Bc1,fitInfoc1] = lasso(c1_data, c1_labels);
% PlotMSE(Bc1, fitInfoc1.MSE, "control versus stage i");
idx = find(fitInfoc1.MSE < 0.02);
selected_c1 = find(Bc1(:,idx(end))~=0);

%try 1 versus evrything else
y1_labels = ExtractLabelsBinarize(y_labels, "stage i");
[B1,fitInfo1] = lasso(x_data, y1_labels);
% PlotMSE(B1, fitInfo1.MSE, "stage i versus everything else");
idx = find(fitInfo1.MSE < 0.035);
selected_1 = find(B1(:,idx(end))~=0);

Plot2MSE(Bc1, fitInfoc1.MSE, B1, fitInfo1.MSE, "Stage I")

%% STAGE II********************************************************************
[c2_data, c2_labels] = ExtractLabels(x_data, y_labels, "control", "stage ii");
[Bc2,fitInfoc2] = lasso(c2_data, c2_labels);
% PlotMSE(Bc2, fitInfoc2.MSE, "control versus stage ii");
idx = find(fitInfoc2.MSE < 0.04);
selected_c2 = find(Bc2(:,idx(end))~=0);

y2_labels = ExtractLabelsBinarize(y_labels, "stage ii");
[B2,fitInfo2] = lasso(x_data, y2_labels);
% PlotMSE(B2, fitInfo2.MSE, "stage ii versus everything else");
idx = find(fitInfo2.MSE < 0.01);
selected_2 = find(B2(:,idx(end))~=0);

Plot2MSE(Bc2, fitInfoc2.MSE, B2, fitInfo2.MSE, "Stage II")

%% STAGE III********************************************************************
[c3_data, c3_labels] = ExtractLabels(x_data, y_labels, "control", "stage iii");
[Bc3,fitInfoc3] = lasso(c3_data, c3_labels);
% PlotMSE(Bc3, fitInfoc3.MSE, "control versus stage iii");
idx = find(fitInfoc3.MSE < 0.03);
selected_c3 = find(Bc3(:,idx(end))~=0);

y3_labels = ExtractLabelsBinarize(y_labels, "stage iii");
[B3,fitInfo3] = lasso(x_data, y3_labels);
% PlotMSE(B3, fitInfo3.MSE, "stage iii versus everything else");
idx = find(fitInfo1.MSE < 0.02);
selected_3 = find(B3(:,idx(end))~=0);

Plot2MSE(Bc3, fitInfoc3.MSE, B3, fitInfo3.MSE, "Stage III")

%% STAGE IV********************************************************************
[c4_data, c4_labels] = ExtractLabels(x_data, y_labels, "control", "stage iv");
[Bc4,fitInfoc4] = lasso(c4_data, c4_labels);
% PlotMSE(Bc4, fitInfoc4.MSE, "control versus stage iv");
idx = find(fitInfoc4.MSE < 0.001);
selected_c4 = find(Bc4(:,idx(end))~=0);

y4_labels = ExtractLabelsBinarize(y_labels, "stage iv");
[B4,fitInfo4] = lasso(x_data, y4_labels);
% PlotMSE(B4, fitInfo4.MSE, "stage iv versus everything else");
idx = find(fitInfo4.MSE < 0.004);
selected_4 = find(B4(:,idx(end))~=0);

Plot2MSE(Bc4, fitInfoc4.MSE, B4, fitInfo4.MSE, "Stage IV")



end
