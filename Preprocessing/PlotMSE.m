function PlotMSE(B, MSE, fig_name)
%% Author: Rachel Theriault
%PURPOSE: The purpose of this function is to produce variable number versus
%MSE plots from lasso results
%INPUT: 
%   B - coefficients 
%   MSE - MSE values
%ENVIRONMENT: MATLAB2020b

%% Plot
norm_values = [];
for i =1:size(B,2)
    num_zeros = find(B(:,i)~=0);
    num_zeros = size(num_zeros,1);
    norm_values = [norm_values; num_zeros];
end
figure
scatter(norm_values, MSE, 'b*')
xlabel('Number of variables selected')
ylabel('MSE')
title(fig_name)