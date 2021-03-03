function Plot2MSE(B1, MSE1, B2, MSE2, fig_name)
%% Author: Rachel Theriault
%PURPOSE: The purpose of this function is to produce variable number versus
%MSE plots from lasso regression data overlaying results of 2 lasso
%calculations
%INPUT: 
%   B1 - coefficients of first calculation
%   MSE1 - MSE values of first calculation
%   B2 - coefficients of second calculation
%   MSE2 - MSE values of second calculation
%ENVIRONMENT: MATLAB2020b

%% Plot
norm_values1 = [];
for i =1:size(B1,2)
    num_zeros = find(B1(:,i)~=0);
    num_zeros = size(num_zeros,1);
    norm_values1 = [norm_values1; num_zeros];
end

norm_values2 = [];
for i =1:size(B2,2)
    num_zeros = find(B2(:,i)~=0);
    num_zeros = size(num_zeros,1);
    norm_values2 = [norm_values2; num_zeros];
end
figure
scatter(norm_values1, MSE1, 'b*', 'LineWidth', 2)
hold on
scatter(norm_values2, MSE2, 'r*','LineWidth', 2)
xlabel('Number of Variables Selected', 'FontName', 'latex', 'FontSize', 24)
ylabel('Mean Squared Error','FontName', 'latex', 'FontSize', 24)
title(fig_name, 'FontName', 'latex', 'FontSize', 24)