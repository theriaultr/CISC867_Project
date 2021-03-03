function dataout = replaceZeros(datain, method)
%% Author: Kathrin Tyryshkin (previous professor)
% dataout = replaceZeros(datain, method) returns datain where all 0-s are
% replaced with either NaNs or very small value

%Input parameters: 
% datain - a numeric matrix,  rows - features, collumns - samples
% method - a string that is either:
% 'nan' -       replace 0's with NaNs
% 'lowval' -    replace 0's with a value that is 10 in the power of the
%               (exponent-1) of the smallest value in the input data.
%               e.g. if the minimum value in the input data. is 0.089, 
%               then the 0s will the replaced with 0.001
% for any other value in method - no change to the input data;
%
%output parameters:
%dataout - a numeric matrix with zeros replace according to the 'method'
%
%example: 
%data_zeros_replaced = replaceZeros(mydata, 'lowval');
%or
%data_zeros_replaced = replaceZeros(mydata, 'nan');
%
%author: Kathrin Tyryshkin
%date: January 2018

%% Code
%initialize the output
dataout = datain;

%mark all samples that are 0
flag = ~datain>0;

if strcmpi(method, 'nan') 
    dataout(flag) = nan;
elseif strcmpi(method, 'lowval')
    %add very small value to the counts with 0 relative to other values
    m = min(dataout(~flag)); % compute the smallest value in the set that is > 0
    exponent=floor(log10(abs(m)))-1; %find its nearest exponent
    dataout(flag) = 10.^exponent; %replace the 0's with 10 in the power of the exponent
end
