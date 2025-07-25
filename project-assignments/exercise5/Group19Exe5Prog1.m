% Daskalopoulos Aristeidis (10640)
% Rousomanis Georgios (10703)

clc, clearvars, close all;
addpath('../lib/');  % Add the parent directory to the path

[data_with_TMS, data_without_TMS] = loadTMSdata('../TMS.xlsx');

% Regression model and diagnostic plot with and without TMS
Group19Exe5Fun1(data_without_TMS{:,{'Setup'}}, data_without_TMS{:,{'EDduration'}}, 'without TMS');
Group19Exe5Fun1(data_with_TMS{:,{'Setup'}}, data_with_TMS{:,{'EDduration'}}, 'with TMS');

% Diagnostic plots for higher order regression with and without TMS
Group19Exe5Fun2(data_without_TMS{:,{'Setup'}}, data_without_TMS{:,{'EDduration'}}, 'without TMS');
Group19Exe5Fun2(data_with_TMS{:,{'Setup'}}, data_with_TMS{:,{'EDduration'}}, 'with TMS');

% Analysis Without TMS:
%
% - The R-squared statistic is close to zero, indicating 
%   that ED duration and Setup are not linearly correlated.
%
% - The diagnostic plot confirms that the model is unsuitable for 
%   estimating the dependent variable (ED duration) from the independent 
%   variable (Setup).
%
% - The residuals exhibit a straight-line pattern, showing strong 
%   correlation. This is further supported by a high correlation coefficient 
%   (r > 0.9). This suggests that the dependent variable (ED duration) may be 
%   influenced by another variable not included in the model.
%
% - The poor fit of the model is further validated by the very low R^2 
%   statistic.
%
% Analysis With TMS:
%
% - In this case the R^2 statistic is slightly higher than without TMS, 
%   indicating a marginal improvement in model performance.
%
% - However, the model still fails to explain the dependent variable (ED 
%   duration) based on the independent variable (Setup), as shown in the 
%   diagnostic plot.
%
% - The residuals once again form a straight-line pattern, suggesting strong 
%   correlation. This is reflected in the high correlation coefficient 
%   (r > 0.9), reinforcing the likelihood of an omitted variable influencing 
%   the model.
%
% Overall Observations:
%
% - In both cases (with and without TMS), the model is unable to predict 
%   the dependent variable (ED duration) using the independent variable 
%   (Setup).
%
% - The straight-line residual pattern highlights the need for an 
%   additional independent variable to account for the missing influence.
%
% - Increasing the polynomial order of the Setup variable provides only a 
%   slight improvement in the adjusted R^2 statistic, as shown in the 
%   diagnostic figures. Thus, a higher-order polynomial transformation is 
%   unlikely to significantly enhance model performance.
%
% - Including another relevant independent variable is critical to improving 
%   the model’s predictive power.
