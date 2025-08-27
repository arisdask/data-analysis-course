function [b, y_pred, R2, adjR2, inmodel] = stepwise_regress_new(y, X)
% stepwise_regress   Performs stepwise linear regression variable selection
%
%   [b, y_pred, R2, adjR2, inmodel] = stepwise_regress(y, X)
%   fits a linear regression model using stepwise selection on the predictors in X.
%
%   Inputs:
%       y - Response vector (n-by-1)
%       X - Predictor matrix (n-by-p)
%
%   Outputs:
%       b       - Regression coefficients vector [intercept; all predictors with zeros for unselected]
%       y_pred  - Predicted response vector from the selected model (n-by-1)
%       R2      - R-squared value of the fitted model
%       adjR2   - Adjusted R-squared of the fitted model
%       inmodel - Logical vector indicating which predictors are included in the model
%
%   Description:
%       - Uses MATLAB's built-in stepwisefit function to select variables based on
%         stepwise regression without displaying output.
%       - Constructs final regression coefficients including intercept and zeros
%         for unselected predictors in their original positions.
%       - Calculates predicted values and model fit statistics.
%
%   Example:
%       y = randn(50,1);
%       X = randn(50,5);
%       [b, y_pred, R2, adjR2, inmodel] = stepwise_regress(y, X);
    
    [b_selected, ~, ~, inmodel, stats] = stepwisefit(X, y, 'display', 'off');

    n = size(X, 1);
    p = size(X, 2);
    intercept = stats.intercept;
    
    % Create coefficient vector with zeros for unselected variables
    b = zeros(p + 1, 1);  % +1 for intercept
    b(1) = intercept;     % Set intercept
    b(2:end) = 0;         % Initialize all predictor coefficients to zero
    b(find(inmodel) + 1) = b_selected(inmodel);  % Set selected coefficients
    
    % Calculate predictions using only selected variables
    X_selected = X(:, inmodel);
    y_pred = [ones(n, 1), X_selected] * [intercept; b_selected(inmodel)];

    R2 = R_squared(y, y_pred);
    adjR2 = adjR_squared(y, y_pred, sum(inmodel));
end