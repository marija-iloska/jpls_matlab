function [J] = pred_error_baselines(y, H, n, n0, theta)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This fn computes the predictive error from scratch for a chosen set of
% indices

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get indices of features used
S_features_used = find(theta ~= 0);

% Initialize model at n0
Dk = inv(H(1:n0, S_features_used)'*H(1:n0, S_features_used));
theta_k = Dk*H(1:n0, S_features_used)'*y(1:n0);

% Predictive residual error
e = [];

for i = n0+1:n

    % Compute predictive residual error
    e(end+1) = y(i) - H(i,S_features_used)*theta_k;

    if (i == n)
        break
    end

    % Update theta in time
    [theta_k, Dk] = RLS(y(i), H(i, S_features_used), theta_k, Dk);

end

% Predictive error
J = sum(e.^2);

end