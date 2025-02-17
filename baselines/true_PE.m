function [J_pred, mse] = true_PE(y, H, n0, N, S_true_features, var_y)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This fn computes the predictive error from scratch for the true model.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialize
Dp = inv(H(1:n0, S_true_features)'*H(1:n0, S_true_features));
theta_p = Dp*H(1:n0, S_true_features)'*y(1:n0);

J = 0;
J_pred = [];
mse = [];

for i = n0+1:N

    J = J + (y(i) - H(i,S_true_features)*theta_p)^2;
    J_pred(end+1) = J;

    mse(end+1) = var_y + var_y*H(i,S_true_features)*Dp*H(i,S_true_features)';

    if (i == N)
        break
    end

    % Compute theta_(k+1, t-1), check Dk indices
    [theta_p, Dp] = RLS(y(i), H(i, S_true_features), theta_p, Dp);


end


end