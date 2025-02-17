function [theta_olin, idx_olin, J, plot_stats, idx_store] = olasso(y, H, n0, epsilon, idx_h)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This fn is a compact implementation of OLinLASSO only for the purpose of
% generating statistical experiments for comparison. Users should see 
% olasso_update.m for a single update step.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Dimensions
N = length(y);
K = length(H(1,:));

% Define initial batch
y0 = y(1:n0);
H0 = H(1:n0, :);

% Define initial batch terms
xy0 = H0'*y0;
xx0 = H0'*H0;

% EIG
a = eig(xx0);
step = 0.001*n0/max(real(a));

% Initial estimate
[B, STATS] = lasso(H0, y0, 'CV', 5);
theta_olin = B(:, STATS.IndexMinMSE);

% Initialize terms
xy = zeros(K,1);
xx = zeros(K,K);

% theta at t0
J = [];

% For plotting
correct = [];
incorrect = [];
theta_store = [];
idx_store = {};

for n = n0+1:N

    % Updates
    xx = xx + H(n,:)'*H(n,:);
    xy = xy + H(n,:)'*y(n);    
    [theta_olin, ~] = olasso_update(xy0, xx0, xy, xx, theta_olin, epsilon, step, n0, n, K);

    % Evaluate model
    idx_olin = find(theta_olin ~= 0)';
    correct(end+1) = sum(ismember(idx_olin, idx_h));
    incorrect(end+1) = length(idx_olin) - correct(end);
    idx_store{end+1} = idx_olin;

    % Pred Error
    [J(end+1)] = pred_error_baselines(y, H, n, n0, theta_olin);

    %theta_store = [theta_store; theta_olasso'];
end

% Concatenate results
plot_stats = {correct, incorrect};


end