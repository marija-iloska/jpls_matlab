function [theta_occd, idx_occd, J, plot_stats, idx_store] = occd(y, H, n0, var_y, idx_h)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This fn is a compact implementation of OLinLASSO only for the purpose of
% generating statistical experiments for comparison. Users should see 
% occd_update.m for a single update step.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Dimensions
N = length(y);
K = length(H(1,:));

% Initial batch start
theta_occd = zeros(K,1);

% Denominators for each feature
for j = 1:K
    % Indexes of all elements except jth
    all_but_j{j} = setdiff(1:K, j);
end

rn = zeros(1,K);
Rn = zeros(K,K);
J = [];
correct = [];
incorrect = [];
idx_store = {};

for n = 1:N
    % Receive new data point Xn, yn
    Xn = H(n,:);
    yn = y(n);

    [theta_occd, rn, Rn] = occd_update(yn, Xn, rn, Rn, n, K, theta_occd, all_but_j, var_y);

    if n>n0
        [J(end+1)] = pred_error_baselines(y, H, n, n0, theta_occd);
        idx_occd = find(theta_occd ~= 0)';
        idx_store{end+1} = idx_occd;
    
        % EVALUATION
        correct(n-n0) = sum(ismember(idx_occd, idx_h));
        incorrect(n-n0) = length(idx_occd) - correct(n-n0);
    end

end



% Concatenate results
plot_stats = {correct, incorrect};

end