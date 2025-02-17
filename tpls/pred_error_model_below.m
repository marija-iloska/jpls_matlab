function [update_difference] = pred_error_model_below(y, Hk, n, n0, rm_idx, model_at_n0)

% INPUT
% y(1:t)
% H(1:t, 1:k+1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This fn that computes the predictive error recursively in dimension when
% decreasing model dimension J_pred(n,k) --> J_pred(n,k-1)) according to
% the expression in equation (6) in the reference paper 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Current model at t0
[theta_k, Dk] = model_at_n0{:};
idx = setdiff(1:length(theta_k), rm_idx);


update_difference = 0;

for i = n0+1:n

    % predictive residual error
    e = y(i) - Hk(i,:)*theta_k;

    % Compute d term
    d_past = - Dk(idx, rm_idx)/Dk(rm_idx,rm_idx);

    % G
    G = Hk(i, [idx rm_idx])*[d_past; -1]*theta_k(rm_idx);

    update_difference = update_difference + G^2 - 2*e*G;

    % Compute theta_(k+1, t-1), check Dk indices
    [theta_k, Dk] = RLS(y(i), Hk(i, :), theta_k, Dk);

end




end