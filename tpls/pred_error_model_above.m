function [update_difference] = pred_error_model_above(y, Hk, n, n0, model_at_n0)

% INPUT
% y(1:t)
% H(1:t, 1:k)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This fn that computes the predictive error recursively in dimension when
% increasing model dimension J_pred(n,k) --> J_pred(n,k+1)) according to
% the expression in equation (6) in the reference paper 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Current model at t0
[theta_k, Dk] = model_at_n0{:};
k = length(theta_k);


update_difference = 0;

for i = n0+1:n

    % residual pred error
    e = y(i) - Hk(i,1:k)*theta_k;

    % Compute d term
    d_past = Dk*Hk(1:i-1, 1:k)'*Hk(1:i-1,k+1);

    % Projection matrix
    P = eye(i-1) - Hk(1:i-1,1:k)*Dk*Hk(1:i-1,1:k)';

    % theta new element
    hP = Hk(1:i-1, k+1)'*P;
    theta_kk_scalar = hP*y(1:i-1)/(hP*Hk(1:i-1, k+1));

    % G
    G = Hk(i, :)*[d_past; -1]*theta_kk_scalar;

    update_difference = update_difference + G^2 + 2*e*G;

    % Compute theta_(k+1, t-1), check Dk indices
    [theta_k, Dk] = RLS(y(i), Hk(i, 1:k), theta_k, Dk);

end





end