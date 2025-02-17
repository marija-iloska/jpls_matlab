function [E_add, E_rmv] = expectations(y, H, n0, N, S_true_features, var_y, theta)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Regret analysis: This fn computes the cumulative differences:
% predictive MSE (Model above) - predictive MSE (True Model) and
% predictive MSE (Model below) - predictive MES (True Model) according 
% to equations (10) and (11) in the reference paper.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialize
K = length(H(1,:));
S_other_features = setdiff(1:K, S_true_features);
p = length(S_true_features);
num_unused_features = K - p;

% Sort H
H = H(:, [S_true_features, S_other_features]);

% Get true estimates
Dp = inv(H(1:n0, 1:p)'*H(1:n0, 1:p));
theta_p = Dp*H(1:n0, 1:p)'*y(1:n0);


% Repetitive term - define
for m = 1:p
    % Indexes of all elements except jth
    all_but_m{m} = setdiff(1:p, m);
end


% Start loop
for n = n0+1:N


    
    % ADDITION  ===================================================
    for m = 1:num_unused_features
        % It updates DIM at n-1
        [~, Dpp] = ascendingORLS(y(1:n-1), H(1:n-1, 1:p), H(1:n-1, p+m), n-1, Dp, theta_p);

        % D(p+1, t-1)
        b_add = - Dpp(1:end-1,end)/Dpp(end, end);
        Q_add = H(n, p+m) - H(n, 1:p)*b_add;

        % Expectation E(p+1) - E(p)   single and batch
        E_add(n,m) = var_y*Q_add^2*Dpp(end,end);


    end

    % REMOVAL  ===================================================
    for m = 1:p

        idx = all_but_m{m};
        
        % Get Dk tilde by swapping 
        Dp_swap = Dp(idx, idx);
        Dp_swap(p, 1:p-1) = Dp(m, idx);
        Dp_swap(:, p) = Dp([idx, m], m);
        
        % D(p, t-1)
        b_rmv = - Dp_swap(1:end-1,end)/Dp_swap(end, end);
        Q_rmv = H(n,m) - H(n, idx)*b_rmv;


        % Expectation E(p-1) - E(p)   single and batch
        E_rmv(n,m) = Q_rmv^2*(theta(m)^2 - var_y*Dp_swap(end,end));
     
    end


    % Compute theta_(k+1, t-1), check Dk indices
    [theta_p, Dp] = RLS(y(n), H(n, 1:p), theta_p, Dp);



end


end