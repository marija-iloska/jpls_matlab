function [theta_k, S_features_used, J_pred, plot_stats] = tpls(y, H, k, n0, idx)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This fn is a compact implementation of TPLS only for the purpose of
% generating statistical experiments comparing to other methods. Users
% should see example_code.m for online implementation of TPLS

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Dimensions
N = length(H(:,1));
K = length(H(1,:));

% Set S 
S_features_used = datasample(1:K, k, 'replace', false);
S_features_unused = setdiff(1:K, S_features_used);

% Initial batch of data
y0 = y(1:n0);
H0 = H(1:n0,:);

[theta_k, Dk] = initialize(y0, H0(:,S_features_used));
model_at_n0 = {theta_k, Dk};

% Model storage
correct = zeros(1,N-n0);
incorrect = zeros(1,N-n0);

% Predictive errors init
J = 0;
J_pred = [];
J_above = Inf;
J_below = Inf;


%% JPLS LOOP
% Start time loop
tic
for n = n0+1:N

    % Data
    Hn = H(1:n, S_features_used);
    yn = y(1:n);

    % CURRENT MODEL PE
    J = J + (y(n) - Hn(n,:)*theta_k)^2;

    % MODEL ABOVE PE
    if k < K
        rm_idx = nan;
        parfor m = 1:K-k
             difference = pred_error_model_above(yn, [Hn H(1:n, S_features_unused(m))], n, n0, model_at_n0);
             J_above(m) = J + difference;     
        end
    end

   % MODEL BELOW PE
   if k > 1
    parfor m = 1:k
         difference = pred_error_model_below(yn, Hn, n, n0, m, model_at_n0);
         J_below(m) = J + difference;
    end
   end

    % Find min PE
    m_above  = find(J_above == min(J_above));
    m_below = find(J_below == min(J_below));
    minJ = min([J_above(m_above), J_below(m_below), J]);

    % IF BELOW
    if minJ == J_below(m_below)


        % Update base model
        [theta0, D0] = descendingORLS(model_at_n0{1}, model_at_n0{2}, m_below);
        model_at_n0 = {theta0, D0};


        % Move to neighbor model below
        [theta_k, Dk] = descendingORLS(theta_k, Dk, m_below);

        % Update predictive error
        J = J_below(m_below);

        % Update feature sets
        S_features_unused(end+1) = S_features_used(m_below);
        S_features_used(m_below) = [];

        % Update model dimension
        k = k - 1;

    % IF ABOVE
    elseif minJ == J_above(m_above)     

        % Update base model
        [theta0, D0] = ascendingORLS(y0, H0(:,S_features_used), H0(:, S_features_unused(m_above)), n0, model_at_n0{2}, model_at_n0{1});
        model_at_n0 = {theta0, D0};

        % Move to neighbor model above
        [theta_k, Dk] = ascendingORLS(yn(1:n-1), Hn(1:n-1,:), H(1:n-1, S_features_unused(m_above)), n-1, Dk, theta_k);
        
        % Update predictive error
        J = J_above(m_above);

        % Update feature sets
        S_features_used(end+1) = S_features_unused(m_above);
        S_features_unused(m_above) = [];

        % Update model dimension
        k = k + 1;

    end

    %[J_scratch] = normal_PE(y, H, S_features_used,n0, n);

    % Place holders
    J_above = Inf;
    J_below = Inf;
    
    % Store predictive error
    J_pred(end+1) = J;


    % TIME UPDATE    
    % theta(k,n) <-- theta(k,n-1) and Dk(n) <-- Dk(n-1)
    [theta_k, Dk] = RLS(y(n), H(n,S_features_used), theta_k, Dk);

    % EVALUATION
    correct(n-n0) = sum(ismember(S_features_used, idx));
    incorrect(n-n0) = length(S_features_used) - correct(n-n0);


end
toc
% Concatenate results
plot_stats = {correct, incorrect};


end