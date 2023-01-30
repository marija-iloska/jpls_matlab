clear all
close all
clc

% Settings
var_y = 0.1; % Variance
g = @(x) x;  % Transition function
p_s = 0.3;   % Sparsity percent
dx = 6;      % System dimension
T = 1000;     % Time series length
r = 0.5;     % Range of input data H
rt = 5;      % Range of theta

%SSM
tr = @(coeff, states) coeff*g(states);

%Create data
[y, H, theta, a] = generate_data(T, dx, r, rt, p_s, var_y, tr, g);

H_true = H;


% Initialize
yt = y(1);

% Initialize first Hk
Hk = H(1,1);

T = length(H(:,1));

% Initialize first Dk
Dk = 1/(Hk'*Hk);

% Compute iniital estimate of theta_k
theta_k = Dk*Hk'*y;

% Initial covariance of data
Sigma = Dk/var_y;

% First gain
K = 0.5*Dk*Hk;

% Initial theta update
theta_k = Dk*Hk*yt;

k = 1;
Hk = H(1:2, k);
% Update indices in original H

% TIME UPDATE
% Get theta dimension from last chosen model
k = length(theta_k);

% Update covariance for new data
Sigma = var_y*Dk;

% Update gain
K = Sigma*Hk(2,:)'/(var_y + Hk(2,:)*Sigma*Hk(2,:)');

% Update estimate
theta_k = theta_k + K*(y(2) - Hk(2,:)*theta_k);

% Update covariance
Sigma = (eye(k) - K*Hk(2,:))*Sigma;


% Model choice count
k_store = ones(1,T);

% Start time
for t = 3 : T-1


    % ORDER UPDATE

    % Compute probability of staying Mk
    log_pk = log_mvnpdf(y(1:t), H(1:t, 1:k), theta_k, var_y);


    % Compute probabilities of moving to Mk+j,
    % j is an index that iterates through all unused basis functions in H
    clear H1_store D1_store theta1_store
    log_pk1 = [];
    if (dx > k)
        for j = 1:(dx - k)
            % Update current theta by jth basis function
            [theta_temp, D_temp, H_temp] = ols_updates(y, H, k, j, t, Dk, theta_k);
    
            % Compute that probability
            log_pk1(j) = log_mvnpdf(y(1:t), H_temp, theta_temp, var_y);
    
            % Corresponding variables store
            H1_store{j} = H_temp;
            D1_store{j} = D_temp;
            theta1_store{j} = theta_temp;
        end
    end
    % Temp set logp0 = [];
    log_pk0 = [];


    % Compute probabilities of moving to Mk-j
    % j is an index that iterates through all current basis functions in H
    if (k > 1)
        clear log_pk0 H0_store D0_store theta0_store
        for j = 1:k
            % Update current theta by remoivng jth basis function
            [theta_temp, D_temp, H_temp] = ols_downdates(theta_k, Dk, j, H, t);
    
            % Compute that probability
            log_pk0(j) = log_mvnpdf(y(1:t), H_temp, theta_temp, var_y);
    
            % Corresponding variables store
            H0_store{j} = H_temp;
            D0_store{j} = D_temp;
            theta0_store{j} = theta_temp;
        end
    end


    % Normalize Probabilities
    log_probs = [log_pk, log_pk0, log_pk1];
    probs = exp(log_probs - max(log_probs));
    probs = probs./sum(probs);

    k_temp = length(log_pk0);

    % Sample model
    ki = datasample(1:length(probs), 1, 'Weights', probs);

    % Update params based on chosen model
    m = ki - 1;

    % If REMOVE COLUMN
    if (ki > 1) && (ki <= k_temp+1)
        % Update original H (swap column order)
        H = H(:, [setdiff(1:dx, m), m]);

        % Update rest
        %Hk = H0_store{m};
        Dk = D0_store{m};
        theta_k = theta0_store{m};

        % IF ADD COLUMN
    elseif (ki > k_temp + 1)
        % Update original available data H (swap column order )
        H = H(:, [1:k, m + k - k_temp, setdiff(k+1:dx, m + k - k_temp)]);

        % Update rest
        %Hk = H1_store{m - k_temp};
        Dk = D1_store{m - k_temp};
        theta_k = theta1_store{m - k_temp};
    end


    % TIME UPDATE
    % Get theta dimension from last chosen model
    k = length(theta_k);
    k_store(t) = k;
    Hk = H(1:t, 1:k);

    % Update covariance for new data
    Sigma = var_y*Dk;

    % Update gain
    K = Sigma*Hk(t,:)'/(var_y + Hk(t,:)*Sigma*Hk(t,:)');

    % Update estimate
    theta_k = theta_k + K*(y(t) - Hk(t,:)*theta_k);

    % Update covariance
    Sigma = (eye(k) - K*Hk(t,:))*Sigma;


end


theta_clean = theta;
theta_clean(theta_clean==0) = [];
dk = length(theta_clean);
dk_est = length(theta_k);
k_store(T) = dk_est;

check = (dk == dk_est);

dk_mode = mode(k_store((T-50):T));
check_mode = (dk == dk_mode);


%scatter(1:length(model), model)


%T = 400;
sz = 40;

rp = [212, 19, 19]/256;
rs = [94, 4, 4]/256;

gp = [87, 194, 105]/ 256;
gs = [5, 102, 37]/256;

yline(dk, 'Color', gp, 'LineWidth',2, 'LineStyle', '-')
hold on
scatter(1:T, k_store(1:T), sz,'filled', 'MarkerFaceColor', rs, 'Linewidth', 1);
set(gca,'FontSize',15, 'Linewidth',1)
xlabel('Time', 'FontSize', 30)
ylabel('Model Choice','FontSize', 30)
title('Convergence', 'FontSize',30)
ylim([0, dx])








