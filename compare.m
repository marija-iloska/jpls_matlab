clear all
close all
clc

% Settings
var_y = 0.1;   % Variance
ps = 2;    % Sparsity percent
dy = 5;      % System dimension
T = 500;      % Time series length
r = 1;       % Range of input data H
rt = 2;      % Range of theta
n = round(0.3*T);
Ns = 2000;


%Create data
[y, H, theta] = generate_data(T, dy, r, rt,  ps, var_y);
idx_h = find(theta ~= 0)';


% PJ ORLS___________________________________________________
tic
[theta_k, Hk, k_store, k_mode, models_orls, count_orls, idx_orls] = pj_orls(y, H, dy, var_y);
toc 

[~, idx_orls_last] = ismember(Hk(1,:), H(1,:));
idx_orls_last = sort(idx_orls_last, 'ascend');



% RJ MCMC ___________________________________________________
% Data partition and Number of sweeps
tic
[idx_mcmc, theta_RJ, models_mcmc, count_mcmc, Nm] = rj_mcmc(y, H, n, Ns);
toc


idx_h
idx_mcmc
%idx_orls
idx_orls_last


for m = 1:Nm
    corr_mcmc = ismember(nonzeros(models_mcmc(m,:)), idx_h);
    if (sum(corr_mcmc) == length(idx_h)) && (length(nonzeros(models_mcmc(m,:))) == length(idx_h))
        idx_corr_mcmc = m;
    end
end

for m = 1:length(models_orls(:,1))
    corr_orls = ismember(nonzeros(models_orls(m,:)), idx_h);
    if (sum(corr_orls) == length(idx_h)) && (length(nonzeros(models_orls(m,:))) == length(idx_h))
        idx_corr_orls = m;
    else
        idx_corr_orls = [];
    end
end

if (length(idx_orls_last) == length(idx_h))
    if (sum(idx_orls_last == idx_h) == length(idx_h))
        idx_corr_orls_last = 1;
    else
        idx_corr_orls_last = [];
    end
else

    idx_corr_orls_last = [];
end

% Bar plot
figure(1)
b_mcmc = bar(count_mcmc/Ns, 'FaceColor', 'flat');
ylabel('Number of Visits')
title('RJMCMC Models visited','FontSize',20)
set(gca, 'FontSize', 20);
grid on
b_mcmc.CData(idx_corr_mcmc,:) = [0, 0, 0];

% filename = join(['figs23/rjmcmc', num2str(run), '.eps']);
% print(gcf, filename, '-depsc2', '-r300');

% Bar plot
figure(2)
b_orls = bar(count_orls/(T-3), 'FaceColor', 'flat');
ylabel('Number of Visits')
title('ORLS Models visited ','FontSize',20)
set(gca, 'FontSize', 20); 
grid on
if (isempty(idx_corr_orls_last) == 0)
    b_orls.CData(idx_corr_orls_last,:) = [0.5, 0, 0];
elseif (isempty(idx_corr_orls) == 0)
    b_orls.CData(idx_corr_orls_last,:) = [0, 0, 0.5];
end

% filename = join(['figs23/pjorls', num2str(run), '.eps']);
% print(gcf, filename, '-depsc2', '-r300');

