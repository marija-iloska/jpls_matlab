function [theta_k, Dk] = descendingORLS(theta_k, Dk, rm_idx)


% Code for D change (before update)
% Let be the column removed (or rm_idx)
k = length(Dk(1,:));
idx_old = setdiff(1:k, rm_idx);
idx_new = [idx_old, rm_idx];


% Get Dk tilde by swapping 
Dk_swap = Dk(idx_old, idx_old);
Dk_swap(k, 1:k-1) = Dk(rm_idx, idx_old);
Dk_swap(:, k) = Dk(idx_new, rm_idx);

% Now Dk downdate code
DK11 = Dk_swap(1:k-1, 1:k-1);
DK12 = Dk_swap(1:k-1, k);
DK22 = Dk_swap(k,k);


% Final DK
Dk = DK11 - DK12*DK12' /( DK22 );


% Ratio
ratio = DK12/DK22;


% Update rest of theta elements
theta_k(idx_old) = theta_k(idx_old) - ratio*theta_k(rm_idx);


% theta(k-1) < --- theta(k)
theta_k(rm_idx) = [];




end