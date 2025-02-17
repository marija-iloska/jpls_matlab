function [theta_k, Dk] = initialize(y, H)


% Initialize first Dk
Dk = inv(H'*H);

% Compute iniital estimate of theta_k
theta_k = Dk*H'*y;


end