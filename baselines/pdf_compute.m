function [ln_dens] = pdf_compute(N, n, p, M, He, H, y, ye)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This fn computes the pdf in log scale for RJMCMC implementation

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

valn = 0.5*(n - p);
valN = 0.5*(N - p);

%gam_ratio = gamma(valN)/gamma(valn);
log_gam_ratio = sum(log(valn+1 : valN));

% Projection matrix
Dk = H(:, M)'*H(:, M);
Pk = eye(N) - H(:,M)*inv(Dk)*H(:, M)';
Dke = He(:, M)'*He(:, M);
Pke = eye(n) - He(:,M)*inv(Dke)*He(:, M)';


% Log compute
ln_top = valn*log(0.5*ye'*Pke*ye) + 0.5*log(det(Dke));
ln_bot = valN*log(0.5*y'*Pk*y) + 0.5*log(det(Dk));


% Compute density
ln_dens = log_gam_ratio + ln_top - ln_bot;


end