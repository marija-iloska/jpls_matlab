function [theta_k, Dk] = RLS(y, hk, theta_k, Dk)


    % Get current dimension
    k = length(theta_k);

    % Current error
    et = (y - hk*theta_k);

    % Update gain
    %K = Sigma*hk'/(var_y + hk*Sigma*hk');
    K = Dk*hk'/(1 + hk*Dk*hk');

    % Update estimate
    theta_k = theta_k + K*et;

    % Update covariance
    Dk = (eye(k) - K*hk)*Dk;
 

end
