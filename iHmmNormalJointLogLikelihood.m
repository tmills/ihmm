function [ logp ] = iHmmNormalJointLogLikelihood( S, Y, Beta, alpha0, mu_0, sigma2_0, sigma2)
%IHMMNORMALJOINTLOGLIKELIHOOD Computes the joint log likelihood of 
% generating a particular hidden state sequence and emission string.
        
K = max(S);
T = length(S);
N = zeros(K);
E = zeros(K,3);

% Compute all transition counts and emission sufficient statistics.
N(1,S(1)) = 1;
E(S(1), 1) = E(S(1), 1) + Y(1);
E(S(1), 2) = E(S(1), 2) + 1;
E(S(1), 3) = E(S(1), 3) + Y(1)^2;
for t=2:T
    N(S(t-1), S(t)) = N(S(t-1), S(t)) + 1;
    E(S(t), 1) = E(S(t), 1) + Y(t);
    E(S(t), 2) = E(S(t), 2) + 1;
    E(S(t), 3) = E(S(t), 3) + Y(t)^2;
end

% Compute the log likelihood.
logp = 0;
for k=1:K
    R = [N(k,:) 0] + alpha0 * Beta;
    ab = alpha0 * Beta;
    nzind = find(R ~= 0);
    % Add transition likelihood.
    logp = logp + gammaln(alpha0) ...
            - gammaln(sum([N(k,:) 0]) + alpha0) ...
            + sum(gammaln( R(nzind)  )) ...
            - sum(gammaln( ab(nzind) ));
    % Add emission likelihood.
    sigma2_n = 1 / (1 / sigma2_0 + E(k, 2) / sigma2);
    mu_n = (mu_0 / sigma2_0 + E(k,1) / sigma2) * sigma2_n;
    logp = logp + 0.5 * log(sigma2_n) ...
                - 0.5 * (log(sigma2_0) + E(k,2)/2 * log(2*pi*sigma2)) ...
                - 0.5 * (E(k,3)/sigma2 + mu_0^2 / sigma2_0 - mu_n^2 / sigma2_n);
end