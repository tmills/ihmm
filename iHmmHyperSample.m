function [sbeta, salpha0, sgamma, N, M] = iHmmHyperSample(S, ibeta, ialpha0, igamma, hypers, numi)
% IHMMHYPERSAMPLE resamples the hyperparameters of an infinite hmm.
%
% [sbeta, salpha0, sgamma, N, M] = ...
%   iHmmHyperSample(S, ibeta, ialpha0, igamma, hypers, numi) resamples the
%   hyperparameters given the state sequence S, the previous
%   hyperparameters ibeta, ialpha0, igamma and their respective
%   hyper-hyperparameters in the structure hypers (needs alpha0_a,
%   alpha0_b, gamma_a and gamma_b fields corresponding to gamma prior on
%   the hyperparameters). If the hyper-hyperparameters are not given,  the
%   estimated alpha0 and gamma will be the same as the input alpha0 and
%   gamma. numi is the number of times we run the Gibbs samplers for alpha0
%   and gamma (see HDP paper or Escobar & West); we recommend a value of
%   around 20. The function returns the new hyperparameters, the CRF counts
%   (N) and the sampled number of tables in every restaurant (M).
%
%   Note that the size of the resampled beta will be the same as the size
%   of the original beta.

K = length(ibeta)-1;        % # of states in iHmm.
T = size(S,2);              % length of iHmm.

% Compute N: state transition counts.
N = zeros(K,K);
N(1,S(1)) = 1;
for t=2:T
    N(S(t-1), S(t)) = N(S(t-1), S(t)) + 1;
end

% Compute M: number of similar dishes in each restaurant.
M = zeros(K);
for j=1:K
    for k=1:K
        if N(j,k) == 0
            M(j,k) = 0;
        else
            for l=1:N(j,k)
                M(j,k) = M(j,k) + (rand() < (ialpha0 * ibeta(k)) / (ialpha0 * ibeta(k) + l - 1));
            end
        end
    end
end

% Resample beta
ibeta = dirichlet_sample([sum(M,1) igamma]);

% Resample alpha
if isfield(hypers, 'alpha0')
    ialpha0 = hypers.alpha0;
else
    for iter = 1:numi
        w = betarnd(ialpha0+1, sum(N,2));
        p = sum(N,2)/ialpha0; p = p ./ (p+1);
        s = binornd(1, p);
        ialpha0 = gamrnd(hypers.alpha0_a + sum(sum(M)) - sum(s), 1.0 / (hypers.alpha0_b - sum(log(w))));
    end
end

% Resample gamma (using Escobar & West 1995)
if isfield(hypers, 'gamma')
    igamma = hypers.gamma;
else
    k = length(ibeta);
    m = sum(sum(M));
    for iter = 1:numi
        mu = betarnd(igamma + 1, m);
        pi_mu = 1 / (1 + (m * (hypers.gamma_b - log(mu))) / (hypers.gamma_a + k - 1)  );
        if rand() < pi_mu
            igamma = gamrnd(hypers.gamma_a + k, 1.0 / (hypers.gamma_b - log(mu)));
        else
            igamma = gamrnd(hypers.gamma_a + k - 1, 1.0 / (hypers.gamma_b - log(mu)));
        end
    end
end


sbeta = ibeta;
salpha0 = ialpha0;
sgamma = igamma;