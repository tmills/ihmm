function [ logp ] = iHmmJointLogLikelihood( S, Y, Beta, alpha0, H)
%IHMMJOINTLOGLIKELIHOOD Computes the joint log likelihood of a
%particular hidden state sequence and emission string.
        
K = max(S);
L = length(H);
T = length(S);
N = zeros(K);
E = zeros(K,L);

% Compute all transition and emission counts.
N(1,S(1)) = 1;
E(1, Y(1)) = E(1, Y(1)) + 1;
for t=2:T
    N(S(t-1), S(t)) = N(S(t-1), S(t)) + 1;
    E(S(t), Y(t)) = E(S(t), Y(t)) + 1;
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
    logp = logp + gammaln(sum(H)) ...
                - sum(gammaln(H)) ...
                + sum(gammaln(H + E(k,:))) ...
                - gammaln(sum(H + E(k,:)));
end