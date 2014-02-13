function [ Pi ] = SampleTransitionMatrix( S, H )
%SAMPLETRANSITIONMATRIX Samples a transition matrix from a state sequence S
% and Dirichlet prior H (row vector).

K = size(H,2);
T = size(S,2);
Pi = zeros(K);

N = zeros(K);
for t=2:T
    N(S(t-1), S(t)) = N(S(t-1), S(t)) + 1;
end

for k=1:K
    Pi(k, :) = dirichlet_sample(N(k,:) + H);
end