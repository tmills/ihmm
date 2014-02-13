function [ Phi ] = SampleEmissionMatrix( S, Y, K, H )
%SAMPLEEMISSIONMATRIX Samples an emission matrix from a given state
% sequence S, a corresponding observation vector Y, a Dirichlet prior H
% (row vector) for a K state HMM.

L = size(H, 2);
Phi = zeros(K,L);

for k=1:K
    emp = zeros(1,L);
    for l=1:L
        emp(l) = sum(Y(S == k) == l);
    end
    Phi(k,:) = dirichlet_sample(emp + H);
end