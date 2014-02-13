function [ Y, S ] = HmmGenerateData(nums, T, pi, A, E, outModel )
%HMMGENERATEDATA Generates data from a Hidden Markov Model.
%
%   [Y,S] = HmmGenerate(nums, T, pi, A, E ), generates nums samples from an
%   HMM of length T with the following parameters:
%       pi is the initial distribution of states,
%       A is the transition probability matrix,
%       E are the emission parameters:
%         - for a multinomial output alphabet, this is an emission
%           probability matrix,
%         - for normal distributed output, this is a structure where the mu
%           part is a vector of means and the sigma2 part is a vector of
%           variances,
%         - for an autoregressive process this is a vector of AR
%           coefficients,
%       outModel is a string specifying what kind of output model one
%       wants:
%         - 'multinomial' (default): multinomial output alphabet,
%         - 'normal': normal distributed alphabet.
%         - 'ar1': autoregressive process of order 1 output.
%   The function returns both the observations and the hidden states that
%   generated the observations.

if nargin < 6
    outModel = 'multinomial';
end

% Start generating samples.
S = zeros(nums,T);
Y = zeros(nums,T);
for i=1:nums
    % Generate initial state and observation.
    S(i,1) = 1+sum(rand()>cumsum(pi));
    if strcmp(outModel, 'multinomial')
        Y(i,1) = 1+sum(rand()>cumsum(E(S(i,1),:)));
    elseif strcmp(outModel, 'normal')
        Y(i,1) = randn() * sqrt(E.sigma2(S(i,1))) + E.mu(S(i,1));
    elseif strcmp(outModel, 'ar1')
        Y(i,1) = randn()*0.01;
    else
        error('Unknown observation model.');
    end
    
    % Generate the rest of the observations.
    for l=2:T
        S(i,l) = 1+sum(rand()>cumsum(A(S(i,l-1),:)));
        if strcmp(outModel, 'multinomial')
            Y(i,l) = 1+sum(rand()>cumsum(E(S(i,l),:)));
        elseif  strcmp(outModel, 'normal')
            Y(i,l) = randn() * sqrt(E.sigma2(S(i,l))) + E.mu(S(i,l));
        elseif  strcmp(outModel, 'ar1')
            Y(i,l) =  E(S(i,l)) * Y(i,l-1) + randn()*0.01;
        else
            error('Unknown observation model.');
        end
    end
end
