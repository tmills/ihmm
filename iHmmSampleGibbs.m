function [S, stats] = iHmmSampleGibbs(Y, hypers, numb, nums, numi, S0)
% IHMMSAMPLEGIBBS Samples states from the iHMM with multinomial output
% using the Gibbs sampler.
%
% [S, stats] = iHmmSampleGibbs(Y, hypers, numb, nums, numi, S0) uses the
% Gibbs sampling training algorithm for the infinite HMM.
%
%   Input Parameters:
%   - Y: training sequence of arbitrary length,
%   - hypers: a structure that describes the hyperparameters for the beam
%             sampler. If this structure contains alpha0 and gamma, it will
%             not resample these during sampling. If these are not
%             specified, one needs to specify hyperparameters for alpha0
%             and gamma (alpha0_a, alpha0_b, gamma_a, gamma_b). hypers
%             should also contain a prior for the emission alphabet in the
%             field H,
%   - numb: the number of burnin iterations,
%   - nums: the number of samples to output,
%   - numi: the number of sampling, iterations between two samples,
%   - S0: is the initial assignment to the sequence.
%
%   Output Parameters:
%   - S: is a cell array of sample structures where each sample contains the
%        hidden state sequence S, the number of states K and the Beta used 
%        for that sample.
%   - stats: is a structure that contains a variety of statistics for every
%            iteration of the sampler: K, alpha0, gamma, the size of the
%            trellis and the joint log likelihood.

% Initialize the sampler.
T = size(Y,2);                      % # of time-steps T.
L = size(hypers.H,2);               % # of emission symbols.

sample.S = S0;
sample.K = max(S0);

% Setup structures to store the output.
S = {};
stats.K = zeros(1,(numb + (nums-1)*numi));
stats.alpha0 = zeros(1,(numb + (nums-1)*numi));
stats.gamma = zeros(1,(numb + (nums-1)*numi));
stats.jll = zeros(1,(numb + (nums-1)*numi));
stats.trellis = zeros(1,(numb + (nums-1)*numi));

% Initialize hypers; resample a few times as our inital guess might be off.
if isfield(hypers, 'alpha0')
    sample.alpha0 = hypers.alpha0;
else
    sample.alpha0 = gamrnd(hypers.alpha0_a, 1.0 / hypers.alpha0_b);
end
if isfield(hypers, 'gamma')
    sample.gamma = hypers.gamma;
else
    sample.gamma = gamrnd(hypers.gamma_a, 1.0 / hypers.gamma_b);
end
for i=1:5
    sample.Beta = ones(1, sample.K+1) / (sample.K+1);
    [sample.Beta, sample.alpha0, sample.gamma] = iHmmHyperSample(sample.S, sample.Beta, sample.alpha0, sample.gamma, hypers, 20);
end

iter = 1;
fprintf('Iteration 0: K = %d, alpha0 = %f, gamma = %f.', sample.K, sample.alpha0, sample.gamma);

while iter <= (numb + (nums-1)*numi)

    % Compute the empirical emission matrix.
    % E(i,j) = number of emmisions of symbol l from state i.
    E = zeros(sample.K, L);
    for t=1:T
        E(sample.S(t), Y(t)) = E(sample.S(t), Y(t)) + 1;
    end
    
    % Compute the empirical transition matrix.
    % N(i,j) = number of transition from state i to j.
    N = zeros(sample.K, sample.K);
    N(1, sample.S(1)) = 1;
    for t=2:T
        N(sample.S(t-1), sample.S(t)) = N(sample.S(t-1), sample.S(t)) + 1;
    end
    
    % Start resampling the hidden state sequence.
    for t=1:T
        % Discount the transition and emission counts for timestep t.
        E(sample.S(t), Y(t)) = E(sample.S(t), Y(t)) - 1;
        if t ~= 1
            N(sample.S(t-1), sample.S(t)) = N(sample.S(t-1), sample.S(t)) - 1;
        else
            N(1, sample.S(t)) = N(1, sample.S(t)) - 1;
        end
        if t ~= T
            N(sample.S(t), sample.S(t+1)) = N(sample.S(t), sample.S(t+1)) - 1;
        end
        
        % Compute the marginal probability for timestep t.
        r = ones(1, sample.K+1);
        for k=1:sample.K
            if t ~= 1
                r(k) = r(k) * ( N(sample.S(t-1), k) + sample.alpha0 * sample.Beta(k) );
            else
                r(k) = r(k) * ( N(1, k) + sample.alpha0 * sample.Beta(k) );
            end
            
            if t ~= T
                if t > 1 && k ~= sample.S(t-1)
                    r(k) = r(k) * ( N(k, sample.S(t+1)) + sample.alpha0 * sample.Beta(sample.S(t+1)) ) / ( sum(N(k, :)) + sample.alpha0 );
                elseif t == 1 && k ~= 1
                    r(k) = r(k) * ( N(k, sample.S(t+1)) + sample.alpha0 * sample.Beta(sample.S(t+1)) ) / ( sum(N(k, :)) + sample.alpha0 );
                elseif t > 1 && k == sample.S(t-1) && k ~= sample.S(t+1)
                    r(k) = r(k) * ( N(k, sample.S(t+1)) + sample.alpha0 * sample.Beta(sample.S(t+1)) ) / ( sum(N(k, :)) + 1 + sample.alpha0 );
                elseif t > 1 && k == sample.S(t-1) && k == sample.S(t+1)
                    r(k) = r(k) * ( N(k, sample.S(t+1)) + 1 + sample.alpha0 * sample.Beta(sample.S(t+1)) ) / ( sum(N(k, :)) + 1 + sample.alpha0 );
                elseif t == 1 && k == 1 && k ~= sample.S(t+1)
                    r(k) = r(k) * ( N(k, sample.S(t+1)) + sample.alpha0 * sample.Beta(sample.S(t+1)) ) / ( sum(N(k, :)) + 1 + sample.alpha0 );
                elseif t == 1 && k == 1 && k == sample.S(t+1)
                    r(k) = r(k) * ( N(k, sample.S(t+1)) + 1 + sample.alpha0 * sample.Beta(sample.S(t+1)) ) / ( sum(N(k, :)) + 1 + sample.alpha0 );
                end
            end

            r(k) = r(k) * ( hypers.H(Y(t)) + E(k,Y(t)) ) / ( sum(E(k,:)) + sum(hypers.H) );
        end
        r(sample.K+1) = ( hypers.H(Y(t)) / (sum(hypers.H)) ) * sample.alpha0 * sample.Beta(sample.K+1);
        if t ~= T
            r(sample.K+1) = r(sample.K+1) * sample.Beta(sample.S(t+1));
        end
        
        % Resample s_t.
        r = r ./ sum(r);
        sample.S(t) = 1 + sum(rand() > cumsum(r));
        
        % Update datastructures if we move to a new state.
        assert(size(N,1) == sample.K);
        assert(size(N,2) == sample.K);
        if sample.S(t) > sample.K
            N(:, sample.S(t)) = 0;                  % We have a new state: augment data structures
            N(sample.S(t), :) = 0;
            E(sample.S(t), :) = 0;
            
            % Extend Beta. Standard stick-breaking construction stuff
            b = betarnd(1, sample.gamma);
            BetaU = sample.Beta(end);
            sample.Beta(end) = b * BetaU;
            sample.Beta(end+1) = (1-b)*BetaU;
            
            sample.K = sample.K + 1;
        end
        
        % Update emission and transition counts.
        E(sample.S(t), Y(t)) = E(sample.S(t), Y(t)) + 1;
        if t ~= 1
            N(sample.S(t-1), sample.S(t)) = N(sample.S(t-1), sample.S(t)) + 1;
        else
            N(1, sample.S(t)) = N(1, sample.S(t)) + 1;
        end
        if t ~= T
            N(sample.S(t), sample.S(t+1)) = N(sample.S(t), sample.S(t+1)) + 1;
        end
        
        % Perform some coherency checks on the datastructures.
        assert(size(N,1) == sample.K);
        assert(size(N,2) == sample.K);
        assert(length(sample.Beta) == sample.K+1);
        assert(sum(sum(N)) == T);
        assert(sum(sum(E)) == T);
    end
        
    % Recompute the number of states.
    zind = sort(setdiff(1:sample.K, unique(sample.S)));
    for i=size(zind,2):-1:1                 % Make sure we delete from the back onwards, otherwise indexing is more complex.
        sample.S(sample.S > zind(i)) = sample.S(sample.S > zind(i)) - 1;
        N(zind(i),:) = [];
        N(:,zind(i)) = [];
        E(zind(i),:) = [];
        sample.Beta(end) = sample.Beta(end) + sample.Beta(zind(i));
        sample.Beta(zind(i)) = [];
    end
    sample.K = size(N,2);
    
    % Resample Beta
    [sample.Beta sample.alpha0 sample.gamma] = iHmmHyperSample(sample.S, sample.Beta, sample.alpha0, sample.gamma, hypers, 20);
    
    % Prepare next iteration.
    stats.alpha0(iter) = sample.alpha0;
    stats.gamma(iter) = sample.gamma;
    stats.K(iter) = sample.K;
    stats.jll(iter) = iHmmJointLogLikelihood(sample.S, Y, sample.Beta, sample.alpha0, hypers.H);
    fprintf('Iteration: %d: K = %d, alpha0 = %f, gamma = %f, JL = %f.\n', ...
        iter, sample.K, sample.alpha0,sample. gamma, stats.jll(iter));
    
    if iter >= numb && mod(iter-numb, numi) == 0
    	S{end+1} = sample;
    end
    iter = iter + 1;
end