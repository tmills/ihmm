function [S, stats] = iHmmSampleBeam(Y, hypers, numb, nums, numi, S0, STrue, optional_params)
% IHMMSAMPLEBEAM Samples states from the iHMM with multinomial output
% using the Beam sampler.
%
% [S, stats] = iHmmSampleBeam(Y, hypers, numb, nums, numi, S0) uses the
% beam sampling training algorithm for the infinite HMM.
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
%        hidden state sequence S, the number of states K, the Beta, Pi,
%        Phi's used for that sample.
%   - stats: is a structure that contains a variety of statistics for every
%            iteration of the sampler: K, alpha0, gamma, the size of the
%            trellis and the marginal likelihood.

% Initialize the sampler.
T = size(Y,2);                      % # of time-steps T
numIters = numb + (nums-1)*numi;

sample.S = S0;
sample.K = max(S0);

% Setup structures to store the output.
S = {};
stats.K = zeros(1,(numb + (nums-1)*numi));
stats.alpha0 = zeros(1,(numb + (nums-1)*numi));
stats.gamma = zeros(1,(numb + (nums-1)*numi));
stats.jml = zeros(1,(numb + (nums-1)*numi));
stats.trellis = zeros(1,(numb + (nums-1)*numi));
stats.ent = zeros(1, (numb + (nums-1)*numi));
stats.vi = zeros(1, numIters);

useNullState = isfield(optional_params, 'nullstate');
nullDone = 0;
uniform = ones(1,max(Y));
uniform(1) = 0;
uniform = uniform ./ sum(uniform);

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
if  ~isfield(hypers, 'discount')
    hypers.discount=0;     %% PY with discount=0 is regular DP
end
for i=1:5
    sample.Beta = ones(1, sample.K+1) / (sample.K+1);
    if(useNullState)
        sample.Beta(2) = sample.Beta(2) + 1;
    end
    [sample.Beta, sample.alpha0, sample.gamma] = iHmmHyperSample(sample.S, sample.Beta, sample.alpha0, sample.gamma, hypers, 20);
end

% Sample the emission and transition probabilities.
sample.Phi = SampleEmissionMatrix( sample.S, Y, sample.K, hypers.H );
sample.Phi(1,:) = [1.0 zeros(1,max(Y)-1)];
if(useNullState)
    fprintf('Using uniform null state\n');
    sample.Phi(2,:) = uniform;
else
    fprintf('Not using null state\n');
end

sample.Pi = SampleTransitionMatrix( sample.S, sample.alpha0 * sample.Beta );
if useNullState
    sample.Pi(2,:) = ones(1,sample.K+1) / (sample.K+1);
end
    
sample.Pi(sample.K+1,:) = [];

iter = 1;
fprintf('Iteration 0: K = %d, alpha0 = %f, gamma = %f.\n', sample.K, sample.alpha0, sample.gamma);

while iter <= (numb + (nums-1)*numi)
%    sample.Beta
    fflush(stdout);
    % Safety check.
    assert(size(sample.Phi,1) == size(sample.Beta,2) - 1);
    
    % Reset the trellis size count in case the previous iteration didn't
    % return a samplable path.
    stats.trellis(iter) = 0;
    
    % Sample the auxilary variables and extend Pi and Phi if necessary.
    u = zeros(1,T);
    for t=1:T
        if t == 1
            u(t) = rand() * sample.Pi(1, sample.S(t));
        else
            u(t) = rand() * sample.Pi(sample.S(t-1), sample.S(t));
        end
    end

    while max(sample.Pi(:, end)) > min(u)     % Break the Pi{k} stick some more.
        pl = size(sample.Pi, 2);
        bl = length(sample.Beta);

        % Safety check.
        assert(bl == pl);

        % Add row to transition matrix.
        sample.Pi(bl,:) = dirichlet_sample(sample.alpha0 * sample.Beta);
        sample.Phi(bl,:) = dirichlet_sample(hypers.H);

        % Break beta stick.
        be = sample.Beta(end);
        bg = betarnd(1, sample.gamma);
        sample.Beta(bl) = bg * be;
        sample.Beta(bl+1) = (1-bg) * be;

        pe = sample.Pi(:, end);
        a = repmat(sample.alpha0 * sample.Beta(end-1), bl, 1);
        b = sample.alpha0 * (1 - sum(sample.Beta(1:end-1)));
        if min(a) < 1e-2 || min(b) < 1e-2       % This is an approximation when a or b are really small.
             pg = binornd(1, a./(a+b));
        else
            pg = betarnd( a, b );
        end
        sample.Pi(:, pl) = pg .* pe;
        sample.Pi(:, pl+1) = (1-pg) .* pe;
    end
    sample.K = size(sample.Pi, 1);
    
    % Safety check.
    assert(sample.K == length(sample.Beta) - 1);
    assert(sample.K == size(sample.Phi, 1));
    
    % Resample the hidden state sequence.
    dyn_prog = zeros(sample.K, T);
    
    dyn_prog(:,1) = sample.Pi(1,1:sample.K) > u(1);
    stats.trellis(iter) = stats.trellis(iter) + sum(sum(dyn_prog(1,:)));
    for k=1:sample.K
        dyn_prog(k,1) = sample.Phi(k, Y(1)) * dyn_prog(k,1);
    end
    dyn_prog(:,1) = dyn_prog(:,1) / sum(dyn_prog(:,1));
    
    for t=2:T
        if(Y(t) == 1)
            %% only possible state is 1
            dyn_prog(:,t) = [1.0 zeros(1,sample.K-1)]';
            %% do not change trellis values
        else
            A = sample.Pi(1:sample.K, 1:sample.K) > u(t);
            dyn_prog(:,t) = A' * dyn_prog(:,t-1);
            stats.trellis(iter) = stats.trellis(iter) + sum(sum(A));
            for k=1:sample.K
                dyn_prog(k,t) = sample.Phi(k, Y(t)) * dyn_prog(k, t);
            end
            dyn_prog(:,t) = dyn_prog(:,t) / sum(dyn_prog(:,t));
        end
    end
    
    % Backtrack to sample a path through the HMM.
    if sum(dyn_prog(:,T)) ~= 0.0 && isfinite(sum(dyn_prog(:,T)))
        sample.S(T) = 1 + sum(rand() > cumsum(dyn_prog(:,T)));
        for t=T-1:-1:1
            r = dyn_prog(:,t) .* (sample.Pi(:, sample.S(t+1)) > u(t+1));
            r = r ./ sum(r);
            sample.S(t) = 1 + sum(rand() > cumsum(r));
        end
        % Safety check.
        assert(~isnan(sum(sample.S(t))));
        
        % Cleanup our state space by removing redundant states.
        zind = sort(setdiff(1:sample.K, unique(sample.S)));
        if useNullState && (sum(find(zind==2)))
            nullDone = true;
        end
        
        for i = length(zind):-1:1
            sample.Beta(end) = sample.Beta(end) + sample.Beta(zind(i));
            sample.Beta(zind(i)) = [];
            sample.Pi(:,zind(i)) = [];
            sample.Pi(zind(i),:) = [];
            sample.Phi(zind(i),:) = [];
            sample.S(sample.S > zind(i)) = sample.S(sample.S > zind(i)) - 1;
        end
        sample.K = size(sample.Pi,1);

        % Resample Beta given the transition probabilities.
        [sample.Beta, sample.alpha0, sample.gamma] = iHmmHyperSample(sample.S, sample.Beta, sample.alpha0, sample.gamma, hypers, 20);
        
        % Resample the Phi's given the new state sequences.
        sample.Phi = SampleEmissionMatrix(sample.S, Y, sample.K,hypers.H);
        sample.Phi(1,:) = [1.0 zeros(1,max(Y)-1)];
        if(useNullState && ~nullDone)
            sample.Phi(2,:) = uniform;
        end
        
        % Resample the transition probabilities.
        sample.Pi = SampleTransitionMatrix(sample.S, sample.alpha0 * sample.Beta);
        if(useNullState && ~nullDone)
            sample.Pi(2,:) = ones(1,sample.K+1) / (sample.K+1);
        end
        sample.Pi(sample.K+1,:) = [];
        fprintf('Resampled..\n');
        fflush(stdout);

        % Safety checks
        assert(size(sample.Pi,1) == sample.K);
        assert(size(sample.Pi,2) == sample.K+1);
        assert(sample.K == length(sample.Beta) - 1);
        assert(min(min(sample.Pi)) >= 0);
        assert(sample.K == max(sample.S));
        
        % Prepare next iteration.
        stats.alpha0(iter) = sample.alpha0;
        stats.gamma(iter) = sample.gamma;
        stats.K(iter) = sample.K;
        stats.jll(iter) = iHmmJointLogLikelihood(sample.S, Y, sample.Beta, sample.alpha0, hypers.H);
        stats.vi(iter) = getVI(STrue, sample.S);
        if(useNullState && ~nullDone)
%            nullIndexes = find(sample.S == 2);
%            nullTokens = Y(nullIndexes);
%            stats.nullIndexes{iter} = nullIndexes;
%            stats.nullTokens{iter} = nullTokens;
        end
        fprintf('Iteration: %d: K = %d, alpha0 = %f, gamma = %f, JL = %f, VI = %f, NS=%d.\n', ...
            iter, sample.K, sample.alpha0,sample. gamma, stats.jll(iter), stats.vi(iter), useNullState);
        fflush(stdout);
        
        if iter >= numb && mod(iter-numb, numi) == 0
            S{end+1} = sample;
        end
        
        if(useNullState && ~nullDone)
            fprintf('Null state used by %d tokens\n', length(find(sample.S==2)));
            fflush(stdout);
        end
        
        iter = iter + 1;
    else
        fprintf('Wasted computation as there were no paths through the iHMM.\n');
    end
end

