% Test the iHMM Beam sampler on the expansive HMM experiment from the
% original iHMM paper [Beal et al. 2002]. The plotted transition and
% emission matrices should be equivalent up to permutation.

load('wsj_pos_120k.mat');
STrue = X(:,1)';
Y = X(:,2)';
%page_output_immediately();

burnin = 1000;    % was 500 at start
numSamples = 1; % default was 1
sampleIters = 500; % defualt was 1 (for one sample)

T = length(Y);                       % Length of HMM
startK = 20;                         % # of states
L = max(Y);                         % # of symbols in emission alphabet
fprintf('Starting the program!\n');
%fflush(stdout);

%stream = RandStream('mt19937ar','seed',21);
%RandStream.setDefaultStream(stream);

% Sample states using the iHmm Gibbs sampler.
%hypers.alpha0_a = 4;
%hypers.alpha0_b = 2;
hypers.alpha0 = 0.8;
%hypers.gamma_a = 3;
%hypers.gamma_b = 6;
hypers.gamma = 5.0;
%hypers.discount = 0.5;
hypers.H = ones(1,L) * 1.0;
%hypers.H(find(mod(1:startK,2) == 1),:) *= 0.01;

%hypers.H(find(mod(1:L,2) == 1)) *= 1.0;
%hypers.H(find(mod(1:L,2) == 0)) *= 0.01;

tic

S0 = 1 + ceil(rand(1,T) * (startK-1));  %% assign everything between 2-K
S0(find(Y==1)) = 1;                     %% overwrite the samples where we have a sentence-start

params = {};


[Sbase stats_base] = iHmmSampleBeam(Y, hypers, burnin, numSamples, sampleIters, S0, STrue, params);
toc
% Plot some stats
figure(1)
subplot(3,2,1)
plot(stats_base.K)
title('K')
subplot(3,2,2)
plot(stats_base.jll)
title('Joint Log Likelihood')
subplot(3,2,3)
plot(stats_base.alpha0)
title('alpha0')
subplot(3,2,4)
plot(stats_base.gamma)
title('gamma')
subplot(3,2,5)
plot(stats_base.vi)
title('VI')


tic
params.nullstate = 1;
[Snull stats_null] = iHmmSampleBeam(Y, hypers, burnin, numSamples, sampleIters, S0, STrue, params);
toc

% Plot some stats
figure(2)
subplot(3,2,1)
plot(stats_null.K)
title('K')
subplot(3,2,2)
plot(stats_null.jll)
title('Joint Log Likelihood')
subplot(3,2,3)
plot(stats_null.alpha0)
title('alpha0')
subplot(3,2,4)
plot(stats_null.gamma)
title('gamma')
subplot(3,2,5)
plot(stats_null.vi)
title('VI')
%hid = S{1}.S;
%idx = 1:max(hid);
%bins = zeros(max(hid),1);
%bins = sum(hid==idx', 2);

vi_random = getVI(STrue, S0);
base_vi = getVI(STrue, Sbase{end}.S);
null_vi = getVI(STrue, Snull{end}.S);

fprintf('VI of random sequence: %f\n', vi_random);
fprintf('VI of baseline induced sequence: %f\n', base_vi);
fprintf('VI of nullstate induced sequence: %f\n', null_vi);




