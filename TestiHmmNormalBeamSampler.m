% Test the iHMM Gibbs sampler with normal output.
T = 1000;                        % Length of HMM
K = 4;                          % Number of states

stream = RandStream('mt19937ar','seed',21);
RandStream.setDefaultStream(stream);

% Parameters for HMM which generates data.
A = [ 0.0 0.5 0.5 0.0;
      0.0 0.0 0.5 0.5;
      0.5 0.0 0.0 0.5;
      0.5 0.5 0.0 0.0 ];
E.mu = [-4.0; 2.0; -1.0; 3.0];
E.sigma2 = [0.5; 0.5; 0.5; 0.5];
pi = [1.0; zeros(K-1,1)];

% Generate data.
[Y, STrue] = HmmGenerateData(1, T, pi, A, E, 'normal');

% Sample states using the iHmm Gibbs sampler.
tic
hypers.alpha0_a = 4;
hypers.alpha0_b = 1;
hypers.gamma_a = 3;
hypers.gamma_b = 6;
hypers.sigma2 = 1.5;
hypers.mu_0 = 0.0;
hypers.sigma2_0 = 1.0;
tic
[S, stats] = iHmmNormalSampleBeam(Y, hypers, 500, 1, 1, ceil(rand(1,T) * 10));
toc

figure(2)
subplot(3,2,1)
plot(stats.K)
title('K')
subplot(3,2,2)
plot(stats.jll)
title('Joint Log Likelihood')
subplot(3,2,3)
plot(stats.alpha0)
title('alpha0')
subplot(3,2,4)
plot(stats.gamma)
title('gamma')
subplot(3,2,5)
imagesc(SampleTransitionMatrix(S{1}.S, zeros(1,S{1}.K))); colormap('Gray');
title('Transition Matrix')