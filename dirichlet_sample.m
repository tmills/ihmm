function dir = dirichlet_sample(alpha)
% Samples a dirichlet distributed random vector.

dir = gamrnd(alpha, 1);
dir = dir ./ sum(dir);