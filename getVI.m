function [vi] =  getVI(Sreal, Shypoth)

fake_inds = find(Sreal==1);
Sreal(fake_inds) = [];
Shypoth(fake_inds) = [];

%% get entropy for hypothesis
idx = 1:max(Shypoth);
bins = zeros(max(Shypoth),1);
bins = sum(Shypoth==idx', 2);

H1 = entropy(bins ./sum(bins));

%% get entropy for real sequence
idx = 1:max(Sreal);
bins = zeros(max(Sreal),1);
bins = sum(Sreal==idx',2);

H2 = entropy(bins ./sum(bins));

%% Get mutual information between two sequences
P = zeros(max(Sreal), max(Shypoth));
for i=1:length(Sreal)
  P(Sreal(i), Shypoth(i)) = P(Sreal(i), Shypoth(i)) + 1;
end

Pxy = P ./ sum(P(:));
Px = P ./ sum(P);
Py = P ./ sum(P,2);

mi = Pxy .* (log(Pxy) - (log(Px) + log(Py)));
mi(find(isnan(mi))) = 0;
mi = sum(mi(:));

vi = H1 + H2 - 2*mi;


