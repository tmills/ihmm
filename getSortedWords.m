function [sorted,indices] = getSortedWords(S,Y,pos_idx)

Y = Y';
hid = S{end}.S;
idx = 1:max(Y);   %% all word indices
pos_inds = find(hid==pos_idx);   %% find all instances of given hmm state
pos_words = Y(pos_inds);         %% get the word indices for those positions
match_mat = (pos_words==idx);    %% build a matrix of counts 
c_w_giv_p = sum(match_mat);      %% sum across word dimension
c_w_giv_p = c_w_giv_p ./ sum(c_w_giv_p);   %% normalize
[sorted,indices] = sort(c_w_giv_p);  %% sort by probability

