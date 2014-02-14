function [sorted,indices] = getSortedWords(S,Y,pos_idx)

hid = S{end}.S;
idx = 1:max(Y);
pos_inds = find(hid==pos_idx);
pos_words = Y(pos_inds);
match_mat = (pos_words==idx);
c_w_giv_p = sum(match_mat);
c_w_giv_p = c_w_giv_p ./ sum(c_w_giv_p);
[sorted,indices] = sort(c_w_giv_p);

