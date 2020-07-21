% Contents.m
% 
%   This folder contains MATLAB code to accompany the paper:
%
%   "Efficient Krylov Subspace Methods for uncertainty quantification in
%       large Bayesian Linear Inverse Problems" 
%           - Saibaba, Chung, and Petroske, 2019
%
% These codes require the following toolboxes:
%     AIRTOOLS: https://github.com/jakobsj/AIRToolsII
%     IRTools: https://github.com/jnagy1/IRtools
%     genHyBR: https://github.com/juliannechung/genHyBR
%
%% Example scripts.
%
% To run the following example scripts, first set paths to the 
%    above toolboxes and folders.
%
% EX_Prec.m - demonstrates preconditioner use (Figure 2 in paper)
%
% EX_PAT.m  - demonstrates sampling for the PAT example (Figure 3 in paper)
%
%% Supporting files.
% Sampling:
%   krylov_sqrt.m    - computes A^{1/2}b using Lanczos approach
%   krylov_invsqrt.m - computes A^{-1/2}b using Lanczos approach
%   sampling.m       - computes samples from the approximate posterior distribution
%                           This corresponds to Method 1 in the paper.
%   postsample.m     - computes samples from the posterior distribution
%                           This corresponds to Method 2 in the paper.
%