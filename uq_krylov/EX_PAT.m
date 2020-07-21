% EX_PAT.m
%
% This script sets up the the PAT 2D problem (PRshperical) from IRTools:
% We approximate the MAP estimate using genHyBR and then generate a sample
% from the prior and posterior distribution using the methods described in
%
%   "Efficient Krylov Subspace Methods for uncertainty quantification in
%       large Bayesian Linear Inverse Problems"
%           - Saibaba, Chung, and Petroske, 2019

rng(0)

%% Setup the forward problem
nx = 128; opts.phantomImage = 'smooth'; opts.sm = true;
[A,b,x_true, ProbInfo] = PRspherical(nx,opts);

%% Noise covariance
n = size(b,1);
nlevel = .02;
[N,sigma] = WhiteNoise(b(:),nlevel);
bn = b + N;
R = nlevel*speye(n,n);

%% Covariance kernel for prior
xmin = [0 0];           % Coordinates of left corner
xmax = [1 1];           % Coordinates of right corner
nvec = [nx, nx];         % Number of points
theta = [1, 1];
sigma2 = 1;
nu = .5; ell = .25; k = @(r) sigma2*matern(r,nu,ell);

Qr = createrow(xmin,xmax,nvec,k,theta);
Qfun = @(x) toeplitzproduct(x, Qr, nvec);
Q = funMat(Qfun,Qfun,nvec.^2);

%% Solve for the MAP and determine lambda (using WGCV)
maxit = 50;
solver = 'tikhonov';
input = HyBR_lsmrset('InSolv', solver,'RegPar','wgcv', 'x_true', x_true(:),'Iter', maxit,'Reorth','On');
[x_hy, output_hy] = genHyBR(A, bn(:), Q, R, input);
fprintf('genHyBR alpha WGCV: %.4e\n',output_hy.alpha)

%% Generate one prior sample
h = 1./nx;
P = gallery('poisson',nx)/(h.^2) +ell.^2*speye(nx.^2);
G = chol(P,'upper');
eps = randn(nx^2,1);
params.maxiter = 200;   params.tol = 1.e-6;
tic; [x12,relres] = krylov_sqrt(Q, G, eps, params.maxiter, params.tol);
psample = x12/sqrt(output_hy.alpha);

%% Method 1: Generate one sample from approximate posterior
disp('Method 1: Sample from Approximate Posterior')
[sample1,iterc1] = sampling(output_hy, Q, G, eps, params);
fprintf(' Iterations for sampling: %g\n',iterc1)

%% Method 2: Generate one sample from posterior
disp('Method 2: Sample from Posterior')
params.maxiter = 500;
alpha = output_hy.alpha;
[sample2,iterc2] = postsample(Q,A,G,alpha,eps, params);
fprintf(' Iterations for sampling: %g\n',iterc2)

%% Generate figure
x = [x_hy, psample, x_hy+sample1, x_hy+sample2];
xmin = min(min(x)); xmax = max(max(x));

fig = figure;
set(gcf, 'Position', [.5 .5 800 300]);

subplot(1,4,1), imagesc(reshape(x_hy,nx,nx)), axis image
colormap(parula); cmin = xmin; cmax = xmax; V=[cmin, cmax]; caxis(V);
title('MAP estimate')

subplot(1,4,2), imagesc(reshape(psample,nx,nx)), axis image,set(gca,'ytick',[])
colormap(parula); cmin = xmin; cmax = xmax; V=[cmin, cmax]; caxis(V);
title('Prior sample')

subplot(1,4,3), imagesc(reshape(x_hy+sample1,nx,nx)), axis image, set(gca,'ytick',[])
colormap(parula); cmin = xmin; cmax = xmax; V=[cmin, cmax]; caxis(V);
title('Method 1')

subplot(1,4,4), imagesc(reshape(x_hy+sample2,nx,nx)), axis image, set(gca,'ytick',[])
colormap(parula); cmin = xmin; cmax = xmax; V=[cmin, cmax]; caxis(V);
title('Method 2')
