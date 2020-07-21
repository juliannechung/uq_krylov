% EX_Prec.m
%
% Comparison of preconditioned vs. unpreconditioned convergence
% for Lanczos based sampling from a large Gaussian as described in
%
%   "Efficient Krylov Subspace Methods for uncertainty quantification in
%       large Bayesian Linear Inverse Problems"
%           - Saibaba, Chung, and Petroske, 2019
%
%  Here we sample from N(0,Q) where Q is the prior covariance matrix, using
%  preconditioners defined by fractional differential operators.
%
rng(0)

%% Test preconditioner
n = 300 ;
xmin = [0 0];           %Coordinates of left corner
xmax = [1 1];           %Coordinates of right corner
nvec = [n, n];         %Number of points. Here this is a 300 x 300 grid

b = randn(prod(nvec),1);

%% Set up Matern covariance matrix and preconditioner
nu = 0.5;
ell = .25;
k = @(r) matern(r,nu,ell);

% Additional parameters governing length scales.
theta = [1.0 1.0];      %For now set them as isotropic

% Build row/column of the Toeplitz matrix
Qr = createrow(xmin,xmax,nvec,k,theta);
Q = @(x) toeplitzproduct(x, Qr, nvec);
Q = funMat(Q,Q,nvec.^2);

% Preconditioner for nu = .5
h = 1./n;
P = gallery('poisson',n)/(h.^2);
G =  chol(P,'upper');

%% Save the history of relative differences
tic
[~,diffp] = krylov_sqrt(Q,G,b,300,1.e-6); % Preconditioned
toc

tic
[~,diff] = krylov_sqrt(Q,speye(n.^2),b,300,1.e-6); % Unpreconditioned
toc

%% Create figure
figure,
semilogy(1:length(diffp), diffp, '-r','LineWidth',4.0); hold on
semilogy(1:length(diff), diff, ':k','LineWidth',4.0);

xlim([0,300])
xlabel('Iteration')
set(gca,'FontSize',16)

legend({'Preconditioned','Unpreconditioned'},'FontSize',18)
ylabel('$\tilde{e}_K$', 'FontSize', 18,'Interpreter' , 'latex')

title(strcat('\nu = ',num2str(nu)))


