function [samples,iterc] = sampling(lbd, Q, G, eps, params)
%
% [samples,iterc] = sampling(lbd, Q, G, eps, params)
%
% Method 1: Compute samples from the approximate posterior distribution 
%   as described in
%   "Efficient Krylov Subspace Methods for uncertainty quantification in
%       large Bayesian Linear Inverse Problems"
%           - Saibaba, Chung, and Petroske, 2019
%
% Inputs:
%             lbd - output structure from genHyBR
%               Q - prior covariance matrix
%               G - preconditioner
%    eps (n x ns) - random vectors drawn from N(0,I)
% params (struct) - parameters for iterative solver including 
%                       maxiter and tol
%
% Outputs:
%     samples (n x ns) - Samples from N(0,\hat\Gamma_post)
%       iterc (ns x 1) - Iteration count for each sample
%

%Set maximum iterations and tolerance for the iterative solver
if isfield(params,'maxiter')
    maxiter = params.maxiter;
else
    maxiter = 200;
end

if isfield(params,'tol')
    tol = params.tol;
else
    tol = 1.e-6;
end

k = lbd.iterations;
Bk = lbd.B(1:k+1,1:k);
V  = lbd.V(:,1:k);
QV  = lbd.QV(:,1:k);

n = size(QV,1);
ns = size(eps,2);
lambda = sqrt(lbd.alpha);

%Counts the number of iterations
iterc = zeros(ns,1);

%Precompute the low-rank approximation
f = @(x) G*(Q*(G'*x));   I = speye(n);
Qh = funMat(f,f,[n,n]);
QhVk = 0*V;

precomp = 0;
for i = 1:k
    z =  G'\V(:,i);
    [x12,relres] = krylov_sqrt(Qh, I, z, maxiter, tol);
    QhVk(:,i) = x12;
    precomp = precomp + size(relres,1);
end
fprintf(' Total number of iterations taken for precomputation was %g\n', precomp);

[~,r] = qr(Bk/lambda,0);
[v,d] = lowrank(QhVk*r');

d12 = woodbury(d,'invsqrt');

samples = 0*eps;
for i = 1:ns
    b = eps(:,i) + v*(diag(d12)*(v'*eps(:,i)));
    [x12,relres] = krylov_sqrt(Q,G,b,maxiter,tol);
    samples(:,i) = x12;
    iterc(i) = size(relres,1);
end
samples = samples/lambda;

end

function [V,D] = lowrank(U)
% Computes low-rank representation
%   VDV' = UU'
[q,r] = qr(U,0);
[v,d] = eig(r*r');
V = q*v; D = diag(d);
end

function dh = woodbury(d, method)
% This function is used to evaluate the Woodbury identity.
% dh = woodbury(d, method)
% Input: Diagonal entries in d
%         method options: 'inv', 'sqrt', 'invsqrt'
if strcmp(method, 'inv')
    dh = -d./(1+d);
elseif strcmp(method, 'sqrt')
    dh = -1+sqrt(1+d);
elseif strcmp(method, 'invsqrt')
    dh = -d./(1+d);
    dh = -1+sqrt(1+dh);
end
end