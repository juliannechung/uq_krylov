function [samples,iterc] = postsample(Q, A, G, lambda, eps, params)
%
% [samples,iterc] = postsample(Q,A,G,lambda,eps,params)
%
% Method 2: Compute samples from the posterior distribution as described in
%   "Efficient Krylov Subspace Methods for uncertainty quantification in
%       large Bayesian Linear Inverse Problems"
%           - Saibaba, Chung, and Petroske, 2019
%
% Inputs:
%      Q (n x n) - prior covariance matrix
%             A  - forward operator
%             G  - preconditioner for (Q+QA'AQ)
%         lambda - regularization parameter
%   eps (n x ns) - random vectors drawn from N(0,I)
% params (struct) - parameters for iterative solver including 
%                       maxiter and tol
%
% Outputs:
%     samples (n x ns) - Samples from N(0,\Gamma_post)
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

% Initialize parameters
samples = 0*eps;
[n, ns] = size(eps);
iterc = zeros(ns,1);

%Construct function handle for matvec with (lambda^2 Q + QA'AQ)
Mx = @(x) matvec(x,Q,A,lambda);
M = funMat(Mx,Mx,[n,n]);

% Iterate over each sample
for i = 1:ns
  [xm12,relres] = krylov_invsqrt(M, G, eps(:,i), maxiter, tol);
  samples(:,i) = Q*xm12;
  iterc(i) = size(relres,1);
end

end

function y = matvec(x,Q,A,lambda)
% Evaluate y = (lambda^2 * Q + Q A' A Q)x
z = Q*x;
y = Q*(A'*(A*z)) + (lambda.^2)*z;
end
