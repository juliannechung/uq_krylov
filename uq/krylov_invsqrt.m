function [xm12,relres] = krylov_invsqrt(A,G,b,maxiter,tol)
%
% [xm12,relres] = krylov_invsqrt(A,G,b,maxiter,tol)
%
% This function computes A^{-1/2}b using Lanczos approach
% described in Algorithm 2.3 of
%
%   "Efficient Krylov Subspace Methods for uncertainty quantification in
%       large Bayesian Linear Inverse Problems"
%           - Saibaba, Chung, and Petroske, 2019
%
% Inputs:
%   A (n x n) - Sparse matrix or funMat type
%   G (n x n) - Sparse matrix or funMat type. Preconditioner, such that G'*G = A^{-1}
%   b (n x 1) - right hand side
%   maxiter   - maximum number of Lanczos iterations
%         tol - tolerance for stopping
%
% Outputs:
%        xm12 - approximation of A^{-1/2} b
%      relres - difference between successive iterations

n = size(b,1);
nrmb = norm(b);

%Initialize Lanczos quantities
V = zeros(n,maxiter);
T = zeros(maxiter+1,maxiter+1);

%First step
vj = b/nrmb;
vjm1 = b*0;
beta = 0;

relres = zeros(maxiter,1);
xp = 0*b;
for j = 1:maxiter
  V(:,j) = vj;
  wj = G*(A*(G'*vj));
  alpha = wj'*vj;
  wj = wj - alpha*vj -beta*vjm1;
  beta = norm(wj);
  
  %Set vectors for new iterations
  vjm1 = vj;
  vj = wj/beta;
  
  %Reorthogonalize vj (CGS2) 
  vj = vj - V(:,1:j)*(V(:,1:j)'*vj);  vj = vj/norm(vj);
  vj = vj - V(:,1:j)*(V(:,1:j)'*vj);  vj = vj/norm(vj);
  
  %Set the tridiagonal matrix
  T(j,j) = alpha;
  T(j+1,j) = beta; T(j,j+1) = beta;
  
  %Compute partial Lanczos solution
  Tk = T(1:j,1:j);    Vk = V(:,1:j);
  e1 = zeros(j,1);  e1(1) = nrmb;
  
  T12 = sqrtm(Tk);
  xm12 = G'*(Vk*(T12\e1));
  
  %Check differences b/w successive iterations
  relres(j) = norm(xm12-xp)/norm(xm12);
  if  relres(j) < tol
    relres = relres(1:j);
    break
  else
    xp = xm12;
  end
  
end

end