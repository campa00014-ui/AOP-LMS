function report = check_orthonormality(Phi, tol)
% Check orthonormality of a basis matrix by computing its Gram matrix.
% Inputs:
%   Phi : NxK basis matrix
%   tol : tolerance for the max-error pass/fail check
% Output:
%   report : struct with Gram matrix and max error

% set default value for tol
if nargin < 2
    tol = 1e-5;
end

G = (Phi' * Phi) / size(Phi, 1);
I = eye(size(G));

errorMatrix = abs(G - I);
maxError = max(errorMatrix(:));
passed = maxError < tol;

report.gramMatrix = G;
report.errorMatrix = errorMatrix;
report.maxError = maxError;
report.tolerance = tol;
report.passed = passed;

end
