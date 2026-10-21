function [coeffs, info] = construct_aop_coeffs(mu, maxOrder, tol)
% Constructs adaptive orthonormal polynomial coefficients.
% Inputs:
%   mu : column vector [mu_2; mu_4; ...; mu_2P]
%   maxOrder : highest nonlinear order (should be an odd integer)
%   tol : numerical tolerance (to safely check if a variable is zero)
%
% Outputs:
%   coeffs : cell array of coefficient row vectors
%   info : diagnostic information

% set default value for tol
if nargin < 3
    tol = 1e-6;
end

mu = mu(:);

if mod(maxOrder, 2) == 0
    error("maxOrder must be an odd integer.");
end

numBasisRequested = (maxOrder + 1) / 2;

coeffs = cell(numBasisRequested, 1);

%% First-order basis:

coeffs{1} = 1 / sqrt(mu(1));

validNumBasis = 1;

Minv = 1 / mu(1);

%% Higher-order AOP coefficients.

for p = 2:numBasisRequested

    % u_p = [mu_2p, mu_(2p+2), ..., mu_(4p-4)]^T
    uIdxStart = p;
    uIdxEnd = 2*p - 2;
    u = mu(uIdxStart:uIdxEnd);

    % Unnormalized monic polynomial coefficient vector.
    cbar = -Minv * u;
    c = [cbar(:).', 1];

    % Normalization:
    normSq = basis_power(c, mu);

    if normSq <= tol
        warning("AOP basis construction stopped at p = %d because the polynomial norm is near zero.", p);
        break;
    end

    coeffs{p} = c / sqrt(normSq);
    validNumBasis = p;

    % No need to update M inverse after the final requested basis.
    if p == numBasisRequested
        break;
    end

    % Schur-complement update:
    % The inverse moment matrix is updated recursively using the
    % Schur-complement expression from Algorithm 1.
    r = mu(2*p - 1);

    utilde = Minv * u;
    s = r - u.' * utilde;

    if abs(s) <= tol
        warning("AOP basis construction stopped after p = %d because the Hankel update became singular.", p);
        break;
    end

    Minv = [Minv + (utilde * utilde.') / s,  -utilde / s; ...
        -utilde.' / s,                    1 / s];

end

coeffs = coeffs(1:validNumBasis);

info.requestedNonlinearOrder = maxOrder;
info.requestedNumBasis = numBasisRequested;
info.validNumBasis = validNumBasis;
info.validNonlinearOrder = 2*validNumBasis - 1;
info.isFullOrder = validNumBasis == numBasisRequested;
info.tolerance = tol;

end


function normSq = basis_power(c, mu)
% Compute E{|phi_p(x;c)|^2}

c = c(:).';

momentCoeffs = conv(conj(c), c);

requiredMoments = numel(momentCoeffs);

if numel(mu) < requiredMoments
    error("Not enough moments supplied for normalization.");
end

normSq = real(momentCoeffs * mu(1:requiredMoments));

end