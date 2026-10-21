function result = run_aop_lms(x, received, coeffs, branchStepSizes, numTaps)
% Run fixed-basis AOP-LMS with one static step size per branch.

x = x(:);
received = received(:);
branchStepSizes = branchStepSizes(:);

if numel(x) ~= numel(received)
    error("x and received must have the same number of samples.");
end

basis = aoplms.basis.evaluate_aop_basis(x, coeffs);
numSamples = numel(x);
numBranches = size(basis, 2);

if numel(branchStepSizes) ~= numBranches
    error("branchStepSizes must contain one value per AOP branch.");
end

if numTaps < 1 || numTaps ~= floor(numTaps)
    error("numTaps must be a positive integer.");
end

weights = zeros(numTaps, numBranches);
tappedBasis = zeros(numTaps, numBranches);
estimate = zeros(numSamples, 1);
residual = zeros(numSamples, 1);

for n = 1:numSamples
    if numTaps > 1
        tappedBasis(2:end, :) = tappedBasis(1:end-1, :);
    end
    tappedBasis(1, :) = basis(n, :);

    % yhat[n] = h[n]^H phi[n]
    estimate(n) = sum(sum(conj(weights) .* tappedBasis));

    % e[n] = y[n] - yhat[n]
    residual(n) = received(n) - estimate(n);

    % h_p[n+1] = h_p[n] + mu_p phi_p[n] conj(e[n])
    weights = weights + tappedBasis .* branchStepSizes.' .* conj(residual(n));
end

result.error = residual;
result.estimate = estimate;
result.weights = weights;
result.branchStepSizes = branchStepSizes;
result.numTaps = numTaps;

end
