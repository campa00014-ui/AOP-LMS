function result = run_aop_lms_lut( ...
    x, received, modOrders, lut, branchStepSizes, numTaps)
% Run AOP-LMS using an offline QAM LUT and known modulation orders.
% modOrders gives M at each sample, or a scalar M for a stationary signal.
% x must use the same unit-average-power constellation as the LUT.

x = x(:);
received = received(:);
modOrders = modOrders(:);
branchStepSizes = branchStepSizes(:);
numSamples = numel(x);
numBranches = (lut.maxOrder + 1) / 2;

if numel(received) ~= numSamples
    error("x and received must have the same number of samples.");
end
if isscalar(modOrders)
    modOrders = repmat(modOrders, numSamples, 1);
end
if numel(modOrders) ~= numSamples
    error("modOrders must be a scalar or contain one QAM order per sample.");
end
if numel(branchStepSizes) ~= numBranches
    error("branchStepSizes must contain one value per requested AOP branch.");
end
if numTaps < 1 || numTaps ~= floor(numTaps)
    error("numTaps must be a positive integer.");
end

weights = zeros(numTaps, numBranches);
tappedBasis = zeros(numTaps, numBranches);
estimate = zeros(numSamples, 1);
residual = zeros(numSamples, 1);
activeNumBranches = zeros(numSamples, 1);
switchSamples = find([true; diff(modOrders) ~= 0]);

for n = 1:numSamples
    if n == 1 || modOrders(n) ~= modOrders(n-1)
        entry = aoplms.lut.get_qam_lut_entry(lut, modOrders(n));
        coeffs = entry.coeffs;
        numActive = numel(coeffs);

        % Evaluate all available delays in the selected basis, as in v6.
        % The weights stay unchanged. No moments or coefficients are learned.
        numAvailable = min(n, numTaps);
        tapIndices = n:-1:(n - numAvailable + 1);
        tappedBasis = zeros(numTaps, numBranches);
        tappedBasis(1:numAvailable, 1:numActive) = ...
            aoplms.basis.evaluate_aop_basis(x(tapIndices), coeffs);
    else
        if numTaps > 1
            tappedBasis(2:end, :) = tappedBasis(1:end-1, :);
        end
        tappedBasis(1, :) = 0;
        tappedBasis(1, 1:numActive) = ...
            aoplms.basis.evaluate_aop_basis(x(n), coeffs);
    end
    activeNumBranches(n) = numActive;

    % Use the same output, error, and LMS update as run_aop_lms.m.
    estimate(n) = sum(sum(conj(weights) .* tappedBasis));
    residual(n) = received(n) - estimate(n);
    weights = weights + tappedBasis .* branchStepSizes.' .* conj(residual(n));
end

% During 4-QAM, inactive columns are zero and their weights are held fixed.
% Those saved weights are used again when a higher-order basis is selected.
result.error = residual;
result.estimate = estimate;
result.weights = weights;
result.branchStepSizes = branchStepSizes;
result.numTaps = numTaps;
result.nonlinearOrder = lut.maxOrder;
result.switchSamples = switchSamples;
result.activeNumBranches = activeNumBranches;

end
