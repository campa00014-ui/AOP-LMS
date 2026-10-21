function result = run_aop_lms_nonstationary( ...
    x, received, maxOrder, branchStepSizes, numTaps, ...
    numLearningSamples, updateInterval, options)
% Run AOP-LMS for non-stationary signals following Algorithm 2 of the paper.
%
% Inputs:
%   x                  : Transmit signal samples [Nx1]
%   received           : Received signal samples [Nx1]
%   maxOrder           : Highest nonlinearity order P (odd integer, e.g. 5)
%   branchStepSizes    : Step size vector mu_p for each branch [numBranches x 1]
%   numTaps            : Number of FIR filter taps L per branch
%   numLearningSamples : Number of samples Nmax used to learn moments (e.g. 55)
%   updateInterval     : Interval Nint at which basis is recomputed (scalar, e.g. 1100 or 3000),
%                        OR a vector of transition start sample indices (e.g. segmentStarts).
%   options            : Optional configuration struct (tol, etc.)
% The requested branches remain allocated when only a reduced basis exists.

x = x(:);
received = received(:);

numSamples = numel(x);
numBranches = (maxOrder + 1) / 2;

if numel(received) ~= numSamples
    error("x and received must have the same number of samples.");
end

if maxOrder < 1 || mod(maxOrder, 2) == 0
    error("maxOrder must be a positive odd integer.");
end

if numTaps < 1 || numTaps ~= floor(numTaps)
    error("numTaps must be a positive integer.");
end

if numLearningSamples < 1 || numLearningSamples ~= floor(numLearningSamples)
    error("numLearningSamples must be a positive integer.");
end

if isscalar(updateInterval)
    if updateInterval < numLearningSamples || updateInterval ~= floor(updateInterval)
        error("updateInterval must be an integer no smaller than numLearningSamples.");
    end
    intervalStarts = 1:updateInterval:numSamples;
else
    if ~isnumeric(updateInterval) || isempty(updateInterval)
        error("updateInterval must be a scalar integer or a numeric vector of transition indices.");
    end
    intervalStarts = sort(unique(floor(updateInterval(:).')));
    if intervalStarts(1) ~= 1
        intervalStarts = [1, intervalStarts];
    end
    if any(intervalStarts < 1) || any(intervalStarts > numSamples)
        error("Transition start indices in updateInterval must be between 1 and numSamples.");
    end
    diffs = [diff(intervalStarts), numSamples - intervalStarts(end) + 1];
    if any(diffs < numLearningSamples)
        error("Interval between transition starts must be at least numLearningSamples.");
    end
end

if nargin < 8 || isempty(options)
    options = struct();
end

if ~isfield(options, "tol")
    options.tol = 1e-6;
end

%% Determine interval / slot boundaries
% If a scalar is provided, statistical information is refreshed every updateInterval samples.
% If a vector of transition starts is provided, information is refreshed at those transitions.
numIntervals = numel(intervalStarts);

if isvector(branchStepSizes) && numel(branchStepSizes) == numBranches
    branchStepSizes = repmat(branchStepSizes(:), 1, numIntervals);
elseif ~isequal(size(branchStepSizes), [numBranches, numIntervals])
    error("branchStepSizes must have size [%d x 1] or [%d x %d].", ...
        numBranches, numBranches, numIntervals);
end

updateStartIndices = intervalStarts(2:end).';
numUpdates = numel(updateStartIndices);

%% Preallocate outputs and diagnostics
estimate = zeros(numSamples, 1);
residual = zeros(numSamples, 1);
weights = zeros(numTaps, numBranches);
tappedBasis = zeros(numTaps, numBranches);
basisVersion = zeros(numSamples, 1);
activeNumBranches = zeros(numSamples, 1);

updateMoments = zeros(maxOrder, max(1, numUpdates));
updateCoeffs = cell(max(1, numUpdates), 1);
updateInfo = cell(max(1, numUpdates), 1);
switchIndices = zeros(max(1, numUpdates), 1);

initialMoments = [];
initialCoeffs = {};
initialInfo = struct();
activeCoeffs = {};
activeBasisVersion = 0;
rebuildTappedBasis = false;

%% Run interval-based AOP-LMS adaptation
for k = 1:numIntervals
    startIdx = intervalStarts(k);
    if k < numIntervals
        endIdx = intervalStarts(k + 1) - 1;
    else
        endIdx = numSamples;
    end
    learnEndIdx = min(startIdx + numLearningSamples - 1, endIdx);

    currentSteps = branchStepSizes(:, k).';
    momentSums = zeros(maxOrder, 1);
    numMomentsCollected = 0;

    % Process samples in the current interval
    for n = startIdx:endIdx
        isLearning = n <= learnEndIdx;

        if isLearning
            absX = abs(x(n));
            momentSums = momentSums + absX .^ (2 * (1:maxOrder)).';
            numMomentsCollected = numMomentsCollected + 1;
        end

        if rebuildTappedBasis
            tappedBasis = build_tapped_basis(x, n, numTaps, activeCoeffs, numBranches);
            rebuildTappedBasis = false;
        elseif ~isempty(activeCoeffs)
            if numTaps > 1
                tappedBasis(2:end, :) = tappedBasis(1:end-1, :);
            end
            tappedBasis(1, :) = 0;
            tappedBasis(1, 1:numel(activeCoeffs)) = ...
                aoplms.basis.evaluate_aop_basis(x(n), activeCoeffs);
        end

        basisVersion(n) = activeBasisVersion;
        activeNumBranches(n) = numel(activeCoeffs);

        % yhat[n] = sum_p h_p^H phi_p[n]
        estimate(n) = sum(sum(conj(weights) .* tappedBasis));

        % e[n] = y[n] - yhat[n]
        residual(n) = received(n) - estimate(n);

        % Freeze the LMS weights while moments are learned causally.
        if ~isLearning
            % h_p[n+1] = h_p[n] + mu_p phi_p[n] conj(e[n])
            weights = weights + tappedBasis .* currentSteps .* conj(residual(n));
        end

        if n == learnEndIdx
            moments = momentSums / numMomentsCollected;

            % A constant-envelope window (including 4-QAM) has only phi_1.
            % Construct its linear basis directly, without singular higher orders.
            basisOrder = maxOrder;
            learningAmplitudes = abs(x(startIdx:learnEndIdx));
            if max(learningAmplitudes) - min(learningAmplitudes) <= options.tol
                basisOrder = 1;
            end
            [newCoeffs, info] = aoplms.basis.construct_aop_coeffs( ...
                moments, basisOrder, options.tol);

            % Keep however many independent branches the constructor returns.
            % Weights are unchanged; zero regressors freeze inactive branches.

            % Record diagnostics after the final learning sample is observed.
            if k == 1
                initialMoments = moments;
                initialCoeffs = newCoeffs;
                initialInfo = info;
            else
                upIdx = k - 1;
                updateMoments(:, upIdx) = moments;
                updateCoeffs{upIdx} = newCoeffs;
                updateInfo{upIdx} = info;
                switchIndices(upIdx) = min(n + 1, numSamples + 1);
            end

            activeCoeffs = newCoeffs;
            activeBasisVersion = k;
            rebuildTappedBasis = n < endIdx;
        end
    end
end

result.error = residual;
result.estimate = estimate;
result.weights = weights;
result.branchStepSizes = branchStepSizes;
result.numTaps = numTaps;
result.nonlinearOrder = maxOrder;

result.diagnostics.initialMoments = initialMoments;
result.diagnostics.initialCoeffs = initialCoeffs;
result.diagnostics.initialInfo = initialInfo;
result.diagnostics.updateStartIndices = updateStartIndices;
result.diagnostics.switchIndices = switchIndices;
result.diagnostics.updateMoments = updateMoments;
result.diagnostics.updateCoeffs = updateCoeffs;
result.diagnostics.updateInfo = updateInfo;
result.diagnostics.basisVersion = basisVersion;
result.diagnostics.activeNumBranches = activeNumBranches;
result.diagnostics.numLearningSamples = numLearningSamples;
result.diagnostics.updateInterval = updateInterval;

end


function tappedBasis = build_tapped_basis(x, n, numTaps, coeffs, numBranches)
% Rebuild the FIR regressor using the active basis for available taps.

tappedBasis = zeros(numTaps, numBranches);
numAvailable = min(n, numTaps);
rawTapIndices = n:-1:(n - numAvailable + 1);

tappedBasis(1:numAvailable, 1:numel(coeffs)) = aoplms.basis.evaluate_aop_basis( ...
    x(rawTapIndices), coeffs);

end
