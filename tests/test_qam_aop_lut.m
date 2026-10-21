function test_qam_aop_lut()
% Check Table I, exact orthonormality, and the LUT LMS switching behavior.

projectRoot = fileparts(fileparts(mfilename("fullpath")));
run(fullfile(projectRoot, "startup.m"));
lut = aoplms.lut.build_qam_aop_lut(5);
orders = [4 16 64 256];
tableMoments = [1 1 1 1; 1 1.32 1.96 3.1248; ...
    1 1.381 2.2258 3.9630; 1 1.3953 2.2922 4.1910];
thirdOrderNorms = [0.2176 0.3188 0.3453];
fifthOrderNorms = [0.0542 0.1421 0.1772];
fifthOrderMonic = [1.30 -2.47 1; 1.6268 -2.7898 1; 1.7189 -2.8747 1];

for k = 1:numel(orders)
    M = orders(k);
    entry = aoplms.lut.get_qam_lut_entry(lut, M);
    [mu, symbols] = aoplms.lut.compute_qam_moments(M, 5);
    assert(max(abs(entry.moments - mu)) < 1e-12);
    assert(max(abs(mu(1:4).' - tableMoments(k, :))) < 5.1e-5);

    % The explicit grid has the same constellation as the existing qammod path.
    referenceSymbols = qammod((0:M-1).', M, "UnitAveragePower", true);
    gridPoints = sortrows([real(symbols), imag(symbols)]);
    referencePoints = sortrows([real(referenceSymbols), imag(referenceSymbols)]);
    assert(max(abs(gridPoints - referencePoints), [], "all") < 1e-12);

    Phi = aoplms.basis.evaluate_aop_basis(symbols, entry.coeffs);
    gram = Phi' * Phi / M;
    assert(max(abs(gram - eye(numel(entry.coeffs))), [], "all") < 1e-10);
    assert(abs(entry.coeffs{1} - 1) < 1e-12);

    if M == 4
        assert(isscalar(entry.coeffs));
        assert(max(abs(mu - 1)) < 1e-12);
    else
        c2 = entry.coeffs{2};
        c3 = entry.coeffs{3};
        assert(abs(-c2(1) / c2(end) - tableMoments(k, 2)) < 5.1e-5);
        assert(abs(1 / c2(end)^2 - thirdOrderNorms(k-1)) < 5.1e-5);
        assert(abs(1 / c3(end)^2 - fifthOrderNorms(k-1)) < 5.1e-5);
        monicTolerance = 5.1e-5;
        if M == 16
            monicTolerance = 5.1e-3;  % Table I rounds these two coefficients to 2 decimals.
        end
        assert(max(abs(c3 / c3(end) - fifthOrderMonic(k-1, :))) < monicTolerance);
    end
end
fprintf("Table I and full-constellation orthonormality checks passed.\n");

%% A stationary LUT run must equal the existing fixed-basis LMS.
rng(123);
x = aoplms.signals.generate_qam_signal(120, 64, 1);
received = filter([0.8; 0.2i; -0.1], 1, 3*x ./ (1 + 0.09*abs(x).^2));
steps = [0.017; 0.0045; 0.0015];
fixed = aoplms.algorithms.run_aop_lms(x, received, lut.qam64.coeffs, steps, 3);
result = aoplms.algorithms.run_aop_lms_lut(x, received, 64, lut, steps, 3);
assert(max(abs(fixed.error - result.error)) < 1e-12);
assert(max(abs(fixed.weights - result.weights), [], "all") < 1e-12);

%% Check every switch using weights from the preceding sample.
segmentOrders = [16 64 4 256 64];
segmentLengths = [30 35 20 25 15];
x = aoplms.signals.generate_nonstationary_qam_signal(segmentOrders, segmentLengths, 1);
modOrders = repelem(segmentOrders, segmentLengths).';
received = filter([0.8; 0.2i; -0.1], 1, 3*x ./ (1 + 0.09*abs(x).^2));
result = aoplms.algorithms.run_aop_lms_lut(x, received, modOrders, lut, steps, 3);
starts = [1; 1 + cumsum(segmentLengths(1:end-1)).'];
assert(isequal(result.switchSamples, starts));
assert(all(result.activeNumBranches(66:85) == 1));
assert(all(result.activeNumBranches([1:65, 86:125]) == 3));

for k = 2:numel(starts)
    n = starts(k);
    previous = aoplms.algorithms.run_aop_lms_lut( ...
        x(1:n-1), received(1:n-1), modOrders(1:n-1), lut, steps, 3);
    entry = aoplms.lut.get_qam_lut_entry(lut, modOrders(n));
    currentTaps = zeros(3, 3);
    currentTaps(:, 1:numel(entry.coeffs)) = ...
        aoplms.basis.evaluate_aop_basis(x(n:-1:n-2), entry.coeffs);
    expectedEstimate = sum(sum(conj(previous.weights) .* currentTaps));
    assert(abs(result.estimate(n) - expectedEstimate) < 1e-12);
end

beforeQam4 = aoplms.algorithms.run_aop_lms_lut( ...
    x(1:65), received(1:65), modOrders(1:65), lut, steps, 3);
afterQam4 = aoplms.algorithms.run_aop_lms_lut( ...
    x(1:85), received(1:85), modOrders(1:85), lut, steps, 3);
assert(isequal(beforeQam4.weights(:, 2:3), afterQam4.weights(:, 2:3)));
assert(all(isfinite(result.error)));

% The runtime needs coefficients only; stored moments can be removed.
lut.qam4 = rmfield(lut.qam4, "moments");
lut.qam16 = rmfield(lut.qam16, "moments");
lut.qam64 = rmfield(lut.qam64, "moments");
lut.qam256 = rmfield(lut.qam256, "moments");
withoutMoments = aoplms.algorithms.run_aop_lms_lut( ...
    x, received, modOrders, lut, steps, 3);
assert(isequal(result.error, withoutMoments.error));

%% The lower-order configurations and all-4-QAM case also run.
for P = [1 3]
    lowerLut = aoplms.lut.build_qam_aop_lut(P);
    lower = aoplms.algorithms.run_aop_lms_lut( ...
        x, received, modOrders, lowerLut, steps(1:(P+1)/2), 3);
    assert(all(isfinite(lower.error)));
end
x4 = aoplms.signals.generate_qam_signal(40, 4, 1);
received4 = filter([0.8; 0.2i; -0.1], 1, x4);
fixed4 = aoplms.algorithms.run_aop_lms(x4, received4, {1}, steps(1), 3);
result4 = aoplms.algorithms.run_aop_lms_lut(x4, received4, 4, lut, steps, 3);
assert(isequal(fixed4.error, result4.error));
assert(all(result4.weights(:, 2:3) == 0, "all"));
fprintf("Fixed-basis equivalence, weight continuity, 4-QAM, and LUT-only runtime checks passed.\n");

end
