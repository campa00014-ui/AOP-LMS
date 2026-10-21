function test_aop_lms_windowing()
% Check causal learning, reduced-order 4-QAM, and retained branch weights.

projectRoot = fileparts(fileparts(mfilename("fullpath")));
run(fullfile(projectRoot, "startup.m"));
rng(70001);
orders = [16 64 4 256 64];
lengths = [2200 5500 3300 4400 2200];
x = aoplms.signals.generate_nonstationary_qam_signal(orders, lengths, 1);
received = filter([0.8; 0.2i; -0.1], 1, 3*x ./ (1 + 0.09*abs(x).^2));
steps = [0.017; 0.0045; 0.0015];
numTaps = 3;
Nmax = 55;
Nint = 1100;
result = aoplms.algorithms.run_aop_lms_nonstationary( ...
    x, received, 5, steps, numTaps, Nmax, Nint);
d = result.diagnostics;
assert(all(isfinite(result.error)));
assert(isequal(result.estimate(1:Nmax), zeros(Nmax, 1)));
assert(all(d.activeNumBranches(1:55) == 0));
assert(all(d.activeNumBranches(56:7755) == 3));
assert(all(d.activeNumBranches(7756:11055) == 1));
assert(all(d.activeNumBranches(11056:end) == 3));
assert(isequal(d.switchIndices, (1156:1100:16556).'));
for k = 7:9
    assert(isscalar(d.updateCoeffs{k}));
    assert(abs(d.updateCoeffs{k}{1} - 1) < 1e-12);
    assert(d.updateInfo{k}.validNumBasis == 1);
end

% Each learned basis is orthonormal on its own 55-sample learning window.
for slot = 1:16
    startIdx = 1 + (slot-1)*Nint;
    if slot == 1
        coeffs = d.initialCoeffs;
    else
        coeffs = d.updateCoeffs{slot-1};
    end
    Phi = aoplms.basis.evaluate_aop_basis(x(startIdx:startIdx+Nmax-1), coeffs);
    gram = Phi' * Phi / Nmax;
    assert(max(abs(gram - eye(numel(coeffs))), [], "all") < 1e-10);
end

% At each modulation's first learned basis, predict with the preceding weights.
% This detects any global reset or inadvertent coordinate transformation.
activationSamples = [1156 2256 3356 7756 11056 12156 15456 16556];
for n = activationSamples
    previous = aoplms.algorithms.run_aop_lms_nonstationary( ...
        x(1:n-1), received(1:n-1), 5, steps, numTaps, Nmax, Nint);
    slot = floor((n-1) / Nint) + 1;
    coeffs = d.updateCoeffs{slot-1};
    tappedBasis = zeros(numTaps, 3);
    tappedBasis(:, 1:numel(coeffs)) = ...
        aoplms.basis.evaluate_aop_basis(x(n:-1:n-numTaps+1), coeffs);
    expectedEstimate = sum(sum(conj(previous.weights) .* tappedBasis));
    assert(abs(result.estimate(n) - expectedEstimate) < 1e-12);
    assert(any(abs(previous.weights) > 0, "all"));
end

before4 = aoplms.algorithms.run_aop_lms_nonstationary( ...
    x(1:7700), received(1:7700), 5, steps, numTaps, Nmax, Nint);
after4 = aoplms.algorithms.run_aop_lms_nonstationary( ...
    x(1:11055), received(1:11055), 5, steps, numTaps, Nmax, Nint);
assert(isequal(before4.weights(:, 2:3), after4.weights(:, 2:3)));
assert(any(abs(before4.weights(:, 2:3)) > 0, "all"));

% Once 4-QAM is learned, only its linear branch contributes to the output.
n = 8000;
previous = aoplms.algorithms.run_aop_lms_nonstationary( ...
    x(1:n-1), received(1:n-1), 5, steps, numTaps, Nmax, Nint);
linearEstimate = sum(conj(previous.weights(:, 1)) .* x(n:-1:n-numTaps+1));
assert(abs(result.estimate(n) - linearEstimate) < 1e-12);

% A stationary 4-QAM run never constructs or adapts higher-order branches.
x4 = aoplms.signals.generate_qam_signal(2200, 4, 1);
received4 = filter([0.8; 0.2i; -0.1], 1, x4);
result4 = aoplms.algorithms.run_aop_lms_nonstationary( ...
    x4, received4, 5, steps, numTaps, Nmax, Nint);
assert(all(result4.diagnostics.activeNumBranches(56:end) == 1));
assert(all(result4.weights(:, 2:3) == 0, "all"));
fprintf("Windowed 4-QAM, empirical orthonormality, branch reactivation, weight continuity, and causal 55/1100 learning checks passed.\n");

end
