function test_ofdm_aop_lut()
% Unit tests for OFDM AOP LUT construction, moments, coefficients, and LMS integration.

projectRoot = fileparts(fileparts(mfilename("fullpath")));
run(fullfile(projectRoot, "startup.m"));

%% 1. Build OFDM LUT and verify structure
maxOrder = 5;
lut = aoplms.lut.build_ofdm_aop_lut(maxOrder, 64, 0, 500000, 12345);

assert(lut.maxOrder == 5, "maxOrder should be 5.");
assert(isequal(lut.modOrders, [4 16 64 256]), "modOrders must be [4 16 64 256].");
assert(isfield(lut, "qam4") && isfield(lut, "qam16") && isfield(lut, "qam64") && isfield(lut, "qam256"));

%% 2. 4QAM-OFDM must NOT be restricted to first order
% Direct QAM has |x|=1 so raw 4QAM only has order 1.
% In OFDM, 4QAM time-domain signal has amplitude variation, so all 3 branches exist.
assert(numel(lut.qam4.coeffs) == 3, "4QAM OFDM LUT must have 3 basis branches for maxOrder=5.");
assert(lut.qam4.info.isFullOrder == 1, "4QAM OFDM basis should achieve full order.");

%% 3. Moments should be approximately Gaussian [1, 2, 6, 24, 120]
% For unit-variance circular complex Gaussian, E|x|^(2k) = k!
gaussianMoments = [1, 2, 6, 24, 120];
for M = lut.modOrders
    entry = aoplms.lut.get_qam_lut_entry(lut, M);
    mu = entry.moments;
    assert(abs(mu(1) - 1.0) < 0.01, "mu_2 should be ~1 for M=%d", M);
    assert(abs(mu(2) - 2.0) < 0.05, "mu_4 should be ~2 for M=%d", M);
    assert(abs(mu(3) - 6.0) < 0.3, "mu_6 should be ~6 for M=%d", M);
    assert(abs(mu(4) - 24.0) < 2.0, "mu_8 should be ~24 for M=%d", M);
    assert(abs(mu(5) - 120.0) < 15.0, "mu_10 should be ~120 for M=%d", M);
end

%% 4. Closeness of OFDM AOP coefficients to Ito-Hermite coefficients
ihCoeffs = {1; [-2 1] / sqrt(2); [6 -6 1] / sqrt(12)};
for M = lut.modOrders
    entry = aoplms.lut.get_qam_lut_entry(lut, M);
    c1 = entry.coeffs{1};
    c2 = entry.coeffs{2};
    c3 = entry.coeffs{3};
    assert(abs(c1 - 1.0) < 0.01, "Linear coefficient should be ~1.");
    assert(max(abs(c2 - ihCoeffs{2})) < 0.05, "Branch 2 should be close to IH basis.");
    assert(max(abs(c3 - ihCoeffs{3})) < 0.15, "Branch 3 should be close to IH basis.");
end

%% 5. Empirical orthonormality on training set and independent validation set
for M = lut.modOrders
    entry = aoplms.lut.get_qam_lut_entry(lut, M);
    % Check mathematical orthonormality using the estimated moments directly
    % Gram matrix from moments:
    numBasis = numel(entry.coeffs);
    gramFromMoments = zeros(numBasis, numBasis);
    for p = 1:numBasis
        for q = 1:numBasis
            cp = entry.coeffs{p}(:).';
            cq = entry.coeffs{q}(:).';
            polyProd = conv(fliplr(conj(cp)), cq);
            % phi_p*(x) phi_q(x) = sum_k polyProd(k) * |x|^(2k)
            % Gram(p,q) = sum_k polyProd(k) * mu_{2k}
            polyProdFull = conv(polyProd, [1 0]); % multiply by |x|^2
            % Actually, phi_p(x) = (sum c_{p,k} |x|^{2k}) * x
            % phi_p*(x) * phi_q(x) = (sum c_{p,k}* |x|^{2k}) * (sum c_{q,m} |x|^{2m}) * |x|^2
            cp_monic = entry.coeffs{p}(:).';
            cq_monic = entry.coeffs{q}(:).';
            convCoeffs = conv(conj(cp_monic), cq_monic);
            gramFromMoments(p, q) = real(convCoeffs * entry.moments(1:numel(convCoeffs)));
        end
    end
    assert(max(abs(gramFromMoments - eye(numBasis)), [], "all") < 1e-10, ...
        "Exact Gram matrix from moments must be identity for M=%d.", M);
end

% On independent validation set, check statistical deviation
rng(98765);
Nval = 200000;
for M = lut.modOrders
    entry = aoplms.lut.get_qam_lut_entry(lut, M);
    xVal = aoplms.signals.generate_ofdm_signal(Nval, M, 1, 64, 0);
    Phi = aoplms.basis.evaluate_aop_basis(xVal, entry.coeffs);
    gram = (Phi' * Phi) / Nval;
    diagErr = max(abs(diag(gram) - 1));
    offDiag = gram - diag(diag(gram));
    offDiagErr = max(abs(offDiag), [], 'all');
    assert(diagErr < 0.20, "Empirical diagonal variance error should be < 0.20 for M=%d", M);
    assert(offDiagErr < 0.10, "Empirical cross-correlation should be < 0.10 for M=%d", M);
end

%% 6. Test run_aop_lms_lut with OFDM signal
rng(111);
orders = [16 64 4 256 64];
lengths = [100 150 100 120 80];
xOfdmNonstat = aoplms.signals.generate_nonstationary_ofdm_signal(orders, lengths, 1, 64, 0);
modOrders = repelem(orders, lengths).';
received = filter([0.8; 0.2i; -0.1], 1, 3*xOfdmNonstat ./ (1 + 0.09*abs(xOfdmNonstat).^2));
steps = [0.017; 0.0045; 1.5e-4];
res = aoplms.algorithms.run_aop_lms_lut(xOfdmNonstat, received, modOrders, lut, steps, 3);
assert(all(isfinite(res.error)), "Residuals must be finite.");
assert(all(res.activeNumBranches == 3), "All 3 branches should be active even during 4-QAM under OFDM.");

% Test IH-LMS on OFDM signal
ihCoeffs = {1; [-2 1] / sqrt(2); [6 -6 1] / sqrt(12)};
resIh = aoplms.algorithms.run_aop_lms(xOfdmNonstat, received, ihCoeffs, steps, 3);
assert(all(isfinite(resIh.error)), "IH-LMS residuals must be finite on OFDM.");

% Test windowed AOP-LMS on OFDM signal
winSteps = [0.015; 0.0012; 1e-5];
resWin = aoplms.algorithms.run_aop_lms_nonstationary( ...
    xOfdmNonstat, received, 5, winSteps, 3, 50, 100);
assert(all(isfinite(resWin.error)), "Windowed AOP-LMS residuals must be finite on OFDM.");
% Verify that under OFDM 4-QAM, windowed AOP-LMS maintains all 3 branches (since not constant-modulus)
assert(all(resWin.diagnostics.activeNumBranches(51:end) == 3), ...
    "Windowed AOP-LMS under OFDM must maintain 3 active branches even during 4-QAM.");

%% 7. Lower order LUTs (P=1, P=3)
lut1 = aoplms.lut.build_ofdm_aop_lut(1, 64, 0, 50000);
assert(numel(lut1.qam4.coeffs) == 1);
lut3 = aoplms.lut.build_ofdm_aop_lut(3, 64, 0, 50000);
assert(numel(lut3.qam4.coeffs) == 2);

fprintf("All test_ofdm_aop_lut tests passed successfully.\n");

end
