function test_ofdm_signal()
% Unit tests for OFDM signal generation functions.

projectRoot = fileparts(fileparts(mfilename("fullpath")));
run(fullfile(projectRoot, "startup.m"));

%% 1. Length, shape, and column vector checks
for N = [1, 64, 100, 2200, 17600]
    x = aoplms.signals.generate_ofdm_signal(N, 16, 1, 64, 0);
    assert(length(x) == N, "Output length must match requested N = %d", N);
    assert(iscolumn(x), "Output must be a column vector.");
end

% Check N = 0 edge case
x0 = aoplms.signals.generate_ofdm_signal(0, 16, 1, 64, 0);
assert(isempty(x0) && iscolumn(x0), "N=0 should return empty column vector.");

%% 2. Average power normalization across all MCS orders
rng(12345);
Npower = 200000;
for M = [4 16 64 256]
    x = aoplms.signals.generate_ofdm_signal(Npower, M, 1, 64, 0);
    pwr = mean(abs(x).^2);
    assert(abs(pwr - 1.0) < 0.02, ...
        "OFDM time-domain average power for %d-QAM should be ~1.0, got %.4f", M, pwr);
end

% Custom power scaling
x2 = aoplms.signals.generate_ofdm_signal(Npower, 16, 2.5, 64, 0);
pwr2 = mean(abs(x2).^2);
assert(abs(pwr2 - 2.5) < 0.05, "OFDM average power should scale with power parameter.");

%% 3. Cyclic prefix support
xCp = aoplms.signals.generate_ofdm_signal(Npower, 64, 1, 64, 16);
assert(length(xCp) == Npower, "CP waveform length must match requested N.");
assert(abs(mean(abs(xCp).^2) - 1.0) < 0.02, "CP waveform power should be ~1.0.");

%% 4. Deterministic seed reproducibility
rng(999);
x_run1 = aoplms.signals.generate_ofdm_signal(500, 16, 1, 64, 0);
rng(999);
x_run2 = aoplms.signals.generate_ofdm_signal(500, 16, 1, 64, 0);
assert(isequal(x_run1, x_run2), "Same seed must produce identical OFDM waveforms.");

%% 5. Nonstationary OFDM generation: exact segment lengths and boundaries
orders = [16 64 4 256 64];
lengths = [2200 5500 3300 4400 2200];
xNonstat = aoplms.signals.generate_nonstationary_ofdm_signal(orders, lengths, 1, 64, 0);
assert(length(xNonstat) == sum(lengths), "Nonstationary length must equal sum of segment lengths.");
assert(iscolumn(xNonstat), "Nonstationary output must be a column vector.");

% Nonstationary with CP > 0
xNonstatCp = aoplms.signals.generate_nonstationary_ofdm_signal(orders, lengths, 1, 64, 16);
assert(length(xNonstatCp) == sum(lengths), "Nonstationary with CP length must equal sum of segment lengths.");

% Empty nonstationary signal
xEmpty = aoplms.signals.generate_nonstationary_ofdm_signal([], [], 1, 64, 0);
assert(isempty(xEmpty) && iscolumn(xEmpty), "Empty orders/lengths should return empty column vector.");

%% 6. Input validation error checks
assertThrows(@() aoplms.signals.generate_ofdm_signal(-1, 16, 1, 64, 0));
assertThrows(@() aoplms.signals.generate_ofdm_signal(100, 8, 1, 64, 0)); % invalid M
assertThrows(@() aoplms.signals.generate_ofdm_signal(100, 16, -1, 64, 0)); % invalid power
assertThrows(@() aoplms.signals.generate_ofdm_signal(100, 16, 1, 0, 0)); % invalid nfft
assertThrows(@() aoplms.signals.generate_ofdm_signal(100, 16, 1, 64, -1)); % invalid cpLen

% Nonstationary input validation
assertThrows(@() aoplms.signals.generate_nonstationary_ofdm_signal([16 64], [100], 1, 64, 0)); % mismatched
assertThrows(@() aoplms.signals.generate_nonstationary_ofdm_signal([16 64], [100 -50], 1, 64, 0)); % negative length
assertThrows(@() aoplms.signals.generate_nonstationary_ofdm_signal([16 64], [100 50.5], 1, 64, 0)); % non-integer length

fprintf("All test_ofdm_signal tests passed successfully.\n");

end

function assertThrows(fh)
    threw = false;
    try
        fh();
    catch
        threw = true;
    end
    assert(threw, "Expected error was not thrown.");
end
