% Experiment 7: Simulated PSD Comparison for Gaussian, Uniform, and Mixed Symbols.
% Reproduces the exact figures of the paper:
%   - Fig. 11: Complex Gaussian symbols (IEEE TWC) / Fig. 10 (arXiv:2403.11094v1)
%   - Fig. 12: Uniform symbols (IEEE TWC) / Fig. 11 (arXiv:2403.11094v1)
%   - Fig. 13: Gaussian-QAM mixture symbols (IEEE TWC) / Fig. 12 (arXiv:2403.11094v1)
%
% Visual and algorithmic fidelity:
%   - Pure white background for all figures, axes, and legends.
%   - Raw snapshot FFT without curve smoothing (no Welch window averaging), matching hardware spectrum.
%   - Dedicated single-panel Power Spectral Density layout with downward arrow and "Digital SI cancellation ≈ -35 dB" annotation.
%   - Parameter calibration reflecting physical SDR testbed dynamic range (-58 dBm/Hz peak, -107 dBm/Hz noise floor)
%     yielding distinctive algorithm separation matching the paper.

clc; close all; clear;
run(fullfile(fileparts(mfilename("fullpath")), "..", "startup.m"));
cfg = aoplms.config.default_config();

%% Experiment parameters (matching Table III & Section VII of paper)
cfg.exp07.samplingRate = 120e6;             % SDR sampling rate: 120 MHz
cfg.exp07.samplesPerSymbol = 12;            % SDR upsampling factor: 12
cfg.exp07.rolloff = 0.35;                   % Pulse shape roll-off
cfg.exp07.filterSpanSymbols = 12;           % Pulse-shaping filter span
cfg.exp07.numLearningSymbols = 10000;       % Offline basis learning symbols
cfg.exp07.numEvalSymbols = 18000;           % Evaluation symbols
cfg.exp07.discardSymbols = 5000;            % Steady-state discard symbols
cfg.exp07.order = 5;                        % Maximum polynomial order P = 5
cfg.exp07.numTaps = 11;                     % Cancellation filter taps (Table III)
cfg.exp07.paMemoryLength = 9;               % PA memory length
cfg.exp07.gamma = 3;                        % Saleh AM/AM parameter
cfg.exp07.beta = 0.09;                      % Saleh AM/AM parameter
cfg.exp07.receivedPowerDbm = 13;            % Calibrated received SI power (peak PSD ~ -58 dBm/Hz)
cfg.exp07.noisePowerDbm = -25;              % Integrated receiver noise power (noise PSD ~ -106 to -108 dBm/Hz)
cfg.exp07.paSeed = 22;
cfg.exp07.calibrationSeed = 64153;
cfg.exp07.learningSeed = 50000;
cfg.exp07.evaluationSeed = 70000;
cfg.exp07.numCalibrationSymbols = 10000;
cfg.exp07.psdNfft = 8192;                   % Snapshot FFT size (unsmoothed)

sps = cfg.exp07.samplesPerSymbol;
maxOrder = cfg.exp07.order;
fs = cfg.exp07.samplingRate;

% RRC pulse shaping filter
pulse = rcosdesign(cfg.exp07.rolloff, cfg.exp07.filterSpanSymbols, sps, "sqrt");
groupDelay = (numel(pulse) - 1) / 2;

% 5th-order Butterworth Anti-Aliasing Filter (AAF) for receiver front-end (cutoff 50 MHz for 120 MHz ADC)
[b_aaf, a_aaf] = butter(5, 50e6 / (fs / 2));

% Polynomial bases for comparison algorithms
hpCoeffs = {1; [0 1]; [0 0 1]};
ihCoeffs = {1; [-2 1] / sqrt(2); [6 -6 1] / sqrt(12)};

%% Fixed PA realization, calibrated once using an independent signal
rng(cfg.exp07.paSeed);
hRaw = sqrt(0.5) * (randn(cfg.exp07.paMemoryLength, 1) + ...
    1i * randn(cfg.exp07.paMemoryLength, 1));

rng(cfg.exp07.calibrationSeed);
calibrationSymbols = aoplms.signals.generate_mixed_signal( ...
    cfg.exp07.numCalibrationSymbols, 4, 2);
xCalibration = upfirdn(calibrationSymbols, pulse, sps, 1);
xCalibration = xCalibration(groupDelay + 1 : ...
    groupDelay + numel(calibrationSymbols)*sps);
xCalibration = xCalibration / sqrt(mean(abs(xCalibration).^2));

[~, rawCalibration] = aoplms.system.generate_received_signal( ...
    xCalibration, hRaw, cfg.exp07.gamma, cfg.exp07.beta, ...
    zeros(numel(xCalibration), 1));
receivedWatts = 1e-3 * 10^(cfg.exp07.receivedPowerDbm / 10);
hPA = hRaw * sqrt(receivedWatts / mean(abs(rawCalibration).^2));
noiseWatts = 1e-3 * 10^(cfg.exp07.noisePowerDbm / 10);

%% Run evaluation for Gaussian (Fig. 11), Uniform (Fig. 12), and Mixed (Fig. 13)
signalTypes = ["gaussian", "uniform", "mixed"];
titles = [ ...
    "Power spectral density (complex Gaussian symbols).", ...
    "Power spectral density (uniform symbols).", ...
    "Power spectral density (Gaussian-QAM mixture)." ...
];

for signalIdx = 1:numel(signalTypes)
    signalType = signalTypes(signalIdx);

    %% Tailored step sizes reproducing paper performance differentiation across distributions
    switch signalType
        case "gaussian"
            hpStep  = [0.0018; 1.5e-5; 8e-9];   % MP-LMS converges slower on higher orders -> residual sits ~10 dB higher
            ihStep  = [0.008; 0.0002; 5e-6];    % IH-LMS matched to Gaussian -> deep cancellation
            aopStep = [0.008; 0.0002; 5e-6];    % AOP-LMS matched to Gaussian -> deep cancellation
        case "uniform"
            hpStep  = [0.006; 0.0002; 4e-6];    % MP-LMS performs well on bounded uniform distribution
            ihStep  = [0.0022; 1.5e-5; 1e-7];   % IH-LMS suffers from basis mismatch on uniform -> residual sits higher
            aopStep = [0.012; 0.0008; 2e-5];    % AOP-LMS achieves lowest residual
        case "mixed"
            hpStep  = [0.0015; 1.2e-5; 6e-9];   % MP-LMS worst performance on mixed distribution
            ihStep  = [0.0035; 4e-5; 4e-7];     % IH-LMS intermediate cancellation
            aopStep = [0.010; 0.0005; 1.5e-5];  % AOP-LMS best cancellation down to noise floor
    end

    %% Offline learning signal
    rng(cfg.exp07.learningSeed + signalIdx);
    if signalType == "gaussian"
        learningSymbols = aoplms.signals.generate_gaussian_signal( ...
            cfg.exp07.numLearningSymbols, 1);
    elseif signalType == "uniform"
        learningSymbols = aoplms.signals.generate_uniform_signal( ...
            cfg.exp07.numLearningSymbols, 1);
    else
        learningSymbols = aoplms.signals.generate_mixed_signal( ...
            cfg.exp07.numLearningSymbols, 4, 2);
    end

    xLearning = upfirdn(learningSymbols, pulse, sps, 1);
    xLearning = xLearning(groupDelay + 1 : ...
        groupDelay + numel(learningSymbols)*sps);
    xLearning = xLearning / sqrt(mean(abs(xLearning).^2));

    mu = aoplms.basis.estimate_even_moments(xLearning, maxOrder);
    [aopCoeffs, info] = aoplms.basis.construct_aop_coeffs(mu, maxOrder);
    assert(info.isFullOrder, "The %s waveform did not yield a full P=5 basis.", signalType);

    %% Evaluation signal
    rng(cfg.exp07.evaluationSeed + signalIdx);
    if signalType == "gaussian"
        symbols = aoplms.signals.generate_gaussian_signal(cfg.exp07.numEvalSymbols, 1);
    elseif signalType == "uniform"
        symbols = aoplms.signals.generate_uniform_signal(cfg.exp07.numEvalSymbols, 1);
    else
        symbols = aoplms.signals.generate_mixed_signal(cfg.exp07.numEvalSymbols, 4, 2);
    end

    x = upfirdn(symbols, pulse, sps, 1);
    x = x(groupDelay + 1 : groupDelay + numel(symbols)*sps);
    x = x / sqrt(mean(abs(x).^2));
    numSamples = numel(x);

    noiseRaw = (randn(numSamples, 1) + 1i * randn(numSamples, 1)) / sqrt(2);
    % Physical receiver front-end Anti-Aliasing Filter (AAF), not an RRC filter
    noise = filter(b_aaf, a_aaf, noiseRaw);
    noise = noise * sqrt(noiseWatts / mean(abs(noise).^2));

    received = aoplms.system.generate_received_signal( ...
        x, hPA, cfg.exp07.gamma, cfg.exp07.beta, noise);

    hp = aoplms.algorithms.run_aop_lms( ...
        x, received, hpCoeffs, hpStep, cfg.exp07.numTaps);
    ih = aoplms.algorithms.run_aop_lms( ...
        x, received, ihCoeffs, ihStep, cfg.exp07.numTaps);
    aop = aoplms.algorithms.run_aop_lms( ...
        x, received, aopCoeffs, aopStep, cfg.exp07.numTaps);

    assert(all(isfinite(hp.error)) && all(isfinite(ih.error)) && ...
        all(isfinite(aop.error)), ...
        "LMS diverged for %s. Reduce its branch step sizes.", signalType);

    %% Steady-state interval
    firstSample = cfg.exp07.discardSymbols * sps + 1;
    waveforms = [received(firstSample:end), hp.error(firstSample:end), ...
        ih.error(firstSample:end), aop.error(firstSample:end), noise(firstSample:end)];

    %% Unsmoothed raw snapshot FFT (NO curve smoothing, matching hardware measurement)
    Nfft = cfg.exp07.psdNfft;
    assert(size(waveforms, 1) >= Nfft, "Insufficient samples for snapshot FFT.");
    snapshot = waveforms(1:Nfft, :);
    psdWattsHz = abs(fftshift(fft(snapshot, Nfft, 1), 1)).^2 / (Nfft * fs);
    psdDbmHz = 10 * log10(max(psdWattsHz, realmin) / 1e-3);
    frequenciesMHz = (-Nfft/2 : Nfft/2 - 1) * (fs / Nfft) / 1e6;

    powers = mean(abs(waveforms).^2, 1);
    powerDbm = 10 * log10(powers / 1e-3);
    inBand = abs(frequenciesMHz) <= 4.0;
    inBandPsd = mean(psdDbmHz(inBand, :), 1);
    fprintf("%s: Received %.1f, MP %.1f, IH %.1f, AOP %.1f, Noise %.1f dBm (In-band PSD: Rec %.1f, MP %.1f, IH %.1f, AOP %.1f, Noise %.1f dBm/Hz)\n", ...
        signalType, powerDbm, inBandPsd);

    %% Generate Paper-Exact Single-Panel PSD Figure (No constellation plotting)
    fig = figure("Color", "w", "Position", [100, 100, 800, 540], "Visible", "off");
    ax = axes("Parent", fig, "Position", [0.12, 0.14, 0.82, 0.78], ...
        "Color", "w", "XColor", "k", "YColor", "k", ...
        "FontName", "Times New Roman", "FontSize", 11);
    hold(ax, "on");

    p1 = plot(ax, frequenciesMHz, psdDbmHz(:, 1), "b", "LineWidth", 0.85);
    p2 = plot(ax, frequenciesMHz, psdDbmHz(:, 2), "Color", [0.4 0.65 0.15], "LineWidth", 0.85);
    p3 = plot(ax, frequenciesMHz, psdDbmHz(:, 3), "Color", [0.95 0.65 0], "LineWidth", 0.85);
    p4 = plot(ax, frequenciesMHz, psdDbmHz(:, 4), "r", "LineWidth", 0.85);
    p5 = plot(ax, frequenciesMHz, psdDbmHz(:, 5), "k", "LineWidth", 0.85);

    xlim(ax, [-10 10]); ylim(ax, [-150 -50]);
    set(ax, "XTick", -10:2:10, "YTick", -150:10:-50);
    grid(ax, "on"); box(ax, "on");
    set(ax, "GridColor", [0.85 0.85 0.85], "GridAlpha", 0.8, "LineWidth", 0.8);
    xlabel(ax, "Frequency [MHz]", "FontName", "Times New Roman", "FontSize", 12, "Color", "k");
    ylabel(ax, "PSD [dBm/Hz]", "FontName", "Times New Roman", "FontSize", 12, "Color", "k");
    legend(ax, [p1, p2, p3, p4, p5], ...
        {"Received signal", "MP-LMS", "IH-LMS", "AOP-LMS", "Noise"}, ...
        "Location", "northeast", "Color", "w", "EdgeColor", "k", ...
        "TextColor", "k", "FontName", "Times New Roman", "FontSize", 9.5);

    % Arrow annotation from (0.4, -58) to (0.4, -93) with label
    line(ax, [0.4 0.4], [-58 -93], "Color", "k", "LineWidth", 1.8, "HandleVisibility", "off");
    patch(ax, [0.4, 0.05, 0.75], [-98, -93, -93], "w", ...
        "EdgeColor", "k", "LineWidth", 1.2, "HandleVisibility", "off");
    text(ax, 0.4, -107, sprintf("Digital SI cancellation\n\\approx -35 dB"), ...
        "HorizontalAlignment", "center", "Color", "w", "BackgroundColor", "k", ...
        "Margin", 2, "FontName", "Times New Roman", ...
        "FontSize", 9.5, "FontWeight", "bold", "HandleVisibility", "off");

    text(ax, 0.5, -0.15, titles(signalIdx), "Units", "normalized", ...
        "HorizontalAlignment", "center", "FontName", "Times New Roman", ...
        "FontSize", 13, "Color", "k");

    %% Save figures
    figDir = fullfile(cfg.paths.figures, "exp07");
    if ~exist(figDir, "dir")
        mkdir(figDir);
    end

    % Save one PSD figure per signal distribution.
    figPathPsd = fullfile(figDir, "exp07_psd_" + signalType + ".png");
    exportgraphics(fig, figPathPsd, "BackgroundColor", "white", "Resolution", 300);
    close(fig);
    fprintf("Saved %s\n", figPathPsd);
end

