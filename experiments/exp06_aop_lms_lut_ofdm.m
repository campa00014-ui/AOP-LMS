% Experiment 6: Figure 8(a) OFDM QAM with LUT and windowed AOP-LMS.

clc; close all; clear;
run(fullfile(fileparts(mfilename("fullpath")), "..", "startup.m"));
cfg = aoplms.config.default_config();

%% Experiment parameters
cfg.exp06.qamOrders = [16 64 4 256 64];
cfg.exp06.segmentLengths = [2200 5500 3300 4400 2200];
cfg.exp06.channelChangeSample = 5000;
cfg.exp06.order = 5;
cfg.exp06.numTrials = 200;
cfg.exp06.signalPower = 1;
cfg.exp06.gamma = cfg.pa.gamma;
cfg.exp06.beta = cfg.pa.beta;
cfg.exp06.paMemoryLength = cfg.pa.memoryLength;
cfg.exp06.siChannelLength = 5;
cfg.exp06.lmsFilterLength = cfg.exp06.paMemoryLength + cfg.exp06.siChannelLength - 1;
cfg.exp06.receivedPowerDbm = -55;
cfg.exp06.noiseFloorDbm = cfg.noise.floorDbm;
cfg.exp06.nfft = cfg.ofdm.nfft;
cfg.exp06.cpLength = cfg.ofdm.cpLength;
% In OFDM signaling, the serialized time-domain waveform is approximately complex
% Gaussian with 10-12 dB PAPR. The 5th-order nonlinear regressor (|x|^4 x) reaches
% peak instantaneous powers > 1500. Retaining the direct SC-FDE branch-3 step size
% (0.0015) violates LMS stability limits on high peaks, causing 30-40 dB spikes.
% Scaling branch 3 step size to 1.5e-4 ensures stable convergence to the noise floor.
cfg.exp06.lutBranchStepSizes = [0.013; 0.0017; 1e-4];
cfg.exp06.ihBranchStepSizes = [0.009; 0.0014; 1e-4];
cfg.exp06.windowedBranchStepSizes = [0.005; 0.001; 1e-6];
cfg.exp06.numLearningSamples = 55;
cfg.exp06.updateInterval = 1100;
cfg.exp06.paSeed = 22;
cfg.exp06.channelSeed = 23;
cfg.exp06.calibrationSeed = 64153;
cfg.exp06.trialSeedBase = 70000;
cfg.exp06.numCalibrationSamples = 5000;
cfg.exp06.lutOfflineSamples = 1000000;

numSamples = sum(cfg.exp06.segmentLengths);
numTrials = cfg.exp06.numTrials;
maxOrder = cfg.exp06.order;
numBranches = (maxOrder + 1) / 2;
lutBranchStepSizes = cfg.exp06.lutBranchStepSizes(1:numBranches);
ihStepSizes = cfg.exp06.ihBranchStepSizes(1:numBranches);
windowedStepSizes = cfg.exp06.windowedBranchStepSizes(1:numBranches);
changeSample = cfg.exp06.channelChangeSample;

%% Offline OFDM LUT and modulation schedule
% The LUT uses moments of the time-domain OFDM waveform, which are approximately
% complex Gaussian by CLT. Therefore, higher-order basis functions exist even for 4-QAM.
lut = aoplms.lut.build_ofdm_aop_lut( ...
    maxOrder, cfg.exp06.nfft, cfg.exp06.cpLength, cfg.exp06.lutOfflineSamples);

segmentStarts = [1, 1 + cumsum(cfg.exp06.segmentLengths(1:end-1))];
cfg.exp06.updateInterval = segmentStarts;

modOrders = zeros(numSamples, 1);
for k = 1:numel(cfg.exp06.qamOrders)
    segmentEnd = segmentStarts(k) + cfg.exp06.segmentLengths(k) - 1;
    modOrders(segmentStarts(k):segmentEnd) = cfg.exp06.qamOrders(k);
end

% Unit-power Gaussian IH basis functions
ihCoeffs = {1; [-2 1] / sqrt(2); [6 -6 1] / sqrt(12)};
ihCoeffs = ihCoeffs(1:numBranches);

%% Fixed PA and two independent SI-channel realizations
rng(cfg.exp06.paSeed);
h_paRaw = sqrt(0.5) * (randn(cfg.exp06.paMemoryLength, 1) + ...
    1i * randn(cfg.exp06.paMemoryLength, 1));

rng(cfg.exp06.channelSeed);
Q = cfg.exp06.siChannelLength;
h_si1 = sqrt(0.5 / Q) * (randn(Q, 1) + 1i * randn(Q, 1));
h_si2 = sqrt(0.5 / Q) * (randn(Q, 1) + 1i * randn(Q, 1));

% Calibrate once with a separate 16-QAM OFDM signal and the first SI channel.
% Do not recalibrate after either a modulation change or the channel change.
rng(cfg.exp06.calibrationSeed);
xCalibration = aoplms.signals.generate_ofdm_signal( ...
    cfg.exp06.numCalibrationSamples, 16, cfg.exp06.signalPower, ...
    cfg.exp06.nfft, cfg.exp06.cpLength);
[~, paCalibration] = aoplms.system.generate_received_signal( ...
    xCalibration, h_paRaw, cfg.exp06.gamma, cfg.exp06.beta, ...
    zeros(cfg.exp06.numCalibrationSamples, 1));
siCalibration = aoplms.system.apply_si_channel(paCalibration, h_si1);
targetWatts = 1e-3 * 10^(cfg.exp06.receivedPowerDbm / 10);
powerScale = sqrt(targetWatts / mean(abs(siCalibration).^2));
h_pa = powerScale * h_paRaw;
noiseWatts = 1e-3 * 10^(cfg.exp06.noiseFloorDbm / 10);

%% Paired Monte Carlo trials using OFDM transmission
errorTrials = zeros(numSamples, numTrials);
ihErrorTrials = zeros(numSamples, numTrials);
windowedErrorTrials = zeros(numSamples, numTrials);

for trial = 1:numTrials
    rng(cfg.exp06.trialSeedBase + trial);
    x = aoplms.signals.generate_nonstationary_ofdm_signal( ...
        cfg.exp06.qamOrders, cfg.exp06.segmentLengths, cfg.exp06.signalPower, ...
        cfg.exp06.nfft, cfg.exp06.cpLength);
    noise = sqrt(noiseWatts / 2) * ...
        (randn(numSamples, 1) + 1i * randn(numSamples, 1));

    [~, paOutput] = aoplms.system.generate_received_signal( ...
        x, h_pa, cfg.exp06.gamma, cfg.exp06.beta, zeros(numSamples, 1));
    siBefore = aoplms.system.apply_si_channel(paOutput, h_si1);
    siAfter = aoplms.system.apply_si_channel(paOutput, h_si2);

    % The new channel applies starting at sample 5000 (one-based).
    % Filtering the full PA output retains its history at the change.
    cleanSI = [siBefore(1:changeSample-1); siAfter(changeSample:end)];
    received = cleanSI + noise;

    result = aoplms.algorithms.run_aop_lms_lut( ...
        x, received, modOrders, lut, lutBranchStepSizes, cfg.exp06.lmsFilterLength);
    ihResult = aoplms.algorithms.run_aop_lms( ...
        x, received, ihCoeffs, ihStepSizes, cfg.exp06.lmsFilterLength);
    windowedResult = aoplms.algorithms.run_aop_lms_nonstationary( ...
        x, received, maxOrder, windowedStepSizes, cfg.exp06.lmsFilterLength, ...
        cfg.exp06.numLearningSamples, cfg.exp06.updateInterval);

    assert(all(isfinite(result.error)) && all(isfinite(ihResult.error)) && ...
        all(isfinite(windowedResult.error)), ...
        "Nonfinite residual in trial %d; inspect the configured step sizes.", trial);
    errorTrials(:, trial) = result.error;
    ihErrorTrials(:, trial) = ihResult.error;
    windowedErrorTrials(:, trial) = windowedResult.error;
end

%% Ensemble MSE: average power before converting to dBm
mseWatts = aoplms.metrics.compute_mse(errorTrials);
msedBm = 10 * log10(mseWatts / 1e-3);
ihMseWatts = aoplms.metrics.compute_mse(ihErrorTrials);
ihMsedBm = 10 * log10(ihMseWatts / 1e-3);
windowedMseWatts = aoplms.metrics.compute_mse(windowedErrorTrials);
windowedMsedBm = 10 * log10(windowedMseWatts / 1e-3);

%% Plot the implemented OFDM comparisons
fig = figure("Position", [100 100 875 662], "Color", "w");
hold on;
pIh  = plot(1:numSamples, ihMsedBm, "Color", 'c', "LineWidth", 1.0);
pWin = plot(1:numSamples, windowedMsedBm, "Color", 'm', "LineWidth", 1.0);
pLut = plot(1:numSamples, msedBm, "Color", 'b', "LineWidth", 1.0);
pNoise = yline(cfg.exp06.noiseFloorDbm, "--k", "LineWidth", 1.1);
pChan  = xline(changeSample, "--r", "LineWidth", 1.1);

for k = 1:numel(segmentStarts)
    xline(segmentStarts(k), ":k", "LineWidth", 1.4);
    labelX = segmentStarts(k) + cfg.exp06.segmentLengths(k) / 3;
    text(labelX, -50, sprintf("%d-QAM\nL=%d", ...
        cfg.exp06.qamOrders(k), cfg.exp06.segmentLengths(k)), ...
        "Color", "k");
end

xlabel("Samples [n]");
ylabel("Mean squared error [dBm]");
grid on; box on;
xlim([1 numSamples]); ylim([-105 -45]);
lgd = legend([pIh, pWin, pLut, pNoise, pChan], ...
    {"IH-LMS", "AOP LMS with windowing", "AOP LMS with LUT", "Noise floor", "SI channel change"}, ...
    "Location", "southoutside", "Orientation", "horizontal");
lgd.Color = "w";
lgd.TextColor = "k";
ax = gca; ax.Toolbar.Visible = "off";
ax.Color = "w"; ax.XColor = "k"; ax.YColor = "k";

%% Save figure
figureDir = fullfile(cfg.paths.figures, "exp06");
if ~exist(figureDir, "dir")
    mkdir(figureDir);
end

figurePath = fullfile(figureDir, "exp06_aop_lms_qam_ofdm_windowed.png");
exportgraphics(fig, figurePath);
fprintf("Figure 8(a) experiment complete. Saved plot to %s\n", figurePath);
