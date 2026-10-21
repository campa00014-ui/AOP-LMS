% Experiment 5: Figure 8(b) SC-FDE QAM with LUT and windowed AOP-LMS.

clc; close all; clear;
run(fullfile(fileparts(mfilename("fullpath")), "..", "startup.m"));
cfg = aoplms.config.default_config();

%% Experiment parameters
cfg.exp05.qamOrders = [16 64 4 256 64];
cfg.exp05.segmentLengths = [2200 5500 3300 4400 2200];
cfg.exp05.channelChangeSample = 5000;
cfg.exp05.order = 5;
cfg.exp05.numTrials = 100;
cfg.exp05.signalPower = 1;
cfg.exp05.gamma = cfg.pa.gamma;
cfg.exp05.beta = cfg.pa.beta;
cfg.exp05.paMemoryLength = cfg.pa.memoryLength;
cfg.exp05.siChannelLength = 5;
cfg.exp05.lmsFilterLength = cfg.exp05.paMemoryLength + cfg.exp05.siChannelLength - 1;
cfg.exp05.receivedPowerDbm = -55;
cfg.exp05.noiseFloorDbm = cfg.noise.floorDbm;
cfg.exp05.branchStepSizes = [0.017; 0.0045; 0.0015]; 
cfg.exp05.numLearningSamples = 55;
cfg.exp05.updateInterval = 1100;
cfg.exp05.ihBranchStepSizes = [0.012; 0.0016; 10e-4];
cfg.exp05.paSeed = 22;
cfg.exp05.channelSeed = 23;
cfg.exp05.calibrationSeed = 64153;
cfg.exp05.trialSeedBase = 70000;
cfg.exp05.numCalibrationSamples = 5000;

numSamples = sum(cfg.exp05.segmentLengths);
numTrials = cfg.exp05.numTrials;
maxOrder = cfg.exp05.order;
numBranches = (maxOrder + 1) / 2;
branchStepSizes = cfg.exp05.branchStepSizes(1:numBranches);

ihStepSizes = cfg.exp05.ihBranchStepSizes(1:numBranches);
changeSample = cfg.exp05.channelChangeSample;

%% Offline LUT and known modulation schedule
lut = aoplms.lut.build_qam_aop_lut(maxOrder);
if ~exist(cfg.paths.lut, "dir")
    mkdir(cfg.paths.lut);
end
if ~exist(cfg.lut.cacheFile, "file")
    save(cfg.lut.cacheFile, "lut");
end

segmentStarts = [1, 1 + cumsum(cfg.exp05.segmentLengths(1:end-1))];
cfg.exp05.updateInterval = segmentStarts;

modOrders = zeros(numSamples, 1);
for k = 1:numel(cfg.exp05.qamOrders)
    segmentEnd = segmentStarts(k) + cfg.exp05.segmentLengths(k) - 1;
    modOrders(segmentStarts(k):segmentEnd) = cfg.exp05.qamOrders(k);
end

% Unit-power Gaussian IH basis, already used in Experiment 3.
ihCoeffs = {1; [-2 1] / sqrt(2); [6 -6 1] / sqrt(12)};
ihCoeffs = ihCoeffs(1:numBranches);

%% Fixed PA and two independent SI-channel realizations
rng(cfg.exp05.paSeed);
h_paRaw = sqrt(0.5) * (randn(cfg.exp05.paMemoryLength, 1) + ...
    1i * randn(cfg.exp05.paMemoryLength, 1));

rng(cfg.exp05.channelSeed);
Q = cfg.exp05.siChannelLength;
h_si1 = sqrt(0.5 / Q) * (randn(Q, 1) + 1i * randn(Q, 1));
h_si2 = sqrt(0.5 / Q) * (randn(Q, 1) + 1i * randn(Q, 1));

% Calibrate once with a separate 16-QAM signal and the first SI channel.
% Do not recalibrate after either a modulation change or the channel change.
rng(cfg.exp05.calibrationSeed);
xCalibration = aoplms.signals.generate_qam_signal( ...
    cfg.exp05.numCalibrationSamples, 16, cfg.exp05.signalPower);
[~, paCalibration] = aoplms.system.generate_received_signal( ...
    xCalibration, h_paRaw, cfg.exp05.gamma, cfg.exp05.beta, ...
    zeros(cfg.exp05.numCalibrationSamples, 1));
siCalibration = aoplms.system.apply_si_channel(paCalibration, h_si1);
targetWatts = 1e-3 * 10^(cfg.exp05.receivedPowerDbm / 10);
powerScale = sqrt(targetWatts / mean(abs(siCalibration).^2));
h_pa = powerScale * h_paRaw;
noiseWatts = 1e-3 * 10^(cfg.exp05.noiseFloorDbm / 10);

%% Paired Monte Carlo trials using direct single-carrier QAM symbols
errorTrials = zeros(numSamples, numTrials);
ihErrorTrials = zeros(numSamples, numTrials);
windowedErrorTrials = zeros(numSamples, numTrials);

for trial = 1:numTrials
    rng(cfg.exp05.trialSeedBase + trial);
    x = aoplms.signals.generate_nonstationary_qam_signal( ...
        cfg.exp05.qamOrders, cfg.exp05.segmentLengths, cfg.exp05.signalPower);
    noise = sqrt(noiseWatts / 2) * ...
        (randn(numSamples, 1) + 1i * randn(numSamples, 1));

    [~, paOutput] = aoplms.system.generate_received_signal( ...
        x, h_pa, cfg.exp05.gamma, cfg.exp05.beta, zeros(numSamples, 1));
    siBefore = aoplms.system.apply_si_channel(paOutput, h_si1);
    siAfter = aoplms.system.apply_si_channel(paOutput, h_si2);

    % The new channel applies starting at sample 5000 (one-based).
    % Filtering the full PA output retains its history at the change.
    cleanSI = [siBefore(1:changeSample-1); siAfter(changeSample:end)];
    received = cleanSI + noise;

    result = aoplms.algorithms.run_aop_lms_lut( ...
        x, received, modOrders, lut, branchStepSizes, cfg.exp05.lmsFilterLength);
    ihResult = aoplms.algorithms.run_aop_lms( ...
        x, received, ihCoeffs, ihStepSizes, cfg.exp05.lmsFilterLength);
    windowedResult = aoplms.algorithms.run_aop_lms_nonstationary( ...
        x, received, maxOrder, branchStepSizes, cfg.exp05.lmsFilterLength, ...
        cfg.exp05.numLearningSamples, cfg.exp05.updateInterval);

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

%% Plot the implemented SC-FDE comparisons
fig = figure("Position", [100 100 875 662], "Color", "w");
hold on;
plot(1:numSamples, ihMsedBm, "Color", 'c');
plot(1:numSamples, windowedMsedBm, "Color", 'm');
plot(1:numSamples, msedBm, "Color", 'b');
xline(changeSample, "--r");
for k = 1:numel(segmentStarts)
    xline(segmentStarts(k), ":k", "LineWidth", 1.4);
    labelX = segmentStarts(k) + cfg.exp05.segmentLengths(k) / 3;
    text(labelX, -50, sprintf("%d-QAM\nL=%d", ...
        cfg.exp05.qamOrders(k), cfg.exp05.segmentLengths(k)), ...
        "Color", "k");
end
xlabel("Samples [n]");
ylabel("Mean squared error [dBm]");
grid on; box on;
xlim([1 numSamples]); ylim([-105 -45]);
lgd = legend("IH-LMS", "AOP LMS with windowing", "AOP LMS with LUT", "SI channel change", ...
    "Location", "southoutside", "Orientation", "horizontal");
lgd.Color = "w";
lgd.TextColor = "k";
ax = gca; ax.Toolbar.Visible = "off";
ax.Color = "w"; ax.XColor = "k"; ax.YColor = "k";

%% Save figure
figureDir = fullfile(cfg.paths.figures, "exp05");
if ~exist(figureDir, "dir")
    mkdir(figureDir);
end

figurePath = fullfile(figureDir, "exp05_aop_lms_qam_scfde_windowed.png");
exportgraphics(fig, figurePath);