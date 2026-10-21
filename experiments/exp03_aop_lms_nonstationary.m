% Experiment 3: Non-Stationary AOP-LMS
% Reproduces Figure 7 of the paper with interval-based basis updates (Algorithm 2).

clc; close all; clear;
run(fullfile(fileparts(mfilename("fullpath")), "..", "startup.m"));
cfg = aoplms.config.default_config();

%% Experiment parameters
cfg.exp03.numUniformSamples = 6000;
cfg.exp03.numMixedSamples = 9000;
cfg.exp03.numSamples = cfg.exp03.numUniformSamples + cfg.exp03.numMixedSamples;
cfg.exp03.numLearningSamples = 55;
cfg.exp03.updateInterval = 3000;
cfg.exp03.numTrials = 50;
cfg.exp03.order = 5;
cfg.exp03.aopBranchStepSizes = [0.015; 0.0012; 0.5e-4];
cfg.exp03.lmsStepSize = 0.055;
cfg.exp03.hpBranchStepSizes = [0.065; 0.00009; 1e-6];
cfg.exp03.ihBranchStepSizes = [0.011; 0.0009; 8e-6];
cfg.exp03.receivedPowerDbm = -55;
cfg.exp03.noiseFloorDbm = -100;
cfg.exp03.totalMixedPower = 2;
cfg.exp03.qamOrder = 4;
cfg.exp03.gamma = 3;
cfg.exp03.beta = 0.09;
cfg.exp03.paMemoryLength = 9;
cfg.exp03.lmsFilterLength = 9;
cfg.exp03.paSeed = 22;
cfg.exp03.calibrationSeed = 64153;
cfg.exp03.trialSeedBase = 70000;
cfg.exp03.numCalibrationSamples = 5000;

numSamples = cfg.exp03.numSamples;
numTrials = cfg.exp03.numTrials;
maxOrder = cfg.exp03.order;
branchStepSizes = cfg.exp03.aopBranchStepSizes;

%% Fixed bases for the comparison algorithms
lmsCoeffs = {1};
hpCoeffs = {1; [0 1]; [0 0 1]};
ihCoeffs = {1; [-2 1] / sqrt(2); [6 -6 1] / sqrt(12)};

%% Generate and calibrate the fixed PA realization
rng(cfg.exp03.paSeed);
hRaw = sqrt(0.5) * (randn(cfg.exp03.paMemoryLength, 1) + ...
    1i * randn(cfg.exp03.paMemoryLength, 1));

rng(cfg.exp03.calibrationSeed);
xCalibration = aoplms.signals.generate_mixed_signal( ...
    cfg.exp03.numCalibrationSamples, cfg.exp03.qamOrder, cfg.exp03.totalMixedPower);

[~, yCalibrationRaw] = aoplms.system.generate_received_signal( ...
    xCalibration, hRaw, cfg.exp03.gamma, cfg.exp03.beta, ...
    zeros(cfg.exp03.numCalibrationSamples, 1));

targetWatts = 1e-3 * 10^(cfg.exp03.receivedPowerDbm / 10);
rawWatts = mean(abs(yCalibrationRaw).^2);
powerScale = sqrt(targetWatts / rawWatts);
hPA = hRaw * powerScale;

%% Run Monte Carlo trials
aopErrorTrials = zeros(numSamples, numTrials);
lmsErrorTrials = zeros(numSamples, numTrials);
hpErrorTrials = zeros(numSamples, numTrials);
ihErrorTrials = zeros(numSamples, numTrials);
noiseWatts = 1e-3 * 10^(cfg.exp03.noiseFloorDbm / 10);

for trial = 1:numTrials
    rng(cfg.exp03.trialSeedBase + trial);

    xUniform = aoplms.signals.generate_uniform_signal( ...
        cfg.exp03.numUniformSamples, 1);
    xMixed = aoplms.signals.generate_mixed_signal( ...
        cfg.exp03.numMixedSamples, cfg.exp03.qamOrder, cfg.exp03.totalMixedPower);
    x = [xUniform; xMixed];

    noise = sqrt(noiseWatts / 2) * ...
        (randn(numSamples, 1) + 1i * randn(numSamples, 1));
    received = aoplms.system.generate_received_signal( ...
        x, hPA, cfg.exp03.gamma, cfg.exp03.beta, noise);

    result = aoplms.algorithms.run_aop_lms_nonstationary( ...
        x, received, maxOrder, branchStepSizes, ...
        cfg.exp03.lmsFilterLength, cfg.exp03.numLearningSamples, ...
        cfg.exp03.updateInterval);

    aopErrorTrials(:, trial) = result.error;

    % Every algorithm receives the same x, PA/channel output, and noise.
    lmsResult = aoplms.algorithms.run_aop_lms( ...
        x, received, lmsCoeffs, cfg.exp03.lmsStepSize, cfg.exp03.lmsFilterLength);
    hpResult = aoplms.algorithms.run_aop_lms( ...
        x, received, hpCoeffs, cfg.exp03.hpBranchStepSizes, cfg.exp03.lmsFilterLength);
    ihResult = aoplms.algorithms.run_aop_lms( ...
        x, received, ihCoeffs, cfg.exp03.ihBranchStepSizes, cfg.exp03.lmsFilterLength);

    lmsErrorTrials(:, trial) = lmsResult.error;
    hpErrorTrials(:, trial) = hpResult.error;
    ihErrorTrials(:, trial) = ihResult.error;

end

%% Compute MSE and transition diagnostics
mseWatts = aoplms.metrics.compute_mse(aopErrorTrials);
aopMsedBm = 10 * log10(mseWatts / 1e-3);
lmsMseWatts = aoplms.metrics.compute_mse(lmsErrorTrials);
hpMseWatts = aoplms.metrics.compute_mse(hpErrorTrials);
ihMseWatts = aoplms.metrics.compute_mse(ihErrorTrials);
lmsMsedBm = 10 * log10(lmsMseWatts / 1e-3);
hpMsedBm = 10 * log10(hpMseWatts / 1e-3);
ihMsedBm = 10 * log10(ihMseWatts / 1e-3);

%% Plot the Figure 7 comparison
fig = figure("Position", [100 100 875 662], "Color", "w");
hold on;

plot(1:numSamples, lmsMsedBm, "Color", "#FF0000");
plot(1:numSamples, hpMsedBm, "Color", "#FFAE00");
plot(1:numSamples, ihMsedBm, "Color", "#00FFDF");
plot(1:numSamples, aopMsedBm, "Color", "#0000FF");
yline(cfg.exp03.noiseFloorDbm, "--k");
xline(cfg.exp03.numUniformSamples, ":k");

xlabel("Samples [n]");
ylabel("Mean squared error [dBm]");
grid on;
box on;
xlim([1 numSamples]);
ylim([-105 -55]);

lgd = legend("LMS", "HP-LMS", "IH-LMS", "AOP-LMS", "Noise", "Distribution change", ...
    "Location", "northeast");
lgd.Color = "w"; lgd.TextColor = "k";

ax = gca;
ax.Color = "w"; ax.XColor = "k"; ax.YColor = "k";
ax.Toolbar.Visible = "off";

%% Save figure
figureDir = fullfile(cfg.paths.figures, "exp03");
if ~exist(figureDir, "dir")
    mkdir(figureDir);
end

figurePath = fullfile(figureDir, "exp03_aop_lms_nonstationary.png");
exportgraphics(fig, figurePath);