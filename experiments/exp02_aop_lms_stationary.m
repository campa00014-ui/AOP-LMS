% Experiment 6: Figure 5 stationary Gaussian-QAM comparison (P=7).
% See docs/figure5_notes.md for paper parameters and reproducibility choices.

clc; close all; clear;
run(fullfile(fileparts(mfilename("fullpath")), "..", "startup.m"));
cfg = aoplms.config.default_config();
cfg.exp02.numSamples = 10000;
cfg.exp02.numTrials = 200;
cfg.exp02.numLearningSamples = 50000;
cfg.exp02.order = 7;
cfg.exp02.totalSignalPower = 2;
cfg.exp02.gamma = 3;
cfg.exp02.beta = 0.09;
cfg.exp02.paMemoryLength = 9;
cfg.exp02.lmsFilterLength = 9;
cfg.exp02.receivedPowerDbm = -43;
cfg.exp02.noiseFloorDbm = -100;
cfg.exp02.paSeed = 22;
cfg.exp02.calibrationSeed = 64153;
cfg.exp02.learningSeed = 30;
cfg.exp02.trialSeedBase = 70000;
cfg.exp02.numCalibrationSamples = 5000;
cfg.exp02.hpBranchStepSizes = [0.005; 1e-4; 1e-7; 2e-10];
cfg.exp02.ihBranchStepSizes = [0.004; 0.004; 0.002; 6e-8];
cfg.exp02.aopBranchStepSizes = [0.02; 0.0035; 0.0005; 0.00005];
% Optional short validation run; preserve separate output names.
cfg.exp02.smokeTest = strcmp(getenv("AOPLMS_FIG5_SMOKE"), "1");
if cfg.exp02.smokeTest
    cfg.exp02.numTrials = 3;
end
numSamples = cfg.exp02.numSamples;
numTrials = cfg.exp02.numTrials;
curveNames = ["HP-LMS", "IH-LMS", "AOP-LMS"];
hpCoeffs = {1; [0 1]; [0 0 1]; [0 0 0 1]};
% Seventh-order monic polynomial has norm squared 4!3! = 144.
ihCoeffs = {
    [1] / sqrt(2);
    [-4, 1] / 4;
    [24, -12, 1] / (8*sqrt(6));
    [-192, 144, -24, 1] / (96*sqrt(2))
};
stepSizes = {cfg.exp02.hpBranchStepSizes, cfg.exp02.ihBranchStepSizes, cfg.exp02.aopBranchStepSizes};

%% Fixed PA/channel realization and independent calibration
rng(cfg.exp02.paSeed);
hRaw = sqrt(0.5) * (randn(cfg.exp02.paMemoryLength, 1) + ...
    1i * randn(cfg.exp02.paMemoryLength, 1));
rng(cfg.exp02.calibrationSeed);
xCalibration = aoplms.signals.generate_mixed_signal( ...
    cfg.exp02.numCalibrationSamples, 4, cfg.exp02.totalSignalPower);
[~, yCalibrationRaw] = aoplms.system.generate_received_signal( ...
    xCalibration, hRaw, cfg.exp02.gamma, cfg.exp02.beta, ...
    zeros(cfg.exp02.numCalibrationSamples, 1));
targetWatts = 1e-3 * 10^(cfg.exp02.receivedPowerDbm / 10);
powerScale = sqrt(targetWatts / mean(abs(yCalibrationRaw).^2));
hPA = hRaw * powerScale;
noiseWatts = 1e-3 * 10^(cfg.exp02.noiseFloorDbm / 10);

%% Paired Monte Carlo evaluation; learn one fixed AOP basis per trial
errorTrials = complex(zeros(numSamples, numTrials, 3));
receivedTrials = complex(zeros(numSamples, numTrials));
noiseTrials = complex(zeros(numSamples, numTrials));


for trial = 1:numTrials
    rng(cfg.exp02.learningSeed + trial - 1);
    xLearn = aoplms.signals.generate_mixed_signal( ...
        cfg.exp02.numLearningSamples, 4, cfg.exp02.totalSignalPower);
    mu = aoplms.basis.estimate_even_moments(xLearn, cfg.exp02.order);
    [aopCoeffs, info] = aoplms.basis.construct_aop_coeffs(mu, cfg.exp02.order);
    assert(info.isFullOrder, "Figure 5 requires a full P=7 AOP basis.");
    rng(cfg.exp02.trialSeedBase + trial);
    x = aoplms.signals.generate_mixed_signal(numSamples, 4, cfg.exp02.totalSignalPower);
    noise = sqrt(noiseWatts / 2) * (randn(numSamples, 1) + 1i * randn(numSamples, 1));
    received = aoplms.system.generate_received_signal( ...
        x, hPA, cfg.exp02.gamma, cfg.exp02.beta, noise);
    receivedTrials(:, trial) = received;
    noiseTrials(:, trial) = noise;
    coeffs = {hpCoeffs, ihCoeffs, aopCoeffs};
    for curve = 1:3
        result = aoplms.algorithms.run_aop_lms( ...
            x, received, coeffs{curve}, stepSizes{curve}, cfg.exp02.lmsFilterLength);
        assert(all(isfinite(result.error)), ...
            "Nonfinite residual in trial %d, curve %s.", trial, curveNames(curve));
        errorTrials(:, trial, curve) = result.error;
    end
end


%% Ensemble MSE: average linear power before conversion to dBm
mseWatts = zeros(numSamples, 3);
for curve = 1:3
    mseWatts(:, curve) = aoplms.metrics.compute_mse(errorTrials(:, :, curve));
end
msedBm = 10 * log10(mseWatts / 1e-3);
receivedMseWatts = aoplms.metrics.compute_mse(receivedTrials);
noiseMseWatts = aoplms.metrics.compute_mse(noiseTrials);
receivedMsedBm = 10 * log10(receivedMseWatts / 1e-3);
noiseMsedBm = 10 * log10(noiseMseWatts / 1e-3);


%% Plot Figure 5 stationary comparison
fig = figure("Position", [100 100 875 662], "Color", "w");
hold on;

receivedHandle = plot(1:numSamples, receivedMsedBm, ...
    "Color", 'k', "LineWidth", 1.0);

colors = ["#EDB120"; "#0FFFFF"; "#000CFF"];
curveHandles = gobjects(3, 1);
for curve = 1:3
    curveHandles(curve) = plot(1:numSamples, msedBm(:, curve), ...
        "Color", colors(curve, :), "LineWidth", 1.1);
end
noiseHandle = yline(cfg.exp02.noiseFloorDbm, "--k", "LineWidth", 1.1);

xlabel("Samples [n]");
ylabel("Mean squared error [dBm]");
grid on;
box on;
xlim([1 numSamples]);
ylim([-110 -40]);

lgd = legend([receivedHandle; curveHandles; noiseHandle], ...
    ["Received signal", curveNames, "Noise"], ...
    "Location", "northeast");
lgd.Color = "w";
lgd.TextColor = "k";

ax = gca; ax.Toolbar.Visible = "off";
ax.Color = "w"; ax.XColor = "k"; ax.YColor = "k";

%% Save figure
figureDir = fullfile(cfg.paths.figures, "exp02");
if ~exist(figureDir, "dir")
    mkdir(figureDir);
end

figurePath = fullfile(figureDir, "exp02_aop_lms_stationary.png");
exportgraphics(fig, figurePath);