% Experiment 4: AOP-LMS Polynomial Order Sweep
% Reproduces the Figure 6 order comparison

clc; close all; clear;
run(fullfile(fileparts(mfilename("fullpath")), "..", "startup.m"));
cfg = aoplms.config.default_config();

%% Experiment parameters
cfg.exp04.numSamples = 10000;
cfg.exp04.numLearningSamples = 5000;
cfg.exp04.numTrials = 150;
cfg.exp04.orders = [1 3 5 7];
cfg.exp04.curveStepSizes = { ...
    [0.04], ...                            % p=1
    [0.027, 0.007], ...                    % p=3
    [0.02, 0.0045, 0.0009], ...            % p=5
    [0.013, 0.0025, 0.0008, 0.00020] ...   % p=7
};
cfg.exp04.receivedPowerDbm = -52;
cfg.exp04.noiseFloorDbm = -100;
cfg.exp04.totalSignalPower = 2;
cfg.exp04.qamOrder = 4;
cfg.exp04.gamma = 3;
cfg.exp04.beta = 0.09;
cfg.exp04.paMemoryLength = 9;
cfg.exp04.lmsFilterLength = 9;
cfg.exp04.paSeed = 22;
cfg.exp04.learningSeed = 30;
cfg.exp04.numCalibrationSamples = 5000;

numSamples = cfg.exp04.numSamples;
numTrials = cfg.exp04.numTrials;
orders = cfg.exp04.orders;
curveStepSizes = cfg.exp04.curveStepSizes;
numOrders = numel(orders);

%% Generate and calibrate the fixed PA realization
rng(cfg.exp04.paSeed);
hRaw = sqrt(0.5) * (randn(cfg.exp04.paMemoryLength, 1) + ...
    1j * randn(cfg.exp04.paMemoryLength, 1));

rng(64153);

xCalibration = aoplms.signals.generate_mixed_signal( ...
    cfg.exp04.numCalibrationSamples, cfg.exp04.qamOrder, cfg.exp04.totalSignalPower);

[~, yCalibrationRaw] = aoplms.system.generate_received_signal( ...
    xCalibration, hRaw, cfg.exp04.gamma, cfg.exp04.beta, zeros(cfg.exp04.numCalibrationSamples, 1));

targetWatts = 1e-3 * 10^(cfg.exp04.receivedPowerDbm / 10);
rawWatts = mean(abs(yCalibrationRaw).^2);
powerScale = sqrt(targetWatts / rawWatts);
hPA = hRaw * powerScale;

%% Run Monte Carlo trials
errorTrials = zeros(numSamples, numTrials, numOrders);
noiseWatts = 1e-3 * 10^(cfg.exp04.noiseFloorDbm / 10);

for trial = 1:numTrials
    % Estimate the moments from a separate learning signal in every trial.
    rng(cfg.exp04.learningSeed + trial - 1);
    xLearn = aoplms.signals.generate_mixed_signal( ...
        cfg.exp04.numLearningSamples, cfg.exp04.qamOrder, cfg.exp04.totalSignalPower);
    mu = aoplms.basis.estimate_even_moments(xLearn, max(orders));

    coeffs = cell(1, numOrders);
    for pIdx = 1:numOrders
        P = orders(pIdx);
        [coeffs{pIdx}, info] = aoplms.basis.construct_aop_coeffs(mu(1:P), P);

        if ~info.isFullOrder
            error("AOP construction stopped at order %d for requested P=%d in trial %d.", ...
                info.validNonlinearOrder, P, trial);
        end
    end

    % Use the same evaluation input and noise for every order in this trial.
    rng(70000 + trial);
    xEval = aoplms.signals.generate_mixed_signal(numSamples, cfg.exp04.qamOrder, cfg.exp04.totalSignalPower);
    noise = sqrt(noiseWatts / 2) * (randn(numSamples, 1) + ...
        1j * randn(numSamples, 1));
    received = aoplms.system.generate_received_signal(xEval, hPA, cfg.exp04.gamma, cfg.exp04.beta, noise);

    for pIdx = 1:numOrders
        stepSize = curveStepSizes{pIdx};
        result = aoplms.algorithms.run_aop_lms( ...
            xEval, received, coeffs{pIdx}, ...
            stepSize, cfg.exp04.lmsFilterLength);
        errorTrials(:, trial, pIdx) = result.error;
    end
end

%% Compute MSE
mseWatts = zeros(numSamples, numOrders);
for pIdx = 1:numOrders
    mseWatts(:, pIdx) = aoplms.metrics.compute_mse(errorTrials(:, :, pIdx));
end
msedBm = 10 * log10(mseWatts / 1e-3);

%% Plot Figure 6 order sweep
fig = figure("Position", [100 100 875 662], "Color", "w");
hold on;
curveHandles = gobjects(numOrders, 1);
colors = ["#FF0000" "#EDB120" "#77AC30" "#0000FF"];
for pIdx = 1:numOrders
    curveHandles(pIdx) = plot(1:numSamples, msedBm(:, pIdx), ...
        "Color", colors(pIdx));
end
noiseHandle = yline(cfg.exp04.noiseFloorDbm, "--k", "LineWidth", 1.1);

xlabel("Samples [n]");
ylabel("Mean squared error [dBm]");
grid on;
box on;
xlim([1 numSamples]);
ylim([-105 -50]);
lgd = legend([curveHandles; noiseHandle], ...
    ["LMS (1st order, P=1)", ...
    "AOP-LMS (3rd order, P=3)", ...
    "AOP-LMS (5th order, P=5)", ...
    "AOP-LMS (7th order, P=7)", ...
    "Noise"], ...
    "Location", "northeast");
lgd.Color = "w"; 
lgd.TextColor = "k";
lgd.EdgeColor = "k"; 

ax = gca; ax.Toolbar.Visible = 'off';
ax.Color = 'w'; ax.XColor = 'k'; ax.YColor = 'k';

%% Save figure
figureDir = fullfile(cfg.paths.figures, "exp04");
if ~exist(figureDir, "dir")
    mkdir(figureDir);
end

figurePath = fullfile(figureDir, "exp04_aop_lms_order_sweep.png");
exportgraphics(fig, figurePath);