% Experiment 1: AOP Basis Orthonormality Validation
% Redesigned to evaluate empirical orthonormality of AOP vs. Hammerstein basis.

clc; close all; clear;

%% Project setup and configuration
run(fullfile(fileparts(mfilename("fullpath")), "..", "startup.m"));
cfg = aoplms.config.default_config();

% Experiment parameters
cfg.exp01.learningSamples = [55, 100, 500, 1000, 5000];
cfg.exp01.nonlinearOrders = [3, 5, 7];
cfg.exp01.Pplot = 5;
cfg.exp01.Nplot = 100;
cfg.exp01.numEvaluationSamples = 5000;
cfg.exp01.numTrials = 20;

% Figure output directory
figDir = fullfile(cfg.paths.figures, "exp01");
if ~exist(figDir, "dir")
    mkdir(figDir);
end

%% Signal definitions
signals = [ ...
    struct("type", "gaussian", "name", "Complex Gaussian",      "tag", "gaussian", "order", NaN), ...
    struct("type", "uniform",  "name", "Real Uniform",          "tag", "uniform",  "order", NaN), ...
    struct("type", "mixed",    "name", "Gaussian-QAM Mixture",  "tag", "mixed",    "order", 4), ...
    struct("type", "qam",      "name", "4QAM",                  "tag", "4qam",     "order", 4), ...
    struct("type", "qam",      "name", "16QAM",                 "tag", "16qam",    "order", 16), ...
    struct("type", "qam",      "name", "64QAM",                 "tag", "64qam",    "order", 64), ...
    struct("type", "qam",      "name", "256QAM",                "tag", "256qam",   "order", 256) ...
    ];

numSignals = numel(signals);
learningSamples = cfg.exp01.learningSamples;
numN = numel(learningSamples);
orders = cfg.exp01.nonlinearOrders;
numP = numel(orders);
numTrials = cfg.exp01.numTrials;
numEval = cfg.exp01.numEvaluationSamples;
maxN = max(learningSamples);

%% Preallocate metric storage
% Off-diagonal erros
% Dimensions: (signal, order, learning_length, trial)
maxOffAop = zeros(numSignals, numP, numN, numTrials);
meanOffAop = zeros(numSignals, numP, numN, numTrials);
maxOffHp = zeros(numSignals, numP, numN, numTrials);
meanOffHp = zeros(numSignals, numP, numN, numTrials);
validK = zeros(numSignals, numP);

% Representative error matrices for heatmaps (stored for Nplot, Pplot, trial 1)
repErrorAop = cell(numSignals, 1);
repErrorHp = cell(numSignals, 1);

%% Run simulation
rng(cfg.sim.randomSeed);
warning("off", "all"); % Prevent rank-deficiency warning spam

for sigIdx = 1:numSignals
    sig = signals(sigIdx);

    for trial = 1:numTrials

        % Generate one realization containing learning pool and separate evaluation portion
        switch sig.type
            case "gaussian"
                xTotal = aoplms.signals.generate_gaussian_signal(maxN + numEval, cfg.signal.power);
            case "uniform"
                xTotal = aoplms.signals.generate_uniform_signal(maxN + numEval, cfg.signal.power);
            case "mixed"
                xTotal = aoplms.signals.generate_mixed_signal(maxN + numEval, sig.order, cfg.signal.power);
            case "qam"
                xTotal = aoplms.signals.generate_qam_signal(maxN + numEval, sig.order, cfg.signal.power);
        end

        xLearnPool = xTotal(1:maxN);
        xEval = xTotal(maxN + 1 : maxN + numEval);

        for pIdx = 1:numP
            P = orders(pIdx);

            for nIdx = 1:numN
                N = learningSamples(nIdx);
                xLearn = xLearnPool(1:N);

                % AOP Basis
                muEst = aoplms.basis.estimate_even_moments(xLearn, P);
                [coeffs, info] = aoplms.basis.construct_aop_coeffs(muEst, P);
                K_valid = info.validNumBasis;
                validK(sigIdx, pIdx) = K_valid;

                Phi_aop = aoplms.basis.evaluate_aop_basis(xEval, coeffs);
                aopReport = aoplms.metrics.check_orthonormality(Phi_aop);
                E_aop = aopReport.errorMatrix;

                % Normalized Hammerstein Basis
                H_learn = hammerstein_basis(xLearn, K_valid);
                rmsFactors = max(sqrt(mean(abs(H_learn).^2, 1)), eps);

                H_eval = hammerstein_basis(xEval, K_valid);
                Phi_hp = H_eval ./ rmsFactors;
                hpReport = aoplms.metrics.check_orthonormality(Phi_hp);
                E_hp = hpReport.errorMatrix;

                % Metrics
                [maxOffAop(sigIdx, pIdx, nIdx, trial), meanOffAop(sigIdx, pIdx, nIdx, trial)] = compute_metrics(E_aop);
                [maxOffHp(sigIdx, pIdx, nIdx, trial),  meanOffHp(sigIdx, pIdx, nIdx, trial)]  = compute_metrics(E_hp);

                % Store representative error matrices for heatmaps
                if N == cfg.exp01.Nplot && P == cfg.exp01.Pplot && trial == 1
                    repErrorAop{sigIdx} = E_aop;
                    repErrorHp{sigIdx} = E_hp;
                end
            end
        end
    end
end

warning("on", "all");

%% Generate Heatmaps (for representative Nplot and Pplot)
hexBlues = ['#f7fbff'; '#deebf7'; '#c6dbef'; '#9ecae1'; '#6baed6'; '#4292c6'; '#2171b5'; '#08519c'; '#08306b'];
rgbBlues = zeros(9, 3);
for k = 1:9
    hStr = hexBlues(k, :);
    rgbBlues(k, :) = [hex2dec(hStr(2:3)), hex2dec(hStr(4:5)), hex2dec(hStr(6:7))] / 255;
end
cmapBlues = interp1(linspace(0, 1, 9), rgbBlues, linspace(0, 1, 256), "pchip");

for sigIdx = 1:numSignals
    sig = signals(sigIdx);
    E_aop = repErrorAop{sigIdx};
    E_hp = repErrorHp{sigIdx};
    cMax = max([max(E_aop(:)), max(E_hp(:)), 1e-4]);

    % Combined side-by-side heatmap figure
    figHeat = figure("Visible", "off", "Position", [100, 100, 1150, 450], "Color", "w");

    % Left panel: AOP Absolute Error
    hAop = heatmap(figHeat, E_aop, "Position", [0.05, 0.12, 0.39, 0.74]);
    hAop.Title = sprintf("AOP Absolute Error (N=%d, P=%d)", cfg.exp01.Nplot, cfg.exp01.Pplot);
    hAop.CellLabelFormat = "%.4f";
    hAop.ColorLimits = [0, cMax];
    hAop.Colormap = cmapBlues;
    hAop.FontColor = "k";

    % Right panel: Normalized Hammerstein Absolute Error
    hHp = heatmap(figHeat, E_hp, "Position", [0.53, 0.12, 0.39, 0.74]);
    hHp.Title = sprintf("Norm. Hammerstein Absolute Error (N=%d, P=%d)", cfg.exp01.Nplot, cfg.exp01.Pplot);
    hHp.CellLabelFormat = "%.4f";
    hHp.ColorLimits = [0, cMax];
    hHp.Colormap = cmapBlues;
    hHp.FontColor = "k";

    % Figure Title
    annotation(figHeat, "textbox", [0, 0.90, 1, 0.09], ...
        "String", sprintf("%s: Basis Orthonormality Error Matrices (N=%d, P=%d)", sig.name, cfg.exp01.Nplot, cfg.exp01.Pplot), ...
        "EdgeColor", "none", "HorizontalAlignment", "center", ...
        "FontWeight", "bold", "FontSize", 13, "Color", "k");

    print(figHeat, fullfile(figDir, sprintf("exp01_heatmap_%s_comparison_N%d_P%d.png", sig.tag, cfg.exp01.Nplot, cfg.exp01.Pplot)), "-dpng", "-r300");
    close(figHeat);
end


%% Generate Grouped-Bar Summaries across Learning Sample Sizes
avgMaxOffAop  = mean(maxOffAop, 4);
avgMeanOffAop = mean(meanOffAop, 4);
avgMaxOffHp   = mean(maxOffHp, 4);
avgMeanOffHp  = mean(meanOffHp, 4);

nCats = categorical(string(learningSamples), string(learningSamples), "Ordinal", true);

for sigIdx = 1:numSignals
    sig = signals(sigIdx);
    summaryFig = figure("Visible", "off", "Position", [100, 100, 920, 260 * numP], "Color", "w");
    tl = tiledlayout(numP, 2, "TileSpacing", "compact", "Padding", "compact");

    for pIdx = 1:numP
        P = orders(pIdx);
        K_val = validK(sigIdx, pIdx);

        maxA = squeeze(avgMaxOffAop(sigIdx, pIdx, :));
        maxH = squeeze(avgMaxOffHp(sigIdx, pIdx, :));
        meanA = squeeze(avgMeanOffAop(sigIdx, pIdx, :));
        meanH = squeeze(avgMeanOffHp(sigIdx, pIdx, :));

        % Max off-diagonal error tile
        nexttile;
        bar(nCats, [maxA(:), maxH(:)], "grouped");
        grid on;
        ax = gca;
        ax.Color = "w";
        ax.XColor = "k";
        ax.YColor = "k";
        xlabel("Learning samples (N)", "Color", "k");
        ylabel("Max Off-Diagonal Error", "Color", "k");
        title(sprintf("P = %d (K = %d): Maximum Off-Diagonal", P, K_val), "Color", "k");
        if all([maxA(:); maxH(:)] == 0)
            ylim([0, 0.1]);
        else
            ylim([0, max(1.15 * max([maxA(:); maxH(:)]), 1e-4)]);
        end
        if pIdx == 1
            lgd = legend(["AOP", "Norm. Hammerstein"], "Orientation", "horizontal");
            lgd.Layout.Tile = "north";
            lgd.Color = "w";
            lgd.TextColor = "k";
            lgd.EdgeColor = "k";
        end

        % Mean off-diagonal error tile
        nexttile;
        bar(nCats, [meanA(:), meanH(:)], "grouped");
        grid on;
        ax = gca;
        ax.Color = "w";
        ax.XColor = "k";
        ax.YColor = "k";
        xlabel("Learning samples (N)", "Color", "k");
        ylabel("Mean Off-Diagonal Error", "Color", "k");
        title(sprintf("P = %d (K = %d): Mean Off-Diagonal", P, K_val), "Color", "k");
        if all([meanA(:); meanH(:)] == 0)
            ylim([0, 0.1]);
        else
            ylim([0, max(1.15 * max([meanA(:); meanH(:)]), 1e-4)]);
        end
    end

    title(tl, sprintf("%s: Basis Orthonormality Error vs. Learning Samples N", sig.name), ...
        "FontWeight", "bold", "Color", "k");

    exportgraphics(summaryFig, fullfile(figDir, sprintf("exp01_barchart_%s_offdiagonal.png", sig.tag)), ...
        "BackgroundColor", "white", "Resolution", 300);
    close(summaryFig);
end

%% Local Functions

function H = hammerstein_basis(x, K)
N = numel(x);
H = zeros(N, K);
absx2 = abs(x).^2;
for k = 1:K
    H(:, k) = (absx2.^(k - 1)) .* x;
end
end

function [maxOff, meanOff] = compute_metrics(E)
K = size(E, 1);
if K <= 1
    maxOff = 0;
    meanOff = 0;
else
    offMask = ~eye(K);
    maxOff = max(E(offMask));
    meanOff = mean(E(offMask));
end
end