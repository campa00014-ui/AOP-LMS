function cfg = default_config()

% This file stores simulation parameters used across the project.
% It does not generate signals or run algorithms.

%% General simulation settings
cfg.sim.numSamples = 10000;
cfg.sim.randomSeed = 1;

%% Transmit signal settings
cfg.signal.type = "gaussian";  % Can be "gaussian", "qam", or "mixed"
cfg.signal.power = 1;  % Normalized average signal power
cfg.signal.qamOrder = 4;  % Default QAM order
cfg.signal.qamOrders = [4 16 64 256];

%% OFDM signal settings
cfg.ofdm.nfft = 64;
cfg.ofdm.cpLength = 0;

%% Saleh PA model settings
cfg.pa.gamma = 3;  % Small-signal gain
cfg.pa.beta = 0.09;  % Transition sharpness
cfg.pa.memoryLength = 9;  % PA memory length M

%% Noise settings
cfg.noise.floorDbm = -100;  % Noise floor in dBm

%% Nonlinear polynomial model settings
cfg.model.nonlinearOrder = 5;  % Maximum nonlinear order, represented as P in the paper
cfg.model.numBasis = (cfg.model.nonlinearOrder + 1) / 2;

%% Moment-estimation settings
cfg.moments.numLearningSamples = 55;
cfg.moments.updateInterval = 3000;

%% LMS settings
cfg.lms.stepSize = 1e-3;  % Common static step size; experiments may use one value per branch

%% Result paths
configDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(fileparts(configDir)));

cfg.paths.projectRoot = projectRoot;
cfg.paths.results = fullfile(projectRoot, "results");
cfg.paths.figures = fullfile(cfg.paths.results, "figures");
cfg.paths.lut = fullfile(cfg.paths.results, "lut");

%% LUT settings
cfg.lut.modOrders = [4 16 64 256];
cfg.lut.cacheFile = fullfile(cfg.paths.lut, "qam_aop_lut.mat");

end
