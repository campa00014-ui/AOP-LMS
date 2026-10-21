% This code initializes the AOP-LMS MATLAB project.
clc;

projectRoot = fileparts(mfilename('fullpath'));

srcDir = fullfile(projectRoot, "src");
expDir = fullfile(projectRoot, "experiments");

addpath(srcDir);
addpath(expDir);

% Create output folders if they do not already exist.
resultsDir = fullfile(projectRoot, "results");
figuresDir = fullfile(resultsDir, "figures");
lutDir = fullfile(resultsDir, "lut");

if ~exist(resultsDir, "dir")
    mkdir(resultsDir);
end

if ~exist(figuresDir, "dir")
    mkdir(figuresDir);
end

if ~exist(lutDir, "dir")
    mkdir(lutDir);
end

fprintf("AOP-LMS project initialized.\n");
