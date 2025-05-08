%% Harmonics Analysis for PD vs ET Tremor Classification
% This script analyzes tremor data from accelerometer and EMG recordings
% to extract harmonic features for differential diagnosis between
% Parkinson's Disease (PD) and Essential Tremor (ET).
%
% Author: Tanmoy Sil
% Affiliation: University Hospital Würzburg, Germany
% Email: sil_t@ukw.de
% Date: May 2025
%
% The script:
% 1. Loads and processes accelerometer and EMG signals
% 2. Extracts peak frequencies using wavelet analysis
% 3. Calculates various harmonic indices for feature extraction
%
% Requirements:
% - Signal Processing Toolbox
% - Parallel Computing Toolbox

%% Initialization
clc
clear

% Constants
SAMPLING_FREQ = 800; % Hz

% Define sensor and disease types
sensor = {'acc', 'emg'};
disease = {'PD', 'ET'};

% Set up paths based on operating system
if ismac
    dir_pats = {
        '', ...
        ''
    };
    addpath('');
    pool_size = 10;
elseif isunix
    dir_pats = {
        '', ...
        ''
    };
    addpath('');
    addpath('');
    pool_size = 32;
end

%% Feature Extraction Loop for Each Sensor and Disease
% Initialize data structures to store results
max_freq = cell(numel(sensor), numel(disease));
power = cell(numel(sensor), numel(disease));
power_avg = cell(numel(sensor), numel(disease));
hoi2 = cell(numel(sensor), numel(disease));
hoi3 = cell(numel(sensor), numel(disease));
hoi_summative = cell(numel(sensor), numel(disease));
hoi_all = cell(numel(sensor), numel(disease));
hoi_non_harmonics = cell(numel(sensor), numel(disease));

for s = 1:numel(sensor)
    for d = 1:numel(disease)
        % Load appropriate data file based on sensor and disease
        data_file = [sensor{s}, '_', disease{d}];
        fprintf('Processing %s data for %s patients...\n', sensor{s}, disease{d});
        
        try
            load(data_file);
            harmonic = exp.frequencymixing.harmonicmixing';
            cd(dir_pats{d});
        catch ME
            warning('Could not load %s. Error: %s', data_file, ME.message);
            continue;
        end
        
        % Load signal files
        files = dir('*.dat');
        signals = cell(length(files), 1);
        
        % Create parallel pool for processing
        if ~isempty(gcp('nocreate'))
            delete(gcp('nocreate'));
        end
        parpool('Threads', pool_size);
        
        % Load signals based on sensor type
        if strcmp(sensor{s}, 'acc')
            parfor i = 1:length(files)
                signals{i} = load(files(i).name, '-ascii');
                signals{i} = signals{i}(:, [1, 4]); % Extract accelerometer channels
            end
        elseif strcmp(sensor{s}, 'emg')
            parfor i = 1:length(files)
                signals{i} = load(files(i).name, '-ascii');
                signals{i} = signals{i}(:, [2, 3, 5, 6]); % Extract EMG channels
            end
        end
        
        % Process signals to extract peak frequencies and power
        fprintf('Extracting peak frequencies using wavelet analysis...\n');
        max_freq_wavelet = cell(length(signals), 1);
        pow = cell(length(signals), 1);
        pow_avg = cell(length(signals), 1);
        
        % Close and restart parallel pool for better memory management
        delete(gcp('nocreate'));
        parpool('Threads', pool_size);
        
        parfor i = 1:length(signals)
            [max_freq_wavelet{i}, pow{i}, pow_avg{i}] = peakfreq_using_wavelet(signals{i}, SAMPLING_FREQ);
        end
        
        % Store results
        max_freq{s, d} = max_freq_wavelet;
        power{s, d} = pow;
        power_avg{s, d} = pow_avg;
        
        % Calculate harmonic indices
        fprintf('Calculating harmonic indices for %s-%s...\n', sensor{s}, disease{d});
        
        % Second harmonic index (distance correlation between fundamental and 2nd harmonic)
        hoi2{s, d} = harmonics(max_freq{s, d}, harmonic, sensor{s});
        
        % Third harmonic index (distance correlation between fundamental and 3rd harmonic)
        hoi3{s, d} = harmonics_higher(max_freq{s, d}, harmonic, sensor{s}, 0);
        
        % Summative harmonic indices (mean of only summatives)
        hoi_summative{s, d} = harmonics_summative_avg(harmonic, 1, sensor{s});
        
        % All harmonics (mean of the whole series)
        hoi_all{s, d} = harmonics_summative_avg(harmonic, 0, sensor{s});
        
        % Non-harmonic indices (mean of only plus_minus from harmonics)
        hoi_non_harmonics{s, d} = harmonics_plus_minus(harmonic, sensor{s});
        
        % Clean up variables for next iteration to manage memory
        clearvars -except hoi2 hoi3 hoi_summative hoi_all hoi_non_harmonics ...
            sensor disease dir_pats d s max_freq power power_avg pool_size SAMPLING_FREQ
    end
end

% Close parallel pool
if ~isempty(gcp('nocreate'))
    delete(gcp('nocreate'));
end

% Save processed data
output_file = 'harmonics_features_PD_ET.mat';
save(output_file, 'hoi2', 'hoi3', 'hoi_summative', 'hoi_all', 'hoi_non_harmonics', 'max_freq', 'power', 'power_avg');
fprintf('Harmonic features saved to %s\n', output_file);

%% Function Descriptions:
% 
% harmonics() - Calculates distance correlation between fundamental and 2nd harmonic
% harmonics_higher() - Calculates distance correlation between fundamental and 3rd harmonic 
% harmonics_summative_avg() - Calculates mean of summative harmonics
% harmonics_plus_minus() - Calculates mean of plus/minus from harmonics
% peakfreq_using_wavelet() - Extracts peak frequencies using wavelet analysis
