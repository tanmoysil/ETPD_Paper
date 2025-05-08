%% Harmonics Analysis for Parkinson's Disease and Essential Tremor Classification
% This script processes tremor data from accelerometers and EMG recordings,
% extracting harmonic features for classification of Parkinson's Disease (PD) 
% vs Essential Tremor (ET).
%
% Author: Tanmoy Sil
% Affiliation: University Hospital Würzburg, Germany
% Email: sil_t@ukw.de
% Date: May 2025
%
% The script extracts various features including:
%   - Harmonic features of order 2 and 3
%   - Summative harmonic indices
%   - Non-harmonic features
%   - Fundamental frequency features
%   - Power-related features
%
% These features are combined to create a feature matrix suitable for
% machine learning classification algorithms.

%% Clear workspace and load data
clearvars;
clc;

% Define data file paths
HOI_FILE_PATH = '';
FREQ_POW_FILE_PATH = '';
OUTPUT_PATH = '';

% Load the pre-processed harmonic feature data
load(HOI_FILE_PATH);
load(FREQ_POW_FILE_PATH);

%% Define constants
% Indices for data access
ACC = 1;  % Accelerometer data
EMG = 2;  % EMG data
PD = 1;   % Parkinson's Disease
ET = 2;   % Essential Tremor

% Get dimensions
num_samples_PD = length(hoi2{ACC}{PD}{1}); 
num_samples_ET = length(hoi2{ACC}{ET}{1}); 
num_features_acc = 2; 
num_features_emg = 4;
num_conditions = 3; 

% Transpose frequency and power data for easier access
max_freq = max_freq';
power = power';

%% Process Parkinson's Disease data
% Initialize arrays
X_PD_acc_2 = zeros(num_samples_PD, num_features_acc, num_conditions);
X_PD_acc_3 = zeros(num_samples_PD, num_features_acc, num_conditions);
X_PD_acc_summative = zeros(num_samples_PD, num_features_acc, num_conditions);
X_PD_acc_non_harmonics = zeros(num_samples_PD, num_features_acc, num_conditions);
X_PD_emg_2 = zeros(num_samples_PD, num_features_emg, num_conditions);
X_PD_emg_3 = zeros(num_samples_PD, num_features_emg, num_conditions);
X_PD_emg_summative = zeros(num_samples_PD, num_features_emg, num_conditions);
X_PD_emg_non_harmonics = zeros(num_samples_PD, num_features_emg, num_conditions);

% Extract features for PD patients
for k = 1:num_conditions
    for i = 1:num_samples_PD
        % Accelerometer features
        for j = 1:num_features_acc
            X_PD_acc_2(i,j,k) = hoi2{ACC}{PD}{k}{i}(j);
            X_PD_acc_3(i,j,k) = hoi3{ACC}{PD}{k}{i}(j);
            X_PD_acc_summative(i,j,k) = hoi_summative{ACC}{PD}{k}{i}(j);
            X_PD_acc_non_harmonics(i,j,k) = hoi_non_harmonics{ACC}{PD}{k}{i}(j);
        end
        
        % EMG features
        for j = 1:num_features_emg
            X_PD_emg_2(i,j,k) = hoi2{EMG}{PD}{k}{i}(j);
            X_PD_emg_3(i,j,k) = hoi3{EMG}{PD}{k}{i}(j);
            X_PD_emg_summative(i,j,k) = hoi_summative{EMG}{PD}{k}{i}(j);
            X_PD_emg_non_harmonics(i,j,k) = hoi_non_harmonics{EMG}{PD}{k}{i}(j);
        end
    end
end

% Reshape the 3D arrays to 2D feature matrices
X_PD_acc_2 = reshape(X_PD_acc_2, [num_samples_PD, num_features_acc * num_conditions]);
X_PD_acc_3 = reshape(X_PD_acc_3, [num_samples_PD, num_features_acc * num_conditions]);
X_PD_acc_summative = reshape(X_PD_acc_summative, [num_samples_PD, num_features_acc * num_conditions]);
X_PD_acc_non_harmonics = reshape(X_PD_acc_non_harmonics, [num_samples_PD, num_features_acc * num_conditions]);
X_PD_emg_2 = reshape(X_PD_emg_2, [num_samples_PD, num_features_emg * num_conditions]);
X_PD_emg_3 = reshape(X_PD_emg_3, [num_samples_PD, num_features_emg * num_conditions]);
X_PD_emg_summative = reshape(X_PD_emg_summative, [num_samples_PD, num_features_emg * num_conditions]);
X_PD_emg_non_harmonics = reshape(X_PD_emg_non_harmonics, [num_samples_PD, num_features_emg * num_conditions]);

% Labels for PD patients (class 0)
y_PD = zeros(num_samples_PD, 1);

%% Process Essential Tremor data
% Initialize arrays
X_ET_acc_2 = zeros(num_samples_ET, num_features_acc, num_conditions);
X_ET_acc_3 = zeros(num_samples_ET, num_features_acc, num_conditions);
X_ET_acc_summative = zeros(num_samples_ET, num_features_acc, num_conditions);
X_ET_acc_non_harmonics = zeros(num_samples_ET, num_features_acc, num_conditions);
X_ET_emg_2 = zeros(num_samples_ET, num_features_emg, num_conditions);
X_ET_emg_3 = zeros(num_samples_ET, num_features_emg, num_conditions);
X_ET_emg_summative = zeros(num_samples_ET, num_features_emg, num_conditions);
X_ET_emg_non_harmonics = zeros(num_samples_ET, num_features_emg, num_conditions);

% Extract features for ET patients
for k = 1:num_conditions
    for i = 1:num_samples_ET
        % Accelerometer features
        for j = 1:num_features_acc 
            X_ET_acc_2(i,j,k) = hoi2{ACC}{ET}{k}{i}(j);
            X_ET_acc_3(i,j,k) = hoi3{ACC}{ET}{k}{i}(j);
            X_ET_acc_summative(i,j,k) = hoi_summative{ACC}{ET}{k}{i}(j);
            X_ET_acc_non_harmonics(i,j,k) = hoi_non_harmonics{ACC}{ET}{k}{i}(j);
        end
        
        % EMG features
        for j = 1:num_features_emg
            X_ET_emg_2(i,j,k) = hoi2{EMG}{ET}{k}{i}(j);
            X_ET_emg_3(i,j,k) = hoi3{EMG}{ET}{k}{i}(j);
            X_ET_emg_summative(i,j,k) = hoi_summative{EMG}{ET}{k}{i}(j);
            X_ET_emg_non_harmonics(i,j,k) = hoi_non_harmonics{EMG}{ET}{k}{i}(j);
        end
    end
end

% Reshape the 3D arrays to 2D feature matrices
X_ET_acc_2 = reshape(X_ET_acc_2, [num_samples_ET, num_features_acc * num_conditions]);
X_ET_acc_3 = reshape(X_ET_acc_3, [num_samples_ET, num_features_acc * num_conditions]);
X_ET_acc_summative = reshape(X_ET_acc_summative, [num_samples_ET, num_features_acc * num_conditions]);
X_ET_acc_non_harmonics = reshape(X_ET_acc_non_harmonics, [num_samples_ET, num_features_acc * num_conditions]);
X_ET_emg_2 = reshape(X_ET_emg_2, [num_samples_ET, num_features_emg * num_conditions]);
X_ET_emg_3 = reshape(X_ET_emg_3, [num_samples_ET, num_features_emg * num_conditions]);
X_ET_emg_summative = reshape(X_ET_emg_summative, [num_samples_ET, num_features_emg * num_conditions]);
X_ET_emg_non_harmonics = reshape(X_ET_emg_non_harmonics, [num_samples_ET, num_features_emg * num_conditions]);

% Labels for ET patients (class 1)
y_ET = ones(num_samples_ET, 1);

%% Extract frequency and power features
% Process frequency data
acc_PD_freq = processFrequencyData(ACC, PD, num_features_acc, max_freq);
emg_PD_freq = processFrequencyData(EMG, PD, num_features_emg, max_freq);
acc_ET_freq = processFrequencyData(ACC, ET, num_features_acc, max_freq);
emg_ET_freq = processFrequencyData(EMG, ET, num_features_emg, max_freq);

% Process power data
acc_PD_power = processPowerData(ACC, PD, num_features_acc, power);
emg_PD_power = processPowerData(EMG, PD, num_features_emg, power);
acc_ET_power = processPowerData(ACC, ET, num_features_acc, power);
emg_ET_power = processPowerData(EMG, ET, num_features_emg, power);

% Process average power data
acc_PD_power_avg = processPowerData(ACC, PD, num_features_acc, power_avg);
emg_PD_power_avg = processPowerData(EMG, PD, num_features_emg, power_avg);
acc_ET_power_avg = processPowerData(ACC, ET, num_features_acc, power_avg);
emg_ET_power_avg = processPowerData(EMG, ET, num_features_emg, power_avg);

%% Concatenate features for all conditions
% Initialize empty matrices
freq_PD_acc = []; freq_PD_emg = []; freq_ET_acc = []; freq_ET_emg = [];
power_PD_acc = []; power_PD_emg = []; power_ET_acc = []; power_ET_emg = [];
power_avg_PD_acc = []; power_avg_PD_emg = []; power_avg_ET_acc = []; power_avg_ET_emg = [];

% Concatenate features across conditions
for i = 1:num_conditions
    freq_PD_acc = [freq_PD_acc, acc_PD_freq{i}];
    freq_PD_emg = [freq_PD_emg, emg_PD_freq{i}];
    freq_ET_acc = [freq_ET_acc, acc_ET_freq{i}];
    freq_ET_emg = [freq_ET_emg, emg_ET_freq{i}];

    power_PD_acc = [power_PD_acc, acc_PD_power{i}];
    power_PD_emg = [power_PD_emg, emg_PD_power{i}];
    power_ET_acc = [power_ET_acc, acc_ET_power{i}];
    power_ET_emg = [power_ET_emg, emg_ET_power{i}];

    power_avg_PD_acc = [power_avg_PD_acc, acc_PD_power_avg{i}];
    power_avg_PD_emg = [power_avg_PD_emg, emg_PD_power_avg{i}];
    power_avg_ET_acc = [power_avg_ET_acc, acc_ET_power_avg{i}];
    power_avg_ET_emg = [power_avg_ET_emg, emg_ET_power_avg{i}];
end

%% Create final feature matrices
% Concatenate all features for PD patients
X_PD = [X_PD_emg_2, ...
    X_PD_emg_3, ...
    X_PD_emg_summative, ...
    X_PD_emg_non_harmonics, ...
    freq_PD_emg, ...
    power_PD_emg, ...
    power_avg_PD_emg, ...
    X_PD_acc_2, ...
    X_PD_acc_3, ...
    X_PD_acc_summative, ...
    X_PD_acc_non_harmonics, ...
    freq_PD_acc, ...
    power_PD_acc, ...
    power_avg_PD_acc];

% Concatenate all features for ET patients
X_ET = [X_ET_emg_2, ...
    X_ET_emg_3, ...
    X_ET_emg_summative, ...
    X_ET_emg_non_harmonics, ...
    freq_ET_emg, ...
    power_ET_emg, ...
    power_avg_ET_emg, ...
    X_ET_acc_2, ...
    X_ET_acc_3, ...
    X_ET_acc_summative, ...
    X_ET_acc_non_harmonics, ...
    freq_ET_acc, ...
    power_ET_acc, ...
    power_avg_ET_acc];

% Combine PD and ET data into a single feature matrix
X = [X_PD; X_ET];
y = [y_PD; y_ET];

% Display information about the feature matrix
fprintf('Feature matrix dimensions: %d samples, %d features\n', size(X, 1), size(X, 2));
fprintf('PD samples: %d, ET samples: %d\n', size(X_PD, 1), size(X_ET, 1));

% Save the feature matrix and labels for classification
save(OUTPUT_PATH, 'X', 'y');
fprintf('Features and labels saved to: %s\n', OUTPUT_PATH);

%% Helper Functions
function freq_data = processFrequencyData(data_type, condition, num_features, max_freq)
    % Extract frequency data for a specific condition and data type
    %
    % Parameters:
    % - data_type: ACC (1) or EMG (2)
    % - condition: PD (1) or ET (2)
    % - num_features: Number of features (2 for ACC, 4 for EMG)
    % - max_freq: Cell array containing max frequency values
    %
    % Returns:
    % - freq_data: Cell array with processed frequency data for 3 conditions
    
    freq_data = cell(1, 3);
    for k = 1:3
        % Get indices for the current condition
        idx = k:3:length(max_freq{data_type}{condition});
        
        % Extract data
        temp_data = max_freq{data_type}{condition}(idx);
        
        % Concatenate along the third dimension
        temp_data = cat(3, temp_data{:});
        
        % Reshape to have samples as rows, features as columns
        freq_data{k} = reshape(temp_data, [], num_features);
    end
end

function power_data = processPowerData(data_type, condition, num_features, power)
    % Extract power data for a specific condition and data type
    %
    % Parameters:
    % - data_type: ACC (1) or EMG (2)
    % - condition: PD (1) or ET (2)
    % - num_features: Number of features (2 for ACC, 4 for EMG)
    % - power: Cell array containing power values
    %
    % Returns:
    % - power_data: Cell array with processed power data for 3 conditions
    
    power_data = cell(1, 3);
    for k = 1:3
        % Get indices for the current condition
        idx = k:3:length(power{data_type}{condition});
        
        % Extract data
        temp_data = power{data_type}{condition}(idx);
        
        % Concatenate along the third dimension
        temp_data = cat(3, temp_data{:});
        
        % Reshape to have samples as rows, features as columns
        power_data{k} = reshape(temp_data, [], num_features);
    end
end
