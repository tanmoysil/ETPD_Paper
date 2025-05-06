function [freq_max, pow, pow_avg] = peakfreq_using_wavelet(signal, fs)
%PEAKFREQ_USING_WAVELET Extract peak frequencies and power using wavelet transform
%
% This function analyzes a multi-channel signal using wavelet transform to 
% determine the frequency components with maximum magnitude for each channel.
%
% Inputs:
%   signal      - Input signal matrix (samples x channels)
%   fs          - Sampling frequency in Hz
%
% Outputs:
%   freq_max    - Peak frequencies for each channel (Hz)
%   pow         - Power at peak frequencies for each channel
%   pow_avg     - Average power across frequency spectrum for each channel
%
% Author: Tanmoy Sil, University Hospital Würzburg
% Date: May 2025

% Import necessary dependencies
import freqmix.spectrum.*

% Define frequency range of interest (2-20 Hz covers most pathological tremors)
freqs = 2:20;  % Increased frequency resolution for more precise peaks

% Calculate wavelet spectrum (signal must be transposed for WaveletSpecEddy)
spectrum = WaveletSpec(signal', 1/fs, freqs);

% Calculate power as squared magnitude of the spectrum
magnitude = abs(spectrum.Spectrum).^2;

% Initialize output arrays
num_channels = size(magnitude, 3);
freq_max = zeros(1, num_channels);
pow = zeros(1, num_channels);
pow_avg = zeros(1, num_channels);

% Find peak frequencies and power for each channel
for i = 1:num_channels
    % Average power across time
    avg_power_per_freq = mean(magnitude(:,:,i), 2);
    
    % Find maximum power and corresponding frequency
    [pow(i), idx] = max(avg_power_per_freq);
    freq_max(i) = spectrum.Freqs(idx);
    
    % Calculate average power across the entire spectrum
    pow_avg(i) = mean(avg_power_per_freq);
end

end