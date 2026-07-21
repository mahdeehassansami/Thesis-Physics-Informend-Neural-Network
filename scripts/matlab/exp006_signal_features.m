function features = exp006_signal_features(signal, sampling_hz)
%EXP006_SIGNAL_FEATURES Match thesis_work.multi_dataset.extract_signal_features.
%
% The implementation deliberately uses population moments and a periodic Hann
% window so its output matches NumPy/SciPy rather than MATLAB's default sample
% standard deviation and symmetric window conventions.

x = double(signal(:));
x = x(isfinite(x));
if numel(x) < 8
    error("EXP006:ShortSignal", "At least eight finite samples are required.");
end

mean_value = mean(x);
centered = x - mean_value;
rms_value = sqrt(mean(x .^ 2));
second_moment = mean(centered .^ 2);
third_moment = mean(centered .^ 3);
fourth_moment = mean(centered .^ 4);

sample_count = numel(centered);
windowed = centered .* hann(sample_count, "periodic");
spectrum = fft(windowed);
positive_count = floor(sample_count / 2) + 1;
power = abs(spectrum(1:positive_count)) .^ 2;
frequencies = (0:(positive_count - 1))' .* (double(sampling_hz) / sample_count);
power_sum = sum(power) + 1e-12;
probability = power ./ power_sum;
centroid = sum(frequencies .* probability);
bandwidth = sqrt(sum(((frequencies - centroid) .^ 2) .* probability));
spectral_entropy = -sum(probability .* log(probability + 1e-12));
spectral_entropy = spectral_entropy / log(max(2, numel(probability)));
high_frequency_threshold = 0.25 * (double(sampling_hz) / 2.0);
high_frequency_ratio = sum(power(frequencies >= high_frequency_threshold)) / power_sum;

features = struct( ...
    "rms", rms_value, ...
    "std", sqrt(second_moment), ...
    "ptp", max(x) - min(x), ...
    "kurtosis", fourth_moment / (second_moment ^ 2 + 1e-12), ...
    "crest_factor", max(abs(x)) / (rms_value + 1e-12), ...
    "mean_abs", mean(abs(x)), ...
    "skewness", third_moment / (second_moment ^ 1.5 + 1e-12), ...
    "spectral_centroid", centroid, ...
    "spectral_bandwidth", bandwidth, ...
    "spectral_entropy", spectral_entropy, ...
    "high_frequency_ratio", high_frequency_ratio);
end
