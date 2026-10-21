function x = generate_nonstationary_ofdm_signal(orders, lengths, power, nfft, cpLen)
% GENERATE_NONSTATIONARY_OFDM_SIGNAL Generates non-stationary OFDM signal with adaptive QAM.
%
% Inputs:
%   orders  : Vector of QAM modulation orders for each segment (e.g., [16 64 4 256 64])
%   lengths : Vector of segment lengths in samples (e.g., [2200 5500 3300 4400 2200])
%   power   : Desired average power of each segment (default: 1)
%   nfft    : FFT size / active subcarriers (default: 64)
%   cpLen   : Cyclic prefix length in samples (default: 0)
%
% Outputs:
%   x       : Column vector of concatenated time-domain OFDM samples.
%
% Details:
%   Follows the Figure 8 specification: each MCS segment is generated independently
%   using complete OFDM symbols, serialized, truncated to its exact segment length,
%   and concatenated. This retains exact sample durations while preventing subcarriers
%   from mixing across different MCS constellations within any OFDM symbol.

if numel(lengths) ~= numel(orders)
    error("Number of segments must match number of orders.");
end
if any(lengths < 0) || any(lengths ~= floor(lengths))
    error("Segment lengths must be non-negative integers.");
end

if nargin < 3 || isempty(power)
    power = 1;
end
if nargin < 4 || isempty(nfft)
    nfft = 64;
end
if nargin < 5 || isempty(cpLen)
    cpLen = 0;
end

total_samples = sum(lengths);
x = zeros(total_samples, 1);
start_index = 1;

for k = 1:length(orders)
    seg_length = lengths(k);
    end_index = start_index + seg_length - 1;

    x_segment = aoplms.signals.generate_ofdm_signal( ...
        seg_length, orders(k), power, nfft, cpLen);

    x(start_index:end_index) = x_segment;
    start_index = end_index + 1;
end

end
