function x = generate_ofdm_signal(N, M, power, nfft, cpLen)
% GENERATE_OFDM_SIGNAL Generates a time-domain OFDM signal from M-QAM subcarriers.
%
% Inputs:
%   N     : Number of output time-domain samples to return.
%   M     : QAM constellation order (4, 16, 64, 256).
%   power : Desired average power of the time-domain signal (default: 1).
%   nfft  : FFT size / number of active subcarriers (default: 64).
%   cpLen : Cyclic prefix length in samples (default: 0).
%
% Outputs:
%   x     : Column vector of length N containing the time-domain OFDM samples.
%
% Details:
%   Each complete OFDM symbol consists of nfft active subcarriers modulated
%   by independent unit-average-power M-QAM symbols. The output of MATLAB's
%   ofdmmod is scaled by sqrt(nfft) to preserve unit average power from the
%   frequency domain to the time domain under MATLAB's 1/nfft IFFT scaling convention.
%   The serialized time-domain waveform is truncated to exactly N samples.

if nargin < 3 || isempty(power)
    power = 1;
end
if nargin < 4 || isempty(nfft)
    nfft = 64;
end
if nargin < 5 || isempty(cpLen)
    cpLen = 0;
end

if N < 0 || N ~= floor(N)
    error("N must be a non-negative integer.");
end
if N == 0
    x = zeros(0, 1);
    return;
end
if ~ismember(M, [4 16 64 256])
    error("M must be 4, 16, 64, or 256.");
end
if ~isscalar(power) || power <= 0
    error("power must be a positive scalar.");
end
if ~isscalar(nfft) || nfft < 1 || nfft ~= floor(nfft)
    error("nfft must be a positive integer.");
end
if ~isscalar(cpLen) || cpLen < 0 || cpLen ~= floor(cpLen)
    error("cpLen must be a non-negative integer.");
end

symbolLen = nfft + cpLen;
numBlocks = ceil(N / symbolLen);
numQam = numBlocks * nfft;

qamSymbols = aoplms.signals.generate_qam_signal(numQam, M, 1);
X = reshape(qamSymbols, nfft, numBlocks);

% MATLAB's ofdmmod divides by nfft during IFFT, reducing average power to 1/nfft.
% Scaling by sqrt(nfft) restores unit average power in the time domain.
y = ofdmmod(X, nfft, cpLen);
x = sqrt(power) * sqrt(nfft) * y(:);

% Truncate serialized waveform to exactly requested length N
x = x(1:N);

end
