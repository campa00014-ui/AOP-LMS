function lut = build_ofdm_aop_lut(maxOrder, nfft, cpLen, numOfflineSamples, seed)
% BUILD_OFDM_AOP_LUT Builds the offline AOP LUT for OFDM waveforms.
%
% Inputs:
%   maxOrder          : Maximum nonlinear polynomial order P (default: 5)
%   nfft              : FFT size / number of subcarriers (default: 64)
%   cpLen             : Cyclic prefix length in samples (default: 0)
%   numOfflineSamples : Number of offline samples for moment estimation (default: 1000000)
%   seed              : Deterministic seed base for reproducibility (default: 12345)
%
% Outputs:
%   lut : Struct containing precomputed moments and AOP coefficients for
%         4-QAM, 16-QAM, 64-QAM, and 256-QAM under OFDM signaling.
%
% Details:
%   Unlike direct single-carrier QAM, the time-domain OFDM waveform formed by
%   summing independent subcarriers has amplitude variation and approximately
%   circularly symmetric complex Gaussian statistics (by CLT). Therefore,
%   even for 4-QAM, higher-order AOP basis functions exist and are well-conditioned.
%   Moments and coefficients are estimated offline from large deterministic datasets
%   and stored for real-time reference.

if nargin < 1 || isempty(maxOrder)
    maxOrder = 5;
end
if nargin < 2 || isempty(nfft)
    nfft = 64;
end
if nargin < 3 || isempty(cpLen)
    cpLen = 0;
end
if nargin < 4 || isempty(numOfflineSamples)
    numOfflineSamples = 1000000;
end
if nargin < 5 || isempty(seed)
    seed = 12345;
end

if maxOrder < 1 || mod(maxOrder, 2) == 0
    error("maxOrder must be a positive odd integer.");
end

% Preserve the caller's random number generator state upon exit
savedRng = rng();
cleanupRng = onCleanup(@() rng(savedRng));

lut.maxOrder = maxOrder;
lut.modOrders = [4 16 64 256];
lut.nfft = nfft;
lut.cpLen = cpLen;
lut.numOfflineSamples = numOfflineSamples;

for M = lut.modOrders
    rng(seed + M);
    xOfdm = aoplms.signals.generate_ofdm_signal( ...
        numOfflineSamples, M, 1, nfft, cpLen);

    entry.moments = aoplms.basis.estimate_even_moments(xOfdm, maxOrder);
    [entry.coeffs, entry.info] = aoplms.basis.construct_aop_coeffs( ...
        entry.moments, maxOrder);

    switch M
        case 4
            lut.qam4 = entry;
        case 16
            lut.qam16 = entry;
        case 64
            lut.qam64 = entry;
        case 256
            lut.qam256 = entry;
    end
end

end
