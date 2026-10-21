function [mu, symbols] = compute_qam_moments(M, maxOrder)
% Exact even moments of equally likely, unit-average-power square QAM.
% mu = [mu_2; mu_4; ...; mu_2P], where P = maxOrder.

if nargin < 2
    maxOrder = 5;
end

if ~ismember(M, [4 16 64 256])
    error("M must be 4, 16, 64, or 256.");
end
if maxOrder < 1 || maxOrder ~= floor(maxOrder)
    error("maxOrder must be a positive integer.");
end

levels = -(sqrt(M) - 1):2:(sqrt(M) - 1);
[inPhase, quadrature] = meshgrid(levels, levels);
symbols = inPhase(:) + 1i * quadrature(:);
symbols = symbols / sqrt(mean(abs(symbols).^2));

mu = zeros(maxOrder, 1);
for k = 1:maxOrder
    mu(k) = mean(abs(symbols).^(2*k));
end

end
