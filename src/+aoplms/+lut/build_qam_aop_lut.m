function lut = build_qam_aop_lut(maxOrder)
% Build the QAM LUT once, before running LMS.
% P=1, 3, and 5 are supported by all nonconstant-envelope entries.

if nargin < 1
    maxOrder = 5;
end
if ~ismember(maxOrder, [1 3 5])
    error("Use P=1, 3, or 5; 16-QAM has only three independent branches.");
end

lut.maxOrder = maxOrder;
lut.modOrders = [4 16 64 256];

for M = lut.modOrders
    entry.moments = aoplms.lut.compute_qam_moments(M, maxOrder);

    if M == 4
        % |x|=1, so every |x|^(2k)*x equals x. Higher AOPs do not exist.
        entry.coeffs = {1};
    else
        entry.coeffs = aoplms.basis.construct_aop_coeffs( ...
            entry.moments, maxOrder);
    end

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
