function entry = get_qam_lut_entry(lut, M)
% Return the precomputed moments and coefficients for the current QAM order.

switch M
    case 4
        entry = lut.qam4;
    case 16
        entry = lut.qam16;
    case 64
        entry = lut.qam64;
    case 256
        entry = lut.qam256;
    otherwise
        error("M must be 4, 16, 64, or 256.");
end

end
