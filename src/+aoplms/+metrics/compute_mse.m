function mse = compute_mse(errorTrials)
% Compute the MSE by averaging over multiple trial runs.
% Each column of errorTrials contains one independent error sequence.

if isvector(errorTrials)
    errorTrials = errorTrials(:);
end

mse = mean(abs(errorTrials).^2, 2);

end
