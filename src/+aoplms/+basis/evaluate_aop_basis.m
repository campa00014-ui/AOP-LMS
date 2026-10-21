function Phi = evaluate_aop_basis(x, coeffs)
    % Evaluates AOP basis functions for a signal.
    % Inputs:
    %   x : transmit signal
    %   coeffs : cell array of AOP coefficient vectors
    % Output:
    %   Phi : NxK matrix ( column p contains phi_p(x[n]) )

    x = x(:);

    N = numel(x);
    K = numel(coeffs);

    Phi = zeros(N, K);

    absx2 = abs(x).^2;

    for p = 1:K
        c = coeffs{p};
        Phi(:, p) = polyval(fliplr(c), absx2) .* x;
    end

end