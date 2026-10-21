function mu = estimate_even_moments(x, P)
    % Estimate even moments of the transmit signal.
    % Inputs:
    %   x : transmit signal, column or row vector
    %   P : highest nonlinear order used by the AOP basis
    % Output:
    %   mu : Px1 vector of estimated even moments

    x = x(:);
    mu = zeros(P, 1);

    absx = abs(x);

    for k = 1:P
        mu(k) = mean(absx.^(2*k));
    end

end