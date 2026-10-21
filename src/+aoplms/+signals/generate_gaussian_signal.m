function x = generate_gaussian_signal(N, power)

    % Generates a complex circularly symmetric Gaussian signal
    % x[n] ~ CN(0, power)

    x = sqrt(power/2) * (randn(N, 1) + 1i * randn(N, 1));

end
