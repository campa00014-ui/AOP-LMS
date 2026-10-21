function x = generate_qam_signal(N, M, power)

    % Generates square M-QAM symbols with the specified average power.
    % Uses direct 4-QAM generation to match the reference implementation.

    if M == 4
        bits = randi([0 1], N, 2);

        x = ((2 * bits(:, 1) - 1) + ...
            1i * (2 * bits(:, 2) - 1)) / sqrt(2);
    else
        data = randi([0 M-1], N, 1);

        x = qammod(data, M, ...
            'InputType', 'integer', ...
            'UnitAveragePower', true);
    end

    x = sqrt(power) * x;

end