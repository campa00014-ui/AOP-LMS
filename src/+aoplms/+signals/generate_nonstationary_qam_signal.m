function x = generate_nonstationary_qam_signal(orders, lengths, power)

% Generates a non-stationary QAM signal.
% Each segment has a different QAM modulation order.

if numel(lengths) ~= numel(orders)
    error("Number of segments must match number of orders.");
end

total_samples = sum(lengths);

x = zeros(total_samples, 1);

start_index = 1;

for k = 1:length(orders)

    M = orders(k);
    segment_length = lengths(k);

    end_index = start_index + segment_length - 1;

    x_segment = aoplms.signals.generate_qam_signal( ...
        segment_length, M, power);

    x(start_index:end_index) = x_segment;

    start_index = end_index + 1;

end

end
