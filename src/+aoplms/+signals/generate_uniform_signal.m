function x = generate_uniform_signal(N, amplitude)

% Generates a real-valued uniform signal.

x = -amplitude + 2 * amplitude * rand(N, 1);

end
