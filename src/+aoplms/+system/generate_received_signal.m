function [received, paOutput] = generate_received_signal( x, h_pa, gamma, beta, noise)
% Apply the Saleh PA-with-memory model.

x = x(:);
h_pa = h_pa(:);
noise = noise(:);

if numel(noise) ~= numel(x)
    error("noise must have the same number of samples as x.");
end

g = gamma * x ./ (1 + beta * abs(x).^2);

paOutput = filter(h_pa, 1, g);

received = paOutput + noise;

end
