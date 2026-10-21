function y = apply_si_channel(x, h_si)
% Apply the self-interference propagation FIR channel.

x = x(:);
h_si = h_si(:);

y = filter(h_si, 1, x);

end
