function x = generate_mixed_signal(N, M, power)

% Generates a Gaussian-QAM mixed signal.
ratio = 6/12;
x_gaussian = aoplms.signals.generate_gaussian_signal(N, power*ratio);
x_qam = aoplms.signals.generate_qam_signal(N, M, power*(1-ratio));

x = x_gaussian + x_qam;

end
