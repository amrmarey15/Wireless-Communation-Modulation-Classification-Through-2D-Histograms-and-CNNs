%This function generates a single random complex noise noise value at specified
%SNR for a channel changing every transmitted symbol.

function N_final = gen_noise_channel_1bit(SNR,h,t)
X = 10^(SNR/10);
variance = ((abs(h))^2)*((abs(t))^2);
variance = variance/X;
N_real = (1/sqrt(2))*randn(1,1); % Var = 0.5
N_Imag = (1/sqrt(2))*randn(1,1);
N = N_real + j*N_Imag;
N_final = sqrt(variance)*N;
end