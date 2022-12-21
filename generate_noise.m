function N_final = generate_noise(SNR,samples,M,key)
%key = 0: Working in Eb_N0
%key = 1; working in SNR

X = 10^(SNR/10);
if(key == 0)
     % convert to SNR without dB
    variance = 1/(X*log2(M));
end
if(key == 1)
        variance = 1/(X);
end
    %generate N = gauss(0,0.5)+j*gauss(0,0.5)
    N_real = (1/sqrt(2))*randn(1,samples); % Var = 0.5
    N_Imag = (1/sqrt(2))*randn(1,samples);
    N = N_real + j*N_Imag;  
    N_final = sqrt(variance)*N;
end