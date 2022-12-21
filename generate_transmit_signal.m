function T = generate_transmit_signal(samples,constellation,M)
for i=1:samples
    % T signal generation 
    pos = randi(M);
    T(i) = constellation(pos);
end
end