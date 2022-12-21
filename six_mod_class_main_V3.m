clear;
clc;
close all
%%%%%%%%%%Parameter Stuff/Initialization%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_of_reps = 100;
ModNo = 6;
training_percent = 0.7;
samples = 8200;
m_naka = 3;
channel_key = 0;
m_naka_key = 0;
Bins_no=80;



Iter_No = 12;
MSE_channel_estimate = 0;
constellation1 = QPSK_generate(2); %Generate BPSK constellation
constellation2 = QPSK_generate(4)*exp(j*(pi/4)); %Generate QPSK constellation
constellation3 = QAM_generate(8); %Generate 8-PSK
constellation4 = QAM_generate(16);
constellation5 = QAM_generate(32);
constellation6 = QAM_generate(64);

SNR_dB_vector = linspace(-6,15,8); %We will be classifying 8dB, 10, 12, ----, until 38dB
M(ModNo) = 0;
ClassPerAvg=zeros(ModNo,length(SNR_dB_vector));
M(1) = length(constellation1); M(2) = length(constellation2); M(3) = length(constellation3);
M(4) = length(constellation4); M(5) = length(constellation5); M(6) = length(constellation6);
length_SNR_vector = length(SNR_dB_vector);
num_of_train_reps= training_percent*num_of_reps; %Numbers of repititions for each modulation technique with constant SNR
num_of_test_reps= num_of_reps-num_of_train_reps;
training_columns = length(SNR_dB_vector)*num_of_train_reps*ModNo;
test_columns = length(SNR_dB_vector)*num_of_test_reps*ModNo;




%%%%%%%%%%%%%%Transmitted&Recieved Signal Generation%%%%%%%%%%%%%%%%%%%%%%%
T_train(samples,training_columns) = 0;
R_train(samples,training_columns) = 0;
T_test(samples,test_columns) = 0;
R_test(samples,test_columns) = 0;
YTrain(training_columns) = 0;
YTest(test_columns) = 0;
eq_R_train(samples,training_columns) = 0;
eq_R_test(samples,test_columns) = 0;

delta_h = 0; % for now

for b=1:length_SNR_vector
    SNR = SNR_dB_vector(b);
    for k=1:ModNo*num_of_train_reps
        current_column = k+(b-1)*ModNo*num_of_train_reps;
        if(k<=1*num_of_train_reps)
            T_train(:,current_column) = generate_transmit_signal(samples,constellation1,M(1));
            YTrain(current_column) = 1;
            for c = 1:samples
                h_observed = (1/sqrt(2))*randn(1,1) + j*(1/sqrt(2))*randn(1,1); %Change Channel Every symbol.
                delta_h = ((1/sqrt(2))*randn(1,1) + j*(1/sqrt(2))*randn(1,1));
                delta_h = sqrt(MSE_channel_estimate)*delta_h;
                h_detected = h_observed + delta_h;
                R_train(c,current_column) = h_observed*T_train(c,current_column)+gen_noise_channel_1bit(SNR,h_observed,T_train(c,current_column));
                eq_R_train(c,current_column) = R_train(c,current_column)/h_detected;
            end
        end
        if( (k>1*num_of_train_reps) && (k<=2*num_of_train_reps))
            T_train(:,current_column) = generate_transmit_signal(samples,constellation2,M(2));
            YTrain(current_column) = 2;
            for c = 1:samples
                h_observed = (1/sqrt(2))*randn(1,1) + j*(1/sqrt(2))*randn(1,1); %Change Channel Every symbol.
                delta_h = ((1/sqrt(2))*randn(1,1) + j*(1/sqrt(2))*randn(1,1));
                delta_h = sqrt(MSE_channel_estimate)*delta_h;
                h_detected = h_observed + delta_h;
                R_train(c,current_column) = h_observed*T_train(c,current_column)+gen_noise_channel_1bit(SNR,h_observed,T_train(c,current_column));
                eq_R_train(c,current_column) = R_train(c,current_column)/h_detected;
            end
        end
        if( (k>2*num_of_train_reps) && (k<=3*num_of_train_reps))
            T_train(:,current_column) = generate_transmit_signal(samples,constellation3,M(3));
            YTrain(current_column) = 3;
            for c = 1:samples
                h_observed = (1/sqrt(2))*randn(1,1) + j*(1/sqrt(2))*randn(1,1); %Change Channel Every symbol.
                delta_h = ((1/sqrt(2))*randn(1,1) + j*(1/sqrt(2))*randn(1,1));
                delta_h = sqrt(MSE_channel_estimate)*delta_h;
                h_detected = h_observed + delta_h;
                R_train(c,current_column) = h_observed*T_train(c,current_column)+gen_noise_channel_1bit(SNR,h_observed,T_train(c,current_column));
                eq_R_train(c,current_column) = R_train(c,current_column)/h_detected;
            end
        end
        if( (k>3*num_of_train_reps) && (k<=4*num_of_train_reps))
            T_train(:,current_column) = generate_transmit_signal(samples,constellation4,M(4));
            YTrain(current_column) = 4;
            for c = 1:samples
                h_observed = (1/sqrt(2))*randn(1,1) + j*(1/sqrt(2))*randn(1,1); %Change Channel Every symbol.
                delta_h = ((1/sqrt(2))*randn(1,1) + j*(1/sqrt(2))*randn(1,1));
                delta_h = sqrt(MSE_channel_estimate)*delta_h;
                h_detected = h_observed + delta_h;
                R_train(c,current_column) = h_observed*T_train(c,current_column)+gen_noise_channel_1bit(SNR,h_observed,T_train(c,current_column));
                eq_R_train(c,current_column) = R_train(c,current_column)/h_detected;
            end
        end
        if( (k>4*num_of_train_reps) && (k<=5*num_of_train_reps))
            T_train(:,current_column) = generate_transmit_signal(samples,constellation5,M(5));
            YTrain(current_column) = 5;
            for c = 1:samples
                h_observed = (1/sqrt(2))*randn(1,1) + j*(1/sqrt(2))*randn(1,1); %Change Channel Every symbol.
                delta_h = ((1/sqrt(2))*randn(1,1) + j*(1/sqrt(2))*randn(1,1));
                delta_h = sqrt(MSE_channel_estimate)*delta_h;
                h_detected = h_observed + delta_h;
                R_train(c,current_column) = h_observed*T_train(c,current_column)+gen_noise_channel_1bit(SNR,h_observed,T_train(c,current_column));
                eq_R_train(c,current_column) = R_train(c,current_column)/h_detected;
            end           
        end
        if( (k>5*num_of_train_reps) && (k<=6*num_of_train_reps))
            T_train(:,current_column) = generate_transmit_signal(samples,constellation6,M(6));
            YTrain(current_column) = 6;
            for c = 1:samples
                h_observed = (1/sqrt(2))*randn(1,1) + j*(1/sqrt(2))*randn(1,1); %Change Channel Every symbol.
                delta_h = ((1/sqrt(2))*randn(1,1) + j*(1/sqrt(2))*randn(1,1));
                delta_h = sqrt(MSE_channel_estimate)*delta_h;
                h_detected = h_observed + delta_h;
                R_train(c,current_column) = h_observed*T_train(c,current_column)+gen_noise_channel_1bit(SNR,h_observed,T_train(c,current_column));
                eq_R_train(c,current_column) = R_train(c,current_column)/h_detected;
            end            
        end
    end
    for k=1:ModNo*num_of_test_reps
        current_column = k+(b-1)*ModNo*num_of_test_reps;
        if(k<=1*num_of_test_reps)
            T_test(:,current_column) = generate_transmit_signal(samples,constellation1,M(1));
            YTest(current_column) = 1;
            for c = 1:samples
                h_observed = (1/sqrt(2))*randn(1,1) + j*(1/sqrt(2))*randn(1,1); %Change Channel Every symbol.
                delta_h = ((1/sqrt(2))*randn(1,1) + j*(1/sqrt(2))*randn(1,1));
                delta_h = sqrt(MSE_channel_estimate)*delta_h;
                h_detected = h_observed + delta_h;
                R_test(c,current_column) = h_observed*T_test(c,current_column)+gen_noise_channel_1bit(SNR,h_observed,T_test(c,current_column));
                eq_R_test(c,current_column) = R_test(c,current_column)/h_detected;
            end    
        end
        if( (k>1*num_of_test_reps) && (k<=2*num_of_test_reps))
            T_test(:,current_column) = generate_transmit_signal(samples,constellation2,M(2));
            YTest(current_column) = 2;
            for c = 1:samples
                h_observed = (1/sqrt(2))*randn(1,1) + j*(1/sqrt(2))*randn(1,1); %Change Channel Every symbol.
                delta_h = ((1/sqrt(2))*randn(1,1) + j*(1/sqrt(2))*randn(1,1));
                delta_h = sqrt(MSE_channel_estimate)*delta_h;
                h_detected = h_observed + delta_h;
                R_test(c,current_column) = h_observed*T_test(c,current_column)+gen_noise_channel_1bit(SNR,h_observed,T_test(c,current_column));
                eq_R_test(c,current_column) = R_test(c,current_column)/h_detected;
            end
        end
        if( (k>2*num_of_test_reps) && (k<=3*num_of_test_reps))
            T_test(:,current_column) = generate_transmit_signal(samples,constellation3,M(3));
            YTest(current_column) = 3;
            for c = 1:samples
                h_observed = (1/sqrt(2))*randn(1,1) + j*(1/sqrt(2))*randn(1,1); %Change Channel Every symbol.
                delta_h = ((1/sqrt(2))*randn(1,1) + j*(1/sqrt(2))*randn(1,1));
                delta_h = sqrt(MSE_channel_estimate)*delta_h;
                h_detected = h_observed + delta_h;
                R_test(c,current_column) = h_observed*T_test(c,current_column)+gen_noise_channel_1bit(SNR,h_observed,T_test(c,current_column));
                eq_R_test(c,current_column) = R_test(c,current_column)/h_detected;
            end           
        end
        if( (k>3*num_of_test_reps) && (k<=4*num_of_test_reps))
            T_test(:,current_column) = generate_transmit_signal(samples,constellation4,M(4));
            YTest(current_column) = 4;
            for c = 1:samples
                h_observed = (1/sqrt(2))*randn(1,1) + j*(1/sqrt(2))*randn(1,1); %Change Channel Every symbol.
                delta_h = ((1/sqrt(2))*randn(1,1) + j*(1/sqrt(2))*randn(1,1));
                delta_h = sqrt(MSE_channel_estimate)*delta_h;
                h_detected = h_observed + delta_h;
                R_test(c,current_column) = h_observed*T_test(c,current_column)+gen_noise_channel_1bit(SNR,h_observed,T_test(c,current_column));
                eq_R_test(c,current_column) = R_test(c,current_column)/h_detected;
            end           
        end
        if( (k>4*num_of_test_reps) && (k<=5*num_of_test_reps))
            T_test(:,current_column) = generate_transmit_signal(samples,constellation5,M(5));
            YTest(current_column) = 5;
            for c = 1:samples
                h_observed = (1/sqrt(2))*randn(1,1) + j*(1/sqrt(2))*randn(1,1); %Change Channel Every symbol.
                delta_h = ((1/sqrt(2))*randn(1,1) + j*(1/sqrt(2))*randn(1,1));
                delta_h = sqrt(MSE_channel_estimate)*delta_h;
                h_detected = h_observed + delta_h;
                R_test(c,current_column) = h_observed*T_test(c,current_column)+gen_noise_channel_1bit(SNR,h_observed,T_test(c,current_column));
                eq_R_test(c,current_column) = R_test(c,current_column)/h_detected;
            end            
        end
        if( (k>5*num_of_test_reps) && (k<=6*num_of_test_reps))
            T_test(:,current_column) = generate_transmit_signal(samples,constellation6,M(6));
            YTest(current_column) = 6;
            for c = 1:samples
                h_observed = (1/sqrt(2))*randn(1,1) + j*(1/sqrt(2))*randn(1,1); %Change Channel Every symbol.
                delta_h = ((1/sqrt(2))*randn(1,1) + j*(1/sqrt(2))*randn(1,1));
                delta_h = sqrt(MSE_channel_estimate)*delta_h;
                h_detected = h_observed + delta_h;
                R_test(c,current_column) = h_observed*T_test(c,current_column)+gen_noise_channel_1bit(SNR,h_observed,T_test(c,current_column));
                eq_R_test(c,current_column) = R_test(c,current_column)/h_detected;
            end      
        end
    end
end


X_Train = eq_R_train;
X_Test = eq_R_test;
YTrain = YTrain';
YTest = YTest';

clear T_train R_test T_test R_test eq_R_train eq_R_test

% Normalize the dataset
for m=1:training_columns
    X_Train(:,m)=X_Train(:,m)./std(X_Train(:,m));
end



I1=real(X_Train);
Q1=imag(X_Train);
I2=real(X_Test);
Q2=imag(X_Test);
Hist_IQH_Train= Fun_2DHistCNN(Bins_no,I1,Q1);
Hist_IQH_Test= Fun_2DHistCNN(Bins_no,I2,Q2);
YTest=categorical(YTest);
YTrain=categorical(YTrain);
YTest_temp = YTest;
YTrain_temp = YTrain;

    layers = [
    imageInputLayer([  Bins_no Bins_no  1])

    convolution2dLayer(3,128,'Padding',2)
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',1)

    convolution2dLayer(3,128,'Padding',2)
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',1)

    fullyConnectedLayer(6)
    softmaxLayer
    classificationLayer];

    options = trainingOptions('adam', ...
    'InitialLearnRate',0.001, ...
    'SquaredGradientDecayFactor',0.99, ...
    'MaxEpochs',7, ...
    'MiniBatchSize',64, ...
    'Shuffle','every-epoch');
%% Train Network
% Create the network training options. Train for 30 epochs. Set the initial
% learn rate to 0.001 and lower the learning rate after 20 epochs. Monitor the
% network accuracy during training by specifying validation data and validation
% frequency. The software trains the network on the training data and calculates
% the accuracy on the validation data at regular intervals during training. The
% validation data is not used to update the network weights. Turn on the training
% progress plot, and turn off the command window output.
%%
for iter=1:Iter_No
    YTest = YTest_temp;
    YTrain = YTrain_temp;
    %%%%%%%%%%%%%%%%%%%%%%Split data to traning and testing

    %% Build 2D Histogram by calling 2D histogram function


    % Combine all the layers together in a |Layer| array.
    %%




    % 'Plots','training-progress',...
    %%
    net = trainNetwork(Hist_IQH_Train,YTrain,layers,options);

    %%
    % Examine the details of the network architecture contained in the |Layers|
    % property of |net|.
    %%
    net.Layers
    %% Test Network
    % Test the performance of the network by evaluating the accuracy on the validation
    % data.
    %
    % Use |predict| to predict the angles of rotation of the validation images.
    %%

    YPred = classify(net,Hist_IQH_Test);
    % plotconfusion(YTest,YPred)
    confusionmat(YTest,YPred)
    YTest=reshape(YTest,180,length_SNR_vector);
    YPred=reshape(YPred,180,length_SNR_vector);
    for i=1:length_SNR_vector
        Conf=confusionmat(YTest(:,i),YPred(:,i));
        ClassPer(:,i)=diag(Conf)/.3;
    end
    ClassPer;
    ClassPerAvg=ClassPerAvg+ClassPer
    temp_class(iter)=mean(mean(ClassPer));
end
ClassPerAvg=ClassPerAvg/iter
temp_class
mean(temp_class)
ClassRes=mean(mean(ClassPerAvg))
mean(ClassPerAvg)
% Comp_Time_F=mean(Comp_Time)
figure
hold
for i=1:ModNo
    plot(SNR_dB_vector,ClassPerAvg(i,:),'-*','linewidth',2)
end
hold off
grid on
legend('BPSK','QPSK','8QAM','16QAM','32QAM','64QAM')
axis([SNR_dB_vector(1) SNR_dB_vector(end) 0 105])
xlabel ('SNR in dB')
ylabel ('Correct recognition rate %')