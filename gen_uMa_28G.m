
clc
tic;
num_samples = 3000;
nRx = 4;
nTx = 36;

H_samples = complex(zeros(num_samples, nRx, nTx), zeros(num_samples, nRx, nTx));

for i = 1:num_samples
    H_samples(i,:,:) = single(GENERATE_CHANNEL('ct', 'UMa', 'TR', 28));
end
toc;

save('UMa_testing_3k_4_36_.mat',"H_samples")