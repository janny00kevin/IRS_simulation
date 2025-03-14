% Base station to the IRS
current_time = datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss');
disp(current_time);
% parpool(14);

tic;
num_samples = 2.4e4;
nTx = 8;
nRx = 4;

H_samples = complex(zeros(num_samples, nRx, nTx), zeros(num_samples, nRx, nTx));

parfor i = 1:num_samples
    H_samples(i,:,:) = single(GENERATE_CHANNEL_IU('ct', 'UMa', 'TR', 28));
end
toc;

save('UMa_IU_test_24k_8_4_.mat',"H_samples")