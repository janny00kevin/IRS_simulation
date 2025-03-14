% Base station to the IRS
current_time = datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss');
disp(current_time);
parpool(14);

tic;
num_samples = 2.4e4;
nTx = 4;
nRx = 8;

H_samples = complex(zeros(num_samples, nRx, nTx), zeros(num_samples, nRx, nTx));

parfor i = 1:num_samples
    H_samples(i,:,:) = single(GENERATE_CHANNEL_BI('ct', 'UMa', 'TR', 28));
end
toc;

% H_samples = squeeze(H_samples(1,:,:));
% disp(H_samples(1,:,:))
save('UMa_BI_test_24k_4_8_.mat',"H_samples")