clc
clear all
close all


model = 'wgan-div_s3_200_5_64_0.0002_0.5_0.999_100_0.01';
k_std = 3;

load('datasets/ms.mat');
r = dir(['results/', model, '/predict']);
mkdir(['results/', model, '/img']);

colormap(jet)

for rr=3:size(r,1)
    filename = r(rr).name;
    disp([num2str(rr - 2) ' begin']);
    namelength = find(filename == '.');
    index = str2double(filename(1:namelength-1));

    load(['results/', model, '/predict/', filename]);
    if exist('real_A', 'var')
        real_A = permute(squeeze(real_A), [1, 3, 4, 2]);
        real_A = real_A .* permute(repmat(squeeze(ms(1, 2, :)), ...
            [1, size(real_A, 1), size(real_A, 2), size(real_A, 3)]), ...
            [2, 3, 4, 1]) * k_std + permute(repmat(squeeze(ms(1, 1, :)), ...
            [1, size(real_A, 1), size(real_A, 2), size(real_A, 3)]), ...
            [2, 3, 4, 1]);
        viewslice(real_A(1,:,:,:), [64, 64, 21], 1, 1, [0, 300],[],[6, 4],0);
        saveas(gcf, ['results/', model, '/img/', num2str(index, '%05d'), '_real_A'], 'jpg');
    end
    if exist('fake_B', 'var')
        fake_B = permute(squeeze(fake_B), [1, 3, 4, 2]);
        fake_B = fake_B .* permute(repmat(squeeze(ms(2, 2, :)), ...
            [1, size(fake_B, 1), size(fake_B, 2), size(fake_B, 3)]), ...
            [2, 3, 4, 1]) * k_std + permute(repmat(squeeze(ms(2, 1, :)), ...
            [1, size(fake_B, 1), size(fake_B, 2), size(fake_B, 3)]), ...
            [2, 3, 4, 1]);
        viewslice(fake_B(1,:,:,:), [64, 64, 21], 1, 1, [0, 2000],[],[6, 4],0);
        saveas(gcf, ['results/', model, '/img/', num2str(index, '%05d'), '_fake_B'], 'jpg');
    end
    if exist('rec_A', 'var')
        rec_A = permute(squeeze(rec_A), [1, 3, 4, 2]);
        rec_A = rec_A .* permute(repmat(squeeze(ms(1, 2, :)), ...
            [1, size(rec_A, 1), size(rec_A, 2), size(rec_A, 3)]), ...
            [2, 3, 4, 1]) * k_std + permute(repmat(squeeze(ms(1, 1, :)), ...
            [1, size(rec_A, 1), size(rec_A, 2), size(rec_A, 3)]), ...
            [2, 3, 4, 1]);
        viewslice(rec_A(1,:,:,:), [64, 64, 21], 1, 1, [0, 300],[],[6, 4],0);
        saveas(gcf, ['results/', model, '/img/', num2str(index, '%05d'), '_rec_A'], 'jpg');
    end
    if exist('real_B', 'var')
        real_B = permute(squeeze(real_B), [1, 3, 4, 2]);
        real_B = real_B .* permute(repmat(squeeze(ms(2, 2, :)), ...
            [1, size(real_B, 1), size(real_B, 2), size(real_B, 3)]), ...
            [2, 3, 4, 1]) * k_std + permute(repmat(squeeze(ms(2, 1, :)), ...
            [1, size(real_B, 1), size(real_B, 2), size(real_B, 3)]), ...
            [2, 3, 4, 1]);
        viewslice(real_B(1,:,:,:), [64, 64, 21], 1, 1, [0, 2000],[],[6, 4],0);
        saveas(gcf, ['results/', model, '/img/', num2str(index, '%05d'), '_real_B'], 'jpg');
    end
    if exist('fake_A', 'var')
        fake_A = permute(squeeze(fake_A), [1, 3, 4, 2]);
        fake_A = fake_A .* permute(repmat(squeeze(ms(1, 2, :)), ...
            [1, size(fake_A, 1), size(fake_A, 2), size(fake_A, 3)]), ...
            [2, 3, 4, 1]) * k_std + permute(repmat(squeeze(ms(1, 1, :)), ...
            [1, size(fake_A, 1), size(fake_A, 2), size(fake_A, 3)]), ...
            [2, 3, 4, 1]);
        viewslice(fake_A(1,:,:,:), [64, 64, 21], 1, 1, [0, 300],[],[6, 4],0);
        saveas(gcf, ['results/', model, '/img/', num2str(index, '%05d'), '_fake_A'], 'jpg');
    end
    if exist('rec_B', 'var')
        rec_B = permute(squeeze(rec_B), [1, 3, 4, 2]);    
        rec_B = rec_B .* permute(repmat(squeeze(ms(2, 2, :)), ...
            [1, size(rec_B, 1), size(rec_B, 2), size(rec_B, 3)]), ...
            [2, 3, 4, 1]) * k_std + permute(repmat(squeeze(ms(2, 1, :)), ...
            [1, size(rec_B, 1), size(rec_B, 2), size(rec_B, 3)]), ...
            [2, 3, 4, 1]);
        viewslice(rec_B(1,:,:,:), [64, 64, 21], 1, 1, [0, 2000],[],[6, 4],0);
        saveas(gcf, ['results/', model, '/img/', num2str(index, '%05d'), '_rec_B'], 'jpg');
    end
end