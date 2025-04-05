function resampling_time_comparison()
    M = 10000;     % Number of particles
    T = 100;       % Number of repetitions for averaging

    % Initialize time accumulators
    total_multinomial = 0;
    total_stratified = 0;
    total_systematic = 0;
    total_isr = 0;
    total_msv = 0;

    for t = 1:T
        w = rand(1, M); w = w / sum(w);
        x = randn(1, M);  % dummy particles

        tic;
        multinomial_resampling_count(x, w);
        total_multinomial = total_multinomial + toc;

        tic;
        stratified_resampling_count(x, w);
        total_stratified = total_stratified + toc;

        tic;
        systematic_resampling_count(x, w);
        total_systematic = total_systematic + toc;

        tic;
        improved_systematic_resampling_count(x, w);
        total_isr = total_isr + toc;

        tic;
        msv_resampling_count(x, w);
        total_msv = total_msv + toc;
    end

    fprintf("Average over %d runs with %d particles:\n", T, M);
    fprintf("Multinomial:         %.6f s\n", total_multinomial / T);
    fprintf("Stratified:          %.6f s\n", total_stratified / T);
    fprintf("Systematic:          %.6f s\n", total_systematic / T);
    fprintf("Improved Systematic: %.6f s\n", total_isr / T);
    fprintf("MSV:                 %.6f s\n", total_msv / T);
end




function [px, count] = multinomial_resampling_count(px, pw)
    N = length(pw);
    cdf = cumsum(pw);
    U = rand(1, N);
    indices = zeros(1, N);
    i = 1;
    for j = 1:N
        while U(j) > cdf(i)
            i = i + 1;
        end
        indices(j) = i;
    end
    count = histcounts(indices, 0.5:1:N+0.5);
    px = px(:, indices);
end

function [px, count] = stratified_resampling_count(px, pw)
    N = length(pw);
    cdf = cumsum(pw);
    U = ((0:N-1) + rand(1, N)) / N;
    indices = zeros(1, N);
    i = 1;
    for j = 1:N
        while U(j) > cdf(i)
            i = i + 1;
        end
        indices(j) = i;
    end
    count = histcounts(indices, 0.5:1:N+0.5);
    px = px(:, indices);
end

function [px, count] = systematic_resampling_count(px, pw)
    N = length(pw);
    cdf = cumsum(pw);
    r = rand / N;
    U = r + (0:N-1) / N;
    indices = zeros(1, N);
    i = 1;
    for j = 1:N
        while U(j) > cdf(i)
            i = i + 1;
        end
        indices(j) = i;
    end
    count = histcounts(indices, 0.5:1:N+0.5);
    px = px(:, indices);
end

function [px, count] = improved_systematic_resampling_count(px, pw)
    rho = 1e-4;
    pw(pw < rho) = rho;
    pw = pw / sum(pw);
    [px, count] = systematic_resampling_count(px, pw);
end

function [pxNew, count] = msv_resampling_count(px, pw)
    N = length(pw);
    pw = pw(:)' / sum(pw);
    N_floor = floor(N * pw);
    R = N - sum(N_floor);
    v = N * pw - N_floor;
    [~, idx] = sort(v, 'descend');
    N_floor(idx(1:R)) = N_floor(idx(1:R)) + 1;
    count = N_floor;
    pxNew = zeros(size(px,1), N);
    idx_out = 1;
    for i = 1:N
        for j = 1:count(i)
            pxNew(:, idx_out) = px(:, i);
            idx_out = idx_out + 1;
        end
    end
end

function L = multinomial_likelihood(counts, weights)
    N = sum(counts);
    weights = weights(:)';
    counts = counts(:)';
    L = factorial(N) / prod(factorial(counts)) * prod(weights .^ counts);
end
