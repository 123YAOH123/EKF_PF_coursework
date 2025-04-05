function resampling_likelihood_comparison()
    M = 20;  % Number of particles
    T = 20;  % Number of random weight sequences
    K = 100; % Number of trials per weight sequence

    % Store likelihood values
    L_m = zeros(1, T);     % Multinomial
    L_s = zeros(1, T);     % Stratified
    L_sys = zeros(1, T);   % Systematic
    L_isr = zeros(1, T);   % Improved Systematic
    L_msv = zeros(1, T);   % MSV

    for t = 1:T
        % Generate random weight and particle set
        w = rand(1, M); w = w / sum(w);
        x = randn(1, M);  % dummy particles

        l_m = 0; l_s = 0; l_sys = 0; l_isr = 0; l_msv = 0;

        for k = 1:K
            [~, count] = multinomial_resampling_count(x, w);
            l_m = l_m + multinomial_likelihood(count, w);

            [~, count] = stratified_resampling_count(x, w);
            l_s = l_s + multinomial_likelihood(count, w);

            [~, count] = systematic_resampling_count(x, w);
            l_sys = l_sys + multinomial_likelihood(count, w);

            [~, count] = improved_systematic_resampling_count(x, w);
            l_isr = l_isr + multinomial_likelihood(count, w);

            [~, count] = msv_resampling_count(x, w);
            l_msv = l_msv + multinomial_likelihood(count, w);
        end

        L_m(t) = l_m / K;
        L_s(t) = l_s / K;
        L_sys(t) = l_sys / K;
        L_isr(t) = l_isr / K;
        L_msv(t) = l_msv / K;
    end

    % Plot
    semilogy(L_m, 'DisplayName', 'multinomial', 'LineWidth', 2); hold on;
    semilogy(L_s, 'DisplayName', 'stratified', 'LineWidth', 2);
    semilogy(L_sys, 'DisplayName', 'systematic', 'LineWidth', 2);
    semilogy(L_isr, 'DisplayName', 'improved systematic', 'LineWidth', 2);
    semilogy(L_msv, 'DisplayName', 'msv', 'LineWidth', 2);
    legend;
    xlabel('Weight sequence'); ylabel('Likelihood'); grid on;
    title('Multinomial Likelihood Comparison of Resampling Methods');
    set(gcf, 'Position', [100, 100, 700, 300]);
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
    rho = 1e-6;
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
