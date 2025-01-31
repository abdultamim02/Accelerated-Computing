threads = [1, 2, 4, 8, 16, 32, 64, 128]; % Powers of 2
time = [7.295530, 4.491734, 1.587605, 0.875662, 0.818440, 0.331504, 0.381048, 0.107883];

% Plot settings
figure;
plot(threads, time, '-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Hardware Threads');
ylabel('Elapsed (wall clock) time, s');
title('DGEMM Parallel Performance');
grid on;

% Customize axis ticks to match power-of-2 spacing
set(gca, 'xtick', 2.^(0:7));
set(gca, 'xticklabels', {'2^0', '2^1', '2^2', '2^3', '2^4', '2^5', '2^6', '2^7'});
