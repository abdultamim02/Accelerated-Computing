threads = [1, 2, 4, 8, 16, 32, 64, 128, 256]; % Powers of 2
time = [4.697196, 2.387700, 1.541056, 0.608784, 0.387454, 0.203777, 0.150876, 0.112527, 0.148393];

% Plot settings
figure;
plot(threads, time, '-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Hardware Threads');
ylabel('Elapsed (wall clock) time, s');
title('DGEMM Parallel Performance');
grid on;

% Customize axis ticks to match power-of-2 spacing
set(gca, 'xtick', 2.^(0:8));
set(gca, 'xticklabels', {'2^0', '2^1', '2^2', '2^3', '2^4', '2^5', '2^6', '2^7', '2^8'});
