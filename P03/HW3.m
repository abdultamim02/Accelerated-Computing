threads = [1, 2, 4, 8, 16, 32, 64, 128, 256]; % Powers of 2
time = [0.237467, 0.143676, 0.0777716, 0.0702072, 0.0951428, 0.110896, 0.146824, 0.216615, 0.350486];

% Plot settings
figure;
plot(threads, time, '-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Hardware Threads');
ylabel('Elapsed Time, s');
title('''Multithreaded Composite 1/3 Simpson''s Method''');
grid on;

% Customize axis ticks to match power-of-2 spacing
set(gca, 'xtick', 2.^(0:8));
set(gca, 'xticklabels', {'2^0', '2^1', '2^2', '2^3', '2^4', '2^5', '2^6', '2^7', '2^8'});
set(gca, 'XScale', 'log');

% Legend
legend('Elapsed Time vs. Threads Count', 'Location', 'northwest');