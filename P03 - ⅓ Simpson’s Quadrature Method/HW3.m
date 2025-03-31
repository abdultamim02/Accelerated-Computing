threads = [1, 2, 4, 8, 16, 32, 64, 128, 256]; % Powers of 2
time = [0.0696441, 0.0631901, 0.0323925, 0.033294, 0.0206279, 0.0216191, 0.0218429, 0.0490641, 0.0943903];

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

% Save Plot
saveas(gcf,'TimevsThreads.png');