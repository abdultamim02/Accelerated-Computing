Nodes = [262144, 262,144, 262,144, 262,144, 262,144, 262,144];
threads = [1, 2, 4, 8, 16, 32];
time = [572.527095, 939.100715, 908.415491, 855.646990, 859.197172, 783.424008];

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


% --------------------------------------------------------------------
% Sample data (replace with your actual data)
N = [100, 500, 1000, 5000, 10000, 50000]; % Node counts
t_serial = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]; % Serial runtime (seconds)

t_parallel = [ % Parallel runtimes for different thread counts
    0.008, 0.04, 0.08, 0.4, 0.8, 4.5; % n = 2
    0.006, 0.03, 0.06, 0.3, 0.6, 3.8; % n = 4
    0.005, 0.025, 0.05, 0.25, 0.5, 3.2; % n = 8
    0.004, 0.02, 0.04, 0.2, 0.4, 2.8  % n = 16
];

threads = [2, 4, 8, 16]; % Thread counts
colors = lines(length(threads)); % Generate distinct colors for each thread count

figure;
hold on;

% Plot serial result in black
plot(N, t_serial, 'k', 'LineWidth', 2, 'DisplayName', 'Serial');

% Plot parallel results with different colors
for i = 1:length(threads)
    plot(N, t_parallel(i, :), 'Color', colors(i, :), 'LineWidth', 2, 'DisplayName', sprintf('n = %d', threads(i)));
end

hold off;
legend('Location', 'NorthWest');
xlabel('Node Count (N)');
ylabel('Runtime (seconds)');
title('Runtime vs. Node Count for Serial and Parallel Execution');
grid on;
