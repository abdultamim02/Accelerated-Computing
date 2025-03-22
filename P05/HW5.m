HostCPU = [0.005000, 0.008000, 0.014000, 0.028000, 0.054000, 0.107000, 0.213000, 0.433000, 0.850000];
DeviceGPU = [0.253952, 0.321536, 0.328704, 0.323584, 0.321536, 0.263584, 0.348672, 0.263808, 0.324832];
N = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]; % Powers of 2

% Plot settings
figure;
hold on;
plot(N, HostCPU, '-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Host CPU');
plot(N, DeviceGPU, '-o', 'LineWidth', 2, 'MarkerSize', 8, 'Color', 'r', 'DisplayName', 'Device GPU');
hold off;

% Formatting
xlabel('Number of Summations (N)');
ylabel('Elapsed Time (s)');
title('CPU vs. GPU Performance');
grid on;

% Customize x-axis ticks and scale
set(gca, 'XScale', 'log');
set(gca, 'xtick', N);
set(gca, 'xticklabels', {'2^{10}', '2^{11}', '2^{12}', '2^{13}', '2^{14}', '2^{15}', '2^{16}', '2^{17}', '2^{18}'});

% Legend
legend('Location', 'northwest');

% Save Plot
saveas(gcf, 'CPUvsGPUPerformance.png');
