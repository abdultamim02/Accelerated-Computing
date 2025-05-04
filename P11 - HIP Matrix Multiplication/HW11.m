rocSPARSE_density = [0.00001, 0.0001 , 0.001 , 0.01 , 0.1 , 1]; % Density

rocSPARSE_time = [139.613327, 143.641190, 165.019653, 162.481857, 277.710938, 12738.810547]; % Time (ms)
rocBLAS_time = [218.827942, 220.072906, 227.438522, 188.462875, 233.201096, 201.580353]; % Time (ms)

% Plot
figure;
semilogx(rocSPARSE_density, rocSPARSE_time, 'ro-', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
semilogx(rocSPARSE_density, rocBLAS_time, 'bs-', 'LineWidth', 2, 'MarkerSize', 8);
hold off;

% Set Y axis to log scale to spread out values
set(gca, 'YScale', 'log');

% Labels and Title
xlabel('Density', 'FontSize', 12);
ylabel('Elapsed Time (ms)', 'FontSize', 12);
title('HIP Matrix Multiplication Performance', 'FontSize', 14);

% Customize axes
set(gca, 'XTick', [1e-5 1e-4 1e-3 1e-2 1e-1 1]);
set(gca, 'XTickLabel', {'10^{-5}','10^{-4}','10^{-3}','10^{-2}','10^{-1}','10^{0}'});
grid on;
grid minor;

% Legend
legend('rocSPARSE', 'rocBLAS', 'Location', 'northwest', 'FontSize', 10);

% Save Plot
saveas(gcf,'HIPMatrixMultiplication.png');