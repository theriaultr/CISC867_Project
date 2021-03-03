%Rachel Theriault
%This script quickly formats figures for Latex documents
labels = string(xticklabels);

labels_keep = [0:10:423];
labels_keep(1) = [];

for idx=1:423
    if ~ismember(idx, labels_keep)
        labels(idx) = '';
    end
end

xticklabels(labels);
xlabel('Sample Number', 'FontName', 'Latex', 'FontSize', 24);
ylabel('Feature-Wise Z-Score Normalized Expression', 'FontName', 'Latex', 'FontSize', 24)

% ylabel('Log Transformed Feature-Wise Z-Score Normalized Expression', 'FontName', 'Latex', 'FontSize', 24)
