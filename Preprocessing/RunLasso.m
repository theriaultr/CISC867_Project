%Author:Rachel Theriault
%PURPOSE: The purpose of this script is to run the lasso function. The data
%is reformatted and lasso is called
%Using same patients as gene paper, but no initial feature reduction (not
%necessary for lasso)

%feature-wise normalization of expression data
expression_data_normal = zscore(s1_expression_data_log_transformed'); %z-score along the features (were rows now columns)

%call lasso and determine important genes
[selected_c1, selected_c2, selected_c3, selected_c4, selected_1, selected_2, selected_3, selected_4] = PerformLasso(expression_data_normal, s1_stage_label);

c1_genes = s1_gene_names(selected_c1);
c2_genes = s1_gene_names(selected_c2);
c3_genes = s1_gene_names(selected_c3);
c4_genes = s1_gene_names(selected_c4);

% Stage 1 —> 0.035
% Stage 2 —> 0.01
% Stage 3 —> 0.02
% Stage 4 —> 0.004
e1_genes = s1_gene_names(selected_1);
e2_genes = s1_gene_names(selected_2);
e3_genes = s1_gene_names(selected_3);
e4_genes = s1_gene_names(selected_4);

%final set of genes
final_gene_set = union(e1_genes, e2_genes);
final_gene_set = union(final_gene_set, e3_genes);
final_gene_set = union(final_gene_set, e4_genes);

%Create a figure to visualize the overlap of genes (which genes are
%selected by multiple stages)
genes_in_e1 = ismember(final_gene_set, e1_genes);
genes_in_e2 = (ismember(final_gene_set, e2_genes))*2;
genes_in_e3 = (ismember(final_gene_set, e3_genes))*3;
genes_in_e4 = (ismember(final_gene_set, e4_genes))*4;

plot(genes_in_e1, 'r*')
hold on
plot(genes_in_e2, 'b*')
plot(genes_in_e3, 'p*')
plot(genes_in_e4, 'k*')
xlabel('Selected Gene Number', 'FontName', 'latex', 'FontSize', 24)
ylabel('Stage Number', 'FontName', 'latex', 'FontSize', 24)




