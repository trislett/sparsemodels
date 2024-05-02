# sparsemodels

Sparsemodels imports the core [RGCCA](https://github.com/rgcca-factory/RGCCA) implementation of the [R Penalized Multivariate Analysis (RGCCA) package](https://rdrr.io/cran/RGCCA/) using [rpy2](https://rpy2.github.io/doc/latest/html/introduction.html). It is used to perform sparse generalized canonical correlation analysis (SGCCA) with optimization for parallel processing. SGCCA uses one of multiple components to maximize the covariance among multiple datasets (called data-views) while imposing an L1 penalty. The package also includes optional functions for SGCCA-regression, permutation testing, bootstrap analysis, and variable stability selection.

### Citations ###

Tenenhaus, A., Philippe, C., Guillemot, V., Le Cao, K.-A., Grill, J., & Frouin, V. (2014). Variable selection for generalized canonical correlation analysis. Biostatistics , 15(3), 569–583.

Tenenhaus, M., Tenenhaus, A., & Groenen, P. J. F. (2017). Regularized Generalized Canonical Correlation Analysis: A Framework for Sequential Multiblock Component Methods. Psychometrika. https://doi.org/10.1007/s11336-017-9573-x
