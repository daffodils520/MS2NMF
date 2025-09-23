# MS2NMF

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17181796.svg)](https://doi.org/10.5281/zenodo.17181796)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)]()


[![描述文字](https://imgtu.com/uploads/0qfrkip8/t-20250923094948.webp)](https://imgtu.com/upload/0qfrkip8/20250923094948)

MS2NMF_data_release
===================

Description
-----------
This repository contains the raw data, processed results, figure source files, 
and global metadata supporting the MS2NMF study. The data package is organized 
to ensure transparency, reproducibility, and re-use of the MS2NMF workflow 
for mass spectrometry-based natural products discovery.

Directory Structure
-------------------
1. raw/
   - Contains the original LC-MS/MS data (mgf, mzML, or equivalent formats) 
     directly acquired from instruments before any preprocessing.

2. processed/
   - Contains results generated during the MS2NMF workflow, ordered by 
     processing steps (Step1–Step6).
   - Includes intermediate files (fragment matrices, filtered matrices, 
     transformation annotations) and final outputs (W/H matrices, 
     visualizations, frequency plots).
   - Files are named systematically and can be sorted by timestamp 
     to follow the processing pipeline.
    To fully reproduce the analyses and plots, please use the complete pipeline available at:  
    https://github.com/daffodils520/MS2NMF  

3. figure_source_data/
   - Provides the exact input files used for generating figures in the main 
     manuscript.
   - Example: 
     * Fig3a-Barplot_mirror → normalized H-matrix top features.
     * Fig3b-ms2_mirror → MS/MS spectra used for mirror plots.
     * Fig4/Input → LSH data, filtered mgf, quant table, and molecular 
       network (.graphml).
   - These inputs can be directly used to reproduce the published figures.

4. GLOBAL_METADATA/
   - Contains global experimental and computational metadata, including:
     * Plant material collection and voucher specimen.
     * Extraction and isolation procedures.
     * General experimental methods (polarimetry, IR, ECD, NMR, X-ray, HPLC).
     * High-resolution LC-MS/MS acquisition parameters.
     * Computational methods (DFT NMR, TDDFT ECD, conformational search).
     * Overview of the MS2NMF analysis pipeline (filtering, matrix 
       optimization, NMF decomposition, database annotation, GNPS integration).
   - Serves as contextual documentation for all datasets.

Files
-----
README.txt (this file): 
   - Provides an overview of the entire dataset structure and content.

Usage
-----
- Start from **raw/** if re-running the MS2NMF workflow from scratch.
- Use **processed/** for benchmarking or validating each step of the pipeline.
- Refer to **figure_source_data/** to reproduce specific figures in the paper.
- Consult **GLOBAL_METADATA/** for complete details on experimental methods, 
  computational approaches, and MS2NMF rationale.

Contact
-------
For questions regarding the dataset or workflow, please contact:
shuchenlan@simm.ac.cn  (Chenlan Shu) or yuzhuohao@simm.ac.cn （Zhuohao Yu）





