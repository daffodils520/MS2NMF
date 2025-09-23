# MS2NMF

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17181796.svg
        
        
        
        )](https://doi.org/10.5281/zenodo.17181796
        
        
        
        )
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

## Directory Structure

```plaintext
MS2NMF/
├── LICENSE          
├── LICENSE-CODE     
├── LICENSE-DATA     
│
├── README.md       
├── requirements.txt
├── templates/
├── src/             
└── data/            
``` 

## DATA
   - Contains the original LC-MS/MS data (mgf, mzML, or equivalent formats) 
     directly acquired from instruments before any preprocessing.

## CONDA
1. Install Python3 within conda
2. Install all packages from the requirements.txt
3. Start the dashboard locally (defaults to http://localhost:5000)

**Example shell**
```shell
# make sure to have Python3 installed via conda (preferably 3.8-11)
# install requirements
pip install -r requirements.txt

# run or debug the MS2NMF Dashboard with Python 3 on http://localhost:5000
python ./app.py
```

Contact
-------
For questions regarding the dataset or workflow, please contact:
shuchenlan@simm.ac.cn  (Chenlan Shu) or yuzhuohao@simm.ac.cn （Zhuohao Yu）





