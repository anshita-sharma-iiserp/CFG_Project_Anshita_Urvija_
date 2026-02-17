# CFG_Project_Anshita_Urvija_
## 1. Requirements
* Language: Python 3.12.4
* Libraries: numpy, pandas, matplotlib, sklearn, pyfaidx

## 2. Import data files
he chromosome `.fa` files are too large for GitHub. Please download them separately:

### Step 1: Download the data
**[Download data.zip from Google Drive](https://drive.google.com/file/d/1JUuClufcE9jrF9ilLhNqedvsy1lxvzsT/view?usp=sharing)**
### Step 2: Place in the right location
After cloning the repository, you'll have:
 ```text
CFG_Project_Anshita_Urvija/
├── MarkovCrossValidation.py
├── SimplerVersion.py
├── README.md
└── (other files)
 ```
### Step 3:**Place the extracted `data` directory from `data.zip` inside this folder.**
Final structure should be:
 ```text
CFG_Project_Anshita_Urvija/
├── MarkovCrossValidation.py
├── SimplerVersion.py
├── README.md
├── Output_curves_for_REST_k5_m3
└── data/
    ├── chr1.fa/
    │   ├── chr1.fa
    │   └── chr1_200bp_bins.tsv
    ├── chr2.fa/
    │   ├── chr2.fa
    │   └── chr2_200bp_bins.tsv
    └── ...
 ```


## 3. File Structure 
* `MarkovCrossValidation.py`     - Main classifier with k-fold CV, ROC, PR
* `SimplerVersion.py`            - Single-model log-likelihood version
* `data/`                        - After importing, folder containing the '.fa' and .tsv files
* Output_curves_for_REST_k5_m3/  - Output curves 

## 4. How to Run
Run the script from terminal using:
python MarkovCrossValidation.p/

Make sure your path in terminal is  ***/**/CFG_Project_Anshita_Urvija* while running the script.

You will be prompted for:
```bash
Enter chromosome number (except 3,10,17,x,y):
Choose TF (CTCF, REST, EP300):
Markov model order m:
k-fold value (>=2):

Example input:-

Enter chromosome number (except 3,10,17,x,y): 4
Choose TF (CTCF, REST, EP300): REST
Markov model order m: 3
k-fold value (>=2): 5
```




