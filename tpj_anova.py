# Seed Based Correlation 2nd Level (Group) Analysis using NiLearn

# Load the necessary libraries
from pathlib import Path
import glob
import re 
import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt

from nilearn import datasets, plotting
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img
from nilearn.reporting import make_glm_report
from nilearn.interfaces.bids import save_glm_to_bids

# Get paths to seed correlation maps 
halfpipe_deriv = "/mnt/HPStorage/CRC_Analysis/Analysis/derivatives/halfpipe/"
motion_correct_method = "seedCorr1"
roi = "LTPJ"

corrmap_paths = glob.glob(halfpipe_deriv + "*" + 
                          "/ses-10/func/task-rest/" + "*" + 
                          motion_correct_method + "*" + 
                          roi + "*" + "z_statmap.nii.gz")

nsubj = len(corrmap_paths)

print("\nStarting second level analysis for seed-based correlation from " + 
      roi + " using " + motion_correct_method + " images")

print("\nNumber of subjects for this analysis = ", nsubj)

# Get list of subjects
subjlist = [re.search("sub-(.*?)/ses", i).group(1) for i in corrmap_paths]

print("\nList of subjects included for this analysis: \n")
print(*subjlist, sep = '\n')

# Read the covariates file
df = pd.read_csv("covs/covs.csv")

print("Here's the raw covariates file, check if its OK\n\n")
print(df)

#--------------------Make covariates design-matrix worthy

# Check duplicates in the covariates file
unique, counts = np.unique(df[['ID']], return_counts=True)
duplicates = unique[np.where(counts > 1)]

if len(duplicates) > 0 :
  print("The following duplicates were found in your covariates file taking only the first entry")
  print(duplicates)

# Subset only the required rows, those subjects for whom .nii maps processed
covs = df[df['ID'].isin(subjlist)]

# Only these columns are needed, change if new covariates are added..
covs = covs[["ID", "Age", "Gender", "Group"]]

# Deleting duplicate rows
covs = covs.drop_duplicates(subset='ID', keep="first")

# Recode categorical variable (gender) to 0 and 1
gender_mapping = {'Male':0,'Female':1}
covs = covs.assign(Gender = covs.Gender.map(gender_mapping))

# Reorder the covariates file based on the exact order from subj list of .nii.gz files
covs = covs.set_index("ID")
covs = covs.reindex(index = subjlist)
covs['index'] =  list(range(0, nsubj))
covs =  covs.set_index("index")

covs = covs.rename_axis('')

# Change Group variable to dummy coded columns
covs  = pd.get_dummies(covs, columns=['Group'], prefix='', prefix_sep='', dtype=int)

# Design matrices for null and Group-only (no covariate) models
null_matrix = pd.DataFrame([1] * nsubj, columns=['intercept'])
grouponly_matrix = covs[['SZP', 'DEP', 'HC']]
full_matrix = covs[['Age', 'Gender', 'SZP', 'DEP', 'HC']]

print("\nHere's the design matrix now..\n\n")
print(full_matrix)  

#------------------ Fit the statistical models
print("\nFitting the main GLM models.. \n\n This might take a while so go grab a coffee while you can!")

# Suffix '0' is for null/intercept-only model, '1' for Group-Only model and '2' for full model adjusted for age/gender

print("\n\tNow fitting the Intercept-only (null) model... \n")
second_level_model0 = SecondLevelModel(n_jobs=8).fit(corrmap_paths, design_matrix = null_matrix)

print("\n\tNow fitting the Group-only (unadjusted) model... \n")
second_level_model1 = SecondLevelModel(n_jobs=8).fit(corrmap_paths, design_matrix = grouponly_matrix)

print("\n\tNow fitting the Full (Age & Gender adjusted) model... \n")
second_level_model2 = SecondLevelModel(n_jobs=8).fit(corrmap_paths, design_matrix = covs)

print("\nModel fitting steps completed successfully.. \n")

# ------------------- # Apply contrasts & Save outputs
print("\nNow applying contrasts and saving the outputs..\n")
print("\n\tStill some more time for coffee..\n")

# Save them to the ouput directory
output_dir = Path.cwd() / "results" / roi
output_dir.mkdir(exist_ok=True, parents=True)

save_glm_to_bids(
    second_level_model0,
    contrasts = ["intercept"],
    out_dir = output_dir / "NullModel",
    prefix = "NullModel",
    height_control = 'fdr', alpha = 0.05, cluster_threshold=20,
    title = "Intercept-only (null) model",)

save_glm_to_bids(
    second_level_model1,
    contrasts=['SZP', 'DEP', 'HC', 'SZP - HC', 'DEP - HC'],
    out_dir=output_dir / "GroupOnlyModel",
    prefix="GroupOnlyModel",
    height_control='fdr', alpha = 0.05, cluster_threshold = 10,
    title = "Group-only (unadjusted) model",)

save_glm_to_bids(
    second_level_model2,
    contrasts = ['SZP', 'DEP', 'HC', 'SZP - HC', 'DEP - HC'],
    out_dir = output_dir / "FullModel",
    prefix = "FullModel",
    height_control = 'fdr', alpha = 0.05, cluster_threshold = 10,
    title = "Full (age & gender adjusted) model",)

print("\n\nReports saved, analysis complete.  Congratulations!")
