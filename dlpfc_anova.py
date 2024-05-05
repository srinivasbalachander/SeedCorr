# 2nd Level (Group) Analysis for CRC-Neuromodulation MRIs using NiLearn

# Specify important variables about the analysis (see if these can be entered from the command line)

halfpipe_deriv = "/mnt/HPStorage/CRC_Analysis/Analysis/derivatives/halfpipe/"
firstlevel_name = "seedCorr1"
roi = "LDLPFC"

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

# Get paths to seed correlation maps 

firstlevel_paths = glob.glob(halfpipe_deriv + "*" + 
                             "/ses-10/func/task-rest/" + "*" + 
                             "_feature-" + firstlevel_name + 
                             "_seed-" + roi + 
                             "_stat-" + "z_statmap.nii.gz")

nsubj = len(firstlevel_paths)

print("\nStarting second level analysis for seed-based correlation from " + 
      roi + " using " + firstlevel_name + " images")

print("\nNumber of subjects for this analysis = ", nsubj)

# Get list of subjects
subjlist = [re.search("sub-(.*?)/ses", i).group(1) for i in firstlevel_paths]

print("\nList of subjects included for this analysis: \n")
print(*subjlist, sep = '\n')

# Read the covariates file
df = pd.read_csv("covs/covs.csv")

print("Here's the raw covariates file, check if its OK\n\n")
print(df)

#------------------- Make covariates design-matrix worthy

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
second_level_model0 = SecondLevelModel(n_jobs=8).fit(firstlevel_paths, design_matrix = null_matrix)

print("\n\tNow fitting the Group-only (unadjusted) model... \n")
second_level_model1 = SecondLevelModel(n_jobs=8).fit(firstlevel_paths, design_matrix = grouponly_matrix)

print("\n\tNow fitting the Full (Age & Gender adjusted) model... \n")
second_level_model2 = SecondLevelModel(n_jobs=8).fit(firstlevel_paths, design_matrix = covs)

# Apply contrasts
print("\nModel fitting steps completed successfully, applying various contrasts now.. \n")
print("\n Still some more time for coffee..\n")

z_map0 = second_level_model0.compute_contrast(output_type="z_score")

z_map1_SZP = second_level_model1.compute_contrast([1, 0, 0], output_type="z_score")
z_map1_DEP = second_level_model1.compute_contrast([0, 1, 0], output_type="z_score")
z_map1_HC = second_level_model1.compute_contrast([0, 0, 1], output_type="z_score")
z_map1_SZPvsHC = second_level_model1.compute_contrast([1, 0, -1], output_type="z_score")
z_map1_DEPvsHC = second_level_model1.compute_contrast([0, 1, -1], output_type="z_score")

z_map2_SZP = second_level_model2.compute_contrast([0, 0, 1, 0, 0], output_type="z_score")
z_map2_DEP = second_level_model2.compute_contrast([0, 0, 0, 1, 0], output_type="z_score")
z_map2_HC = second_level_model2.compute_contrast([0, 0, 0, 0, 1], output_type="z_score")
z_map2_SZPvsHC = second_level_model2.compute_contrast([0, 0, 1, 0, -1], output_type="z_score")
z_map2_DEPvsHC = second_level_model2.compute_contrast([0, 0, 0, 1, -1], output_type="z_score")

# ------------------- Save outputs
print("\nAnalysis complete, now saving the outputs..\n")

# Get list of z_map names and store them as a single dictionary 'z_maps'
zmap_names = [x for x in locals() if re.match('^z_map.*', x)]

z_maps = {}
for i in zmap_names:      
    z_maps[i] = eval(i) 

# Save them to the ouput directory
output_dir = Path.cwd() / "results" / roi / firstlevel_name
output_dir.mkdir(exist_ok=True, parents=True)

# Run a for loop to save all z_maps to the output directory
for i in zmap_names:
  z_maps[i].to_filename(Path(output_dir, i + "_" + roi + "_" + firstlevel_name + '.nii.gz'))

# Save GLM reports as html files
icbm152_2009 = datasets.fetch_icbm152_2009()

report0 = make_glm_report(model=second_level_model0, contrasts = ["intercept"], 
                          height_control='fwer', alpha = 0.05, cluster_threshold=20,
                          title = "Intercept-only (null) model")

report1 = make_glm_report(model=second_level_model1, contrasts = ['SZP', 'DEP', 'HC', 'SZP - HC', 'DEP - HC'], 
                          height_control='fdr', alpha = 0.05, cluster_threshold=10,
                          title = "Group-only (unadjusted) model")

report2 = make_glm_report(model=second_level_model2, contrasts = ['SZP', 'DEP', 'HC', 'SZP - HC', 'DEP - HC'], 
                          height_control='fdr', alpha = 0.05, cluster_threshold=10,
                          title = "Full (age & gender adjusted) model")

report0.save_as_html(Path(output_dir, "NullModel" + "_" + roi + "_" + firstlevel_name + '.html'))
report1.save_as_html(Path(output_dir, "GroupOnly" + "_" + roi + "_" + firstlevel_name + '.html'))
report2.save_as_html(Path(output_dir, "FullModel" + "_" + roi + "_" + firstlevel_name + '.html'))

# Saving a few other important info

import json

motion_int = re.findall('\d+', firstlevel_name)

if (len(motion_int) == 0)  : 
  motion_correct_method = "ICA AROMA"
elif (motion_int[0] == '1'):
  motion_correct_method = "aCompCor"
elif (motion_int[0] == '2'):
  motion_correct_method = "ICA AROMA+aCompCor"
else :
  motion_correct_method = ""
  
description = {'SeedRegion': roi,
                'motion_correction' : motion_correct_method,
                'NSubjects' : nsubj,
                'N_SZP' : int(covs['SZP'].sum()),
                'N_DEP' : int(covs['DEP'].sum()),
                'N_HC' : int(covs['HC'].sum()),
                'subjpaths': firstlevel_paths}

with open(Path(output_dir, "description" + "_" + roi + "_" + firstlevel_name + ".json"), "a") as outfile :
    outfile.write(json.dumps(description, indent = 2))

print("\n\nReports saved, analysis complete.  Congratulations!")
