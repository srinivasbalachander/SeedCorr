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

# Get paths to seed correlation maps 
halfpipe_deriv = "/mnt/HPStorage/CRC_Analysis/Analysis/derivatives/halfpipe/"
motion_correct_method = "seedCorr1"
roi = "LDLPFC"

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
null_matrix = pd.DataFrame([1] * nsubj, columns=["intercept"])
grouponly_matrix = covs[['SZP', 'DEP', 'HC']]
full_matrix = covs[['Age', 'Gender', 'SZP', 'DEP', 'HC']]

print("\nHere's the design matrix now..\n\n")
print(full_matrix)  

# Fit the statistical models

# Suffix '0' is for null/intercept-only model, '1' for Group-Only model and '2' for full model adjusted for age/gender
second_level_model0 = SecondLevelModel(n_jobs=8).fit(corrmap_paths, design_matrix = null_matrix)
second_level_model1 = SecondLevelModel(n_jobs=8).fit(corrmap_paths, design_matrix = grouponly_matrix)
second_level_model2 = SecondLevelModel(n_jobs=8).fit(corrmap_paths, design_matrix = covs)

# Apply contrasts
z_map0 = second_level_model0.compute_contrast(output_type="z_score")

z_map1_SCZ = second_level_model1.compute_contrast([1, 0, 0], output_type="z_score")
z_map1_DEP = second_level_model1.compute_contrast([0, 1, 0], output_type="z_score")
z_map1_HC = second_level_model1.compute_contrast([0, 0, 1], output_type="z_score")
z_map1_SCZvsHC = second_level_model1.compute_contrast([1, 0, -1], output_type="z_score")
z_map1_DEPvsHC = second_level_model1.compute_contrast([0, 1, -1], output_type="z_score")

z_map2_SCZ = second_level_model2.compute_contrast([0, 0, 1, 0, 0], output_type="z_score")
z_map2_DEP = second_level_model2.compute_contrast([0, 0, 0, 1, 0], output_type="z_score")
z_map2_HC = second_level_model2.compute_contrast([0, 0, 0, 0, 1], output_type="z_score")
z_map2_SCZvsHC = second_level_model2.compute_contrast([0, 0, 1, 0, -1], output_type="z_score")
z_map2_DEPvsHC = second_level_model2.compute_contrast([0, 0, 0, 1, -1], output_type="z_score")

# Save outputs

zmap_names = [zmap0, 
             zmap1_SCZ, zmap1_DEP, zmap1_HC,
             zmap1_SCZvsHC, zmap1_DEPvsHC,
             zmap2_SCZ, zmap2_DEP, zmap2_HC,
             zmap2_SCZvsHC, zmap2_DEPvsHC]

# Get list of zmap names and store them as a single dictionary 'zmaps'
zmap_names = [x for x in locals() if re.match('^z_map.*', x)]

z_maps = {}
for i in zmap_names:      
    z_maps[i] = eval(i) 

# Save them to the ouput directory
output_dir = Path.cwd() / "results" / roi
output_dir.mkdir(exist_ok=True, parents=True)

# Run a for loop to save all zmaps to the output directory
for i in zmap_names:
  z_maps[i].to_filename(Path(output_dir, i + '.nii.gz'))

# Save GLM reports as html files
icbm152_2009 = datasets.fetch_icbm152_2009()

report0 = make_glm_report(model=second_level_model0, contrasts=["intercept"], 
                          height_control='fdr', alpha = 0.05, cluster_threshold=20,
                          title = "Intercept-only (null) model")

report1 = make_glm_report(model=second_level_model1, contrasts=["SZP", "DEP", "HC"], 
                          height_control='fdr', alpha = 0.05, cluster_threshold=10,
                          title = "Group-only (unadjusted) model")

report2 = make_glm_report(model=second_level_model2, contrasts=["SZP", "DEP", "HC"], 
                          height_control='fdr', alpha = 0.05, cluster_threshold=10,
                          title = "Full (age & gender adjusted) model")

report0.save_as_html(output_dir / "NullModel" + '.html')
report1.save_as_html(output_dir / "GroupOnly" + '.html')
report2.save_as_html(output_dir / "FullModel" + '.html')

