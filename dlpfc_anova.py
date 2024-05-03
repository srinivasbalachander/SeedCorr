# Seed Based Correlation 2nd Level (Group) Analysis using NiLearn

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
motion_correct = "seedCorr1"
roi = "LDLPFC"

corrmap_paths = glob.glob(halfpipe_deriv + "*" + 
                          "/ses-10/func/task-rest/" + "*" + 
                          motion_correct + "*" + 
                          roi + "*" + "z_statmap.nii.gz")

nsubj = len(corrmap_paths)

print("Number of subjects for this analysis = ", nsubj)

# Get list of subjects
subjlist = [re.search("sub-(.*?)/ses", i).group(1) for i in corrmap_paths]
subjlist

# Read the covariates file
df = pd.read_csv("covs/covs.csv")
df

# Check duplicates in the covariates file
unique, counts = np.unique(df[['ID']], return_counts=True)
duplicates = unique[np.where(counts > 1)]
print(duplicates)

# Make covariates design-matrix worthy

# Subset only the required rows and columns
covs = df[df['ID'].isin(subjlist)]
covs = covs[["ID", "Age", "Gender", "Group"]]

# Deleting duplicate rows, check the values with Harsh
covs = covs.drop_duplicates(subset='ID', keep="first")

# Recode categorical variables to 0 and 1
gender_mapping = {'Male':0,'Female':1}
covs = covs.assign(Gender = covs.Gender.map(gender_mapping))

# Reorder the covariates file based on the subj list
covs = covs.set_index("ProjectNumber")
covs = covs.reindex(index = subjlist)
covs['index'] =  list(range(0, nsubj))
covs =  covs.set_index("index")

covs = covs.rename_axis('')

# Make a column for intercept also
# covs['intercept'] = list([1]*nsubj)

# Change Group variable to dummy coded columns
covs  = pd.get_dummies(covs, columns=['Group'], prefix='', prefix_sep='', dtype=int)

# Design matrices for null and Group-only (no covariate) models
null_matrix = pd.DataFrame([1] * nsubj, columns=["intercept"])
grouponly_matrix = covs[['SCZ', 'DEP', 'HC']]
full_matrix = covs[['Age', 'Gender', 'SCZ', 'DEP', 'HC']]

# Fit the statistical models
# Suffix '0' is for null/intercept-only model, '1' for Group-Only model and '2' for full model adjusted for age/gender
second_level_model0 = SecondLevelModel(n_jobs=4).fit(dlpfcmaps, design_matrix = null_matrix)
second_level_model1 = SecondLevelModel(n_jobs=4).fit(dlpfcmaps, design_matrix = grouponly_matrix)
second_level_model2 = SecondLevelModel(n_jobs=4).fit(dlpfcmaps, design_matrix = covs)

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
zmap_names = [x for x in locals() if re.match('^zmap.*', x)]

for i in zmap_names:      
    zmaps[i] = eval(i) 

# Save them to the ouput directory
output_dir = Path.cwd() / "results" / roi
output_dir.mkdir(exist_ok=True, parents=True)

# Run a for loop to save all zmaps to the output directory
z_maps[i].to_filename(join(output_dir, roi, i + '.nii.gz'))

# Save GLM reports as html files
icbm152_2009 = datasets.fetch_icbm152_2009()

report0 = make_glm_report(model=second_level_model0, contrasts=["intercept"], 
                          height_control='fdr', alpha = 0.05, cluster_threshold=20,
                          title = "Intercept-only (null) model")

report1 = make_glm_report(model=second_level_model1, contrasts=["SCZ", "DEP", "HC"], 
                          height_control='fdr', alpha = 0.05, cluster_threshold=10,
                          title = "Group-only (unadjusted) model")

report1 = make_glm_report(model=second_level_model2, contrasts=["SCZ", "DEP", "HC"], 
                          height_control='fdr', alpha = 0.05, cluster_threshold=10,
                          title = "Group-only (unadjusted) model")

report.save_as_html(output_dir/ roi / i + '.nii.gz')

