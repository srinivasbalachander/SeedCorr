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
from nilearn.interfaces.bids import save_glm_to_bids

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
df = pd.read_csv("covs/Demogs_analysis.csv")
df

# Check duplicates in the covariates file
unique, counts = np.unique(df[['ProjectNumber']], return_counts=True)
duplicates = unique[np.where(counts > 1)]
print(duplicates)

# Make covariates design-matrix worthy

# Subset only the required rows and columns
covs = df[df['ProjectNumber'].isin(subjlist)]
covs = covs[["ProjectNumber", "Age", "Gender", "Group"]]

# Deleting duplicate rows, check the values with Harsh
covs = covs.drop_duplicates(subset='ProjectNumber', keep="first")

# Recode categorical variables to 0 and 1
gender_mapping = {'Male':0,'Female':1}
#group_mapping = {'HC':-1, 'SCZ':1}

covs = covs.assign(Gender = covs.Gender.map(gender_mapping))
#covs = covs.assign(Group = covs.Group.map(group_mapping))

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

covs

# Design matrices for null and Group-only (no covariate) models
null_matrix = pd.DataFrame([1] * nsubj, columns=["intercept"])
grouponly_matrix = covs[['SCZ', 'HC']]


# Fit the statistical models
second_level_model0 = SecondLevelModel(n_jobs=4).fit(dlpfcmaps, design_matrix = null_matrix)
second_level_model1 = SecondLevelModel(n_jobs=4).fit(dlpfcmaps, design_matrix = grouponly_matrix)
second_level_model2 = SecondLevelModel(n_jobs=4).fit(dlpfcmaps, design_matrix = covs)

# Apply contrasts
z_map0 = second_level_model0.compute_contrast(output_type="z_score")
z_map1 = second_level_model1.compute_contrast([-1, 1], output_type="z_score")
z_map2 = second_level_model2.compute_contrast([0, 0, -1, 1], output_type="z_score")

# Save outputs

icbm152_2009 = datasets.fetch_icbm152_2009()


output_dir = Path.cwd() / "results"
output_dir.mkdir(exist_ok=True, parents=True)

save_glm_to_bids(
    second_level_model0,
    contrasts="intercept", prefix="nullmodel",
    out_dir=output_dir / "derivatives" / "seedcorr_dlpfc" / "nullmodel",
    bg_img=icbm152_2009["t1"],
)

save_glm_to_bids(
    second_level_model1,
    contrasts="intercept", prefix="nullmodel",
    out_dir=output_dir / "derivatives" / "seedcorr_dlpfc" / "grouponlymodel",
    bg_img=icbm152_2009["t1"],
)

save_glm_to_bids(
    second_level_model2,
    contrasts="intercept", prefix="nullmodel",
    out_dir=output_dir / "derivatives" / "seedcorr_dlpfc" / "fullmodel",
    bg_img=icbm152_2009["t1"],
)
