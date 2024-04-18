"""
Generate a Murcko scaffold for every molecule
(converting all scaffold atoms to C and single bonds)
and then classify
"""
# %%
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd  # type: ignore
from rdkit import Chem  # type: ignore
from rdkit.Chem import AllChem, Draw  # type: ignore
from rdkit.Chem import rdFMCS
from rdkit.DataStructs import BulkTanimotoSimilarity  # type: ignore
from rdkit.ML.Cluster import Butina  # type: ignore
from sklearn.cluster import KMeans  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
from sklearn.manifold import TSNE

from helpers import *

# %%
# Load the data and clean it up
hits = pd.read_csv("HTS_SG.csv", header=None)
lib = pd.read_csv("Chembrigde_Div.csv", header=None)

# %%
hits = organize_df(hits)
# %%
# The library CSV is organized a bit differently, so manually organize it
lib = lib.iloc[:, :1]
lib.rename(columns={0: "SMILES"}, inplace=True)
lib.dropna(inplace=True)
lib.reset_index(drop=True, inplace=True)

# %%
lib_generic_scaffs, lib_scaffs = (
    extract_murcko_scaff(lib.SMILES, generic=True),
    extract_murcko_scaff(lib.SMILES, generic=False)
)
hit_generic_scaffs, hit_scaffs = (
    extract_murcko_scaff(hits.SMILES, generic=True),
    extract_murcko_scaff(hits.SMILES, generic=False)
)

# %%
lib["Scaffold"] = lib_scaffs
lib["Scaffold_Generic"] = lib_generic_scaffs
hits["Scaffold"] = hit_scaffs
hits["Scaffold_Generic"] = hit_generic_scaffs

# %%
# Butina clustering on the Murcko scaffolds
lib_butina = ButinaCluster()
# %%
lib["Cluster"] = lib_butina.cluster_smiles(lib, sim_cutoff=0.6, column_name="Scaffold")
# %%
lib = lib_butina.add_fps_to_df(lib, column_name="FP_Scaffold")
# %%
lib["Cluster_Generic"] = lib_butina.cluster_smiles(lib, column_name="Scaffold_Generic")
lib = lib_butina.add_fps_to_df(lib, column_name="FP_Generic")

# %%
hits_butina = ButinaCluster()
hits["Cluster"] = hits_butina.cluster_smiles(hits, sim_cutoff=0.6, column_name="Scaffold")
hits = hits_butina.add_fps_to_df(hits, column_name="FP_Scaffold")
hits["Cluster_Generic"] = hits_butina.cluster_smiles(hits, sim_cutoff=0.9, column_name="Scaffold_Generic")
hits = hits_butina.add_fps_to_df(hits, column_name="FP_Generic")

# %%
lib_hits = get_hits_from_lib(lib, hits)

# %%
lib_hits_clusters = describe_cluster_counts(lib_hits)


# %%
"""
KMeans clustering on the Murcko scaffolds
"""
# First, silhoutte analysis of the scaffolds
lib_silhoutte = calc_silhoutte(np.stack(lib.FP_Scaffold), 75, 120)
hits_silhoutte = calc_silhoutte(np.stack(hits.FP_Scaffold), 10, 200)

# %%
# Plot the silhoutte analyses to identify K with max score
ax = plt.plot(lib_silhoutte.K, lib_silhoutte.Silhoutte_Score)
plt.title("Library Silhoutte Scores")
plt.xlabel("K")
plt.ylabel("Silhoutte Score")
plt.show()

# %%
ax = plt.plot(hits_silhoutte.K, hits_silhoutte.Silhoutte_Score)
plt.title("Hits Silhoutte Scores")
plt.xlabel("K")
plt.ylabel("Silhoutte Score")
plt.show()

# %%
# Can also use PCA to reduce dimensionality before KMeans, rerun silhoutte
pca = PCA(n_components=0.95)
reduced_lib = pca.fit_transform(np.stack(lib.FP_Scaffold))

# %%
# Re-run silhoutte analysis on reduced lib
lib_reduced_silhoutte = calc_silhoutte(reduced_lib, 100, 200)

ax = plt.plot(lib_reduced_silhoutte.K, lib_reduced_silhoutte.Silhoutte_Score)
plt.title("Library Silhoutte Scores")
plt.xlabel("K")
plt.ylabel("Silhoutte Score")
plt.show()

# %%
# Do KMeans on Scaffold and reduced lib 
km = KMeans(n_clusters=200, random_state=42, n_init='auto')
clusters = km.fit_predict(np.stack(lib.FP_Scaffold))
clusters_reduced = km.fit_predict(reduced_lib)

lib["K_Clusters"] = clusters
lib["K_Clusters_reduced"] = clusters_reduced

# %%
# Get hits from new lib
lib_hits = get_hits_from_lib(lib, hits)

# %%
lib_hits_clusters = describe_cluster_counts(lib_hits)
lib_hits_Kclusters = describe_cluster_counts(lib_hits, cluster="K_Clusters")
lib_hits_reduced_Kclusters = describe_cluster_counts(lib_hits, cluster="K_Clusters_reduced")
# %%
imgs = get_cluster_imgs(lib_hits, cluster="K_Clusters")
imgs_reduced = get_cluster_imgs(lib_hits, cluster="K_Clusters_reduced")


# %%
# Save all the cluster images to current directory
save_cluster_imgs(imgs, path='scaffold_clusters')
save_cluster_imgs(imgs_reduced, path='scaffold_generic_clusters')
# %%
lib_hits_butina = describe_cluster_counts(lib_hits, cluster="Cluster")
# %%
imgs_from_Butina = get_cluster_imgs(lib_hits, cluster="Cluster")
save_cluster_imgs(imgs_from_Butina, path='scaffold_butina_clusters')

# %%
