"""
Cluster a chemical library using the Butina clustering algorithm
Plot the hits onto the resulting library clusters
"""
# %%
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd  # type: ignore # type: ignore
from sklearn.cluster import KMeans  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
from sklearn.manifold import TSNE

from helpers import *

# %%
# Seaborn plot aesthetics 
sns.set()
sns.set(rc={'figure.figsize': (10, 10)})
sns.set_style('whitegrid')
sns.set_context('talk')

# %%
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
# Do Butina clustring on the library and on the hits
lib_butina = ButinaCluster()
lib["Cluster"] = lib_butina.cluster_smiles(lib)
lib = lib_butina.add_fps_to_df(lib)

hits_butina = ButinaCluster()
hits["Cluster"] = hits_butina.cluster_smiles(hits)
hits = hits_butina.add_fps_to_df(hits)

# %%
"""
Exploratory KMeans
"""
# Entire library
num_clusters = 10
k_means = KMeans(num_clusters, random_state=42, n_init='auto')
k_means.fit(np.stack(lib.rdkit))
lib_k_clusters = k_means.predict(np.stack(lib.rdkit))

# %%
# Hits
k_means_hits = KMeans(num_clusters, random_state=42, n_init='auto')
k_means_hits.fit(np.stack(hits.rdkit))
hits_k_clusters = k_means_hits.predict(np.stack(hits.rdkit))

# %%
plot_cluster_pops(lib_k_clusters)
plot_cluster_pops(hits_k_clusters)
# %%
lib_silhoutte = calc_silhoutte(np.stack(lib.rdkit), 10, 100)

# %%
hits_silhoutte = calc_silhoutte(np.stack(hits.rdkit), 10, 200)

# %%
# Plot the silhoutte results as a lineplot
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
# Get K at max silhoutte score
max_lib_silhoutte = lib_silhoutte[
    lib_silhoutte.Silhoutte_Score == max(
        lib_silhoutte.Silhoutte_Score)
].K.values
max_hits_silhoutte = hits_silhoutte[
    hits_silhoutte.Silhoutte_Score == max(
        hits_silhoutte.Silhoutte_Score)
].K.values

# %%
max_lib_silhoutte = int(max_lib_silhoutte[0])
max_hits_silhoutte = int(max_hits_silhoutte[0])

# %%
# KMeans clustering of the library with optimal cluster number
km_lib = KMeans(n_clusters=max_lib_silhoutte, random_state=42, n_init='auto')
clusters_lib_opt = km_lib.fit_predict(np.stack(lib.rdkit))

# %%
# Cluster the KMeans data of the lib with T-SNE
tsne = TSNE(n_components=2, init='pca', random_state=42)
crds = tsne.fit_transform(np.stack(lib.rdkit), clusters_lib_opt)
color_list = [cm.nipy_spectral(float(i) / max_lib_silhoutte) 
              for i in range(0, max_lib_silhoutte)]

# %%
ax = sns.scatterplot(
    x=crds[:,0], 
    y=crds[:,1], 
    hue=clusters_lib_opt, 
    palette=color_list,
    legend=False,
    alpha=0.2
)

# %%
# KMeans clustering of the hits with optimal cluster number, followed by T-SNE
km_hits = KMeans(n_clusters=max_hits_silhoutte, random_state=42, n_init='auto')
clusters_hits_opt = km_hits.fit_predict(np.stack(hits.rdkit))
# %%
tsne_hits = TSNE(n_components=2, init='pca', random_state=42)
crds_hits = tsne_hits.fit_transform(np.stack(hits.rdkit), clusters_hits_opt)
color_list = [cm.nipy_spectral(float(i) / max_hits_silhoutte) 
              for i in range(0, max_hits_silhoutte)]

ax = sns.scatterplot(
    x=crds_hits[:,0], 
    y=crds_hits[:,1], 
    hue=clusters_hits_opt, 
    palette=color_list,
    legend=False
)

# %%
# TSNE clustering with optimal KMeans clusters (91) looks ok, add cluster column to lib DF and extract the hits
lib["K_Cluster"] = clusters_lib_opt
# %%
lib_hits = get_hits_from_lib(lib, hits)
# %%
plot_cluster_pops(lib_hits.K_Cluster)
# %%
# Grab all the clusters with >5 members
cluster_info = describe_cluster_counts(lib_hits, cluster="K_Cluster")

# %%
imgs = get_cluster_imgs(lib_hits, cluster="K_Cluster")

# %%
# Get all cluster 15 hits as its the largest cluster
c_15 = lib_hits[lib_hits.K_Cluster == 15]
c_49 = lib_hits[lib_hits.K_Cluster == 49]

# %%
# Find maximum common substructure of cluster 15
c_15_MCS = SmilesMCStoGridImage(c_15.SMILES, align_substructure=False, verbose=True)
# %%
c_49_MCS = SmilesMCStoGridImage(c_49.SMILES, align_substructure=False, verbose=True)

# %%
