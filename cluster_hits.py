"""
Cluster a chemical library using the Butina clustering algorithm
Plot the hits onto the resulting library clusters
"""
# %%
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd  # type: ignore
from rdkit import Chem  # type: ignore
from rdkit.Chem import Draw  # type: ignore
from rdkit.Chem import AllChem  # type: ignore
from rdkit.DataStructs import BulkTanimotoSimilarity  # type: ignore
from rdkit.ML.Cluster import Butina  # type: ignore
from sklearn.cluster import KMeans  # type: ignore
from sklearn.decomposition import PCA  # type: ignore

from helpers import *

# %%
class ButinaCluster:

    def __init__(self, fptype: str="rdkit"):
        self.fptype = fptype

    
    def cluster_smiles(self, df: pd.DataFrame, sim_cutoff: float=0.8) -> list:
        """
        Cluster the SMILES strings in the passed DataFrame
        Add new columns to the passed DataFrame
        """
        assert "SMILES" in df.columns

        # Get Morgan Fingerprint for each SMILES string
        mols = [Chem.MolFromSmiles(x) for x in df["SMILES"]]
        return self.cluster_mols(mols, sim_cutoff)
    

    def get_fps(self, mols: list[Chem.rdchem.Mol]) -> list:
        """
        Get fingerprint of defined type (rdkit or Morgan)
        from the SMILES
        """
        fp_dict = {
            "rdkit": [Chem.RDKFingerprint(x) for x in mols],
            "morgan": [AllChem.GetMorganFingerprintAsBitVect(x, 2) for x in mols]
        }

        if fp_dict[self.fptype] is None:
            raise ValueError(f"Fingerprint method {self.fptype} not supported")
        
        return fp_dict[self.fptype]
    

    def add_fps_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add new columns to the passed DataFrame
        """
        # Get Morgan Fingerprint for each SMILES string
        df[self.fptype] = self.fp_list
        return df
        

    def cluster_mols(
            self, 
            mols: list[Chem.rdchem.Mol], 
            sim_cutoff: float=0.8
        ) -> list:
        """
        Cluster the Mols from the SMILES strings
        using the Butina algorithm
        """
        dist_cutoff = 1 - sim_cutoff

        # Get fingerprint bits
        self.fp_list = self.get_fps(mols)
        
        # Cluster using Butina
        dists = []
        nfps = len(self.fp_list)
        for i in range(1, nfps):
            sims = BulkTanimotoSimilarity(self.fp_list[i], self.fp_list[:i])
            dists.extend([1 - x for x in sims])

        self.dist_matrix = dists

        # Cluster
        mol_clusters = Butina.ClusterData(
            dists, 
            nfps, 
            dist_cutoff, 
            isDistData=True
        )
        cluster_ids = [0] * nfps
        for idx, cluster in enumerate(mol_clusters):
            for mol_idx in cluster:
                cluster_ids[mol_idx] = idx

        return [x - 1 for x in cluster_ids]
    
# %%
def organize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Modify a dataframe in place, from the HTS data at Penn
    """
    df = df.iloc[:, :2]
    df.columns = ["Index", "SMILES"]
    df.set_index("Index", inplace=True)
    return df

# %%
def describe_cluster_counts(df: pd.DataFrame) -> dict[int]:
    """
    Describe the cluster counts for a dataframe
    """
    assert "Cluster" in df.columns
    
    counts = {}
    counts["total"] = len(df.Cluster.unique())
    for n in [1, 2, 5, 10, 25, 50]:
        if n == 1:
            counts[n] = (
                sum(1 for c in df.Cluster.unique()
                    if df.Cluster[df.Cluster == c].value_counts().values == 1)
            )
        else:
            counts[n] = (
                sum(1 for c in df.Cluster.unique()
                    if df.Cluster[df.Cluster == c].value_counts().values >= n)
            )

    return counts

# %%
def draw_cluster(df: pd.DataFrame, cluster: int) -> None:
    """
    Draw the cluster
    """
    smiles = df[df.Cluster == cluster].SMILES
    Draw.MolsToGridImage(smiles.apply(Chem.MolFromSmiles), molsPerRow=5)

# %%
def get_hits_from_lib(lib: pd.DataFrame, hits: pd.DataFrame) -> pd.DataFrame:
    """
    Get the hits from the library with clusters
    """
    assert "SMILES" in lib.columns and "SMILES" in hits.columns
    
    matching = lib.SMILES.isin(hits.SMILES)
    lib_hits = lib.loc[matching, :]
    return lib_hits

# %%
def get_centroids(arr: np.ndarray) -> np.ndarray:
    """
    Get the centroids of the clusters
    """
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return np.array([sum_x / length, sum_y / length])

# %%
def get_cluster_imgs(df: pd.DataFrame) -> dict:
    """
    Create a dict of cluster images
    since images cannot be printed from a loop or function

    Returns a dictionary
        :k: cluster number
        :img: cluster image
    """
    assert "K_Means" in df.columns
    assert "SMILES" in df.columns
    
    imgs = {}
    for k in df.K_Means.unique():
        smiles = df[df.K_Means == k].SMILES
        imgs[k] = Draw.MolsToGridImage(
                smiles.apply(Chem.MolFromSmiles),
                molsPerRow=5, 
                maxMols=100,
                returnPNG=True
        )
    return imgs

# %%
def save_cluster_imgs(imgs: dict):
    """
    Simply save cluster images in current directory
    """
    for k, img in imgs.items():
        with open(f"cluster_{k}.png", "wb") as f:
            f.write(img.data)
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
# Try with RDKit fingerprint
butina_cluster = ButinaCluster("rdkit")
lib["Cluster"] = butina_cluster.cluster_smiles(lib)
lib = butina_cluster.add_fps_to_df(lib)
# lib.sort_values("Cluster", inplace=True)

# print(lib.Cluster.value_counts())  # 31,298 unique clusters

# %%
# Try with Morgan fingerprint
butina_cluster = ButinaCluster("morgan")
lib["Cluster_Morgan"] = butina_cluster.cluster_smiles(lib)
lib = butina_cluster.add_fps_to_df(lib)
# lib.sort_values("Cluster_Morgan", inplace=True)

# print(lib.Cluster_Morgan.value_counts()) 
# print(f"{len(lib['Cluster_Morgan'].unique())} unique clusters")

# %%
# Reduce to two dimensions and plot by cluster
pca = PCA(n_components=2)
reduced_lib = pca.fit_transform(
    np.array(list(lib["rdkit"]))
)

# %%
fig, ax = plt.subplots(figsize=(10, 10))
for cluster in lib["Cluster"].unique():
    ax.scatter(
        reduced_lib[lib["Cluster"] == cluster, 0],
        reduced_lib[lib["Cluster"] == cluster, 1],
        label=f"Cluster {cluster}",
    )
ax.set_title("RDKit fingerprint clustering")
plt.show()


# %%
# Try clustering just the hits
butina_hits = ButinaCluster()
hits["Cluster"] = butina_hits.cluster_smiles(hits)
hits = butina_hits.add_fps_to_df(hits)

# %%
reduced_hits = pca.fit_transform(
    np.array(list(hits["rdkit"]))
)

# %%
fig, ax = plt.subplots(figsize=(10, 10))
for cluster in hits["Cluster"].unique():
    ax.scatter(
        reduced_hits[hits["Cluster"] == cluster, 0],
        reduced_hits[hits["Cluster"] == cluster, 1],
        label=f"Cluster {cluster}",
    )
ax.set_title("RDKit fingerprint clustering")
plt.show()

# %%
hits.Cluster.value_counts()

# %%
lib.Cluster.value_counts()

# %%
# Get number of clusters with 1 compound, more than 5, and more than 25
total_clusters = len(lib.Cluster.unique())

cluster_1 = sum(1 for c in lib.Cluster 
                if lib.Cluster[lib.Cluster == c].value_counts().values == 1)
cluster_10 = sum(1 for c in lib.Cluster 
                if lib.Cluster[lib.Cluster == c].value_counts().values >= 10)
cluster_25 = sum(1 for c in lib.Cluster
                if lib.Cluster[lib.Cluster == c].value_counts().values >= 25)

print(f"{total_clusters} total clusters")
print(f"{cluster_1} clusters with 1 compound")
print(f"{cluster_10} clusters with 10 or more compounds")
print(f"{cluster_25} clusters with 25 or more compounds")

# %%
# Get the compounds from the lib that match with the hits
lib[lib.SMILES.isin(hits.SMILES)]


# %%
matching = lib.SMILES.isin(hits.SMILES)
matching.value_counts()

# %%
filtered_lib = lib.loc[matching, :]
filtered_lib
# %%
# What clusters are the hits in?
filtered_lib.Cluster.value_counts()

# %%
total_filtered_cluters = len(filtered_lib.Cluster.unique())
cluster_filter_1 = sum(1 for c in filtered_lib.Cluster
                    if filtered_lib.Cluster[filtered_lib.Cluster == c].value_counts().values == 1)
cluster_filter_more = sum(1 for c in filtered_lib.Cluster
                       if filtered_lib.Cluster[filtered_lib.Cluster ==c].value_counts().values > 1)
# %%
print(f"{total_filtered_cluters} total clusters")
print(f"{cluster_filter_1} clusters with 1 compound")
print(f"{cluster_filter_more} clusters with more than 1 compound")

# %%
# Sort the filtered_lib by cluster value counts
filtered_lib.Cluster.value_counts()
# %%
c_2 = filtered_lib[filtered_lib.Cluster == 2]
c_2
# %%
Draw.MolsToGridImage(c_2.iloc[:, 0].apply(Chem.MolFromSmiles), molsPerRow=5)
# %%
c_650 = filtered_lib[filtered_lib.Cluster == 650]
c_650
# %%
Draw.MolsToGridImage(c_650.iloc[:, 0].apply(Chem.MolFromSmiles), molsPerRow=5)


# %%
# Recluster with a less stringent similarity cutoff
lib_loose = lib.iloc[:,[0]]
butina_cluster = ButinaCluster("rdkit")
lib_loose["Cluster"] = butina_cluster.cluster_smiles(
                                        lib_loose, 
                                        sim_cutoff=0.6
                                    )
lib_loose = butina_cluster.add_fps_to_df(lib_loose)
# %%
lib_loose_clusters = describe_cluster_counts(lib_loose)
# %%
hits_loose = get_hits_from_lib(lib_loose, hits)

# %%
hits_loose

# %%
hits_clusters = describe_cluster_counts(hits_loose)
hits_clusters
# %%
draw_cluster(hits_loose, 1378)

# %%
smiles = hits_loose[hits_loose.Cluster == 681].SMILES
Draw.MolsToGridImage(smiles.apply(Chem.MolFromSmiles), molsPerRow=5)
# %%
og_hits_clusters = describe_cluster_counts(filtered_lib)
og_hits_clusters

# %%
# Relax the distance cutoff even more to 0.4 
lib_loose_04 = lib.iloc[:,[0]]
butina_cluster = ButinaCluster("rdkit")
lib_loose_04["Cluster"] = butina_cluster.cluster_smiles(
                                        lib_loose, 
                                        sim_cutoff=0.4
                                    )
lib_loose_04 = butina_cluster.add_fps_to_df(lib_loose_04)

# %%
lib_loose_04_clusters = describe_cluster_counts(lib_loose_04)
hits_loose_04 = get_hits_from_lib(lib_loose_04, hits)
hits_04_clusters = describe_cluster_counts(hits_loose_04)
# %%
lib_loose_04_clusters, hits_04_clusters
# %%
hits_loose_04.Cluster.value_counts()
# %%
c = 6
smiles = hits_loose_04[hits_loose_04.Cluster == c].SMILES
Draw.MolsToGridImage(smiles.apply(Chem.MolFromSmiles), molsPerRow=5, maxMols=100)

# %%
# PCA on the filtered hits and plot
reduced_hits = pca.fit_transform(
    np.array(list(hits_loose_04["rdkit"]))
)

# %%
fig, ax = plt.subplots(figsize=(10, 10))
for cluster in hits_loose_04["Cluster"].unique():
    ax.scatter(
        reduced_hits[hits_loose_04["Cluster"] == cluster, 0],
        reduced_hits[hits_loose_04["Cluster"] == cluster, 1],
        label=f"Cluster {cluster}",
    )
ax.set_title("RDKit fingerprint clustering")
plt.show()

# %%
k_means = KMeans(n_clusters=20, random_state=42)
k = k_means.fit_predict(
    np.array(list(hits_loose_04["rdkit"]))
)
hits_loose_04["K_Means"] = k
# %%
fig, ax = plt.subplots(figsize=(10,10))
reduced_centroids = []
for cluster in hits_loose_04["K_Means"].unique():
    centroid = get_centroids(
        reduced_hits[hits_loose_04["K_Means"] == cluster]
    )
    reduced_centroids.append(centroid)

    ax.scatter(
        reduced_hits[hits_loose_04["K_Means"] == cluster, 0],
        reduced_hits[hits_loose_04["K_Means"] == cluster, 1],
        label=f"Cluster {cluster}",
    )

reduced_centroids = np.array(reduced_centroids)
plt.scatter(reduced_centroids[:,0] , reduced_centroids[:,1] , s = 80, color = 'k')
ax.set_title("K-Means clustering")
plt.show()
# %%
imgs = {}
for k in hits_loose_04.K_Means.unique():
   smiles = hits_loose_04[hits_loose_04.K_Means == k].SMILES
   imgs[k] = Draw.MolsToGridImage(smiles.apply(Chem.MolFromSmiles), molsPerRow=5, maxMols=100, returnPNG=True)

# %%
for k, img in imgs.items():
    with open(f"cluster_{k}.png", "wb") as f:
        f.write(img.data)
# %%
k = 14
smiles = hits_loose_04[hits_loose_04.K_Means == k].SMILES
Draw.MolsToGridImage(smiles.apply(Chem.MolFromSmiles), molsPerRow=5, maxMols=100, returnPNG=False)

# %%
# Get missing hits, add to hits table, recluster
matching = hits.SMILES.isin(hits_loose_04.SMILES)
missing_hits = hits.loc[~matching, :]
combined_hits = pd.concat([hits_loose_04, missing_hits])
# %%
# PCA on the combined hits
reduced_hits = pca.fit_transform(
    np.array(list(combined_hits["rdkit"]))
)


# %%
k_means = KMeans(n_clusters=20, random_state=42)
k = k_means.fit_predict(
    np.array(list(combined_hits["rdkit"]))
)
combined_hits["K_Means"] = k
# %%
fig, ax = plt.subplots(figsize=(10,10))
reduced_centroids = []
for cluster in combined_hits["K_Means"].unique():
    centroid = get_centroids(
        reduced_hits[combined_hits["K_Means"] == cluster]
    )
    reduced_centroids.append(centroid)

    ax.scatter(
        reduced_hits[combined_hits["K_Means"] == cluster, 0],
        reduced_hits[combined_hits["K_Means"] == cluster, 1],
        label=f"Cluster {cluster}",
    )

reduced_centroids = np.array(reduced_centroids)
plt.scatter(reduced_centroids[:,0] , reduced_centroids[:,1], s=80, color='k', marker='x')
ax.set_title("K-Means clustering")
leg = ax.legend(loc = 'center left', bbox_to_anchor = (1.0, 0.5))
plt.savefig('PCA.png', bbox_inches='tight', dpi=300)
plt.show()

# %%
imgs = {}
for k in combined_hits.K_Means.unique():
   smiles = combined_hits[combined_hits.K_Means == k].SMILES
   imgs[k] = Draw.MolsToGridImage(
        smiles.apply(Chem.MolFromSmiles),
        molsPerRow=5, 
        maxMols=100,
        returnPNG=True
    )

# %%
for k, img in imgs.items():
    with open(f"cluster_{k}.png", "wb") as f:
        f.write(img.data)
# %%
# Get the 10 closest molecules to each centroid


# %%
# Cluster just the hits with the Butina Cluster
butina_hits = ButinaCluster()
combined_hits["Cluster"] = butina_hits.cluster_smiles(combined_hits, sim_cutoff=0.4)
combined_hits = butina_hits.add_fps_to_df(combined_hits)
# %%
smiles = combined_hits[combined_hits.Cluster == -1].SMILES
Draw.MolsToGridImage(smiles.apply(Chem.MolFromSmiles), molsPerRow=5, maxMols=100, returnPNG=False)




# %%
# PCA to reduce dimensions to keep 95% variance, then cluster with KMeans
pca = PCA(n_components=0.95)
reduced_lib = pca.fit_transform(
    np.array(list(lib["rdkit"]))
)

# %%
k_means = KMeans(n_clusters=20, random_state=42)
k = k_means.fit_predict(
    reduced_lib
)
lib["K_Means"] = k

# %%
# PCA and KMeans on just the hits
pca = PCA(n_components=0.95)
reduced_hits = pca.fit_transform(
    np.array(list(combined_hits["rdkit"]))
)

k_means = KMeans(n_clusters=20, random_state=42)
k = k_means.fit_predict(
    reduced_hits
)

# %%
combined_hits["K_Means"] = k

# %%
imgs = get_cluster_imgs(combined_hits)

# %%
# Get logP of original hits that were saved and sent to Sarah
hits = pd.read_csv('clustered_hits.csv')
# %%
logp = calc_logp(hits.SMILES)
hits["logP"] = logp
# %%
# Save the logP
hits.to_csv('clustered_hits.csv')
# %%
