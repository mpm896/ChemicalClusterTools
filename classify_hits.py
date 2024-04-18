"""
Use a KNN classifier to cluster the hits by chemical structure

"""
# %%
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd  # type: ignore
from rdkit import Chem  # type: ignore
from rdkit.Chem import Draw  # type: ignore
from rdkit.Chem import AllChem  # type: ignore
from rdkit.DataStructs import BulkTanimotoSimilarity  # type: ignore
from sklearn.cluster import KMeans  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
from sklearn.neighbors import NearestNeighbors  # type: ignore

# %%
def smile_to_fp(df: pd.DataFrame) -> None:
    """
    Convert SMILES to fingerprints
    Add new column to the passed DataFrame
    """
    # Get Morgan Fingerprint for each SMILES string
    fingerprints = []
    for smiles in df["SMILES"]:
        mol = Chem.MolFromSmiles(smiles)
        fp = fpgen.GetFingerprint(mol)
        fingerprints.append(fp)

    df["Fingerprint"] = fingerprints



# %%
# Read in csv and draw chemical structure from SMILES data
hits = pd.read_csv("HTS_SG.csv", header=None)
hits.head()
# %%
# Keep only the first two columns
hits = hits.iloc[:, :2]
hits.head()
# %%
# Change column names
hits.columns = ["Index", "SMILES"]
hits.head()
# %%
hits.set_index("Index", inplace=True)
hits.head()

# %%
# Draw the first 5 SMILES
Draw.MolsToGridImage(hits.iloc[:5, 0].apply(Chem.MolFromSmiles), molsPerRow=5)

# %%
fpgen = AllChem.GetMorganGenerator()
# %%
# Get Morgan Fingerprint for each SMILES string
fingerprints = []
for smiles in hits["SMILES"]:
    mol = Chem.MolFromSmiles(smiles)
    fp = fpgen.GetFingerprint(mol)
    fingerprints.append(fp)

# %%
# Convert fingerprints to numpy array
fingerprints = np.array(fingerprints)

# %% 
# Add fingerprint column to dataframe
smile_to_fp(hits)
hits
# %%
# Setup a KNN classifier for the SMILES data
kmeans = KMeans(random_state=42, n_clusters=4).fit(fingerprints)


# %%
kmeans.labels_
# %%
# Reduce the data to 2 dimensions
reduced_data = PCA(n_components=2).fit_transform(fingerprints)
# %%
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=169, linewidths=3, color="w", zorder=10)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans.labels_)

# %%
lib = pd.read_csv("Chembrigde_Div.csv", header=None)
lib
# %%
lib = lib.iloc[:, :1]  # Keep onle the index and SMILES columns
lib.rename(columns={0: "SMILES"}, inplace=True)
lib
# %%
# Remove NaN rows
lib.dropna(inplace=True)
# %%
lib.reset_index(drop=True, inplace=True)
lib
# %%
# Add fingerprint to dataframe
smile_to_fp(lib)
lib

# %%
# Convert lib fingerprints to np array
lib_fp = list(lib["Fingerprint"])


# %%
# Setup KMean classifier for entire library

kmeans = KMeans(random_state=42, n_clusters=10).fit(lib_fp)

# %%
reduced_lib = PCA(n_components=100).fit_transform(lib_fp)
# %%
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=169, linewidths=3, color="w", zorder=10)
plt.scatter(reduced_lib[:, 0], reduced_lib[:, 1], c=kmeans.labels_, alpha=0.4)

# %%
dists = []
nfps = len(lib_fp)
for i in range(0,nfps-1):
    dist = BulkTanimotoSimilarity(lib_fp[i], lib_fp)
    dists.append(dist)

#lib["mol_dist"] = dists
# %%
# %%
