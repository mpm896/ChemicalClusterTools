"""
Cluster a chemical library using the Butina clustering algorithm
Plot the hits onto the resulting library clusters
"""
# %%
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore

import numpy as np
import pandas as pd  # type: ignore # type: ignore
from sklearn.cluster import KMeans  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
from sklearn.neighbors import NearestNeighbors  # type: ignore

from helpers import (
    ButinaCluster,
    organize_df,
    ensure_hits_in_lib,
    describe_cluster_counts,
    draw_cluster,
    get_hits_from_lib,
    get_centroids,
    get_cluster_imgs,
    get_imgs,
    save_cluster_imgs,
    plot_cluster_pops,
    calc_silhoutte,
    calc_logp,
    smiles_MCS_to_grid_image
)

# %%
from rdkit import Chem  # type: ignore
from rdkit.Chem import PandasTools  # type: ignore

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
lib = ensure_hits_in_lib(lib, hits)

# %%
# Get fingerprints of lib and hits
lib_butina = ButinaCluster()
lib_mols = lib_butina.get_mols(lib)
lib_butina.set_fps(lib_mols)

hits_butina = ButinaCluster()
hits_mols = hits_butina.get_mols(hits)
hits_butina.set_fps(hits_mols)

# %%
# Add chemical fingerprints to the hits
lib = lib_butina.add_fps_to_df(lib, "rdkit")
hits = hits_butina.add_fps_to_df(hits, "rdkit")

# %%
# Perform various K-Means clustering on the library and hits independently
k_means = KMeans(n_clusters=20, random_state=42)
k_lib = k_means.fit_predict(
    np.array(list(lib["rdkit"]))
)
lib["K_Means"] = k_lib

k_hits = k_means.fit_predict(
    np.array(list(hits["rdkit"]))
)
hits["K_Means"] = k_hits

# PCA reduction to 0.95 variance, then K-Means
pca = PCA(n_components=0.95)
reduced_lib = pca.fit_transform(
    np.array(list(lib["rdkit"]))
)
reduced_hits = pca.fit_transform(
    np.array(list(hits["rdkit"]))
)

k_lib_reduced = k_means.fit_predict(reduced_lib)
lib["K_Means_reduced"] = k_lib_reduced

k_hits_reduced = k_means.fit_predict(reduced_hits)
hits["K_Means_reduced"] = k_hits_reduced

# %%
"""
Create new DataFrame, grabbing all the hits from the library
"""
hits_from_lib = get_hits_from_lib(lib, hits)

# %%
# Initialize an unsupervised nearest neighbors search
neigh = NearestNeighbors(n_neighbors=50)
neigh.fit(np.array(list(lib["rdkit"])))

# %%
# Query for the nearest neighbors of our specific compounds
# Example: Query for neighbors of:
query_SMILES = "CC1=C(O)C=C2NC(=O)CC(C3=CC=C(C=C3)C#CCCO)C2=C1"
query = hits[hits["SMILES"] == query_SMILES]

# %%
kneighbors = neigh.kneighbors(
    np.array(list(query["rdkit"]))
)
# %%
query_neighbors = lib.iloc[kneighbors[1][0], :]
# %%
imgs = get_imgs(query_neighbors)
# %%
# Now, determine for each neighbor if it is a hit or not
query_neighbors["IS_HIT"] = ""
for compound in query_neighbors.SMILES:
    if compound == query_SMILES:
        query_neighbors.loc[
            query_neighbors.SMILES == compound,
            "IS_HIT"
        ] = "QUERY"
    elif hits.SMILES.eq(compound).any():
        query_neighbors.loc[
            query_neighbors.SMILES == compound,
            "IS_HIT"
        ] = "HIT"

# %%
# Add compound image column
query_neighbors["Mol Image"] = [
    Chem.MolFromSmiles(s) for s in query_neighbors.SMILES
]
PandasTools.SaveXlsxFromFrame(
    query_neighbors, "Compound26_Neighbors.xlsx", molCol="Mol Image"
)

# %%
# Get nearest neighbors with k = library number total library compounds
# find the index of the farthest hit in the cluster
# Grab that meny nearest neighbors to send to Sarah
neigh_all = NearestNeighbors(n_neighbors=len(lib))
neigh_all.fit(np.array(list(lib["rdkit"])))
kneighbors_all = neigh_all.kneighbors(
    np.array(list(query["rdkit"]))
)

query_neighbors_all = lib.iloc[kneighbors_all[1][0], :]

# %%
query_desired_SMILES = [
    "CCC1=NC(C)=C(N1)C1CC(=O)NC2=CC(O)=C(C)C=C12",
    "CC1=C(O)C=C2NC(=O)CC(C3=CC=C(O3)C3=NNC=C3)C2=C1",
    "COC1=CC=C(N2C=CC=N2)C(=C1)C1CC(=O)NC2=CC(O)=C(C)C=C12",
    "CN1C=NN=C1SC1=CC=C(O1)C1CC(=O)NC2=CC(O)=C(C)C=C12",
   "COC1=C(O)C=C2NC(=O)CC(C3=C(Cl)N(C)N=C3C)C2=C1",
    "COC1=CC2=C(CCC2)C=C1C1CC(=O)NC2=CC(O)=C(C)C=C12",
    "COC1=CC=C(C=C1)N1C=CN=C1C1CC(=O)NC2=CC(O)=C(C)C=C12",
    "CC1=C(O)C=C2NC(=O)CC(C3=CC=C(C=C3)C#CCCO)C2=C1",
    "CC1=C(O)C=C2NC(=O)CC(C3=CNN=C3C3=CC=C(F)C=C3)C2=C1",
    "CC1=C(O)C=C2NC(=O)CC(C3=NC(=NO3)C3=CC=C(Cl)C=C3)C2=C1",
    "CC1=C(O)C=C2NC(=O)CC(C3=NNC(=C3)C(C)(C)C)C2=C1"
]

query_hits = hits[hits["SMILES"].isin(query_desired_SMILES)]
# %%
query_hits_from_neighbors = query_neighbors_all[
    query_neighbors_all.SMILES.isin(query_hits.SMILES)
]
# %%
last = int(query_hits_from_neighbors.iloc[-1].name)
desired_neighbors = query_neighbors_all.iloc[:last+1]

# %%
desired_neighbors["IS_HIT"] = ""
for compound in desired_neighbors.SMILES:
    if compound == query_SMILES:
        desired_neighbors.loc[
            desired_neighbors.SMILES == compound,
            "IS_HIT"
        ] = "QUERY"
    elif hits.SMILES.eq(compound).any():
        desired_neighbors.loc[
            desired_neighbors.SMILES == compound,
            "IS_HIT"
        ] = "HIT"

# %%
# Draw MOL images, save excel file
desired_neighbors["Mol Image"] = [
    Chem.MolFromSmiles(s) for s in desired_neighbors.SMILES
]
PandasTools.SaveXlsxFromFrame(
    desired_neighbors, "Compound26_Neighbors.xlsx", molCol="Mol Image"
)
# %%
""" Query against specific backbones for nearest neighbors """
query_backbones = [
    "CC1=C(C=C2NC(CC(C2=C1)C3=CCC=C3)=O)O",
    "O=C(NC1=CC(O)=C(C=C21)C)CC2C3=CC=CC=C3"
]

neigh_all = NearestNeighbors(n_neighbors=50)
neigh_all.fit(np.array(list(lib["rdkit"])))

# %%
# Create dataframe for the queries - SMILES and rdkit fingerprint
data = []
for query in query_backbones:
    new_df = pd.DataFrame({'SMILES': [query], 'rdkit': [np.nan]})
    data.append(new_df)
query_backbone_df = pd.concat(data).reset_index()
# %%
# Get rdkit fingerprint for the query backbones
query_butina = ButinaCluster()
query_mols = query_butina.get_mols(query_backbone_df)
query_butina.set_fps(query_mols)
query_backbone_df = query_butina.add_fps_to_df(
    query_backbone_df, "rdkit"
)
# %%
# Get closest neighbors for each query backbone 
query_backbone_neighbors = {}
for i in range(len(query_backbone_df)):
    query = query_backbone_df.iloc[[i]]
    kneighs = neigh_all.kneighbors(
        np.array(list(query.rdkit)).reshape(1, -1)
    )
    neighbors = lib.iloc[kneighs[1][0], :]

    # See if each neighbor is a hit or not
    neighbors["IS_HIT"] = ""
    for compound in neighbors.SMILES:
        if compound == query.SMILES:
            neighbors.loc[
                neighbors.SMILES == compound,
                "IS_HIT"
            ] = "QUERY"
        elif hits.SMILES.eq(compound).any():
            neighbors.loc[
                neighbors.SMILES == compound,
                "IS_HIT"
            ] = "HIT"

    # Add query backbone as compound 0
    query["IS_HIT"] = "QUERY"
    neighbors = pd.concat([query, neighbors])

    # Add chemical structure image to DataFrame
    neighbors["Mol Image"] = [
        Chem.MolFromSmiles(s) for s in neighbors.SMILES
    ]

    # Write to Excel file
    PandasTools.SaveXlsxFromFrame(
        neighbors, f'backboone_{i}.xlsx', molCol="Mol Image"
    )

    query_backbone_neighbors[query.SMILES] = {
        "kneighs": kneighs,
        "neighbors": lib.iloc[kneighs[1][0], :]}

# %%
