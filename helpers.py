"""
Helper classes and functions for chemical clustering
"""
# %%
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import numpy as np
import pandas as pd  # type: ignore
from rdkit import Chem  # type: ignore
from rdkit.Chem import AllChem, Draw, Descriptors  # type: ignore
from rdkit.Chem import rdFMCS # type: ignore
from rdkit.Chem.Scaffolds.MurckoScaffold import MakeScaffoldGeneric, MurckoScaffoldSmiles  # type: ignore
from rdkit.DataStructs import BulkTanimotoSimilarity  # type: ignore
from rdkit.ML.Cluster import Butina  # type: ignore
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

class ButinaCluster:

    def __init__(self, fptype: str="rdkit"):
        self.fptype = fptype

    
    def cluster_smiles(
            self, df: pd.DataFrame, 
            sim_cutoff: float=0.8, 
            column_name: str="SMILES"
        ) -> list:
        """
        Cluster the SMILES strings in the passed DataFrame
        Add new columns to the passed DataFrame
        """
        assert column_name in df.columns

        # Get Morgan Fingerprint for each SMILES string
        mols = [Chem.MolFromSmiles(x) for x in df[column_name]]
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
    

    def add_fps_to_df(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """
        Add new columns to the passed DataFrame
        """
        # Get Morgan Fingerprint for each SMILES string
        if column_name:
            df[column_name] = self.fp_list
        else:
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
    

def organize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Modify a dataframe in place, from the HTS data at Penn
    """
    df = df.iloc[:, :2]
    df.columns = ["Index", "SMILES"]
    df.set_index("Index", inplace=True)
    return df

# %%
def describe_cluster_counts(
        df: pd.DataFrame, 
        cluster: str="Cluster"
    ) -> dict[int, int]:
    """
    Describe the cluster counts for a dataframe
    """
    assert cluster in df.columns

    if cluster != "Cluster":
        df["Cluster"] = df[cluster]
    
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


def draw_cluster(df: pd.DataFrame, cluster: int) -> None:
    """
    Draw the cluster
    """
    smiles = df[df.Cluster == cluster].SMILES
    Draw.MolsToGridImage(smiles.apply(Chem.MolFromSmiles), molsPerRow=5)


def get_hits_from_lib(lib: pd.DataFrame, hits: pd.DataFrame) -> pd.DataFrame:
    """
    Get the hits from the library with clusters
    """
    assert "SMILES" in lib.columns and "SMILES" in hits.columns
    
    matching = lib.SMILES.isin(hits.SMILES)
    lib_hits = lib.loc[matching, :]
    return lib_hits


def get_centroids(arr: np.ndarray) -> np.ndarray:
    """
    Get the centroids of the clusters
    """
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return np.array([sum_x / length, sum_y / length])


def get_cluster_imgs(df: pd.DataFrame, cluster: str="K_Means") -> dict:
    """
    Create a dict of cluster images
    since images cannot be printed from a loop or function

    Returns a dictionary
        :k: cluster number
        :img: cluster image
    """
    assert cluster in df.columns
    if cluster != "K_Means":
        df["K_Means"] = df[cluster]

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
def save_cluster_imgs(imgs: dict, path: str=".") -> None:
    """
    Simply save cluster images in current directory
    """
    for k, img in imgs.items():
        with open(f"{path}/cluster_{k}.png", "wb") as f:
            f.write(img.data)


def plot_cluster_pops(clusters: list) -> None:
    """
    Plot the cluster populations from the predicted cluster list
    """
    ax = pd.Series(clusters).value_counts().sort_index().plot(kind='bar')
    ax.set_title("Cluster Populations")
    ax.set_xlabel("Cluster Number")
    ax.set_ylabel("Cluster Size")
    plt.show()

# %%
def calc_silhoutte(X: np.stack, min: int=5, max: int=30) -> pd.DataFrame:
    """
    Calculate the Silhoutte scores over a range of n_clusters
    to find the optimal n_clusters for KMeans
    """
    clusters = range(min, max)
    score_list: list = []
    for k in tqdm(clusters):
        km = KMeans(k, random_state=42, n_init='auto')
        cluster_labels = km.fit_predict(X)
        score = silhouette_score(X, cluster_labels)
        score_list.append([k, score])

    return pd.DataFrame(score_list, columns=["K", "Silhoutte_Score"])

# %%
def smiles_MCS_to_grid_image(
        smiles: list[str] | dict[str, str], 
        align_substructure: bool = True, 
        verbose: bool = False, 
        **kwargs
    ):
     """
     Convert a list (or dictionary) of SMILES strings to an RDKit grid image of the maximum common substructure (MCS) match between them

     :returns: RDKit grid image, and (if verbose=True) MCS SMARTS string and molecule, and list of molecules for input SMILES strings
     :rtype: RDKit grid image, and (if verbose=True) string, molecule, and list of molecules
     :param molecules: The SMARTS molecules to be compared and drawn
     :type molecules: List of (SMARTS) strings, or dictionary of (SMARTS) string: (legend) string pairs
     :param align_substructure: Whether to align the MCS substructures when plotting the molecules; default is True
     :type align_substructure: boolean
     :param verbose: Whether to return verbose output (MCS SMARTS string and molecule, and list of molecules for input SMILES strings); default is False so calling this function will present a grid image automatically
     :type verbose: boolean
     """
     mols = [Chem.MolFromSmiles(smile) for smile in smiles]
     res = rdFMCS.FindMCS(mols, **kwargs)
     mcs_smarts = res.smartsString
     mcs_mol = Chem.MolFromSmarts(res.smartsString)
     smarts = res.smartsString
     smart_mol = Chem.MolFromSmarts(smarts)
     smarts_and_mols = [smart_mol] + mols

     smarts_legend = "Max. substructure match"

     # If user supplies a dictionary, use the values as legend entries for molecules
     if isinstance(smiles, dict):
          mol_legends = [smiles[molecule] for molecule in smiles]
     else:
          mol_legends = ["" for mol in mols]

     legends =  [smarts_legend] + mol_legends
     matches = [""] + [mol.GetSubstructMatch(mcs_mol) for mol in mols]
     subms = [x for x in smarts_and_mols if x.HasSubstructMatch(mcs_mol)]

     Chem.rdDepictor.Compute2DCoords(mcs_mol)
     if align_substructure:
          for m in subms:
               _ = Chem.rdDepictor.GenerateDepictionMatching2DStructure(m, mcs_mol)

     drawing = Draw.MolsToGridImage(smarts_and_mols, highlightAtomLists=matches, legends=legends)
     if verbose:
          return drawing, mcs_smarts, mcs_mol, mols
     else:
          return drawing

# %%
def extract_murcko_scaff(
        smiles: list[str] | pd.Series, 
        generic: bool = True
    ) -> list[str]:
    """
    Extract Murcko scaffolds from a list of SMILES strings

    :returns: list of Murcko scaffolds
    :rtype: list
    :param smiles: list of SMILES strings
    :type smiles: list or pd.Series
    :param generic: whether to make scaffold generic (all C and single bonds)
    :type generic: boolean
    """
    if isinstance(smiles, pd.Series):
        smiles = smiles.tolist()

    scaffs = [MurckoScaffoldSmiles(smile) for smile in smiles]
    if generic:
        scaffs = [Chem.MolToSmiles(MakeScaffoldGeneric(Chem.MolFromSmiles(smile))) for smile in scaffs]
    return scaffs


def calc_logp(smiles: list[str] | pd.Series,) -> list[float]:
    """
    Calculate the molecular logP
    """
    if isinstance(smiles, pd.Series):
        smiles = smiles.tolist()

    logp = [Descriptors.MolLogP(Chem.MolFromSmiles(smile)) for smile in smiles]
    return logp
# %%
