from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from scipy.stats import zscore
from scipy.spatial import Delaunay
import numpy as np
import pandas as pd

# Apply DBSCAN:
def cluster(points, epsilon = None, min_samples = 2):

    if epsilon == None:
        # use k-distance to select epsilon 
        # based on https://stats.stackexchange.com/questions/88872/a-routine-to-choose-eps-and-minpts-for-dbscan
        nearest_neighbors = NearestNeighbors(n_neighbors=min_samples)
        nearest_neighbors.fit(points)
        distances, indices = nearest_neighbors.kneighbors(points)
        distances_k = np.sort(distances[:, min_samples - 1])
        
        kneedle = KneeLocator(range(1,len(distances_k)+1),  #x values
                              distances_k, # y values
                              S=1.0, #parameter suggested from paper
                              curve="convex", #parameter from figure
                              direction="increasing") #parameter from figure

        # multiply with 2 so that clusters are not too small
        epsilon = 2*kneedle.elbow_y
        
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    return dbscan.fit_predict(points)

def get_z_scores(dataset, min_cluster_size = 2):        
    # Arrays to score z scores and lsw normalized values
    result = pd.DataFrame()    
    for sample, value in dataset.groupby("sample"):
        sample_type = value.sample_type.unique()[0]
        clusters    = value["nuc_id"]
        radii       = value["radius"]

        for cluster_num, counts in Counter(clusters).items():
            if counts >= min_cluster_size and cluster_num >= 0:
                tmp    = radii[clusters == cluster_num].to_numpy()
                result = pd.concat([
                    result, 
                    pd.DataFrame({
                        "z_vals": zscore(tmp),
                        "z_logs": zscore(np.log(tmp)),
                        "z_mean": tmp/np.mean(tmp),
                        "sample": sample,
                        "sample_type": sample_type,
                        "nuc_id": cluster_num
                    })
                ])                
        
    return result

# analyze distance distributions
def analyze_distances(dataset):

    nearest_distances   = {sample_type: []  for sample_type in dataset['sample_type'].unique()}
    all_distances       = {sample_type: []  for sample_type in dataset['sample_type'].unique()}
    rotated_coordinates = {sample_type: []  for sample_type in dataset['sample_type'].unique()}
    num_neighbors       = {sample_type: []  for sample_type in dataset['sample_type'].unique()}
    
    for sample, value in dataset.groupby("sample"):
        sample_type = value.sample_type.unique()[0]
        points = np.stack(value.coordinate)

        num_neighbors_inner, nearest_distances_inner, all_distances_inner, rotated_coordinates_inner = analyze_pointcloud(points)

        nearest_distances[sample_type] += nearest_distances_inner
        all_distances[sample_type] += all_distances_inner
        rotated_coordinates[sample_type] += rotated_coordinates_inner
        num_neighbors[sample_type] += num_neighbors_inner
    
    return num_neighbors, nearest_distances, all_distances, rotated_coordinates

# analyze distance distributions
def analyze_pointcloud(points):

    nearest_distances   = []
    all_distances       = []
    rotated_coordinates = []
    
    tri = Delaunay(points)
    neighbors = [set() for _ in range(len(points))]
    
    for simplex in tri.simplices:
        for i in simplex:
            s = set(simplex)
            s.remove(i)
            neighbors[i].update(s)
    
    for i, neighborhood in enumerate(neighbors):
        dx = np.array([points[i] - points[tmp] for tmp in neighborhood])
        dx_norm = np.linalg.norm(dx, axis = -1)
    
        if dx_norm.size:
            nearest_distances.append(np.min(dx_norm))
            all_distances += list(dx_norm)

        i_min = np.argmin(dx_norm)
        phi_0 = np.arctan2(dx[i_min, 1], dx[i_min, 0])
        dx    = np.delete(dx, i_min, axis=0)
        dx_norm =  np.delete(dx_norm, i_min, axis=0)        
        phi   = np.arctan2(dx[:, 1], dx[:, 0]) - phi_0
        
        rotated_coordinates += list(np.transpose(dx_norm * np.array([np.cos(phi), np.sin(phi)])))

    num_neighbors = [len(tmp) for tmp in neighbors]
    return num_neighbors, nearest_distances, all_distances, rotated_coordinates