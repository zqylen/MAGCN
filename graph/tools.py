import numpy as np
import csv


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A): 
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A


def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx
                       for idx, i in enumerate(f.read().strip().split('\n'))}
        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)
        A_ = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)
        A__ = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                A_[id_dict[j], id_dict[i]] = 1
        for key,val in id_dict.items():
            A__[val, val] = 1
        A = normalize_digraph(A)
        A_ = normalize_digraph(A_)
        return np.stack((A__, A, A_))
    with open(distance_df_filename, 'r') as f:
        reader = csv.reader(f)
        header = f.__next__()
        edges_in = [(int(i[0]), int(i[1])) for i in reader]
        edges_out = [(int(i[1]), int(i[0])) for i in reader]
    self_link = [(i, i) for i in range(num_of_vertices)]
    return get_spatial_graph(num_of_vertices,  self_link, edges_in, edges_out)

