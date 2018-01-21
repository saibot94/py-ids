from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd

IDS_COLUMNS = [
    'duration',
    'protocol_type',
    'service',
    'flag',
    'src_bytes',
    'dst_bytes',
    'land',
    'wrong_fragment',
    'urgent',
    'hot',
    'num_failed_logins',
    'logged_in',
    'num_compromised',
    'root_shell',
    'su_attempted',
    'num_root',
    'num_file_creations',
    'num_shells',
    'num_access_files',
    'num_outbound_cmds',
    'is_host_login',
    'is_guest_login',
    'count',
    'srv_count',
    'serror_rate',
    'srv_serror_rate',
    'rerror_rate',
    'srv_rerror_rate',
    'same_srv_rate',
    'diff_srv_rate',
    'srv_diff_host_rate',
    'dst_host_count',
    'dst_host_srv_count',
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    'outcome'
]


def df_to_tf_matrix(df):
    """
    Convert a Pandas df to a matrix of floats
    """
    result = []
    for x in df.columns:
        result.append(x)
    return df.as_matrix(result).astype(np.float64)


def get_label_column(dataset_file):
    return np.genfromtxt(dataset_file,
         delimiter=',', usecols=(41), dtype=str)


def load_ids_df(path):
    testdf = pd.read_csv(path)
    testdf.dropna(inplace=True,axis=1)
    testdf.columns = IDS_COLUMNS
    return testdf


def map_winner_neurons(som, data, target_index=-1):
    """
    Creates a matrix containing the responses (a map) for each of the neurons that are in the som
    """
    vals = []
    mappings_list = np.zeros(np.shape(som.get_weights())[:2]).tolist()
    for i in range(len(mappings_list)):
        for j in range(len(mappings_list[i])):
            mappings_list[i][j] = {}
    for val in data:
        winner = som.winner(val[:target_index])
        target_dict = mappings_list[winner[0]][winner[1]]
        target = val[target_index]
        if target not in target_dict:
            target_dict[target] = 1
        else:
            target_dict[target] += 1
    return mappings_list
