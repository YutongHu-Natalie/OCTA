import argparse
import numpy as np
import os
from tqdm import tqdm
import time
import pickle as pkl
import json
import requests
import sys
import geopandas as gpd
import networkx as nx

import torch
import copy
import torch_geometric.utils
from torch_geometric.data import Data, DataLoader
from torch_scatter import scatter

from utils.utils import build_spanning_tree_edge, find_higher_order_neighbors, add_self_loops
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--model', type=str, default='SGMP')
    parser.add_argument('--dataset', type=str, default='BACE')
    parser.add_argument('--split', type=str, default='811')
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--readout', type=str, default='add')
    parser.add_argument('--spanning_tree', type=str, default='False')
    parser.add_argument('--structure', type=str, default='sc')

    parser.add_argument('--random_seed', type=int, default=12345)
    parser.add_argument('--random_seed_2', type=int, default=12345)
    parser.add_argument('--label', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--test_per_round', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--cutoff', type=float, default=10.0)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    args = parser.parse_args()

    return args

def load_data(args):
    if args.dataset == 'synthetic':
        with open(os.path.join(args.data_dir, 'synthetic', 'synthetic.pkl'), 'rb') as file:
            dataset = pkl.load(file)
        dataset = dataset[torch.randperm(len(dataset))]    
        train_valid_split = int( int(args.split[0]) / 10 * len(dataset) )
        valid_test_split = int( int(args.split[1]) / 10 * len(dataset) )

    elif args.dataset == 'QM9':
        from torch_geometric.datasets import QM9
        dataset = QM9(root=os.path.join(args.data_dir, 'QM9'))
        random_state = np.random.RandomState(seed=42)
        perm = torch.from_numpy(random_state.permutation(np.arange(130831)))
        dataset = dataset[perm]
        train_valid_split, valid_test_split = 110000, 10000

    elif args.dataset == 'brain':
        from utils.brain_load_data import load_brain_data
        dataset = load_brain_data(data_dir=args.data_dir, structure='sc', threshold=5e5, random_seed=args.random_seed) 
        train_valid_split = int( int(args.split[0]) / 10 * len(dataset) )
        valid_test_split = int( int(args.split[1]) / 10 * len(dataset) )

    elif args.dataset == 'NeuroMorpho':
        if os.path.exists('./SGMP_code-main/data/NeuroMorpho/neuro_cell_human.pkl'):
            with open('./SGMP_code-main/data/NeuroMorpho/neuro_cell_human.pkl', 'rb') as file:
                dataset = pkl.load(file)
        else:
            dataset = process_swc_files()
            with open('./SGMP_code-main/data/NeuroMorpho/neuro_cell_human.pkl', 'wb') as file:
                pkl.dump(dataset, file)

        train_valid_split = int( int(args.split[0]) / 10 * len(dataset) )
        valid_test_split = int( int(args.split[1]) / 10 * len(dataset) )

    elif args.dataset == 'Neuro_exp_cond':
        if os.path.exists('/scratch1/azhan82/neuro_5x_pc.pkl'):
            with open('/scratch1/azhan82/neuro_5x_pc.pkl', 'rb') as file:
                dataset = pkl.load(file)
        else:
            five_glia, five_pc, lps_glia, lps_inter, lps_pc = process_swc_files_exp_cond()
            with open('./SGMP_code-main/data/NeuroMorpho/neuro_5x_glia.pkl', 'wb') as file:
                pkl.dump(five_glia, file)
            with open('./SGMP_code-main/data/NeuroMorpho/neuro_5x_pc.pkl', 'wb') as file:
                pkl.dump(five_pc, file)
            with open('./SGMP_code-main/data/NeuroMorpho/neuro_lps_glia.pkl', 'wb') as file:
                pkl.dump(lps_glia, file)
            with open('./SGMP_code-main/data/NeuroMorpho/neuro_lps_inter.pkl', 'wb') as file:
                pkl.dump(lps_inter, file)
            with open('./SGMP_code-main/data/NeuroMorpho/neuro_lps_pc.pkl', 'wb') as file:
                pkl.dump(lps_pc, file)
            dataset = five_glia

        train_valid_split = int( int(args.split[0]) / 10 * len(dataset) )
        valid_test_split = int( int(args.split[1]) / 10 * len(dataset) )

    elif args.dataset == 'NHD':
        if os.path.exists('./SGMP_code-main/data/NHD_2/nhd_cluster.pkl'):
            with open('./SGMP_code-main/data/NHD_2/nhd_cluster.pkl', 'rb') as file:
                dataset = pkl.load(file)
        else:
            nhd_cluster, nhd_diameter, nhd_radius = process_nhd()
            with open('./SGMP_code-main/data/NHD_2/nhd_cluster13.pkl', 'wb') as file:
                pkl.dump(nhd_cluster, file)
            with open('./SGMP_code-main/data/NHD_2/nhd_diameter13.pkl', 'wb') as file:
                pkl.dump(nhd_diameter, file)
            with open('./SGMP_code-main/data/NHD_2/nhd_radius13.pkl', 'wb') as file:
                pkl.dump(nhd_radius, file)
            dataset = nhd_cluster

        train_valid_split = int( int(args.split[0]) / 10 * len(dataset) )
        valid_test_split = int( int(args.split[1]) / 10 * len(dataset) )

    elif args.dataset in ["ESOL", "Lipo", "BACE", "BBBP"]:
        from utils.moleculenet import MoleculeNet
        dataset = MoleculeNet(root=os.path.join(args.data_dir, 'MoleculeNet', args.dataset),name=args.dataset)
        dataset = dataset[torch.randperm(len(dataset))]    
        train_valid_split = int( int(args.split[0]) / 10 * len(dataset) )
        valid_test_split = int( int(args.split[1]) / 10 * len(dataset) )
    else:
        raise Exception('Dataset not recognized.')

    
    train_dataset = dataset[:train_valid_split]
    valid_dataset = dataset[train_valid_split:train_valid_split+valid_test_split]
    test_dataset = dataset[train_valid_split+valid_test_split:]
        
    print('======================')
    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of valid graphs: {len(valid_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


def read_swc_file(filename):
    header = ''
    scale = [1.0, 1.0, 1.0]

    neuron_array = np.array([], dtype=float).reshape(0, 7)

    with open(filename, 'r') as f:
        for line in f:
            if line[0] == '#':
                header += line
                if 'SCALE' in line:
                    curr_line = line.split()
                    if curr_line[1] == 'SCALE':
                        scale[0] = float(curr_line[2])
                        scale[1] = float(curr_line[3])
                        scale[2] = float(curr_line[4])
                    else:
                        scale[0] = float(curr_line[1])
                        scale[1] = float(curr_line[2])
                        scale[2] = float(curr_line[3])

            elif len(line) > 1:
                curr_line = line.strip().split()

                curr_line_list = [int(curr_line[0]) - 1,
                                  int(curr_line[1]),
                                  float(curr_line[2]) * scale[0],
                                  float(curr_line[3]) * scale[1],
                                  float(curr_line[4]) * scale[2],
                                  float(curr_line[5]),
                                  int(curr_line[6]) - 1]

                neuron_array = np.vstack([neuron_array, curr_line_list])
        
    return neuron_array


def process_swc_files():
    directory = './SGMP_code-main/data/NeuroMorpho/human'

    possible_classes = ['glia', 'interneuron', 'long-range non-principal gabaergic', 'principal cell', 'sensory', 'null']

    problem_data = []

    brain_region_list = []
    # with os.scandir(directory) as data_folder:
    #     for file in data_folder:
    #         if file.name.endswith('.swc') and file.is_file():
    #             response = requests.get('https://neuromorpho.org/api/neuron/name/' + file.name.split('.swc', 1)[0])

    #             if not response.ok:
    #                 print('Response not ok: ' + str(file.name))

    #             # Parsing requested metadata text to find primary brain region
    #             try:
    #                 split1 = response.text.split("\"brain_region\":", 1)[1]
    #                 split1 = split1.split(",\"", 1)[0]
    #                 split1 = split1.strip("\"[]").lower()
    #             except:
    #                 e = sys.exc_info()
    #                 print(e)
    #                 print('Response split not working 166: '+ str(file.name))
    #                 problem_data.append('Response split fail 166: ' + str(file.name))
    #                 continue

    #             # Add parsed brain region to list
    #             brain_region_list.append(split1)

    # # Removing duplicates from brain region list
    # brain_region_list = list(set(brain_region_list))

    with open('./SGMP_code-main/out_count_class_cell_human', 'r') as file:
        for line in file:
            brain_region_list.append(line.split(':')[0].lower())

    # Writing .txt file of brain region class mappings, to be used later in constructing y
    brain_region_list.sort()
    with open('./SGMP_code-main/class_mapping.txt', 'w') as out:
        for i in range(len(brain_region_list)):
            out.write(str(i) + ' ' + brain_region_list[i] + '\n')


    graph_list = []
    with os.scandir(directory) as data_folder:
        for file in data_folder:
            if file.name.endswith('.swc') and file.is_file():
                # Load file as numpy array
                file_name = os.path.join(directory, file.name)
                try:
                    neuron_arr = read_swc_file(file_name)
                # If error occurs, add file_name to list and keep going
                except:
                    print('read_swc_file() fail: ')
                    e = sys.exc_info()
                    print(e)
                    problem_data.append('read_swc_fail: ' + str(file_name))
                    continue

                # Dealing with pos
                new_pos = []
                for i in neuron_arr:
                    new_pos.append([i[2], i[3], i[4]])

                # Dealing with edge_index
                new_edge_index = []
                new_edge_index_1 = []
                new_edge_index_2 = []
                for i in neuron_arr:
                    if (i[6] < 0):
                        continue

                    new_edge_index_1.append(i[6])
                    new_edge_index_1.append(i[0])
                    new_edge_index_2.append(i[0])
                    new_edge_index_2.append(i[6])
                
                new_edge_index.append(new_edge_index_1)
                new_edge_index.append(new_edge_index_2)
                
                # Dealing with x
                new_x = []
                for i in neuron_arr:
                    new_x.append([i[1], i[5]])

                # Dealing with y
                new_y = [0]

                response = requests.get('https://neuromorpho.org/api/neuron/name/' + file.name.split('.swc', 1)[0])

                if not response.ok:
                    print('Response not ok: ' + str(file.name))

                # Parsing requested metadata text to find primary cell type
                try:
                    split1 = response.text.split("\"cell_type\":", 1)[1]
                    split1 = split1.split("],\"", 1)[0]
                    split1 = split1.split(",")
                    split1 = [x.strip('\"[]').lower() for x in split1]

                    if split1[0] in possible_classes:
                        cell_class = split1[0]
                    elif split1[-1] in possible_classes:
                        cell_class = split1[-1]
                    else:
                        for i in split1:
                            if i in possible_classes:
                                cell_class = i
                                break
                        else:
                            print('Class not found: ' + str(file.name))
                            continue
                except:
                    e = sys.exc_info()
                    print(e)
                    print('Response split not working 252: ' + str(file_name))
                    problem_data.append('Response split fail 252: ' + str(file_name))
                    continue

                try:
                    new_y[0] = brain_region_list.index(cell_class)
                    new_y = [new_y]
                except:
                    e = sys.exc_info()
                    print(e)
                    print('Cannot find corresponding brain region 276: ' + str(file_name))
                    problem_data.append('Cannot find corresponding brain region 276: ' + str(file_name))
                    continue

                # Initialize Data object with respective tensors
                new_data = Data(
                    x = torch.tensor(new_x, dtype=torch.float),
                    edge_index = torch.tensor(new_edge_index, dtype=torch.long),
                    pos = torch.tensor(new_pos, dtype=torch.float),
                    y = torch.tensor(new_y, dtype=torch.long)
                )

                # Add new graph to graph_list
                graph_list.append(new_data)

    # Write problem data names to file
    with open("problem_data.txt", 'a') as file:
        for neuron in problem_data:
            file.write(str(neuron) + '\n')

    return graph_list


def process_swc_files_exp_cond():
    directory = './SGMP_code-main/data/NeuroMorpho/mouse'

    possible_exp_conds = ['control', '5xfad alzheimer model (app overexpression)', 'lipopolysaccharide injection']
    possible_cell_types = ['glia', 'interneuron', 'principal cell']

    problem_data = []

    graph_list_5x_glia = []
    graph_list_5x_pc = []
    graph_list_lps_glia = []
    graph_list_lps_inter = []
    graph_list_lps_pc = []

    with os.scandir(directory) as data_folder:
        for file in data_folder:
            if file.name.endswith('.swc') and file.is_file():
                # Dealing with y
                new_y = [0]

                response = requests.get('https://neuromorpho.org/api/neuron/name/' + file.name.split('.swc', 1)[0])

                if not response.ok:
                    print('Response not ok: ' + str(file.name))

                # Parsing requested metadata text to find BOTH experimental condition and primary cell type
                try:
                    split1 = response.text.split("\"experiment_condition\":", 1)[1]
                    split1 = split1.split("],\"", 1)[0]
                    split1 = split1.split(",")
                    split1 = [x.strip('\"[]').lower() for x in split1]
                    exp_cond = split1
                
                except:
                    e = sys.exc_info()
                    print(e)
                    print('Exp cond response split not working 252: ' + str(file_name))
                    problem_data.append('Exp cond response split fail 252: ' + str(file_name))
                    continue

                try:
                    split1 = response.text.split("\"cell_type\":", 1)[1]
                    split1 = split1.split("],\"", 1)[0]
                    split1 = split1.split(",")
                    split1 = [x.strip('\"[]').lower() for x in split1]

                    cell_type = ''
                    if split1[0] in possible_cell_types:
                        cell_type = split1[0]
                    elif split1[-1] in possible_cell_types:
                        cell_type = split1[-1]
                    else:
                        for i in split1:
                            if i in possible_cell_types:
                                cell_type = i
                                break
                    
                    if cell_type == '':
                        continue

                except:
                    e = sys.exc_info()
                    print(e)
                    print('Cell type response split not working 252: ' + str(file_name))
                    problem_data.append('Cell type response split fail 252: ' + str(file_name))
                    continue

                # If exp cond is control
                if exp_cond[0] == possible_exp_conds[0]:
                    new_y[0] = 0
                # If exp cond is 5xfad
                elif exp_cond[0] == possible_exp_conds[1]:
                    new_y[0] = 1
                # If exp cond is lps
                elif exp_cond[0] == possible_exp_conds[2]:
                    new_y[0] = 1
                # Unwanted exp cond
                else:
                    continue
                new_y = [new_y]

                # Load file as numpy array
                file_name = os.path.join(directory, file.name)
                try:
                    neuron_arr = read_swc_file(file_name)
                # If error occurs, add file_name to list and keep going
                except:
                    print('read_swc_file() fail: ')
                    e = sys.exc_info()
                    print(e)
                    problem_data.append('read_swc_fail: ' + str(file_name))
                    continue

                # Dealing with pos
                new_pos = []
                for i in neuron_arr:
                    new_pos.append([i[2], i[3], i[4]])

                # Dealing with edge_index
                new_edge_index = []
                new_edge_index_1 = []
                new_edge_index_2 = []
                for i in neuron_arr:
                    if (i[6] < 0):
                        continue

                    new_edge_index_1.append(i[6])
                    new_edge_index_1.append(i[0])
                    new_edge_index_2.append(i[0])
                    new_edge_index_2.append(i[6])
                
                new_edge_index.append(new_edge_index_1)
                new_edge_index.append(new_edge_index_2)
                
                # Dealing with x
                new_x = []
                for i in neuron_arr:
                    new_x.append([i[1], i[5]])

                # Initialize Data object with respective tensors
                new_data = Data(
                    x = torch.tensor(new_x, dtype=torch.float),
                    edge_index = torch.tensor(new_edge_index, dtype=torch.long),
                    pos = torch.tensor(new_pos, dtype=torch.float),
                    y = torch.tensor(new_y, dtype=torch.long)
                )

                # Add new graphs to lists depending on cell type and condition
                # If exp cond is control
                if exp_cond[0] == possible_exp_conds[0]:
                    # If control and glia
                    if cell_type == possible_cell_types[0]:
                        graph_list_5x_glia.append(new_data)
                        graph_list_lps_glia.append(new_data)
                    
                    # If control and interneuron
                    elif cell_type == possible_cell_types[1]:
                        graph_list_lps_inter.append(new_data)
                    
                    # If control and principal cell
                    elif cell_type == possible_cell_types[2]:
                        graph_list_5x_pc.append(new_data)
                        graph_list_lps_pc.append(new_data)
                    
                # If exp cond is 5xfad
                elif exp_cond[0] == possible_exp_conds[1]:
                    # If 5xfad and glia
                    if cell_type == possible_cell_types[0]:
                        graph_list_5x_glia.append(new_data)

                    # If 5xfad and interneuron
                    elif cell_type == possible_cell_types[1]:
                        print('5xfad interneuron??: ' + str(file_name))
                        problem_data.append('5xfad interneuron??: ' + str(file_name))
                        continue

                    # If 5xfad and principal cell
                    elif cell_type == possible_cell_types[2]:
                        graph_list_5x_pc.append(new_data)

                # If exp cond is lps
                elif exp_cond[0] == possible_exp_conds[2]:
                    # If lps and glia
                    if cell_type == possible_cell_types[0]:
                        graph_list_lps_glia.append(new_data)

                    # If lps and interneuron
                    elif cell_type == possible_cell_types[1]:
                        graph_list_lps_inter.append(new_data)
                    
                    # If lps and principal cell
                    elif cell_type == possible_cell_types[2]:
                        graph_list_lps_pc.append(new_data)

    return graph_list_5x_glia, graph_list_5x_pc, graph_list_lps_glia, graph_list_lps_inter, graph_list_lps_pc


def process_nhd():
    directory = './SGMP_code-main/data/NHD_2'

    graph_list_cluster = []
    graph_list_diameter = []
    graph_list_radius = []

    with os.scandir(directory) as data_folder:
        for entry in data_folder:
            if entry.is_dir():
                # Generating list of geometries
                gdf = gpd.read_file(os.path.join(directory, entry.name, 'Shape', 'NHDFlowline.shp'))

                ####### DEBUG
                print(f'{entry.name} start', end=' >>> ', flush=True)
                start_time = time.time()

                geometries_list = []
                for idx, row in gdf.iterrows():
                    coord_list = [(x, y) for x, y, _ in gdf.loc[idx, 'geometry'].coords]
                    geometries_list.append(coord_list)

                G = nx.Graph()

                # Tracking node/position to index mapping
                node_to_index = {}

                # Adding only endpoints as nodes
                for my_list in geometries_list:
                    if my_list:
                        endpoint1 = my_list[0]
                        endpoint2 = my_list[-1]

                        if endpoint1 not in node_to_index:
                            index1 = len(node_to_index)
                            node_to_index[endpoint1] = index1
                            G.add_node(index1, pos=endpoint1)

                        if endpoint2 not in node_to_index:
                            index2 = len(node_to_index)
                            node_to_index[endpoint2] = index2
                            G.add_node(index2, pos=endpoint2)

                        G.add_edge(node_to_index[endpoint1], node_to_index[endpoint2])

                # # Adding ALL coordinates as nodes
                # for coord_list in geometries_list:
                #     for i in range(len(coord_list)):
                #         node1 = coord_list[i]
                #         G.add_node(node1, pos=(node1))

                #         if i < len(coord_list) - 1:
                #             node2 = coord_list[i + 1]
                #             G.add_node(node2, pos=(node2))
                #             G.add_edge(node1, node2)

                # Find the connected components
                connected_components = list(nx.connected_components(G))

                # Determine the largest connected component
                largest_component = max(connected_components, key=len)

                # Create a subgraph with only the largest component
                G_sub = G.subgraph(largest_component).copy()

                # Convert NetworkX into PyTorchGeo Data with pos and edge_index data
                data = torch_geometric.utils.from_networkx(G_sub)

                ##### DEBUG
                # edge_index = data.edge_index
                # coo = torch.stack([edge_index[0], edge_index[1]], dim=0)
                # unique_coo, counts = coo.unique(dim=1, return_counts=True)

                # # Find duplicate edges (if any)
                # duplicate_edges = unique_coo[:, counts > 1]
                # num_duplicates = duplicate_edges.size(1)
                # print(num_duplicates)

                # self_loops = (edge_index[0] == edge_index[1]).sum().item()
                # print(self_loops)

                # Calculating clustering coefficient, diameter, radius and packaging as y
                new_y_cluster = [0]
                new_y_diameter = [0]
                new_y_radius = [0]

                new_y_cluster[0] = nx.average_clustering(G_sub)
                new_y_diameter[0] = nx.diameter(G_sub)
                new_y_radius[0] = nx.radius(G_sub)

                new_y_cluster = [new_y_cluster]
                new_y_diameter = [new_y_diameter]
                new_y_radius = [new_y_radius]

                # Setting node features to 1
                new_x = torch.ones(G_sub.number_of_nodes(), 1)
                data.x = new_x

                # Adding 0 as 3rd dimension to pos
                data.pos = torch.cat([data.pos, torch.zeros(data.pos.size(0), 1)], dim=1)

                # # Setting pos as node coordinates
                # new_pos = torch.tensor([G_sub.nodes[node]['pos'] for node in G_sub], dtype=torch.float)

                # for node, data in G.nodes(data=True):
                #     index = node_to_index[data['pos']]  # Get the index of the node based on its pos attribute
                #     pos = new_pos[index].tolist()  # Get the pos from new_pos tensor
                #     if pos != data['pos']:
                #         print(f"Node {node} does not match its pos attribute in new_pos.")

                # # Set edge index
                # edges = list(G_sub.edges())
                # edges += [(edge[1], edge[0]) for edge in edges]
                # new_edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

                # Create Data objects
                # new_data_cluster = Data(x = new_x, edge_index = new_edge_index, pos = new_pos, y = torch.tensor(new_y_cluster, dtype=torch.float))
                # new_data_diameter = Data(x = new_x, edge_index = new_edge_index, pos = new_pos, y = torch.tensor(new_y_diameter, dtype=torch.float))
                # new_data_radius = Data(x = new_x, edge_index = new_edge_index, pos = new_pos, y = torch.tensor(new_y_radius, dtype=torch.float))
                new_data_cluster = data.clone()
                new_data_diameter = data.clone()
                new_data_radius = data.clone()

                new_data_cluster.y = torch.tensor(new_y_cluster, dtype=torch.float)
                new_data_diameter.y = torch.tensor(new_y_diameter, dtype=torch.float)
                new_data_radius.y = torch.tensor(new_y_radius, dtype=torch.float)

                # Add new graphs to graph_lists
                graph_list_cluster.append(new_data_cluster)
                graph_list_diameter.append(new_data_diameter)
                graph_list_radius.append(new_data_radius)

                print(f'{entry.name} done, {(time.time() - start_time):.4f}', flush=True)

    return graph_list_cluster, graph_list_diameter, graph_list_radius


def main(data, args):
    # logging
    if args.spanning_tree == 'True':
        LOG_DIR = os.path.join(args.save_dir, args.dataset, args.model+'_st')
    else:
        LOG_DIR = os.path.join(args.save_dir, args.dataset, args.model)
    if args.dataset == 'QM9':
        LOG_DIR = os.path.join(LOG_DIR, str(args.label))
        
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    log_file = os.path.join(LOG_DIR, 'log.csv')
    result_file = os.path.join(LOG_DIR, 'results.txt')
    
    # device
    if args.device == 'cpu':
        device = torch.device("cpu")
    elif args.device == 'gpu':
        device = torch.device("cuda:0")
    else:
        raise Exception('Please assign the type of device: cpu or gpu.')
    
    if args.dataset in ['brain', 'NHD']:
        input_channels_node, hidden_channels, readout = 1, 64, args.readout
    elif args.dataset == 'QM9':
        input_channels_node, hidden_channels, readout = 11, 64, args.readout
    elif args.dataset in ["NeuroMorpho", "Neuro_exp_cond"]:
        input_channels_node, hidden_channels, readout = 2, 64, args.readout
    else:
        input_channels_node, hidden_channels, readout = 9, 64, args.readout
    
    if args.dataset in [ 'BACE', 'BBBP', 'NeuroMorpho', 'Neuro_exp_cond']:
        task = 'classification'
    elif args.dataset in ['QM9', 'ESOL', 'Lipo', 'NHD']:
        task = 'regression'
    elif args.dataset in ['brain']:
        task = 'regression'
            
    if task == 'regression':
        output_channels = 1
    else:
        output_channels = 2

    # Special case for NeuroMorpho
    if args.dataset == 'NeuroMorpho':
        with open('./SGMP_code-main/out_count_class_cell_human', 'r') as file:
            lines = file.readlines()
        output_channels = len(lines)
        
    # select model and its parameter
    print(args.model)
    if args.model == 'GIN':
        from models.GIN import GINNet
        net = GINNet(input_channels_node, hidden_channels, output_channels, readout=readout, eps=0., num_layers=args.num_layers)
    elif args.model == 'GCN':
        from models.GCN import GCNNet
        net = GCNNet(input_channels_node, hidden_channels, output_channels, readout=readout, num_layers=args.num_layers)
    elif args.model == 'PointNetraw':
        from models.PointNetraw import PointNetraw
        net = PointNetraw(input_channels_node, hidden_channels, output_channels, readout=readout, num_layers=args.num_layers)
    elif args.model == 'GAT':
        from models.GAT import GATNet
        net = GATNet(input_channels_node, hidden_channels, output_channels, readout=readout, num_layers=args.num_layers)
    elif args.model == 'GatedGraphConv':
        from models.GatedGraphConv import GatedNet
        net = GatedNet(input_channels_node, hidden_channels, output_channels, readout=readout, num_layers=args.num_layers)
    elif args.model == 'PointNet':
        from models.PointNet import PointNet
        net = PointNet(input_channels_node, hidden_channels, output_channels, readout=readout, num_layers=args.num_layers)
    elif args.model == 'PPFNet':
        from models.PPFNet import PPFNet
        net = PPFNet(input_channels_node, hidden_channels, output_channels, readout=readout, num_layers=args.num_layers)
    elif args.model == 'SGCN':
        from models.SGCN import SGCN
        net = SGCN(input_channels_node, hidden_channels, output_channels, readout=readout, num_layers=args.num_layers)
    elif args.model == 'Schnet':
        from models.Schnet import Schnet
        net = Schnet(input_channels_node=input_channels_node, 
                    hidden_channels=hidden_channels, output_channels=output_channels, num_interactions=args.num_layers,
                    num_gaussians=hidden_channels, cutoff=args.cutoff, readout=readout)
    elif args.model == 'Dimenet':
        from models.Dimenet import Dimenet
        net = Dimenet(input_channels_node=input_channels_node, 
                    hidden_channels=hidden_channels, output_channels=output_channels, num_blocks=args.num_layers,
                    cutoff=args.cutoff)
    elif args.model == 'SGMP':
        from models.SGMP import SGMP
        net = SGMP(input_channels_node=input_channels_node, 
            hidden_channels=hidden_channels, output_channels=output_channels,
            num_interactions=args.num_layers, cutoff=args.cutoff,
            readout=readout)
        
    model = net.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=10, min_lr=5e-5)
    
    if task == 'regression':
        criterion = torch.nn.MSELoss()
        from sklearn.metrics import mean_absolute_error
        measure = mean_absolute_error
    elif task == 'classification':
        criterion = torch.nn.CrossEntropyLoss()
        from sklearn.metrics import accuracy_score
        measure = accuracy_score
        
    # get train/valid/test data
    train_loader, valid_loader, test_loader = data        
        
    def train(loader, model, args):
        model.train()
        for data in (loader):  # Iterate in batches over the training dataset.
            x, pos, edge_index, batch = data.x.float(), data.pos, data.edge_index, data.batch
            if args.dataset == 'brain':
                x = x.long()
            if args.dataset == 'QM9':
                y = data.y[:, args.label]
            else:
                y = data.y.long() if task == 'classification' else data.y
            num_nodes = data.num_nodes
            num_edges = data.num_edges
            if args.spanning_tree == 'True':
                edge_index = build_spanning_tree_edge(edge_index.cpu(), algo='scipy', num_nodes=num_nodes, num_edges=num_edges)
            x, pos, edge_index, batch, y = x.to(device), pos.to(device), edge_index.to(device), batch.to(device), y.to(device)
            if args.model == 'SGMP':
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes) # add self loop to avoid crash on specific data point with longest path < 3
                _, _, edge_index_3rd, _, _, _, _, _ = find_higher_order_neighbors(edge_index, num_nodes, order=3)
                out = model(x, pos, batch, edge_index_3rd)
            else:
                out = model(x, pos, edge_index, batch)
                
            if task == 'classification':
                loss = criterion(out, y.reshape(-1))  # Compute the loss.
            else:
                loss = criterion(out.reshape(-1, 1), y.reshape(-1, 1))  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

    def test(loader, model, args):
        model.eval()
        y_hat, y_true, y_probs = [], [], []
        loss_total, total_graph = 0, 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            x, pos, edge_index, batch = data.x.float(), data.pos, data.edge_index, data.batch
            if args.dataset == 'brain':
                x = x.long()
            if args.dataset == 'QM9':
                y = data.y[:, args.label]
            else:
                y = data.y.long() if task == 'classification' else data.y
            num_nodes = data.num_nodes
            num_edges = data.num_edges
            if args.spanning_tree == 'True':
                edge_index = build_spanning_tree_edge(edge_index.cpu(), algo='scipy', num_nodes=num_nodes, num_edges=num_edges)
            x, pos, edge_index, batch, y = x.to(device), pos.to(device), edge_index.to(device), batch.to(device), y.to(device)
            if args.model == 'SGMP':
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes, fill_value=-1.)
                _, _, edge_index_3rd, _, _, _, _, _ = find_higher_order_neighbors(edge_index, num_nodes, order=3)
                out = model(x, pos, batch, edge_index_3rd)
            else:
                out = model(x, pos, edge_index, batch)
                
            if task == 'classification':
                loss = criterion(out, y.reshape(-1))  # Compute the loss.
            else:
                loss = criterion(out.reshape(-1, 1), y.reshape(-1, 1))  # Compute the loss.
            loss_total += loss.detach().cpu() * data.num_graphs
            total_graph += data.num_graphs
            if task == 'classification':
                pred = out.argmax(dim=1)  # Use the class with highest probability.
                y_hat += list(pred.cpu().detach().numpy().reshape(-1))
            else:
                y_hat += list(out.cpu().detach().numpy().reshape(-1))
            y_true += list(y.cpu().detach().numpy().reshape(-1))

            if task == 'classification':
                # L1 norm for out
                out_data = out.detach().cpu().numpy()
                min_values = np.min(out_data, axis=1)
                temp = out_data - min_values[:, np.newaxis]
                row_sums = temp.sum(axis=1)
                normalized_out = temp / row_sums[:, np.newaxis]

                for i in normalized_out:
                    y_probs.append(list(i))

            ##### DEBUG
            # print(normalized_out)
            # print(np.sum(normalized_out))
            # print('\n')

        return loss_total/total_graph, y_hat, y_true, y_probs
            
    with open(log_file, 'a') as f:
        print(f"Epoch, Valid loss, Valid score, --- %s seconds ---", file=f) 

    start_time = time.time()
    best_valid_score = 1e10 if task == 'regression' else 0
    best_model = None
    for epoch in (range(1, args.epoch)):
        # training
        train(train_loader, model, args)
        
        if epoch % args.test_per_round == 0:
            valid_loss, yhat_valid, ytrue_valid, yhat_probs = test(valid_loader, model, args)
            valid_score = measure(ytrue_valid, yhat_valid)

            ##### DEBUG
            # print(ytrue_valid)
            # print(len(ytrue_valid))
            # print(yhat_probs)
            # print(type(yhat_probs))
            # yhat_probs = np.array(yhat_probs)
            # print(yhat_probs[:, 1])
            # print(type(yhat_probs))
            # print(len(yhat_probs))

            # Only do ROC calc for classification
            if task == 'classification':
                labels = np.arange(np.shape(yhat_probs)[1])
                # If binary classification, exclude multi-class parameter
                if output_channels == 2:
                    yhat_probs = np.array(yhat_probs)
                    valid_roc_score = roc_auc_score(ytrue_valid, yhat_probs[:, 1])
                else:
                    valid_roc_score = roc_auc_score(ytrue_valid, yhat_probs, multi_class='ovo', labels=labels)

            if epoch >= 100:
                lr = scheduler.optimizer.param_groups[0]['lr']
                scheduler.step(valid_loss)

            with open(log_file, 'a') as f:
                if task == 'classification':
                    print(f"{epoch:03d}, {valid_loss:.4f}, {valid_score:.4f}, {valid_roc_score:.4f}, {(time.time() - start_time):.4f}", file=f)
                else:
                    print(f"{epoch:03d}, {valid_loss:.4f}, {valid_score:.4f} ,{(time.time() - start_time):.4f}", file=f) 

            if task == 'regression':
                if valid_score < best_valid_score:
                    best_valid_score = valid_score
                    best_model = copy.deepcopy(model)
            else:
                if valid_score > best_valid_score:
                    best_valid_score = valid_score
                    best_model = copy.deepcopy(model)
    

    train_loss, yhat_train, ytrue_train, yhat_probs = test(train_loader, model, args)
    train_score = measure(ytrue_train, yhat_train)
    if task == 'classification':
        if output_channels == 2:
            yhat_probs = np.array(yhat_probs)
            train_roc_score = roc_auc_score(ytrue_train, yhat_probs[:, 1])
        else:
            labels = np.arange(np.shape(yhat_probs)[1])
            train_roc_score = roc_auc_score(ytrue_train, yhat_probs, multi_class='ovo', labels=labels)

    valid_loss, yhat_valid, ytrue_valid, yhat_probs = test(valid_loader, model, args)
    valid_score = measure(ytrue_valid, yhat_valid)
    if task == 'classification':
        if output_channels == 2:
            yhat_probs = np.array(yhat_probs)
            valid_roc_score = roc_auc_score(ytrue_valid, yhat_probs[:, 1])
        else:
            valid_roc_score = roc_auc_score(ytrue_valid, yhat_probs, multi_class='ovo', labels=labels)

    test_loss, yhat_test, ytrue_test, yhat_probs = test(test_loader, model, args)
    test_score = measure(ytrue_test, yhat_test)
    if task == 'classification':
        if output_channels == 2:
            yhat_probs = np.array(yhat_probs)
            test_roc_score = roc_auc_score(ytrue_test, yhat_probs[:, 1])
        else:
            test_roc_score = roc_auc_score(ytrue_test, yhat_probs, multi_class='ovo', labels=labels)

    with open(result_file, 'a') as f:
        if task == 'regression':
            print(f"Final, Train RMSE: {np.sqrt(train_loss):.4f}, Train MAE: {train_score:.4f}, Valid RMSE: {np.sqrt(valid_loss):.4f}, Valid MAE: {valid_score:.4f}, Test RMSE: {np.sqrt(test_loss):.4f}, Test MAE: {test_score:.4f}", file=f) 
        elif task == 'classification':
            print(f"Final, Train loss: {train_loss:.4f}, Train acc: {train_score:.4f}, Train roc: {train_roc_score:.4f}, Valid loss: {valid_loss:.4f}, Valid acc: {valid_score:.4f}, Valid roc: {valid_roc_score:.4f}, Test loss: {test_loss:.4f}, Test acc: {test_score:.4f}, Test roc: {test_roc_score:.4f}", file=f) 
        
    
    train_loss, yhat_train, ytrue_train, yhat_probs = test(train_loader, best_model, args)
    train_score = measure(ytrue_train, yhat_train)
    if task == 'classification':
        if output_channels == 2:
            yhat_probs = np.array(yhat_probs)
            train_roc_score = roc_auc_score(ytrue_train, yhat_probs[:, 1])
        else:
            labels = np.arange(np.shape(yhat_probs)[1])
            train_roc_score = roc_auc_score(ytrue_train, yhat_probs, multi_class='ovo', labels=labels)

    valid_loss, yhat_valid, ytrue_valid, yhat_probs = test(valid_loader, best_model, args)
    valid_score = measure(ytrue_valid, yhat_valid)
    if task == 'classification':
        if output_channels == 2:
            yhat_probs = np.array(yhat_probs)
            valid_roc_score = roc_auc_score(ytrue_valid, yhat_probs[:, 1])
        else:
            valid_roc_score = roc_auc_score(ytrue_valid, yhat_probs, multi_class='ovo', labels=labels)

    test_loss, yhat_test, ytrue_test, yhat_probs = test(test_loader, best_model, args)
    test_score = measure(ytrue_test, yhat_test)
    if task == 'classification':
        if output_channels == 2:
            yhat_probs = np.array(yhat_probs)
            test_roc_score = roc_auc_score(ytrue_test, yhat_probs[:, 1])
        else:
            test_roc_score = roc_auc_score(ytrue_test, yhat_probs, multi_class='ovo', labels=labels)

    ##### DEBUG
    # with open("test_SGMP_probs.txt", 'w') as file:
    #     for i in yhat_probs:
    #         file.write(str(i) + '\n')

    with open(result_file, 'a') as f:
        if task == 'regression':
            print(f"Best Model, Train RMSE: {np.sqrt(train_loss):.4f}, Train MAE: {train_score:.4f}, Valid RMSE: {np.sqrt(valid_loss):.4f}, Valid MAE: {valid_score:.4f}, Test RMSE: {np.sqrt(test_loss):.4f}, Test MAE: {test_score:.4f}") 
            print(f"Best Model, Train RMSE: {np.sqrt(train_loss):.4f}, Train MAE: {train_score:.4f}, Valid RMSE: {np.sqrt(valid_loss):.4f}, Valid MAE: {valid_score:.4f}, Test RMSE: {np.sqrt(test_loss):.4f}, Test MAE: {test_score:.4f}", file=f) 
        elif task == 'classification':
            print(f"Best Model, Train loss: {train_loss:.4f}, Train acc: {train_score:.4f}, Train roc: {train_roc_score:.4f}, Valid loss: {valid_loss:.4f}, Valid acc: {valid_score:.4f}, Valid roc: {valid_roc_score:.4f}, Test loss: {test_loss:.4f}, Test acc: {test_score:.4f}, Test roc: {test_roc_score:.4f}") 
            print(f"Best Model, Train loss: {train_loss:.4f}, Train acc: {train_score:.4f}, Train roc: {train_roc_score:.4f}, Valid loss: {valid_loss:.4f}, Valid acc: {valid_score:.4f}, Valid roc: {valid_roc_score:.4f}, Test loss: {test_loss:.4f}, Test acc: {test_score:.4f}, Test roc: {test_roc_score:.4f}", file=f) 

    ##### DEBUG
    """
    with open('SGCN_cell_human_ytrue_test.txt', 'w') as file:
        for i in ytrue_test:
            file.write(str(i) + '\n')
    with open('SGCN_cell_human_yhat_test.txt', 'w') as file:
        for i in yhat_test:
            file.write(str(i) + '\n')
    test_conf_matrix = confusion_matrix(ytrue_test, yhat_test)
    with open('SGCN_cell_human_conf_matrix.txt', 'w') as file:
        for i in test_conf_matrix:
            file.write(str(i) + '\n')
    """

    
if __name__ == '__main__':
    args = get_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    torch.manual_seed(args.random_seed)  
    data = load_data(args)
    main(data, args)
    
    
