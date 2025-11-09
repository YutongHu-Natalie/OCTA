import os
import sys
import argparse
import requests
import torch
import numpy as np
import pickle as pkl

from torch_geometric.data import Data, DataLoader


# Create argument parser for file
parser = argparse.ArgumentParser()

# Add required parameter argument
parser.add_argument("--dataset", type=str, choices=['null', 'labeled_brain', 'labeled_full'], required=True, help="'null', 'labeled_brain', 'labeled_full' datasets")

# Parse command-line arguments
args = parser.parse_args()


# Brain region mapping
brain_region_mapping = {
    '["basal ganglia","substantia nigra"]': 0,
    '["brainstem","cochlear nucleus"]': 1,
    '["cerebellum","cerebellar cortex"]': 2,
    '["hippocampus","dentate gyrus"]': 3,
    '["main olfactory bulb"]': 4,
    '["neocortex","frontal"]': 5,
    '["neocortex","somatosensory","primary somatosensory"]': 6
}

# Full class list
full_class_mapping = {
    'CB_5xFAD_3mpos_F': 0,
    'CB_5xFAD_3mpos_M': 1,
    'CB_5xFAD_6mpos_F': 2,
    'CB_5xFAD_6mpos_M': 3,
    'CB_Adulthood_Control_F': 4,
    'CB_Adulthood_Control_M': 5,
    'CB_CKp25_1w_F': 6,
    'CB_CKp25_1w_M': 7,
    'CB_CKp25_2w_F': 8,
    'CB_CKp25_2w_M': 9,
    'CB_CKp25_6w_F': 10,
    'CB_CKp25_6w_M': 11,
    'CB_Development_P15_F': 12,
    'CB_Development_P15_M': 13,
    'CB_Development_P22_F': 14,
    'CB_Development_P22_M': 15,
    'CB_Development_P7_F': 16,
    'CB_Development_P7_M': 17,
    'CB_Ovariectomy': 18,
    'CN_5xFAD_3mpos_F': 19,
    'CN_5xFAD_3mpos_M': 20,
    'CN_5xFAD_6mpos_F': 21,
    'CN_5xFAD_6mpos_M': 22,
    'CN_Adulthood_Control_F': 23,
    'CN_Adulthood_Control_M': 24,
    'CN_CKp25_1w_F': 25,
    'CN_CKp25_1w_M': 26,
    'CN_CKp25_2w_F': 27,
    'CN_CKp25_2w_M': 28,
    'CN_CKp25_6w_F': 29,
    'CN_CKp25_6w_M': 30,
    'CN_Development_P15_F': 31,
    'CN_Development_P15_M': 32,
    'CN_Development_P22_F': 33,
    'CN_Development_P22_M': 34,
    'CN_Development_P7_F': 35,
    'CN_Development_P7_M': 36,
    'CN_Ovariectomy': 37,
    'DG_5xFAD_3mpos_F': 38,
    'DG_5xFAD_3mpos_M': 39,
    'DG_5xFAD_6mpos_F': 40,
    'DG_5xFAD_6mpos_M': 41,
    'DG_Adulthood_Control_F': 42,
    'DG_Adulthood_Control_M': 43,
    'DG_CKp25_1w_F': 44,
    'DG_CKp25_1w_M': 45,
    'DG_CKp25_2w_F': 46,
    'DG_CKp25_2w_M': 47,
    'DG_CKp25_6w_F': 48,
    'DG_CKp25_6w_M': 49,
    'DG_Development_P15_F': 50,
    'DG_Development_P15_M': 51,
    'DG_Development_P22_F': 52,
    'DG_Development_P22_M': 53,
    'DG_Development_P7_F': 54,
    'DG_Development_P7_M': 55,
    'DG_Ovariectomy': 56,
    'FC_5xFAD_3mpos_F': 57,
    'FC_5xFAD_3mpos_M': 58,
    'FC_5xFAD_6mpos_F': 59,
    'FC_5xFAD_6mpos_M': 60,
    'FC_Adulthood_Control_F': 61,
    'FC_Adulthood_Control_M': 62,
    'FC_CKp25_1w_F': 63,
    'FC_CKp25_1w_M': 64,
    'FC_CKp25_2w_F': 65,
    'FC_CKp25_2w_M': 66,
    'FC_CKp25_6w_F': 67,
    'FC_CKp25_6w_M': 68,
    'FC_Development_P15_F': 69,
    'FC_Development_P15_M': 70,
    'FC_Development_P22_F': 71,
    'FC_Development_P22_M': 72,
    'FC_Development_P7_F': 73,
    'FC_Development_P7_M': 74,
    'FC_Ovariectomy': 75,
    'OB_5xFAD_3mpos_F': 76,
    'OB_5xFAD_3mpos_M': 77,
    'OB_5xFAD_6mpos_F': 78,
    'OB_5xFAD_6mpos_M': 79,
    'OB_Adulthood_Control_F': 80,
    'OB_Adulthood_Control_M': 81,
    'OB_CKp25_1w_F': 82,
    'OB_CKp25_1w_M': 83,
    'OB_CKp25_2w_F': 84,
    'OB_CKp25_2w_M': 85,
    'OB_CKp25_6w_F': 86,
    'OB_CKp25_6w_M': 87,
    'OB_Development_P15_F': 88,
    'OB_Development_P15_M': 89,
    'OB_Development_P22_F': 90,
    'OB_Development_P22_M': 91,
    'OB_Development_P7_F': 92,
    'OB_Development_P7_M': 93,
    'OB_Ovariectomy': 94,
    'S1_1xKXA_F': 95,
    'S1_1xKXA_M': 96,
    'S1_2xKXA_F': 97,
    'S1_2xKXA_M': 98,
    'S1_3xKXA_F': 99,
    'S1_3xKXA_M': 100,
    'S1_5xFAD_3mpos_F': 101,
    'S1_5xFAD_3mpos_M': 102,
    'S1_5xFAD_6mpos_F': 103,
    'S1_5xFAD_6mpos_M': 104,
    'S1_Adulthood_Control_F': 105,
    'S1_Adulthood_Control_M': 106,
    'S1_CKp25_1w_F': 107,
    'S1_CKp25_1w_M': 108,
    'S1_CKp25_2w_F': 109,
    'S1_CKp25_2w_M': 110,
    'S1_CKp25_6w_F': 111,
    'S1_CKp25_6w_M': 112,
    'S1_Development_P15_F': 113,
    'S1_Development_P15_M': 114,
    'S1_Development_P22_F': 115,
    'S1_Development_P22_M': 116,
    'S1_Development_P7_F': 117,
    'S1_Development_P7_M': 118,
    'S1_Ovariectomy': 119,
    'S1_Recovery_1w_F': 120,
    'S1_Recovery_1w_M': 121,
    'S1_Recovery_2w_F': 122,
    'S1_Recovery_2w_M': 123,
    'S1_Recovery_3d_F': 124,
    'S1_Recovery_3d_M': 125,
    'SN_5xFAD_3mpos_F': 126,
    'SN_5xFAD_3mpos_M': 127,
    'SN_5xFAD_6mpos_F': 128,
    'SN_5xFAD_6mpos_M': 129,
    'SN_Adulthood_Control_F': 130,
    'SN_Adulthood_Control_M': 131,
    'SN_CKp25_1w_F': 132,
    'SN_CKp25_1w_M': 133,
    'SN_CKp25_2w_F': 134,
    'SN_CKp25_2w_M': 135,
    'SN_CKp25_6w_F': 136,
    'SN_CKp25_6w_M': 137,
    'SN_Development_P15_F': 138,
    'SN_Development_P15_M': 139,
    'SN_Development_P22_F': 140,
    'SN_Development_P22_M': 141,
    'SN_Development_P7_F': 142,
    'SN_Development_P7_M': 143,
    'SN_Ovariectomy': 144
}



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

def process_swc_files(directory):
    problem_data = []

    graph_list = []
    with os.scandir(directory) as data_folder:
        for file in data_folder:
            if file.name.endswith('.swc') and file.is_file():
                # Load file as numpy array
                full_file_name = os.path.join(directory, file.name)
                try:
                    neuron_arr = read_swc_file(full_file_name)
                # If error occurs, add file_name to list and keep going
                except:
                    print('read_swc_file() fail: ')
                    e = sys.exc_info()
                    print(e)
                    problem_data.append('read_swc_fail: ' + str(full_file_name))
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

                if args.dataset == 'labeled_brain':
                    response = requests.get('http://cng.gmu.edu:8080/api/neuron/name/' + file.name.split('.swc', 1)[0])

                    if not response.ok:
                        print('Response not ok: ' + str(file.name))

                    try:
                        # Brain region
                        split_brain = response.text.split("\"brain_region\":", 1)[1]
                        split_brain = split_brain.split("],\"", 1)[0]
                        split_brain += ']'
                    except:
                        print('Split not ok: ' + str(file.name))
                        continue

                    if split_brain in brain_region_mapping:
                        new_y[0] = brain_region_mapping[split_brain]
                    else:
                        print('Brain region not found: ' + str(file.name))
                        continue

                if args.dataset == 'labeled_full':
                    for key, value in full_class_mapping.items():
                        if key in file.name:
                            new_y[0] = value
                            break
                    else:
                        print('File name match not found: ' + str(file.name))
                        continue

                
                new_y = [new_y]

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


if args.dataset == 'null':
    pkl_name = 'siegert_null.pkl'
elif args.dataset == 'labeled_brain':
    pkl_name = 'siegert_labeled_brain.pkl'
elif args.dataset == 'labeled_full':
    pkl_name = 'siegert_labeled_full.pkl'

if os.path.exists(pkl_name):
    print("Siegert pickle already exists")
else:
    dataset = process_swc_files("data")
    with open(pkl_name, 'wb') as file:
        pkl.dump(dataset, file)