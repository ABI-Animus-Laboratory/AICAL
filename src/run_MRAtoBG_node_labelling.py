"""
Run brain vasculature labelling on a BIDS-formatted dataset using the FPNL algorithms.

This script labels graph networks stored in BIDS format, producing CoW key node labels
for each subject. Currently, the only available node labelling module is FPNL (and associated GNNART), but the design
allows for future expansion to support other models.

The script reads user-provided configuration parameters from a JSON config file, which must include the model
selection and paths to the model weights and Python environment for inference. Output segmentations are saved
to:
	<dataset_root_path>/derivatives/segmentations/

Usage:
	python run_segmentation_BIDS.py config.json

Configuration (JSON) requirements:
	{
		"dataset_root_path": "<path to BIDS/SPARC dataset root>",
		"do_ICA_init": "0"/"1", whether or not do perform ICA key node initialisation
		"do_FPNL": "0"/"1", whether or not do perform FPNL given an ICA key node initialisation
		"node_labelling_dist_thresholds_root": "<path to distance thresholds file>",
		"node_labelling_ICA_init_model": name of the ICA key node initialisation model
		"node_labelling_ICA_init_model_path": path to the ICA key node initialisation model checkpoint
	}

Inputs:
	- A BIDS-compliant dataset under the specified `dataset_root_path` containing vessel graphs
	- Configuration file as described above

Outputs:
	- Vascular node labelling lists stored in:
		<dataset_root_path>/derivatives/node_labelling/sub-*/

Notes:
	- This script must be run from within an environment compatible with the selected model.
	- For the GNNART ICA initialisation model, ensure you follow the steps at https://github.com/jshe690/GNN-ART-LABEL.

Author: Jiantao Shen
Date: 20.05.25
Version: 1.0.0
"""

import os
import time
import json
import sys
import shutil

# import utils
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(script_dir)
from utils_FPNL import *
from utils_GNNART import *

random.seed(0)

class NodeLabelling():
	def __init__(self, cfg, subj_ix, tag, module_start_time):

		self.cfg = cfg
		self.tag = tag
		self.subj_ix = subj_ix
		self.module_start_time = module_start_time

		self.dataset_structure = cfg['dataset_structure']
		self.dataset_root_path = cfg['dataset_root_path']
		self.gpu_num = cfg['gpu_selection']
		self.image_to_segment_suffix = cfg['image_to_segment_suffix']
		self.verbose = int(cfg['node_labelling_verbose'])

		self.nodelist_path = get_path_to_item(cfg, 'nodelist_path', tag=tag)
		self.edgelist_path = get_path_to_item(cfg, 'edgelist_path', tag=tag)
		self.node_labelling_ICA_init_output_path = get_path_to_item(cfg, 'node_labelling_ICA_init_output_path', tag=tag) 
		self.node_labelling_ICA_init_HR_visualisation_output_path = get_path_to_item(cfg, 'node_labelling_ICA_init_HR_visualisation_output_path', tag=tag)
		self.node_labelling_ICA_init_raw_visualisation_output_path = get_path_to_item(cfg, 'node_labelling_ICA_init_raw_visualisation_output_path', tag=tag)
		self.node_labelling_output_path = get_path_to_item(cfg, 'node_labelling_output_path', tag=tag)
		self.node_labelling_output_path_prunedGen1 = get_path_to_item(cfg, 'node_labelling_output_path_prunedGen1', tag=tag)
		self.node_labelling_visualisation_output_path = get_path_to_item(cfg, 'node_labelling_visualisation_output_path', tag=tag)

		self.do_ICA_init = int(cfg['do_ICA_init'])
		self.do_FPNL = int(cfg['do_FPNL'])
		self.node_labelling_ICA_init_model = cfg['node_labelling_ICA_init_model']
		self.node_labelling_ICA_init_model_path = cfg['node_labelling_ICA_init_model_path']
		self.node_labelling_dist_thresholds_root = cfg['node_labelling_dist_thresholds_root']
		self.tags_to_exclude = []
		self.do_output_HR_html = True
		self.do_output_raw_html = True

	def ICA_init(self):
		"""
		Load a vessel graph from its nodelist and edgelist in BIDS format, and run inference using a pre-trained GNNART
		model to obtain the ICA key node initialisation solution. Note that GNNART provides a full labelling, but only
		the ICA key node solutions are used.
		"""

		def load_nx_g():
			if self.tag in self.tags_to_exclude:
				print('\t\t Skipping', self.tag)
				return
			assert os.path.isfile(self.edgelist_path), f'edgelist.txt not found for {self.tag}'

			g_nx = nx.read_edgelist(self.edgelist_path, nodetype=int, data=True)
			g_nx = g_nx.to_directed()  # to_directed ensures that dgl.from_networkx will auto read in edge_attr
			g = dgl.from_networkx(g_nx, edge_attrs=['length', 'radius', 'length_euclid', 'tortuosity', 'torsion',
													'curvature'])  # try radius, torsion

			deleted_nodes = []
			with open(self.nodelist_path, 'r') as f:
				node_data = {'class': [], 'x': [], 'y': [], 'z': [], 'all_features': []}
				c = 0
				for line_raw in f.readlines():
					line = line_raw.rstrip().split(',')
					line = [float(v) for v in line]
					# if line[0] != c:
					#     gg=5
					# assert line[0] == c, 'nodelist file nodes must be in consecutive order from zero'
					c += 1
					node_id = line[0]
					if node_id not in deleted_nodes:
						# if self.split_scheme == 'midas':
						#     line[2] *= -1
						node_data['class'].append(0)
						node_data['x'].append(line[1])
						node_data['y'].append(line[2])
						node_data['z'].append(line[3])
						node_data['all_features'].append([line[iii] for iii in range(1, 4)])

			# standardise for GNNART
			x = np.array(node_data['x'])
			y = np.array(node_data['y'])
			z = np.array(node_data['z'])

			# Calculate mean and standard deviation
			mean_x = np.mean(x)
			mean_y = np.mean(y)
			mean_z = np.mean(z)

			# std_dev_x = np.std(x)
			# std_dev_y = np.std(y)
			# std_dev_z = np.std(z)

			# Standardize x, y, and z
			# standardized_x = (x - mean_x) / std_dev_x / 10
			# standardized_y = (y - mean_y) / std_dev_y / 10
			# standardized_z = (z - mean_z) / std_dev_z / 10
			# GC_offsets = [0, 0.1, 0.05] # GC guess
			# GC_offsets = [0.006142, 0.072918, 0.072772] # GC ixi TRAIN first 5
			# GC_offsets = [0.0051095	,0.0598915	,0.069809] # GC ixi TRAIN first 10
			GC_offsets = [0,0,0] # GC ixi TRAIN first 5
			standardized_x = (x - mean_x)/200 + GC_offsets[0]
			standardized_y = (y - mean_y)/200 + GC_offsets[1]
			standardized_z = (z - mean_z)/200 + GC_offsets[2]

			# Update node_data with standardized values
			node_data['x'] = standardized_x.tolist()
			node_data['y'] = standardized_y.tolist()
			node_data['z'] = standardized_z.tolist()
			node_data['z'] = standardized_z.tolist()
			node_data['all_features'] = np.column_stack((node_data['x'], node_data['y'], node_data['z']))

			for node_attr, vals in node_data.items():
				if node_attr == 'class':
					g.ndata[node_attr] = th.tensor(vals, dtype=int)
				else:
					g.ndata[node_attr] = th.tensor(vals, dtype=th.float)

			# normalise coordinates BY SUBJECT
			# g.ndata['all_features'] = g.ndata['all_features'] - g.ndata['all_features'].min(axis=0).values
			# g.ndata['all_features'] = g.ndata['all_features'] / g.ndata['all_features'].max(axis=0).values

			for edge_feat in ['length', 'radius']:
				edge_feat_norm = edge_feat + '_norm'
				if g.edata[edge_feat].min() < 0:
					g.edata[edge_feat_norm] = g.edata[edge_feat] + abs(g.edata[edge_feat].min())
					g.edata[edge_feat_norm] = g.edata[edge_feat_norm] / g.edata[edge_feat_norm].max()
				else:
					g.edata[edge_feat_norm] = g.edata[edge_feat] - g.edata[edge_feat].min()
					g.edata[edge_feat_norm] = g.edata[edge_feat_norm] / g.edata[edge_feat_norm].max()
			# for edge_feat in ['length']:
			#     g.edata[edge_feat_norm] = g.edata[edge_feat_norm] * 1

			g_nx = dgl.to_networkx(g, node_attrs=['class', 'all_features'],
								   edge_attrs=['length', 'radius', 'length_euclid', 'tortuosity', 'torsion',
											   'curvature', 'length_norm', 'radius_norm', ])
			g_nx_undirected = g_nx.to_undirected()

			# def break_cycle_with_smallest_radius(nx_g):
			#     print('BREAKING CYCLES')
			#     # Step 1: Convert MultiGraph to Graph to find cycles
			#     if isinstance(nx_g, nx.MultiGraph) or isinstance(nx_g, nx.MultiDiGraph):
			#         simple_graph = nx.Graph(nx_g)  # Convert to simple graph
			#     else:
			#         simple_graph = nx_g  # Already a simple graph
			#
			#     # Step 2: Find all cycles in the simple graph
			#     cycles = nx.cycle_basis(simple_graph)
			#
			#     for cycle in cycles:
			#         min_radius_edge = None
			#         min_radius = float('inf')
			#
			#         # Step 3: Identify the edge with the smallest radius in the original MultiGraph
			#         for i in range(len(cycle)):
			#             u = cycle[i]
			#             v = cycle[(i + 1) % len(cycle)]  # Wrap around the cycle
			#
			#             if nx_g.has_edge(u, v):
			#                 # MultiGraph can have multiple edges between the same nodes
			#                 # We iterate through all parallel edges and find the one with the smallest radius
			#                 for key in nx_g[u][v]:  # Access parallel edges
			#                     radius = nx_g[u][v][key].get('radius', float('inf'))  # Get radius attribute
			#                     if radius < min_radius:
			#                         min_radius = radius
			#                         min_radius_edge = (u, v, key)  # Store edge with the smallest radius
			#
			#         # Step 4: Remove the edge with the smallest radius
			#         if min_radius_edge:
			#             u, v, key = min_radius_edge
			#             nx_g.remove_edge(u, v, key)  # Remove the specific edge in MultiGraph
			#
			def find_cycles_in_multigraph(nx_g):
				def dfs_cycle_detector(v, visited, parent, cycle_stack, cycles):
					visited[v] = True
					cycle_stack.append(v)

					for u in nx_g[v]:
						if not visited[u]:
							if dfs_cycle_detector(u, visited, v, cycle_stack, cycles):
								return True
						elif parent != u and u in cycle_stack:
							# Found a cycle, extract it
							cycle_start_idx = cycle_stack.index(u)
							cycle = cycle_stack[cycle_start_idx:] + [u]
							cycles.append(cycle)

					cycle_stack.pop()
					return False

				cycles = []
				visited = {node: False for node in nx_g}

				for node in nx_g:
					if not visited[node]:
						cycle_stack = []
						dfs_cycle_detector(node, visited, None, cycle_stack, cycles)

				return cycles

			def break_cycle_with_smallest_radius(nx_g):
				if self.verbose:
					print('\t BREAKING CYCLES')

				while True:
					# Find cycles using custom cycle detection for MultiGraph
					cycles = find_cycles_in_multigraph(nx_g)
					if not cycles:
						break

					for cycle in cycles:
						min_radius_edge = None
						min_radius = float('inf')

						# Identify the edge with the smallest radius in the original MultiGraph
						for i in range(len(cycle)):
							u = cycle[i]
							v = cycle[(i + 1) % len(cycle)]  # Wrap around the cycle

							if nx_g.has_edge(u, v):
								# MultiGraph can have multiple edges between the same nodes
								# We iterate through all parallel edges and find the one with the smallest radius
								for key in nx_g[u][v]:  # Access parallel edges
									radius = nx_g[u][v][key].get('radius', float('inf'))  # Get radius attribute
									if radius < min_radius:
										min_radius = radius
										min_radius_edge = (u, v, key)  # Store edge with the smallest radius

						# Remove the edge with the smallest radius
						if min_radius_edge:
							u, v, key = min_radius_edge
							nx_g.remove_edge(u, v, key)  # Remove the specific edge in MultiGraph

				return nx_g

			def break_cycle_with_smallest_radius_deg2_fix(nx_g):
				if self.verbose:
					print('BREAKING CYCLES')

				# Store the maximum node index before we start adding artificial nodes
				max_node_index = max(nx_g.nodes()) if nx_g.nodes else 0
				artificial_node_counter = max_node_index + 1

				while True:
					# Find cycles using custom cycle detection for MultiGraph
					cycles = find_cycles_in_multigraph(nx_g)
					if not cycles:
						break

					for cycle in cycles:
						min_radius_edge = None
						min_radius = float('inf')

						# Identify the edge with the smallest radius in the original MultiGraph
						for i in range(len(cycle)):
							u = cycle[i]
							v = cycle[(i + 1) % len(cycle)]  # Wrap around the cycle

							if nx_g.has_edge(u, v):
								# MultiGraph can have multiple edges between the same nodes
								# We iterate through all parallel edges and find the one with the smallest radius
								for key in nx_g[u][v]:  # Access parallel edges
									radius = nx_g[u][v][key].get('radius', float('inf'))  # Get radius attribute
									if radius < min_radius:
										min_radius = radius
										min_radius_edge = (u, v, key)  # Store edge with the smallest radius

						# Remove the edge with the smallest radius
						if min_radius_edge:
							u, v, key = min_radius_edge
							edge_data = nx_g[u][v][key]  # Store the edge attributes before removing
							nx_g.remove_edge(u, v, key)  # Remove the specific edge in MultiGraph

							# Check if either node u or v has degree 2 after edge removal
							for node in [u, v]:
								if nx_g.degree[node] == 2:
									# Add an artificial node and connect it to the node with degree 2
									pos = nx_g.nodes[node]['all_features']
									artificial_pos = [pos[0], pos[1], pos[2] + 0.005]

									# Add the artificial node
									nx_g.add_node(artificial_node_counter)
									nx_g.nodes[artificial_node_counter]['all_features'] = artificial_pos
									nx_g.nodes[artificial_node_counter]['class'] = 0

									# Create an artificial edge between the node and the new artificial node
									nx_g.add_edge(node, artificial_node_counter, **edge_data)  # Inherit edge features

									# Increment artificial node counter for the next new node
									artificial_node_counter += 1

				return nx_g

			# only break cycles for pre-MVC GNN init!!! MVC must have broken all cycles.
			break_cycle_with_smallest_radius(g_nx_undirected)

			for n in g_nx_undirected.nodes:
				g_nx_undirected.nodes[n]['boitype'] = int(g_nx_undirected.nodes[n]['class'])
				g_nx_undirected.nodes[n]['pos'] = [float(g_nx_undirected.nodes[n]['all_features'][0]),
												   float(g_nx_undirected.nodes[n]['all_features'][1]),
												   float(g_nx_undirected.nodes[n]['all_features'][2])]
				g_nx_undirected.nodes[n]['swcid'] = n
				g_nx_undirected.nodes[n]['deg'] = g_nx_undirected.degree[n]
				del g_nx_undirected.nodes[n]['class']
				del g_nx_undirected.nodes[n]['all_features']

				# calcualte node radius
				neighbours = list(g_nx_undirected.neighbors(n))
				node_radius_norm = []
				for nb in neighbours:
					edges = g_nx_undirected.get_edge_data(n, nb)
					for edge_key in edges:
						edge = edges[edge_key]
						# node_radius_norm.append(float(edge['radius_norm']))
						node_radius_norm.append(float(edge['radius']))
				if node_radius_norm:
					g_nx_undirected.nodes[n]['rad'] = np.array(node_radius_norm).mean()
				else:
					g_nx_undirected.nodes[n]['rad'] = 1e-4

			def posMatch(n, points):
				bestmatchscore = 0
				bestid = -1
				for idx, pi in enumerate(points):
					cmatchscore = np.sum(n * pi)
					if cmatchscore > bestmatchscore:
						bestmatchscore = cmatchscore
						bestid = idx
				return bestid

			availdirs = [(0, 0, -1), (0, 0, 1)]
			for ag1 in range(-45, 46, 45):
				for ag2 in range(0, 360, 45):
					xi = np.cos(ag1 / 180 * np.pi) * np.sin(ag2 / 180 * np.pi)
					yi = np.cos(ag1 / 180 * np.pi) * np.cos(ag2 / 180 * np.pi)
					zi = np.sin(ag1 / 180 * np.pi)
					cdir = (xi, yi, zi)  # .norm()
					availdirs.append(cdir)

			# calcualte node direction
			for n in g_nx_undirected.nodes:
				neighbours = list(g_nx_undirected.neighbors(n))
				node_dir = np.zeros(len(availdirs))
				n_pos = np.array(g_nx_undirected.nodes[n]['pos'])
				for nb in neighbours:
					nb_pos = np.array(g_nx_undirected.nodes[nb]['pos'])
					nb_minus_n = nb_pos - n_pos
					nb_minus_n_norm = np.linalg.norm(nb_minus_n)
					if np.sum(nb_minus_n_norm) == 0:
						nb_minus_n_normed = np.array([0,0,0])
					else:
						nb_minus_n_normed = nb_minus_n / nb_minus_n_norm

					nb_dir_ix = posMatch(nb_minus_n_normed, availdirs)
					node_dir[nb_dir_ix] = 1

				g_nx_undirected.nodes[n]['dir'] = node_dir

			for edge in g_nx_undirected.edges:
				u = edge[0]
				v = edge[1]
				# g_nx_undirected.edges[edge]['rad'] = float(g_nx_undirected.edges[edge]['radius_norm'])#.numpy()  # !!! or mean rad of two nodes?
				g_nx_undirected.edges[edge]['rad'] = float(g_nx_undirected.edges[edge]['radius'])#.numpy()  # !!! or mean rad of two nodes?
				# g_nx_undirected.edges[edge]['dist'] = float(g_nx_undirected.edges[edge]['length_norm'])#.numpy()
				g_nx_undirected.edges[edge]['dist'] = float(g_nx_undirected.edges[edge]['length'])#.numpy()
				u_pos = np.array(g_nx_undirected.nodes[u]['pos'])
				v_pos = np.array(g_nx_undirected.nodes[v]['pos'])
				v_pos_minus_u_pos = v_pos - u_pos
				v_pos_minus_u_pos_norm = np.linalg.norm(v_pos_minus_u_pos)
				if np.sum(v_pos_minus_u_pos_norm) == 0:
					v_pos_minus_u_pos_normed = np.array([0,0,0])
				else:
					v_pos_minus_u_pos_normed = v_pos_minus_u_pos / v_pos_minus_u_pos_norm
				if v_pos_minus_u_pos_normed[2] < 0:
					v_pos_minus_u_pos_normed = -1 * v_pos_minus_u_pos_normed

				g_nx_undirected.edges[edge]['dir'] = v_pos_minus_u_pos_normed.tolist()
				g_nx_undirected.edges[edge]['vestype'] = np.zeros(25)  # !!!

				del g_nx_undirected.edges[edge]['id']
				del g_nx_undirected.edges[edge]['length']
				del g_nx_undirected.edges[edge]['radius']
				del g_nx_undirected.edges[edge]['length_euclid']
				del g_nx_undirected.edges[edge]['tortuosity']
				del g_nx_undirected.edges[edge]['torsion']
				del g_nx_undirected.edges[edge]['curvature']
				del g_nx_undirected.edges[edge]['length_norm']
				del g_nx_undirected.edges[edge]['radius_norm']

			return g_nx_undirected
		if self.verbose:
			print('\t *** Performing ICA key node initialisation ***')
		#####################################################################################
		# Load raw nx_g #####################################################################
		#####################################################################################
		nx_g = load_nx_g()

		#####################################################################################
		# Initialise inputs #################################################################
		#####################################################################################
		BOITYPENUM = 23
		VESTYPENUM = 25
		input_ph, target_ph = create_placeholders_single_graph(nx_g)
		input_ph, target_ph = make_all_runnable_in_session(input_ph, target_ph)
		feed_dict, raw_graphs = create_feed_dict_single_graph(nx_g, input_ph, target_ph)
		model = models.EncodeProcessDecode(edge_output_size=VESTYPENUM, node_output_size=BOITYPENUM)
		output_ops_ge = model(input_ph, 10)

		#####################################################################################
		# Load model and inference ##########################################################
		#####################################################################################
		st = time.time()
		sess = load_sess_and_model(self.node_labelling_ICA_init_model_path, do_verbose=self.verbose)
		targets, outputs = run_inference(sess, target_ph, output_ops_ge, feed_dict)
		# rawoutputs = get_rawoutputs(outputs) # direct output from GNN

		#####################################################################################
		# Hierarchical Refinement ###########################################################
		#####################################################################################
		fitacc = 0
		errfitacc = 0
		errlist = []
		graph = raw_graphs[0]
		target = targets[0]
		output = outputs[0]
		all_db = {'test':self.tag+'       '}
		HR_output = GNNART_Hierarchical_Refinement(graph, target, output,
												   all_db, 'test', [0], 0,
												   edgefromnode, nodeconnection,
												   center_node_prob_thres,
												   errlist, fitacc, errfitacc, BOITYPENUM, VESTYPENUM)

		#####################################################################################
		# Prepare outputs ###################################################################
		#####################################################################################
		pred = [np.argmax(ni) for ni in HR_output["nodes"]]
		rawprednodes = np.argmax(output[-1]["nodes"], axis=1)
		rawpred = list(rawprednodes)

		edges, lookup_coord_class, node_labelling_predictions, MCA_nodes_pos_attributes_hr, ACA_nodes_pos_attributes_hr, \
		PComA_anterior_nodes_pos_attributes_hr, PComA_posterior_nodes_pos_attributes_hr, BAT_node_pos_attributes_hr, \
		BAB_node_pos_attributes_hr, ICA_nodes_pos_attributes_hr, MCA_nodes_pos_attributes_raw, ACA_nodes_pos_attributes_raw, \
		PComA_anterior_nodes_pos_attributes_raw, PComA_posterior_nodes_pos_attributes_raw, BAT_node_pos_attributes_raw, \
		BAB_node_pos_attributes_raw, ICA_nodes_pos_attributes_raw = prepare_GNNART_outputs(nx_g, pred, rawpred, self.tag,
																						   do_ixi=True)

		#####################################################################################
		# Outputting ########################################################################
		#####################################################################################
		initialise_node_labelling_output_file(self.node_labelling_ICA_init_output_path, True)
		output_curr_node_labelling_predictions(self.tag, None, node_labelling_predictions,
											   self.node_labelling_ICA_init_output_path, True, do_tag_only=True)

		html_path_hr = self.node_labelling_ICA_init_HR_visualisation_output_path
		if self.do_output_HR_html:
			save_network_graph_GNNART(edges, 23, lookup_coord_class,
							   html_path_hr, self.tag, fiducial_nodes=[],
							   fiducial_nodes2=[],
							   fiducial_nodes3=[],
							   MCA=MCA_nodes_pos_attributes_hr, ACA=ACA_nodes_pos_attributes_hr,
							   PComA_anterior=PComA_anterior_nodes_pos_attributes_hr,
							   PComA_posterior=PComA_posterior_nodes_pos_attributes_hr,
							   BAT_node=BAT_node_pos_attributes_hr, BAB_node=BAB_node_pos_attributes_hr,
							   ICA=ICA_nodes_pos_attributes_hr
							   )

		html_path_raw = self.node_labelling_ICA_init_raw_visualisation_output_path
		if self.do_output_raw_html:
			save_network_graph_GNNART(edges, 23, lookup_coord_class,
							   html_path_raw, self.tag, fiducial_nodes=[],
							   fiducial_nodes2=[],
							   fiducial_nodes3=[],
							   MCA=MCA_nodes_pos_attributes_raw, ACA=ACA_nodes_pos_attributes_raw,
							   PComA_anterior=PComA_anterior_nodes_pos_attributes_raw,
							   PComA_posterior=PComA_posterior_nodes_pos_attributes_raw,
							   BAT_node=BAT_node_pos_attributes_raw, BAB_node=BAB_node_pos_attributes_raw,
							   ICA=ICA_nodes_pos_attributes_raw
							   )

		return sess

	def FPNL(self):
		"""
		Load a vessel graph from its nodelist and edgelist in BIDS format along with its ICA key node initialisation,
		and run the FPNL algorithms to label the remaining CoW key nodes.
		"""

		def load_ICA_labels(node_labelling_predictions):
			with open(self.node_labelling_ICA_init_output_path, 'r') as f:
				csv_lines = f.readlines()
				assert len(csv_lines) == 2, 'xxx'
				assert 'Name' in csv_lines[0], 'xxx'
				assert 'ICAL' in csv_lines[0], 'xxx'
				assert 'ICAR' in csv_lines[0], 'xxx'
				line_raw =  csv_lines[1]
				line = line_raw.rstrip()
				line = re.split(r'[,.]', line)
				tag = line[0].split('_')[0]
				assert tag == self.tag, f'{line}, {tag}, {self.tag}'
				if line[1] != '':
					node_labelling_predictions[1] = int(line[1])
				else:
					raise ValueError(
						f'{tag} is missing the ICAL ground truth label in ICA init! Can not be processed')
				if line[2] != '':
					node_labelling_predictions[2] = int(line[2])
				else:
					raise ValueError(
						f'{tag} is missing the ICAR ground truth label in ICA init! Can not be processed')
			return node_labelling_predictions
		def load_nx_g(node_labelling_predictions):
			nx_g = nx.read_edgelist(self.edgelist_path, nodetype=int, data=True)
			with open(self.nodelist_path, 'r') as f:
				node_count = 0
				for line_raw in f.readlines():
					line = line_raw.rstrip().split(',')
					line = [float(v) for v in line]

					node_id = int(line[0])
					assert node_id == node_count, 'nodelist file nodes must be in consecutive order from zero'
					nx_g.nodes[node_id]['pos'] = [line[1], line[2], line[3]]
					nx_g.nodes[node_id]['boitype'] = 0

					# populate ICA init into nx_g
					if node_id == node_labelling_predictions[1]:
						nx_g.nodes[node_id]['boitype'] = 1
					elif node_id == node_labelling_predictions[2]:
						nx_g.nodes[node_id]['boitype'] = 2
					node_count += 1

			for u, v, data in nx_g.edges(data=True):
				data['dist'] = data.pop('length', None)
			for u, v, data in nx_g.edges(data=True):
				data['rad'] = data.pop('radius', None)

			# Set min z to 0
			z_coordinates = [data['pos'][2] for _, data in nx_g.nodes(data=True)]
			min_z = min(z_coordinates)
			for node, data in nx_g.nodes(data=True):
				x, y, z = data['pos']
				data['pos'] = [x, y, z - min_z]

			return nx_g

		do_output_node_labelling_predictions = True
		do_save_network_graph = True
		html_type = 'detailed'
		do_verbose_progress = self.verbose
		do_verbose_errors = self.verbose
		do_suppress_skips = False
		cohort = 'Any'
		train_or_test = ''
		xval_split = ''
		c = 0   

		if self.verbose:
			# print('\n')
			print('\t *** Performing FPNL CoW key node labelling ***')

		node_labelling_predictions = {i:None for i in range (1,13)}    # node labelling predictions saved here for outputting
		node_labelling_predictions = load_ICA_labels(node_labelling_predictions)
		nx_g = load_nx_g(node_labelling_predictions)
		nx_g, artificial_leaves, artificial_leaves2, artificial_bridges = add_artificials(nx_g)   # add artificial leaves or bridges to test robustness
		dist_thresholds_dict = load_ACA_PComA_dist_thresholds(self.node_labelling_dist_thresholds_root, cohort, xval_split, do_verbose_progress)
		params_dict = load_params()

		PCAL_edge_eix, PCAR_edge_eix = None, None
		while True: # decide to use this loop to break out of at the "if do_continue:" lines
			################################################################################
			# ICA ##########################################################################
			################################################################################
			ICAL_candidates, do_continue, c = get_ICA_candidates(nx_g, cohort, c, do_suppress_skips, 1, do_verbose_errors)
			if do_continue: raise ValueError('xxx')
			ICAR_candidates, do_continue, c = get_ICA_candidates(nx_g, cohort, c, do_suppress_skips, 2, do_verbose_errors)
			if do_continue: raise ValueError('xxx')
			ICAL_node, ICAR_node = get_ICA_nodes(ICAL_candidates, ICAR_candidates, cohort, train_or_test, c, do_verbose_errors)
			ICAL_pos, ICAR_pos, centroid_ICAL_ICAR, centroid_glob, centroid_ICAL_ICAR_at_centroid_glob_zpos = get_ICA_node_params(nx_g, ICAL_node, ICAR_node)
			node_labelling_predictions[1] = ICAL_node
			node_labelling_predictions[2] = ICAR_node

			################################################################################
			# MCA ##########################################################################
			################################################################################
			MCAL_landmark, MCAR_landmark, MCA_landmark_pos_attributes = find_MCA_landmarks(nx_g, ICAL_node, ICAR_node, ICAL_pos, ICAR_pos)
			SP_ICAL_MCAL_landmark = find_shortest_path(nx_g, ICAL_node, MCAL_landmark)
			SP_ICAR_MCAR_landmark = find_shortest_path(nx_g, ICAR_node, MCAR_landmark)

			ACA_cone_nodes, ACA_cone_nodes_pos_attributes, ACA_cone_base, ACA_cone_radius = find_ACA_cone_nodes(nx_g, ICAL_pos, ICAR_pos, centroid_ICAL_ICAR, centroid_ICAL_ICAR_at_centroid_glob_zpos, params_dict['cone_radius_multiplier'])
			exists_valid_ACA_cone_node = at_least_one_ACA_cone_node_reachable_from_MCA_landmarks(nx_g, ACA_cone_nodes, MCAL_landmark, MCAR_landmark)
			ACA_cone_nodes, ACA_cone_nodes_pos_attributes, ACA_cone_base, ACA_cone_radius, do_continue, c = check_and_update_ACA_cone_nodes(nx_g, ACA_cone_nodes, ACA_cone_nodes_pos_attributes, ACA_cone_base, ACA_cone_radius, params_dict['cone_radius_multiplier'], exists_valid_ACA_cone_node, do_verbose_errors, c, self.tag, node_labelling_predictions, None, False, do_suppress_skips, ICAL_pos, ICAR_pos, centroid_ICAL_ICAR, centroid_ICAL_ICAR_at_centroid_glob_zpos, MCAL_landmark, MCAR_landmark)
			if do_continue: break
			MCAL_node, MCAR_node, contralateral_MCA_phantom_pos, do_continue, c = find_MCA_nodes(nx_g, ICAL_node, ICAR_node, ACA_cone_nodes, MCAL_landmark, MCAR_landmark, centroid_ICAL_ICAR, do_verbose_errors, self.tag, c, node_labelling_predictions, None, False, do_suppress_skips)
			if do_continue: break
			MCAL_node, MCAR_node, MCA_cycle_nodes = adjust_for_MCA_cycles(nx_g, MCAL_node, MCAR_node, do_verbose_errors)
			MCAL_pos, MCAR_pos, MCA_nodes_pos_attributes = get_MCA_node_params(nx_g, MCAL_node, MCAR_node)
			node_labelling_predictions[3] = MCAL_node
			node_labelling_predictions[4] = MCAR_node

			################################################################################
			# ACA ##########################################################################
			################################################################################
			nx_g_ACA = gen_nx_g_ACA(nx_g, SP_ICAL_MCAL_landmark, SP_ICAR_MCAR_landmark, MCAL_node, MCAR_node)
			ACAL_node, ACAR_node, SP_MCAL_MCAR = find_ACA_nodes(nx_g, nx_g_ACA, ACA_cone_nodes, MCAL_node, MCAR_node, MCAL_pos, MCAR_pos, dist_thresholds_dict, ICAL_pos, ICAR_pos, SP_ICAL_MCAL_landmark, SP_ICAR_MCAR_landmark, do_verbose_errors)
			node_labelling_predictions[5] = ACAL_node
			node_labelling_predictions[6] = ACAR_node
			# calc_ACA_segment_dists_here()
			_, _, ACA_nodes_pos_attributes = get_ACA_node_params(nx_g, ACAL_node, ACAR_node)

			################################################################################
			# PComA ROUND A ################################################################
			################################################################################
			BAB_candidates, BAB_candidates_pos_attributes, BAB_cylinder_params = find_BAB_candidates(nx_g, params_dict, ICAL_node, ICAL_pos, ICAR_node, ICAR_pos, MCAL_node, MCAR_node, MCA_cycle_nodes, ACAL_node, ACAR_node, MCAL_landmark, MCAR_landmark, SP_ICAL_MCAL_landmark, SP_ICAR_MCAR_landmark, do_verbose_errors)
			nx_g_PComAL = gen_nx_g_PComA(nx_g, [ACAL_node, ACAR_node, MCAR_node])     # Slice the network to break contralateral anterior circulation for PComA SP search
			PComAL_anterior_node, PComAL_posterior_node = find_PComA_nodes(nx_g_PComAL, ICAL_node, MCAL_landmark, BAB_candidates, params_dict, do_verbose_errors)
			nx_g_PComAR = gen_nx_g_PComA(nx_g, [ACAL_node, ACAR_node, MCAL_node])     # Slice the network to break contralateral anterior circulation for PComA SP search
			PComAR_anterior_node, PComAR_posterior_node = find_PComA_nodes(nx_g_PComAR, ICAR_node, MCAR_landmark, BAB_candidates, params_dict, do_verbose_errors)
			calc_PComA_segment_dists_here()

			################################################################################
			# PComA ROUND B ################################################################
			################################################################################
			# Actually, for fetal PComA situations, the ICA-rooted PCAs still have a PComa anterior node to label, that probably wasnt labelled during PComA Round A!
			# So need a second set of logic targeting all L/R sides where PComA ant was NOT yet found.
			# Instead of using BAB candidate for divergence node finding, use rerarmost PCA landmarks.
			# PCA landmarks used as a backup for find PComA anteriors in case of fetal pcoma, and also later for PCA anchor localisation. Ellipsoids to envelope PCA subtrees.
			PCAL_landmarks, PCAR_landmarks, PCAL_landmarks_pos_attributes, PCAR_landmarks_pos_attributes, PCAL_landmarks_sphere, PCAR_landmarks_sphere, PCAL_landmarks_ellipsoid, PCAR_landmarks_ellipsoid = find_PCA_landmarks(nx_g, params_dict, centroid_ICAL_ICAR, ICAL_pos, ICAR_pos, MCAL_pos, MCAR_pos, do_verbose_errors)
			if PComAL_anterior_node == None: # XAXA done
				PComAL_anterior_node, PComAL_posterior_node = find_PComA_nodes_round_B(nx_g_PComAL, ICAL_node, MCAL_node, MCAL_landmark, PCAL_landmarks, params_dict, dist_thresholds_dict['PComAL_dist_thresh_lower'], dist_thresholds_dict['PComAL_dist_thresh_upper'], do_verbose_errors)
			if PComAR_anterior_node == None:
				PComAR_anterior_node, PComAR_posterior_node = find_PComA_nodes_round_B(nx_g_PComAR, ICAR_node, MCAR_node, MCAR_landmark, PCAR_landmarks, params_dict, dist_thresholds_dict['PComAR_dist_thresh_lower'], dist_thresholds_dict['PComAR_dist_thresh_upper'], do_verbose_errors) # JJJJJ
			PComA_anterior_nodes_pos_attributes, PComA_posterior_nodes_pos_attributes = get_PComA_node_params(nx_g, PComAL_anterior_node, PComAR_anterior_node, PComAL_posterior_node, PComAR_posterior_node)
			node_labelling_predictions[9] = PComAL_posterior_node
			node_labelling_predictions[10] = PComAR_posterior_node
			node_labelling_predictions[11] = PComAL_anterior_node
			node_labelling_predictions[12] = PComAR_anterior_node
			# By this point, any PComAs have been found with great certainty. Proceed to start chopping PComAs to fully isolate posterior circulation for BAT/BAB, and PCA calculation

			################################################################################
			# BAT ROUND A B C ##############################################################
			################################################################################
			BAT_node, BAT_node_pos_attributes, PCAL_anchor, PCAR_anchor, PCAL_anchor_pos_attributes, PCAR_anchor_pos_attributes, chosen_BAB_candidate, BAB_candidates_connected_to_PCA_landmarks, nx_g_BAT \
				= find_BAT_round_A(nx_g, BAB_candidates, PCAL_landmarks, PCAR_landmarks, centroid_ICAL_ICAR, MCAL_pos, MCAR_pos, PComAL_anterior_node, PComAR_anterior_node, do_verbose_errors)
			BAT_node, BAT_node_pos_attributes = find_BAT_round_B(nx_g, nx_g_BAT, PComAL_posterior_node, PComAR_posterior_node, BAT_node, BAT_node_pos_attributes)
			BAT_node, BAT_node_pos_attributes = find_BAT_round_C(nx_g, nx_g_BAT, chosen_BAB_candidate, MCAL_node, MCAR_node, ACAL_node, ACAR_node, PComAL_anterior_node, PComAR_anterior_node, PComAL_posterior_node, PComAR_posterior_node,
								 PCAL_landmarks, PCAR_landmarks, BAB_candidates, BAB_candidates_connected_to_PCA_landmarks, MCAL_pos, MCAR_pos, BAT_node, BAT_node_pos_attributes, do_verbose_errors)
			node_labelling_predictions[8] = BAT_node

			################################################################################
			# BAB ##########################################################################
			################################################################################
			BAB_node, BAB_node_pos_attributes = find_BAB_node(nx_g_BAT, BAB_candidates, BAT_node, centroid_ICAL_ICAR, do_verbose_errors)
			node_labelling_predictions[7] = BAB_node

			################################################################################
			# PCAL/R edge ##################################################################
			################################################################################
			PCAL_edge, PCAL_edge_eix, PCAL_edge_pos, PCAL_edge_landmark_pos, PCAR_edge, PCAR_edge_eix, PCAR_edge_pos, PCAR_edge_landmark_pos = find_PCA_edges(nx_g, PComAL_posterior_node, PComAR_posterior_node, BAT_node)

			################################################################################
			# Sanity Checks ################################################################
			################################################################################
			if do_verbose_errors:
				sanity_check_BAT(nx_g, BAT_node, BAB_node, PComAL_anterior_node, PComAR_anterior_node, PComAL_posterior_node, PComAR_posterior_node, MCAL_pos, MCAR_pos, BAT_node_pos_attributes, params_dict, do_verbose_errors)
				sanity_check_ACA(nx_g, nx_g_ACA, dist_thresholds_dict, ICAL_node, ICAR_node, MCAL_node, MCAR_node, ACAL_node, ACAR_node, BAT_node, PComAL_anterior_node, PComAR_anterior_node, PComAL_posterior_node, PComAR_posterior_node)
				sanity_check_PCA(nx_g, nx_g_BAT, PComAL_posterior_node, PComAR_posterior_node, BAT_node, PCAL_edge, PCAR_edge, PCAL_landmarks, PCAR_landmarks, BAB_candidates)
				sanity_check_MCA(nx_g, ICAL_node, ICAR_node, MCAL_node, MCAR_node, ACA_cone_nodes, MCAL_landmark, MCAR_landmark, PCAL_landmarks, PCAR_landmarks, PComAL_anterior_node, PComAR_anterior_node)
				sanity_check_PComA(nx_g, nx_g_PComAL, nx_g_PComAR, dist_thresholds_dict, PComAL_anterior_node, PComAL_posterior_node, PComAR_anterior_node, PComAR_posterior_node, BAT_node, BAB_node, MCAL_node, MCAR_node, ICAL_node, ICAR_node, ACAL_node, ACAR_node, do_verbose_errors)

			break
			
		################################################################################
		# Outputting ###################################################################
		################################################################################
		def output_node_labelling(node_labelling_predictions, PCAL_edge_eix, PCAR_edge_eix, save_path):
			data = f"ICAL,{'' if node_labelling_predictions[1] == None else node_labelling_predictions[1]}\n" \
				f"ICAR,{'' if node_labelling_predictions[2] == None else node_labelling_predictions[2]}\n" \
				f"MCAL,{'' if node_labelling_predictions[3] == None else node_labelling_predictions[3]}\n" \
				f"MCAR,{'' if node_labelling_predictions[4] == None else node_labelling_predictions[4]}\n" \
				f"ACAL,{'' if node_labelling_predictions[5] == None else node_labelling_predictions[5]}\n" \
				f"ACAR,{'' if node_labelling_predictions[6] == None else node_labelling_predictions[6]}\n" \
				f"BA_BASE,{'' if node_labelling_predictions[7] == None else node_labelling_predictions[7]}\n" \
				f"BA_TOP,{'' if node_labelling_predictions[8] == None else node_labelling_predictions[8]}\n" \
				f"PCOML_PCAL,{'' if node_labelling_predictions[9] == None else node_labelling_predictions[9]}\n" \
				f"PCOML_ICAL,{'' if node_labelling_predictions[11] == None else node_labelling_predictions[11]}\n" \
				f"PCOMR_PCAR,{'' if node_labelling_predictions[10] == None else node_labelling_predictions[10]}\n" \
				f"PCOMR_ICAR,{'' if node_labelling_predictions[12] == None else node_labelling_predictions[12]}\n" \
				f"PCAL_eix,{'' if PCAL_edge_eix == None else PCAL_edge_eix}\n" \
				f"PCAR_eix,{'' if PCAR_edge_eix == None else PCAR_edge_eix}\n" \
				"BAL_eix,\n" \
				"BAR_eix,\n" \
				"ACA_split_edges,[[]]\n" \
				"delete_edges,[]\n" \
				"delete_edges_without_adding_phantoms,[]\n" \
				"split_edges,[[]]\n" \
				"split_edges_double,[[]]\n" \
				"detach_edges,[[]]\n" \
				"add_PCOML,[[]]\n" \
				"add_PCOMR,[[]]\n" \
				"add_BAL,[[]]\n" \
				"add_BAR,[[]]\n" \
				"adjust_radius_eix_eix,[[]]"
			with open(save_path, 'w') as file:
				file.write(data)
		def output_node_labelling_visualisation(nx_g, node_labelling_predictions):
			for knix, kn_id in node_labelling_predictions.items():
				if kn_id != None:
					nx_g.nodes[kn_id]['boitype'] = knix
			edges, lookup_coord_class = generate_plotting_data(nx_g)
			new_tag = self.tag
			html_path = self.node_labelling_visualisation_output_path
			save_network_graph_FPNL(edges, 23, lookup_coord_class,
							   html_path, new_tag, html_type, fiducial_nodes=[centroid_glob],
							   fiducial_nodes2=[centroid_ICAL_ICAR, centroid_ICAL_ICAR_at_centroid_glob_zpos, ACA_cone_base, PCAL_edge_landmark_pos, PCAR_edge_landmark_pos],
							   fiducial_nodes3=BAB_candidates_pos_attributes + ACA_cone_nodes_pos_attributes, MCA_landmarks=MCA_landmark_pos_attributes, MCA=MCA_nodes_pos_attributes,
							   ACA=ACA_nodes_pos_attributes, PComA_anterior = PComA_anterior_nodes_pos_attributes, PComA_posterior = PComA_posterior_nodes_pos_attributes,
							   PCAL_landmarks=PCAL_landmarks_pos_attributes, PCAR_landmarks=PCAR_landmarks_pos_attributes,
							   PCAL_landmarks_sphere=PCAL_landmarks_sphere, PCAR_landmarks_sphere=PCAR_landmarks_sphere,
							   PCAL_landmarks_ellipsoid=PCAL_landmarks_ellipsoid, PCAR_landmarks_ellipsoid=PCAR_landmarks_ellipsoid,
							   PCAL_anchor=PCAL_anchor_pos_attributes, PCAR_anchor=PCAR_anchor_pos_attributes,
							   BAT_node=BAT_node_pos_attributes, BAB_node=BAB_node_pos_attributes,
							   BAB_cylinder=BAB_cylinder_params,
							   ACA_cone=[centroid_ICAL_ICAR_at_centroid_glob_zpos, ACA_cone_base, ACA_cone_radius],
							   MCA_phantom=[contralateral_MCA_phantom_pos], PCAL_edge=PCAL_edge_pos, PCAR_edge=PCAR_edge_pos,
							   PCAL_edge_landmark=PCAL_edge_landmark_pos, PCAR_edge_landmark=PCAR_edge_landmark_pos,
							   ICA=[ICAL_pos, ICAR_pos], artificial_leaves=artificial_leaves, artificial_leaves2=artificial_leaves2, artificial_bridges=artificial_bridges)

		output_node_labelling(node_labelling_predictions, PCAL_edge_eix, PCAR_edge_eix, self.node_labelling_output_path)
		output_node_labelling(node_labelling_predictions, PCAL_edge_eix, PCAR_edge_eix, self.node_labelling_output_path_prunedGen1)
		output_node_labelling_visualisation(nx_g, node_labelling_predictions)

	def run(self):
		if self.dataset_structure == 'BIDS':
			node_labelling_folder = get_path_to_item(self.cfg, 'node_labelling_folder', tag=self.tag)
			node_labelling_subj_folder = get_path_to_item(self.cfg, 'node_labelling_subj_folder', tag=self.tag)
			make_directory(node_labelling_folder, 'Node labelling')
			make_directory(node_labelling_subj_folder, 'Node labelling for subj')
		elif self.dataset_structure == 'SPARC':
			node_labelling_subj_folder = get_path_to_item(self.cfg, 'node_labelling_subj_folder', tag=self.tag)
			make_directory(node_labelling_subj_folder, 'Node Labelling for subj')

		if self.do_ICA_init:
			self.ICA_init()
		if self.do_FPNL:
			self.FPNL()

def get_path_to_item(cfg, item, **kwargs):
	# item adopts BIDS naming convention

	dataset_structure = cfg["dataset_structure"]
	dataset_root_path = cfg["dataset_root_path"]
	image_to_segment_suffix = cfg['image_to_segment_suffix']

	if dataset_structure == 'BIDS':
		if item == 'image_to_segment_file':
			a_path = os.path.join(dataset_root_path, 'rawdata', kwargs['tag'], "anat", f"{kwargs['tag']}_{image_to_segment_suffix}.nii.gz")
		elif item == 'node_labelling_folder':
			a_path = os.path.join(dataset_root_path, 'derivatives', 'MRAtoBG', 'node_labelling')
		elif item == 'node_labelling_subj_folder':
			a_path = os.path.join(dataset_root_path, 'derivatives', 'MRAtoBG', 'node_labelling', kwargs['tag'])
		elif item == 'nodelist_path':
			a_path = os.path.join(dataset_root_path, 'derivatives', 'MRAtoBG', 'node_labelling', kwargs['tag'], f"{kwargs['tag']}_{image_to_segment_suffix}_nodelist.txt")
		elif item == 'edgelist_path':
			a_path = os.path.join(dataset_root_path, 'derivatives', 'MRAtoBG', 'node_labelling', kwargs['tag'], f"{kwargs['tag']}_{image_to_segment_suffix}_edgelist.txt")
		elif item == 'node_labelling_ICA_init_output_path':
			a_path = os.path.join(dataset_root_path, 'derivatives', 'MRAtoBG', 'node_labelling', kwargs['tag'], f"{kwargs['tag']}_{image_to_segment_suffix}_node_labelling_ICA_init.csv")
		elif item == 'node_labelling_ICA_init_HR_visualisation_output_path':
			a_path = os.path.join(dataset_root_path, 'derivatives', 'MRAtoBG', 'node_labelling', kwargs['tag'], f"{kwargs['tag']}_{image_to_segment_suffix}_node_labelling_ICA_init_HR.html")
		elif item == 'node_labelling_ICA_init_raw_visualisation_output_path':
			a_path = os.path.join(dataset_root_path, 'derivatives', 'MRAtoBG', 'node_labelling', kwargs['tag'], f"{kwargs['tag']}_{image_to_segment_suffix}_node_labelling_ICA_init_raw.html")
		elif item == 'node_labelling_output_path':
			a_path = os.path.join(dataset_root_path, 'derivatives', 'MRAtoBG', 'node_labelling', kwargs['tag'], f"{kwargs['tag']}_{image_to_segment_suffix}_node_labelling.txt")
		elif item == 'node_labelling_output_path_prunedGen1':
			a_path = os.path.join(dataset_root_path, 'derivatives', 'MRAtoBG', 'node_labelling', kwargs['tag'], f"{kwargs['tag']}_{image_to_segment_suffix}_node_labelling_prunedGen1.txt")
		elif item == 'node_labelling_visualisation_output_path':
			a_path = os.path.join(dataset_root_path, 'derivatives', 'MRAtoBG', 'node_labelling', kwargs['tag'], f"{kwargs['tag']}_{image_to_segment_suffix}_node_labelling.html")

	elif dataset_structure == 'SPARC':
		if item == 'image_to_segment_file':
			a_path = os.path.join(dataset_root_path, 'primary', kwargs['tag'], "sam-anat", f"{kwargs['tag']}_{image_to_segment_suffix}.nii.gz")
		elif item == 'node_labelling_folder':
			assert False, f'Invalid dataset_structure / item combination: {dataset_structure} / {item}'
		elif item == 'node_labelling_subj_folder':
			a_path = os.path.join(dataset_root_path, 'derivative', kwargs['tag'], 'MRAtoBG', 'node_labelling')
		elif item == 'nodelist_path':
			a_path = os.path.join(dataset_root_path, 'derivative', kwargs['tag'], 'MRAtoBG', 'node_labelling', f"{kwargs['tag']}_{image_to_segment_suffix}_nodelist.txt")
		elif item == 'edgelist_path':
			a_path = os.path.join(dataset_root_path, 'derivative', kwargs['tag'], 'MRAtoBG', 'node_labelling', f"{kwargs['tag']}_{image_to_segment_suffix}_edgelist.txt")
		elif item == 'node_labelling_ICA_init_output_path':
			a_path = os.path.join(dataset_root_path, 'derivative', kwargs['tag'], 'MRAtoBG', 'node_labelling', f"{kwargs['tag']}_{image_to_segment_suffix}_node_labelling_ICA_init.csv")
		elif item == 'node_labelling_ICA_init_HR_visualisation_output_path':
			a_path = os.path.join(dataset_root_path, 'derivative', kwargs['tag'], 'MRAtoBG', 'node_labelling', f"{kwargs['tag']}_{image_to_segment_suffix}_node_labelling_ICA_init_HR.html")
		elif item == 'node_labelling_ICA_init_raw_visualisation_output_path':
			a_path = os.path.join(dataset_root_path, 'derivative', kwargs['tag'], 'MRAtoBG', 'node_labelling', f"{kwargs['tag']}_{image_to_segment_suffix}_node_labelling_ICA_init_raw.html")
		elif item == 'node_labelling_output_path':
			a_path = os.path.join(dataset_root_path, 'derivative', kwargs['tag'], 'MRAtoBG', 'node_labelling', f"{kwargs['tag']}_{image_to_segment_suffix}_node_labelling.txt")
		elif item == 'node_labelling_output_path_prunedGen1':
			a_path = os.path.join(dataset_root_path, 'derivative', kwargs['tag'], 'MRAtoBG', 'node_labelling', f"{kwargs['tag']}_{image_to_segment_suffix}_node_labelling_prunedGen1.txt")
		elif item == 'node_labelling_visualisation_output_path':
			a_path = os.path.join(dataset_root_path, 'derivative', kwargs['tag'], 'MRAtoBG', 'node_labelling', f"{kwargs['tag']}_{image_to_segment_suffix}_node_labelling.html")

	else:
		raise ValueError('Invalid dataset_structure. Select either BIDS or SPARC.')

	return a_path
def load_tags(cfg):
	"""
	Load subject tags from a JSON file at the specified dataset root path.

	Parameters:
		dataset_root_path (str): Path to the root of the dataset.

	Returns:
		list[str]: A list of subject tags if found.

	Raises:
		FileNotFoundError: If the tags JSON file does not exist.
		ValueError: If the file is not a JSON file.
	"""

	dataset_root_path = cfg["dataset_root_path"]
	tags_path = os.path.join(dataset_root_path, "code", "tags.json")

	if not os.path.exists(tags_path):
		raise FileNotFoundError(f"Tags list does not exist. Please create a tags list at {tags_path}.")

	if not tags_path.endswith('.json'):
		raise ValueError("Only JSON tags files are supported.")

	with open(tags_path, 'r') as f:
		tags = json.load(f)

	return tags
def make_directory(dir_path, msg):
	"""
	Create a directory if it does not already exist, and print a message.

	Parameters:
		dir_path (str): Path of the directory to create.
		msg (str): Descriptive message to display when creating the directory.
	"""
	if not os.path.exists(dir_path):
		os.mkdir(dir_path)
		print(f'Directory created for {msg}')
def verify_raw_image_existence(cfg):
	"""
	Verify that all required raw image files exist before segmentation.

	Parameters:
		cfg (dict): Configuration dictionary containing paths and settings.

	Raises:
		AssertionError: If any expected input file does not exist.
	"""

	tags = load_tags(cfg)
	for tag in tags:
		image_to_segment_file = get_path_to_item(cfg, 'image_to_segment_file', tag=tag)
		assert os.path.exists(image_to_segment_file), f"Missing image: {image_to_segment_file}. Are all paths, and image_to_segment_suffix, set correctly?"
def verify_dataset_structure(cfg):
	"""
	Verify that the selected dataset_structure type actually matches the dataset folder. To be make more robust.

	Parameters:
		cfg (dict): Configuration dictionary containing paths and settings.

	Raises:
		AssertionError: If dataset structure mismatches.
	"""

	dataset_structure = cfg["dataset_structure"]
	dataset_root_path = cfg["dataset_root_path"]
	if dataset_structure == 'BIDS':
		assert os.path.isdir(os.path.join(dataset_root_path, 'rawdata')), "'BIDS' is selected as the data structure, but 'rawdata' folder does not exist! Is the correct data structure selected?"
		assert not os.path.isdir(os.path.join(dataset_root_path, 'primary')), "'BIDS' is selected as the data structure, but 'primary' folder (SPARC version of 'rawdata') exists! Is the correct data structure selected?"
	elif dataset_structure == 'SPARC':
		assert os.path.isdir(os.path.join(dataset_root_path, 'primary')), "'SPARC' is selected as the data structure, but 'primary' folder does not exist! Is the correct data structure selected?"
		assert not os.path.isdir(os.path.join(dataset_root_path, 'rawdata')), "'SPARC' is selected as the data structure, but 'rawdata' folder (BIDS version of 'primary') exists! Is the correct data structure selected?"

def main():
	if len(sys.argv) < 2:
		print("Usage: python run_MRAtoBG_node_labelling.py <path_to_config.json>")
		sys.exit(1)

	# load config file
	with open(sys.argv[1]) as config_file:
		cfg = json.load(config_file)
	verify_dataset_structure(cfg)
	verify_raw_image_existence(cfg)
	tags = load_tags(cfg)

	do_node_labelling = 1

	start_time_global = time.perf_counter()
	for subj_ix, tag in enumerate(tags):
		print(f'Processing started for subject {subj_ix + 1} of {len(tags)}: {tag}')

		if do_node_labelling:
			node_labelling_start_time = time.perf_counter()
			NodeLabelling_obj = NodeLabelling(cfg, subj_ix, tag, node_labelling_start_time)
			NodeLabelling_obj.run()
			print('\t Node Labelling completed in {0:.2f} seconds'.format(time.perf_counter() - node_labelling_start_time))

	print(f'Node labelling module processed {len(tags)} subjects in {(time.perf_counter() - start_time_global) / 60:.2f} minutes')

if __name__ == '__main__':
	main()