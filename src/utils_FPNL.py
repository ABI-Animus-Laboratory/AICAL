"""
Utilities for running brain vessel node labelling using FPNL algorithms

Author: Jiantao Shen
Date: 20.05.25
Version: 1.0.0
"""

import os
import re
import copy
import math
import pickle
from decimal import Decimal
import random
import networkx as nx
import numpy as np
from pylab import cm, unique
import plotly.graph_objs as go
import plotly.offline as pyoffline
import json
import time
import pandas as pd
import csv
from timeit import default_timer as timer
import sys

def save_network_graph_FPNL(edges, n_unique_classes, unique_nodes_coord_id, save_path, tag, html_type, fiducial_nodes=[],
                       fiducial_nodes2=[], fiducial_nodes3=[], MCA_landmarks=[], MCA=[], ACA=[], PComA_anterior = [], PComA_posterior = [],
                       PCAL_landmarks=[], PCAR_landmarks=[], PCAL_landmarks_sphere=[], PCAR_landmarks_sphere=[],
                       PCAL_landmarks_ellipsoid=[], PCAR_landmarks_ellipsoid=[],
                       PCAL_anchor=[], PCAR_anchor=[], BAT_node=[], BAB_node=[], BAB_cylinder=[], ACA_cone=[], MCA_phantom=[],
                       PCAL_edge=[], PCAR_edge=[], PCAL_edge_landmark=[], PCAR_edge_landmark=[], ICA=[], artificial_leaves=[],
                       artificial_leaves2=[], artificial_bridges=[]):
    cmap = cm.get_cmap('plasma', n_unique_classes)
    edge_cols = [cmap(i) for i in range(cmap.N)]

    if html_type in ["detailed", "coloured_nodes_only", "coloured_nodes_geometries"]:
        normal_node_size = 2*2
        big_node_size = 3*2
    elif html_type in ["bare"]:
        normal_node_size = 3*2
        big_node_size = 10*2
    else:
        raise ValueError('xxx')

    Xf = [n[0] for n in fiducial_nodes]  # x-coordinates of nodes
    Yf = [n[1] for n in fiducial_nodes]  # y-coordinates
    Zf = [n[2] for n in fiducial_nodes]  # z-coordinates
    Xf2 = [n[0] for n in fiducial_nodes2 if np.array(n).any()]  # x-coordinates of nodes
    Yf2 = [n[1] for n in fiducial_nodes2 if np.array(n).any()]  # y-coordinates
    Zf2 = [n[2] for n in fiducial_nodes2 if np.array(n).any()]  # z-coordinates
    Xf3 = [n[0] for n in fiducial_nodes3]  # x-coordinates of nodes
    Yf3 = [n[1] for n in fiducial_nodes3]  # y-coordinates
    Zf3 = [n[2] for n in fiducial_nodes3]  # z-coordinates
    Xf_MCA_landmaks = [n[0] for n in MCA_landmarks]  # x-coordinates of nodes
    Yf_MCA_landmaks = [n[1] for n in MCA_landmarks]  # y-coordinates
    Zf_MCA_landmaks = [n[2] for n in MCA_landmarks]  # z-coordinates
    Xf_MCA = [n[0] for n in MCA]  # x-coordinates of nodes
    Yf_MCA = [n[1] for n in MCA]  # y-coordinates
    Zf_MCA = [n[2] for n in MCA]  # z-coordinates
    Xf_ACA = [n[0] for n in ACA]  # x-coordinates of nodes
    Yf_ACA = [n[1] for n in ACA]  # y-coordinates
    Zf_ACA = [n[2] for n in ACA]  # z-coordinates
    Xf_ICA = [n[0] for n in ICA]  # x-coordinates of nodes
    Yf_ICA = [n[1] for n in ICA]  # y-coordinates
    Zf_ICA = [n[2] for n in ICA]  # z-coordinates
    Xf_PComA_anterior = [n[0] for n in PComA_anterior]  # x-coordinates of nodes
    Yf_PComA_anterior = [n[1] for n in PComA_anterior]  # y-coordinates
    Zf_PComA_anterior = [n[2] for n in PComA_anterior]  # z-coordinates
    Xf_PComA_posterior = [n[0] for n in PComA_posterior]  # x-coordinates of nodes
    Yf_PComA_posterior = [n[1] for n in PComA_posterior]  # y-coordinates
    Zf_PComA_posterior = [n[2] for n in PComA_posterior]  # z-coordinates
    Xf_PCAL_landmarks = [n[0] for n in PCAL_landmarks]  # x-coordinates of nodes
    Yf_PCAL_landmarks = [n[1] for n in PCAL_landmarks]  # y-coordinates
    Zf_PCAL_landmarks = [n[2] for n in PCAL_landmarks]  # z-coordinates
    Xf_PCAR_landmarks = [n[0] for n in PCAR_landmarks]  # x-coordinates of nodes
    Yf_PCAR_landmarks = [n[1] for n in PCAR_landmarks]  # y-coordinates
    Zf_PCAR_landmarks = [n[2] for n in PCAR_landmarks]  # z-coordinates
    Xf_PCAL_anchor = [n[0] for n in PCAL_anchor]  # x-coordinates of nodes
    Yf_PCAL_anchor = [n[1] for n in PCAL_anchor]  # y-coordinates
    Zf_PCAL_anchor = [n[2] for n in PCAL_anchor]  # z-coordinates
    Xf_PCAR_anchor = [n[0] for n in PCAR_anchor]  # x-coordinates of nodes
    Yf_PCAR_anchor = [n[1] for n in PCAR_anchor]  # y-coordinates
    Zf_PCAR_anchor = [n[2] for n in PCAR_anchor]  # z-coordinates
    Xf_BAT_node = [n[0] for n in BAT_node]  # x-coordinates of nodes
    Yf_BAT_node = [n[1] for n in BAT_node]  # y-coordinates
    Zf_BAT_node = [n[2] for n in BAT_node]  # z-coordinates
    Xf_BAB_node = [n[0] for n in BAB_node]  # x-coordinates of nodes
    Yf_BAB_node = [n[1] for n in BAB_node]  # y-coordinates
    Zf_BAB_node = [n[2] for n in BAB_node]  # z-coordinates
    Xf_MCA_phantom = [n[0] for n in MCA_phantom if MCA_phantom != [[]]]  # x-coordinates of nodes
    Yf_MCA_phantom = [n[1] for n in MCA_phantom if MCA_phantom != [[]]]  # y-coordinates
    Zf_MCA_phantom = [n[2] for n in MCA_phantom if MCA_phantom != [[]]]  # z-coordinates
    # Cf = [unique_nodes_coord_id[n] for n in fiducial_nodes]
    Xn = [k[0] for k, v in unique_nodes_coord_id.items()]
    Yn = [k[1] for k, v in unique_nodes_coord_id.items()]
    Zn = [k[2] for k, v in unique_nodes_coord_id.items()]
    Cn = [f'{v[0]}.{v[1]}' for k, v in unique_nodes_coord_id.items()]
    Xe = []
    Ye = []
    Ze = []
    Ce = []
    Le = []
    X_artificial_leaves = []
    Y_artificial_leaves = []
    Z_artificial_leaves = []
    X_artificial_leaves2 = []
    Y_artificial_leaves2 = []
    Z_artificial_leaves2 = []
    X_artificial_bridges = []
    Y_artificial_bridges = []
    Z_artificial_bridges = []

    for e in edges:
        Xe += [e[3], e[6], None]  # x-coordinates of edge ends
        Ye += [e[4], e[7], None]  # x-coordinates of edge ends
        Ze += [e[5], e[8], None]  # x-coordinates of edge ends
        Ce += 3 * [edge_cols[int(e[11])]]  # [-1, -1, -1]
        Le += 3 * [str(float(e[9]))]

    trace1 = go.Scatter3d(x=Xe,
                          y=Ye,
                          z=Ze,
                          mode='lines',
                          # mode='lines+text',
                          # line=dict(color=Ce, width=5),
                          text=Le,
                          # textfont_size=1,
                          hoverinfo='text'
                          )

    for e in artificial_leaves:
        X_artificial_leaves += [e[0][0], e[1][0], None]  # x-coordinates of edge ends
        Y_artificial_leaves += [e[0][1], e[1][1], None]  # x-coordinates of edge ends
        Z_artificial_leaves += [e[0][2], e[1][2], None]  # x-coordinates of edge ends
    trace_artificial_leaves = go.Scatter3d(x=X_artificial_leaves,
                          y=Y_artificial_leaves,
                          z=Z_artificial_leaves,
                          mode='lines',
                          # mode='lines+text',
                          line=dict(color='red', width=5),
                          # text=Le,
                          # textfont_size=1,
                          hoverinfo='text'
                          )
    for e in artificial_leaves2:
        X_artificial_leaves2 += [e[0][0], e[1][0], None]  # x-coordinates of edge ends
        Y_artificial_leaves2 += [e[0][1], e[1][1], None]  # x-coordinates of edge ends
        Z_artificial_leaves2 += [e[0][2], e[1][2], None]  # x-coordinates of edge ends
    trace_artificial_leaves2 = go.Scatter3d(x=X_artificial_leaves2,
                          y=Y_artificial_leaves2,
                          z=Z_artificial_leaves2,
                          mode='lines',
                          # mode='lines+text',
                          line=dict(color='yellow', width=5),
                          # text=Le,
                          # textfont_size=1,
                          hoverinfo='text'
                          )

    for e in artificial_bridges:
        X_artificial_bridges += [e[0][0], e[1][0], None]  # x-coordinates of edge ends
        Y_artificial_bridges += [e[0][1], e[1][1], None]  # x-coordinates of edge ends
        Z_artificial_bridges += [e[0][2], e[1][2], None]  # x-coordinates of edge ends
    trace_artificial_bridges = go.Scatter3d(x=X_artificial_bridges,
                          y=Y_artificial_bridges,
                          z=Z_artificial_bridges,
                          mode='lines',
                          # mode='lines+text',
                          line=dict(color='orange', width=5),
                          # text=Le,
                          # textfont_size=1,
                          hoverinfo='text'
                          )

    orange_rgb = 'rgb(255,153,0)'
    if html_type in ["coloured_nodes_only", "coloured_nodes_geometries"]:
        mode = 'markers'
        colour_MCA = 'rgb(68,114,196)'
        colour_ACA = 'rgb(117,180,74)'
        colour_PComA_anterior = 'rgb(45,230,255)'
        colour_PComA_posterior = 'rgb(255,102,204)'
        colour_ICA = 'rgb(191,191,191)'
        colour_BAT = 'rgb(255,5,5)'
        # colour_orange = 'rgb(255,153,0)'
        colour_orange = [[0, 'rgb(255,153,0)'], [1, 'rgb(255,153,0)']]
    elif html_type in ["detailed", "bare"]:
        mode = 'markers+text'
        colour_MCA = 'rgb(255,0,0)'
        colour_ACA = 'rgb(255,255,0)'
        colour_PComA_anterior = 'rgb(0,255,255)'
        colour_PComA_posterior = 'rgb(255,102,204)'
        colour_ICA = 'rgb(191,191,191)'
        colour_BAT = 'rgb(255,5,5)'
        # colour_orange = 'rgb(255,153,0)'
        colour_orange = 'oranges'


    else:
        raise ValueError('xxx')
    trace2 = go.Scatter3d(x=Xn,
                          y=Yn,
                          z=Zn,
                          mode=mode,
                          # name='actors',
                          marker=dict(symbol='circle',
                                      size=normal_node_size,
                                      color='rgb(0,0,0)',
                                      # colorscale='Viridis',
                                      line=dict(color='rgb(50,50,50)',
                                                width=0.5)
                                      ),
                          text=Cn,
                          textfont_size=10,
                          hoverinfo= 'text'
                          )

    trace3 = go.Scatter3d(x=Xf,
                          y=Yf,
                          z=Zf,
                          mode='markers',
                          # name='actors',
                          marker=dict(symbol='circle',
                                      size=big_node_size,
                                      color='rgb(255,0,0)',
                                      # colorscale='Viridis',
                                      line=dict(color='rgb(50,50,50)',
                                                width=0.5)
                                      ),
                          # text=Cf,
                          # hoverinfo='text'
                          )
    trace4 = go.Scatter3d(x=Xf2,
                          y=Yf2,
                          z=Zf2,
                          mode='markers',
                          # name='actors',
                          marker=dict(symbol='circle',
                                      size=big_node_size,
                                      color='rgb(0,255,0)',
                                      # colorscale='Viridis',
                                      line=dict(color='rgb(50,50,50)',
                                                width=0.5)
                                      ),
                          # text=Cf,
                          # hoverinfo='text'
                          )
    trace5 = go.Scatter3d(x=Xf3,
                          y=Yf3,
                          z=Zf3,
                          mode='markers',
                          # name='actors',
                          marker=dict(symbol='circle',
                                      size=normal_node_size,
                                      color='rgb(0,255,0)',
                                      # colorscale='Viridis',
                                      line=dict(color='rgb(50,50,50)',
                                                width=0.5)
                                      ),
                          # text=Cf,
                          # hoverinfo='text'
                          )
    trace_MCA_landmarks = go.Scatter3d(x=Xf_MCA_landmaks,
                          y=Yf_MCA_landmaks,
                          z=Zf_MCA_landmaks,
                          mode='markers',
                          # name='actors',
                          marker=dict(symbol='circle',
                                      size=normal_node_size,
                                      color='rgb(0,0,255)',
                                      # colorscale='Viridis',
                                      line=dict(color='rgb(50,50,50)',
                                                width=0.5)
                                      ),
                          # text=Cf,
                          # hoverinfo='text'
                          )
    trace_MCA = go.Scatter3d(x=Xf_MCA,
                          y=Yf_MCA,
                          z=Zf_MCA,
                          mode='markers',
                          # name='actors',
                          marker=dict(symbol='circle',
                                      size=big_node_size,
                                      color=colour_MCA,
                                      # colorscale='Viridis',
                                      line=dict(color='rgb(50,50,50)',
                                                width=0.5)
                                      ),
                          # text=Cf,
                          # hoverinfo='text'
                          )
    trace_ACA = go.Scatter3d(x=Xf_ACA,
                          y=Yf_ACA,
                          z=Zf_ACA,
                          mode='markers',
                          # name='actors',
                          marker=dict(symbol='circle',
                                      size=big_node_size,
                                      color=colour_ACA,
                                      # colorscale='Viridis',
                                      line=dict(color='rgb(50,50,50)',
                                                width=0.5)
                                      ),
                          # text=Cf,
                          # hoverinfo='text'
                          )
    trace_ICA = go.Scatter3d(x=Xf_ICA,
                          y=Yf_ICA,
                          z=Zf_ICA,
                          mode='markers',
                          # name='actors',
                          marker=dict(symbol='circle',
                                      size=big_node_size,
                                      color=colour_ICA,
                                      # colorscale='Viridis',
                                      line=dict(color='rgb(50,50,50)',
                                                width=0.5)
                                      ),
                          # text=Cf,
                          # hoverinfo='text'
                          )
    trace_PComA_anterior = go.Scatter3d(
                            x=Xf_PComA_anterior,
                            y=Yf_PComA_anterior,
                            z=Zf_PComA_anterior,
                          mode='markers',
                          # name='actors',
                          marker=dict(symbol='circle',
                                      size=big_node_size,
                                      color=colour_PComA_anterior,
                                      # colorscale='Viridis',
                                      line=dict(color='rgb(50,50,50)',
                                                width=0.5)
                                      ),
                          # text=Cf,
                          # hoverinfo='text'
                          )
    trace_PComA_posterior = go.Scatter3d(
                            x=Xf_PComA_posterior,
                            y=Yf_PComA_posterior,
                            z=Zf_PComA_posterior,
                          mode='markers',
                          # name='actors',
                          marker=dict(symbol='circle',
                                      size=big_node_size,
                                      color=colour_PComA_posterior,
                                      # colorscale='Viridis',
                                      line=dict(color='rgb(50,50,50)',
                                                width=0.5)
                                      ),
                          # text=Cf,
                          # hoverinfo='text'
                          )
    trace_PCAL_landmarks = go.Scatter3d(
                            x=Xf_PCAL_landmarks,
                            y=Yf_PCAL_landmarks,
                            z=Zf_PCAL_landmarks,
                          mode='markers',
                          # name='actors',
                          marker=dict(symbol='circle',
                                      size=normal_node_size,
                                      color='rgb(255,0,255)',
                                      # colorscale='Viridis',
                                      line=dict(color='rgb(50,50,50)',
                                                width=0.5)
                                      ),
                          # text=Cf,
                          # hoverinfo='text'
                          )
    trace_PCAR_landmarks = go.Scatter3d(
                            x=Xf_PCAR_landmarks,
                            y=Yf_PCAR_landmarks,
                            z=Zf_PCAR_landmarks,
                          mode='markers',
                          # name='actors',
                          marker=dict(symbol='circle',
                                      size=normal_node_size,
                                      color='rgb(100,0,100)',
                                      # colorscale='Viridis',
                                      line=dict(color='rgb(50,50,50)',
                                                width=0.5)
                                      ),
                          # text=Cf,
                          # hoverinfo='text'
                          )


    trace_PCAL_anchor = go.Scatter3d(
                            x=Xf_PCAL_anchor,
                            y=Yf_PCAL_anchor,
                            z=Zf_PCAL_anchor,
                          mode='markers',
                          # name='actors',
                          marker=dict(symbol='circle',
                                      size=normal_node_size,
                                      color='rgb(255,150,0)',
                                      # colorscale='Viridis',
                                      line=dict(color='rgb(50,50,50)',
                                                width=0.5)
                                      ),
                          # text=Cf,
                          # hoverinfo='text'
                          )

    trace_PCAR_anchor = go.Scatter3d(
                            x=Xf_PCAR_anchor,
                            y=Yf_PCAR_anchor,
                            z=Zf_PCAR_anchor,
                          mode='markers',
                          # name='actors',
                          marker=dict(symbol='circle',
                                      size=normal_node_size,
                                      color='rgb(255,100,0)',
                                      # colorscale='Viridis',
                                      line=dict(color='rgb(50,50,50)',
                                                width=0.5)
                                      ),
                          # text=Cf,
                          # hoverinfo='text'
                          )
    trace_BAT_node = go.Scatter3d(
                            x=Xf_BAT_node,
                            y=Yf_BAT_node,
                            z=Zf_BAT_node,
                          mode='markers',
                          # name='actors',
                          marker=dict(symbol='circle',
                                      size=big_node_size,
                                      color=colour_BAT,
                                      # colorscale='Viridis',
                                      line=dict(color='rgb(50,50,50)',
                                                width=0.5)
                                      ),
                          # text=Cf,
                          # hoverinfo='text'
                          )

    trace_BAB_node = go.Scatter3d(
                            x=Xf_BAB_node,
                            y=Yf_BAB_node,
                            z=Zf_BAB_node,
                          mode='markers',
                          # name='actors',
                          marker=dict(symbol='circle',
                                      size=big_node_size,
                                      color='rgb(200,200,200)',
                                      # colorscale='Viridis',
                                      line=dict(color='rgb(50,50,50)',
                                                width=0.5)
                                      ),
                          # text=Cf,
                          # hoverinfo='text'
                          )

    trace_MCA_phantom = go.Scatter3d(
                            x=Xf_MCA_phantom,
                            y=Yf_MCA_phantom,
                            z=Zf_MCA_phantom,
                          mode='markers',
                          # name='actors',
                          marker=dict(symbol='circle',
                                      size=normal_node_size,
                                      color='rgb(150,0,0)',
                                      # colorscale='Viridis',
                                      line=dict(color='rgb(50,50,50)',
                                                width=0.5)
                                      ),
                          # text=Cf,
                          # hoverinfo='text'
                          )

    Xe = [node[0] for node in PCAL_edge]
    Ye = [node[1] for node in PCAL_edge]
    Ze = [node[2] for node in PCAL_edge]
    trace_PCAL_edge = go.Scatter3d(x=Xe,
                          y=Ye,
                          z=Ze,
                          mode='lines',
                          # mode='lines+text',
                          line=dict(color='rgb(255,0,255)', width=5),
                          text=Le,
                          # textfont_size=1,
                          hoverinfo='text'
                          )
    Xe = [node[0] for node in PCAR_edge]
    Ye = [node[1] for node in PCAR_edge]
    Ze = [node[2] for node in PCAR_edge]
    trace_PCAR_edge = go.Scatter3d(x=Xe,
                          y=Ye,
                          z=Ze,
                          mode='lines',
                          # mode='lines+text',
                          line=dict(color='rgb(200,0,200)', width=5),
                          text=Le,
                          # textfont_size=1,
                          hoverinfo='text'
                          )

    ##############################################################################
    ACA_cone_tip = np.array(ACA_cone[0])
    ACA_cone_base = np.array(ACA_cone[1])
    ACA_cone_radius = ACA_cone[2]

    # Calculate the direction vector of the line
    cone_axis = ACA_cone_base - ACA_cone_tip
    # Calculate two orthogonal vectors to the line
    v1 = np.array([1, 0, 0])  # An arbitrary vector
    v2 = np.cross(cone_axis, v1)
    v2 = v2 / np.linalg.norm(v2)
    v3 = np.cross(cone_axis, v2)
    v3 = v3 / np.linalg.norm(v3)

    # Create points for the tilted circle
    theta = np.linspace(0, 2 * np.pi, 365)
    x_circle = ACA_cone_base[0] + ACA_cone_radius * (v2[0] * np.cos(theta) + v3[0] * np.sin(theta))
    y_circle = ACA_cone_base[1] + ACA_cone_radius * (v2[1] * np.cos(theta) + v3[1] * np.sin(theta))
    z_circle = ACA_cone_base[2] + ACA_cone_radius * (v2[2] * np.cos(theta) + v3[2] * np.sin(theta))

    trace_ACA_cone_circle = go.Scatter3d(
        x=x_circle,
        y=y_circle,
        z=z_circle,
        mode='lines',
        line=dict(color=orange_rgb, width=2),
        # name='Tilted Circle'
    )
    # Select equally spaced points around the circle to draw cone surface
    num_points = 30
    indices = np.linspace(0, len(theta) - 1, num_points, dtype=int)
    x_points = x_circle[indices]
    y_points = y_circle[indices]
    z_points = z_circle[indices]

    trace_ACA_cone_axis = go.Scatter3d(
        x=[ACA_cone_tip[0], ACA_cone_base[0]],
        y=[ACA_cone_tip[1], ACA_cone_base[1]],
        z=[ACA_cone_tip[2], ACA_cone_base[2]],
        mode='lines',
        line=dict(color=orange_rgb, width=3),
        # name='Line p1-p2'
    )

    traces_lines_to_ACA_cone_tip = []
    # Loop to draw lines from each selected point to p1
    for i in range(num_points):
        line_to_p1_trace = go.Scatter3d(
            x=[x_points[i], ACA_cone_tip[0]],
            y=[y_points[i], ACA_cone_tip[1]],
            z=[z_points[i], ACA_cone_tip[2]],
            mode='lines',
            line=dict(color=orange_rgb, width=2, dash='dot'),
            # name=f'Line to p1 from Point {i + 1}'
        )
        traces_lines_to_ACA_cone_tip.append(line_to_p1_trace)

    #########################################################################################
    cylinder_center = BAB_cylinder[0]
    cylinder_height = BAB_cylinder[1]
    cylinder_radius1 = BAB_cylinder[2]
    cylinder_radius2 = BAB_cylinder[3]
    cylinder_vector1 = BAB_cylinder[4]
    cylinder_vector2 = BAB_cylinder[5]

    # Create a mesh grid for the surface
    u_vals = np.linspace(0, 2 * np.pi, 100)  # Angle parameter
    z_vals = np.linspace(0, cylinder_height, 50)  # Height parameter
    u_grid, z_grid = np.meshgrid(u_vals, z_vals)

    # Calculate the surface points of the elliptic cylinder
    x_cylinder = cylinder_center[0] + cylinder_radius1 * np.cos(u_grid) * cylinder_vector1[0] + cylinder_radius2 * np.sin(u_grid) * cylinder_vector2[0]
    y_cylinder = cylinder_center[1] + cylinder_radius1 * np.cos(u_grid) * cylinder_vector1[1] + cylinder_radius2 * np.sin(u_grid) * cylinder_vector2[1]
    z_cylinder = cylinder_center[2] + cylinder_radius1 * np.cos(u_grid) * cylinder_vector1[2] + cylinder_radius2 * np.sin(u_grid) * cylinder_vector2[2] + z_grid

    trace_BAB_cylinder = go.Surface(x=x_cylinder, y=y_cylinder, z=z_cylinder,
                     colorscale = colour_orange,
                     showscale=False,
                     opacity=0.5,
                     hoverinfo='none')

    #########################################################################################
    # BAB circular cylinder
    # cylinder_center = BAB_cylinder[0]
    # cylinder_height = BAB_cylinder[1]
    # cylinder_radius = BAB_cylinder[2]
    # nt = 100
    # nv = 50
    # theta = np.linspace(0, 2 * np.pi, nt)
    # v = np.linspace(cylinder_center[2], cylinder_center[2] + cylinder_height, nv)
    # theta, v = np.meshgrid(theta, v)
    # x_cylinder = cylinder_radius * np.cos(theta) + cylinder_center[0]
    # y_cylinder = cylinder_radius * np.sin(theta) + cylinder_center[1]
    # z_cylinder = v
    # trace_BAB_cylinder = go.Surface(x=x_cylinder, y=y_cylinder, z=z_cylinder,
    #                  colorscale = 'gray',
    #                  showscale=False,
    #                  opacity=0.5,
    #                  hoverinfo='none'
    #                                 )

    # phiL, thetaL = np.mgrid[0.0:2.0 * np.pi:100j, 0.0:np.pi:50j]
    # x_sphereL = PCAL_landmarks_sphere[3] * np.sin(thetaL) * np.cos(phiL) + PCAL_landmarks_sphere[0]
    # y_sphereL = PCAL_landmarks_sphere[3] * np.sin(thetaL) * np.sin(phiL) + PCAL_landmarks_sphere[1]
    # z_sphereL = PCAL_landmarks_sphere[3] * np.cos(thetaL) + PCAL_landmarks_sphere[2]
    # trace_PCAL_landmarks_sphere = go.Surface(x=x_sphereL,
    #                                          y=y_sphereL,
    #                                          z=z_sphereL,
    #                                          opacity=0.3,  # Set opacity to make it semi-transparent
    #                                          showscale=False,  # Turn off the color scale for solid color
    #                                          colorscale='Viridis',
    #                                          # Set a colorscale (you can adjust or remove this line)
    #                                          )
    #
    # phiR, thetaR = np.mgrid[0.0:2.0 * np.pi:100j, 0.0:np.pi:50j]
    # x_sphereR = PCAR_landmarks_sphere[3] * np.sin(thetaR) * np.cos(phiR) + PCAR_landmarks_sphere[0]
    # y_sphereR = PCAR_landmarks_sphere[3] * np.sin(thetaR) * np.sin(phiR) + PCAR_landmarks_sphere[1]
    # z_sphereR = PCAR_landmarks_sphere[3] * np.cos(thetaR) + PCAR_landmarks_sphere[2]
    # trace_PCAR_landmarks_sphere = go.Surface(x=x_sphereR,
    #                                          y=y_sphereR,
    #                                          z=z_sphereR,
    #                                          opacity=0.3,  # Set opacity to make it semi-transparent
    #                                          showscale=False,  # Turn off the color scale for solid color
    #                                          colorscale='Viridis',
    #                                          # Set a colorscale (you can adjust or remove this line)
    #                                          )

    phiL, thetaL = np.mgrid[0.0:2.0 * np.pi:100j, 0.0:np.pi:50j]
    x_ellipsoidL = PCAL_landmarks_ellipsoid[3] * np.sin(thetaL) * np.cos(phiL) + PCAL_landmarks_ellipsoid[0]
    y_ellipsoidL = PCAL_landmarks_ellipsoid[4] * np.sin(thetaL) * np.sin(phiL) + PCAL_landmarks_ellipsoid[1]
    z_ellipsoidL = PCAL_landmarks_ellipsoid[5] * np.cos(thetaL) + PCAL_landmarks_ellipsoid[2]
    trace_PCAL_landmarks_ellipsoid = go.Surface(x=x_ellipsoidL,
                                             y=y_ellipsoidL,
                                             z=z_ellipsoidL,
                                             opacity=0.25,  # Set opacity to make it semi-transparent
                                             showscale=False,  # Turn off the color scale for solid color
                                             # colorscale='Viridis',
                                             colorscale=colour_orange,
                                             hoverinfo='none'
                                                # Set a colorscale (you can adjust or remove this line)
                                             )

    phiR, thetaR = np.mgrid[0.0:2.0 * np.pi:100j, 0.0:np.pi:50j]
    x_ellipsoidR = PCAR_landmarks_ellipsoid[3] * np.sin(thetaR) * np.cos(phiR) + PCAR_landmarks_ellipsoid[0]
    y_ellipsoidR = PCAR_landmarks_ellipsoid[4] * np.sin(thetaR) * np.sin(phiR) + PCAR_landmarks_ellipsoid[1]
    z_ellipsoidR = PCAR_landmarks_ellipsoid[5] * np.cos(thetaR) + PCAR_landmarks_ellipsoid[2]
    trace_PCAR_landmarks_ellipsoid = go.Surface(x=x_ellipsoidR,
                                             y=y_ellipsoidR,
                                             z=z_ellipsoidR,
                                             opacity=0.25,  # Set opacity to make it semi-transparent
                                             showscale=False,  # Turn off the color scale for solid color
                                             # colorscale='Viridis',
                                             colorscale=colour_orange,
                                             hoverinfo='none'
                                             # Set a colorscale (you can adjust or remove this line)
                                             )


    axis = dict(showbackground=True,
                showline=True,
                zeroline=True,
                showgrid=True,
                showticklabels=True,
                title=''
                )
    layout = go.Layout(
        # title=tag,
        width=1000,
        height=500,
        showlegend=False,
        scene=dict(
            xaxis=dict(axis),
            yaxis=dict(axis),
            zaxis=dict(axis),
        ),
        margin=dict(
            t=0
        ),
        hovermode='closest',
        annotations=[
            dict(
                showarrow=False,
                text=tag,  # 'ce',
                xref='paper',
                yref='paper',
                x=0,
                y=0.1,
                xanchor='left',
                yanchor='bottom',
                font=dict(
                    size=14
                )
            )
        ], )


    # x = [259, 246, 126]  # Replace with the coordinates of point x
    # y = [259, 174, 201] # Replace with the coordinates of point y
    # r = 40  # Replace with your desired radius
    # height = np.linalg.norm(np.array(y) - np.array(x))
    #
    # # Calculate the direction vector from x to y
    # direction_vector = np.array(y) - np.array(x)
    # direction_vector = direction_vector / np.linalg.norm(direction_vector)  # Normalize the vector
    #
    # # Create vertices for the cone
    # num_points = 100  # Number of points to create the cone's surface
    # theta = np.linspace(0, 2 * np.pi, num_points)
    # z = np.linspace(0, height, num_points)
    # theta, z = np.meshgrid(theta, z)
    #
    # # Cone vertices
    # x_cone = r * (1 - z / height) * np.cos(theta)  + x[0]* direction_vector[0]
    # y_cone = r * (1 - z / height) * np.sin(theta) + x[1]* direction_vector[1]
    # z_cone = z + x[2]* direction_vector[2]
    #
    # # Translate the cone to position it at point x
    # x_cone += direction_vector[0]
    # y_cone += direction_vector[1]
    # z_cone += direction_vector[2]
    #
    # # Define the faces of the cone
    # faces = []
    # for i in range(num_points - 1):
    #     for j in range(num_points - 1):
    #         a = i * num_points + j
    #         b = i * num_points + j + 1
    #         c = (i + 1) * num_points + j
    #         d = (i + 1) * num_points + j + 1
    #         faces.extend([[a, b, c], [b, c, d]])
    #
    # # Create a mesh3d trace for the cone
    # cone_trace = go.Mesh3d(
    #     x=x_cone.flatten(),
    #     y=y_cone.flatten(),
    #     z=z_cone.flatten(),
    #     i=[face[0] for face in faces],
    #     j=[face[1] for face in faces],
    #     k=[face[2] for face in faces],
    #     opacity=0.5,  # Adjust opacity to make the cone semi-transparent
    #     colorscale='Viridis',  # You can change the colorscale if needed
    # )
    #
    # r = 120
    # axis_vector = [y[0] - x[0], y[1] - x[1], y[2] - x[2]]
    # r = sum([a ** 2 for a in axis_vector]) ** 0.5
    # # Create a cone trace
    # cone_trace = go.Cone(
    #     x=[x[0]],
    #     y=[x[1]],
    #     z=[x[2]],
    #     u=[y[0] - x[0]],
    #     v=[y[1] - x[1]],
    #     w=[y[2] - x[2]],
    #     sizemode="absolute",
    #     sizeref=r,
    #     showscale=False,
    #     colorscale='Viridis',  # You can change the colorscale if needed
    #     opacity=0.5,  # Adjust opacity to make the cone semi-transparent
    # )
    if html_type in ["detailed"]:
        data = [trace1, trace2, trace3, trace4, trace5, trace_PCAL_landmarks, trace_PCAR_landmarks,
            trace_MCA_landmarks, trace_ACA, trace_PComA_anterior, trace_PComA_posterior,
            trace_PCAL_landmarks_ellipsoid, trace_PCAR_landmarks_ellipsoid, trace_PCAL_anchor, trace_PCAR_anchor,
            trace_BAT_node, trace_BAB_node, trace_BAB_cylinder, trace_ACA_cone_circle, trace_ACA_cone_axis, *traces_lines_to_ACA_cone_tip, trace_MCA_phantom,
            trace_PCAL_edge, trace_PCAR_edge, trace_ICA, trace_MCA]
    elif html_type in ["bare"]:
        data = [trace1, trace2, trace_ACA, trace_PComA_anterior, trace_PComA_posterior, trace_MCA,
            trace_BAT_node, trace_BAB_node,
            trace_PCAL_edge, trace_PCAR_edge]
    elif html_type in ["coloured_nodes_only"]:
        data = [trace1, trace2, trace_ACA, trace_PComA_anterior, trace_PComA_posterior, trace_MCA,
            trace_BAT_node, trace_BAB_node, trace_BAB_cylinder, trace_ICA, trace_ACA_cone_circle,
                trace_ACA_cone_axis, *traces_lines_to_ACA_cone_tip, trace_artificial_leaves, trace_artificial_leaves2, trace_artificial_bridges]
    elif html_type in ["aaa"]:
        data = [trace1, trace2, trace_ACA, trace_PComA_anterior, trace_PComA_posterior, trace_MCA,
            trace_BAT_node, trace_BAB_node, trace_BAB_cylinder, trace_PCAL_landmarks_ellipsoid,
                trace_PCAR_landmarks_ellipsoid, trace_ACA_cone_circle, trace_ACA_cone_axis, *traces_lines_to_ACA_cone_tip, trace_ICA]
    elif html_type in ["coloured_nodes_geometries"]:
        pass
    else:
        raise ValueError('xxx')

    fig = go.Figure(data=data, layout=layout)

    if html_type in ["coloured_nodes_only", "coloured_nodes_geometries"]:
        fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)

    fig.write_html(save_path)

# ICA functions #####################################################################
def get_ICA_candidates(nx_g, cohort, c, do_suppress_skips, nix, do_verbose_errors):
    assert nix in [1, 2], 'ICA index must be either 1 or 2'
    if nix == 1:
        ICA_node_name = 'ICAL'
    elif nix == 2:
        ICA_node_name = 'ICAR'

    do_continue = False
    ICA_candidates = [node for node, data in nx_g.nodes(data=True) if 'boitype' in data and data['boitype'] == nix]
    if len(ICA_candidates) == 0:
        if not do_suppress_skips:
            print(f'\t Missing {ICA_node_name} node. Skipping this subject')
        c += 1
        do_continue = True
        return [], do_continue, c
    elif len(ICA_candidates) > 1:
        if do_verbose_errors:
            print(
                f'\t Multiple {ICA_node_name} candidates found: {ICA_candidates}. Correct one has been hard coded.')
        if cohort == 'CROPCheck' and c == 32:
            assert nix == 2, 'xxx'
            ICA_candidates = [81]
            if do_verbose_errors:
                print(f'\t WARNING! ICA candidates hard coded.')

    assert len(ICA_candidates) == 1, 'xxx'
    do_continue = False
    return ICA_candidates, do_continue, c
    # MCAL_candidates = [node for node, data in nx_g.nodes(data=True) if 'boitype' in data and data['boitype'] == 3]
    # if len(MCAL_candidates) == 0:
    #     print('MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM')
    #     print('MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM')
    #     print('MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM')
    #     print('MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM')
    #     print('MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM')

def get_ICA_nodes(ICAL_candidates, ICAR_candidates, cohort, train_or_test, c, do_verbose_errors):
    ICAL_node = ICAL_candidates[0]
    ICAR_node = ICAR_candidates[0]

    if cohort == 'ArizonaCheck' and train_or_test == 'train' and c == 93:
        ICAL_node = 34
        ICAR_node = 83
        if do_verbose_errors:
            print(f'\t WARNING! ICA nodes hard coded.')
    if cohort == 'ArizonaCheck' and train_or_test == 'train' and c == 111:
        ICAL_node = 22
        ICAR_node = 45
        if do_verbose_errors:
            print(f'\t WARNING! ICA nodes hard coded.')
    if cohort == 'ArizonaCheck' and train_or_test == 'train' and c == 140:
        ICAL_node = 83
        ICAR_node = 37
        if do_verbose_errors:
            print(f'\t WARNING! ICA nodes hard coded.')
    if cohort == 'ArizonaCheck' and train_or_test == 'all' and c == 107:
        ICAL_node = 34
        ICAR_node = 83
        if do_verbose_errors:
            print(f'\t WARNING! ICA nodes hard coded.')
    if cohort == 'ArizonaCheck' and train_or_test == 'all' and c == 125:
        ICAL_node = 22
        ICAR_node = 45
        if do_verbose_errors:
            print(f'\t WARNING! ICA nodes hard coded.')
    if cohort == 'ArizonaCheck' and train_or_test == 'all' and c == 157:
        ICAL_node = 83
        ICAR_node = 37
        if do_verbose_errors:
            print(f'\t WARNING! ICA nodes hard coded.')

    return ICAL_node, ICAR_node

def get_ICA_node_params(nx_g, ICAL_node, ICAR_node):
        ICAL_pos = nx_g.nodes[ICAL_node]['pos']
        ICAR_pos = nx_g.nodes[ICAR_node]['pos']
        centroid_glob = np.mean(np.array([data['pos'] for _, data in nx_g.nodes(data=True)]), axis=0)  # print("Center of Mass (centroid_glob):", centroid_glob)

        assert ICAL_pos[0] > ICAR_pos[ 0], f'\t ICAL is to the right of ICAR! Skipping this subject ICAL_node{ICAL_node} ICAR_node{ICAR_node}'
        centroid_ICAL_ICAR = np.mean([ICAL_pos, ICAR_pos], axis=0)
        centroid_ICAL_ICAR_at_centroid_glob_zpos = copy.deepcopy(centroid_ICAL_ICAR)
        centroid_ICAL_ICAR_at_centroid_glob_zpos[2] = centroid_glob[2]
        # inter_ICA_distance = np.linalg.norm(np.array(ICAR_pos) - np.array(ICAL_pos))
        # if np.abs(ICAL_pos[2] - ICAR_pos[2]) > 0.5 * inter_ICA_distance:
        #     print('\t WARNING: inter_ICA S-I distance exceeds 0.5 * inter_ICA Euclidean distance')
        # if np.abs(ICAL_pos[1] - ICAR_pos[1]) > 0.5 * inter_ICA_distance:
        #     print('\t WARNING: inter_ICA A-P distance exceeds 0.5 * inter_ICA Euclidean distance')

        return ICAL_pos, ICAR_pos, centroid_ICAL_ICAR, centroid_glob, centroid_ICAL_ICAR_at_centroid_glob_zpos

# MCA functions #####################################################################
def find_MCA_landmarks(nx_g, ICAL_node, ICAR_node, ICAL_pos, ICAR_pos):
    # There's got to be an MCA landmark connected to ipsi ICA if perform exhaustive peripheral to medial x axis SP search
    # Bug when the PCA extends laterally out peripherally interfering with current logic
    # max_xpos_nodes = max(nx_g.nodes(data=True), key=lambda x: x[1].get('pos', 0))
    # min_xpos_nodes = min(nx_g.nodes(data=True), key=lambda x: x[1].get('pos', 0))
    # assert len(max_xpos_nodes) == 2, 'fwf'
    # assert len(min_xpos_nodes) == 2, 'fwf'
    #
    # MCAL_landmark = max_xpos_nodes[0]
    # MCAR_landmark = min_xpos_nodes[0]
    # MCA_landmark_pos_attributes = [nx_g.nodes[node]['pos'] for node in [MCAL_landmark, MCAR_landmark] if
    #                                  'pos' in nx_g.nodes[node]]

    MCAL_landmark_candidates = [node for node, _ in sorted(nx_g.nodes(data='pos'), key=lambda x: x[1][0], reverse=True)]
    found_MCAL = False
    MCAL_landmark = None
    for cand in MCAL_landmark_candidates:
        if not found_MCAL:
            try:
                # Find the shortest path
                shortest_path = nx.shortest_path(nx_g, source=ICAL_node, target=cand)
                found_MCAL = True
                MCAL_landmark = cand
            except:
                pass
                # Handle other exceptions that might occur
                # print("path not found or node not found")

    if not found_MCAL:
        raise ValueError('this should not be possible')

    MCAR_landmark_candidates = [node for node, _ in sorted(nx_g.nodes(data='pos'), key=lambda x: x[1][0], reverse=False)]
    found_MCAR = False
    MCAR_landmark = None
    for cand in MCAR_landmark_candidates:
        if not found_MCAR:
            try:
                # Find the shortest path
                shortest_path = nx.shortest_path(nx_g, source=ICAR_node, target=cand)
                found_MCAR = True
                MCAR_landmark = cand
            except:
                pass
                # Handle other exceptions that might occur
                # print("path not found or node not found")

    if not found_MCAR:
        raise ValueError('this should not be possible')

    MCA_landmark_pos_attributes = [nx_g.nodes[node]['pos'] for node in [MCAL_landmark, MCAR_landmark] if
                                     'pos' in nx_g.nodes[node]]

    assert MCA_landmark_pos_attributes[0][0] > ICAL_pos[0], ''
    assert MCA_landmark_pos_attributes[1][0] < ICAR_pos[0], f'{f}\n{MCA_landmark_pos_attributes[1][0]}\n{ICAR_pos[0]}'

    return MCAL_landmark, MCAR_landmark, MCA_landmark_pos_attributes

def find_MCA_nodes(nx_g, ICAL_node, ICAR_node, ACA_cone_nodes, MCAL_landmark, MCAR_landmark, centroid_ICAL_ICAR, do_verbose_errors,
                   tag, c, node_labelling_predictions, node_labelling_output_path, do_output_node_labelling_predictions, do_suppress_skips):
    def distance(point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    # SP from ICAL to MCAL landmark
    try:
        sp_ICAL_MCAL_landmark = nx.shortest_path(nx_g, source=ICAL_node, target=MCAL_landmark, weight='dist') # liz 1133'3'33 ww.qp.wr
    except:
        # Handle other exceptions that might occur
        raise ValueError("No path found from ICAL to MCAL landmark!")

    # SP from ICAR to MCAR landmark
    try:
        sp_ICAR_MCAR_landmark = nx.shortest_path(nx_g, source=ICAR_node, target=MCAR_landmark, weight='dist')
    except:
        # Handle other exceptions that might occur
        raise ValueError("No path found from ICAR to MCAR landmark!")

    # MCAL #############################################################################
    # Slice nx_g for MCA node localisation by deleting contralateral ICA-MCA landmark SP, to avoid finding SPs from current MCA to bunch of contralateral ACA cone nodes
    nx_g_MCA = copy.deepcopy(nx_g)
    nx_g_MCA.remove_nodes_from(sp_ICAR_MCAR_landmark)

    # Find divergence nodes of SPs from ICAL to each ACA cone node
    MCAL_node = None
    divergence_nodes = {}
    divergence_nodes_x_pos = {}
    n_reachable_cone_nodes_left = 0
    for ACA_cone_node in ACA_cone_nodes:
        try:
            sp_ICAL_ACA_cone_node = nx.shortest_path(nx_g_MCA, source=ICAL_node, target=ACA_cone_node)
            divergence_node, divergence_type = find_divergence_node(sp_ICAL_MCAL_landmark, sp_ICAL_ACA_cone_node, do_verbose_errors)
            assert divergence_type not in ['straggler', 'identical'], 'xxx'
            ACA_cone_node_x_pos = nx_g.nodes[ACA_cone_node]['pos'][0]
            if divergence_node not in divergence_nodes:
                divergence_nodes[divergence_node] = 1
            else:
                divergence_nodes[divergence_node] += 1
            if divergence_node not in divergence_nodes_x_pos:
                divergence_nodes_x_pos[divergence_node] = [ACA_cone_node_x_pos]
            else:
                divergence_nodes_x_pos[divergence_node] += [ACA_cone_node_x_pos]
            n_reachable_cone_nodes_left += 1
        except:
            pass
            # raise ValueError("No path found from ICAL to ACA cone node!")
    divergence_nodes_x_pos_min = {k:min(v) for k, v in divergence_nodes_x_pos.items()}
    if len(divergence_nodes) > 0:
        # if len(divergence_nodes) > 1:
        #     print('Multiple divergence node candidates found. Most frequent one is selected.')
        # MCAL_node = max(divergence_nodes, key=divergence_nodes.get)

        # Select the divergence node with maximum number of ACA landmark supports. If tie, select the one with minimum x pos ACA landmark support to try escape from ophthalmic arteries reaching into ACA cone
        # MCAL_node = max(divergence_nodes, key=lambda k: (divergence_nodes[k], -divergence_nodes_x_pos_min[k]))

        # Select the divergence node with the minimum of the minima x-coord of the ACA landmark support cluster of each divergence node.
        # If tie, select the one with max number of support nodes
        MCAL_node = min(divergence_nodes_x_pos_min, key=lambda k: (divergence_nodes_x_pos_min[k], divergence_nodes[k]))
    # if len(divergence_nodes) > 1:
        # print('L', divergence_nodes)
        # print('L', divergence_nodes_x_pos_min)

    # MCAR #############################################################################
    # Slice nx_g for MCA node localisation by deleting contralateral MCA-ICA SP, to avoid finding SPs from current MCA to bunch of contralateral ACA cone nodes
    nx_g_MCA = copy.deepcopy(nx_g)
    nx_g_MCA.remove_nodes_from(sp_ICAL_MCAL_landmark)

    # Find divergence nodes of SPs from ICAR to each ACA cone node
    MCAR_node = None
    divergence_nodes = {}
    divergence_nodes_x_pos = {}
    n_reachable_cone_nodes_right = 0
    for ACA_cone_node in ACA_cone_nodes:
        try:
            sp_ICAR_ACA_cone_node = nx.shortest_path(nx_g_MCA, source=ICAR_node, target=ACA_cone_node)
            divergence_node, divergence_type = find_divergence_node(sp_ICAR_MCAR_landmark, sp_ICAR_ACA_cone_node, do_verbose_errors)
            assert divergence_type not in ['straggler', 'identical'], 'xxx'
            ACA_cone_node_x_pos = nx_g.nodes[ACA_cone_node]['pos'][0]
            if divergence_node not in divergence_nodes:
                divergence_nodes[divergence_node] = 1
            else:
                divergence_nodes[divergence_node] += 1
            if divergence_node not in divergence_nodes_x_pos:
                divergence_nodes_x_pos[divergence_node] = [ACA_cone_node_x_pos]
            else:
                divergence_nodes_x_pos[divergence_node] += [ACA_cone_node_x_pos]
            n_reachable_cone_nodes_right += 1
        except:
            pass
            # raise ValueError("No path found from ICAR to ACA cone node!")
    # print(f'xoxo xoxo xoxo {n_reachable_cone_nodes_left}, {n_reachable_cone_nodes_right}')
    divergence_nodes_x_pos_max = {k:max(v) for k, v in divergence_nodes_x_pos.items()}
    if len(divergence_nodes) > 0:
        # if len(divergence_nodes) > 1:
        #     print('Multiple divergence node candidates found. Most frequent one is selected.')
        # MCAR_node = max(divergence_nodes, key=divergence_nodes.get)

        # Select the divergence node with maximum ACA landmark support. If tie, select the one with minimum x pos ACA landmark support to try escape from ophthalmic arteries reaching into ACA cone
        # MCAR_node = max(divergence_nodes, key=lambda k: (divergence_nodes[k], divergence_nodes_x_pos_max[k]))

        # Select the divergence node with the maximum of the maxima x-coord of the ACA landmark support cluster of each divergence node.
        # If tie, select the one with max number of support nodes
        MCAR_node = max(divergence_nodes_x_pos_max, key=lambda k: (divergence_nodes_x_pos_max[k], divergence_nodes[k]))
    # if len(divergence_nodes) > 1:
    #     print('R', divergence_nodes)
    #     print('R', divergence_nodes_x_pos_max)

    # assert MCAL_node != None or MCAR_node != None, 'Both MCAs were not found!!!'
    #######################################
    # Sometimes when there is no connection from ICA to ACA cone nodes on one side (aka A1 segment absent) MCA not found
    # Use symetry to define an MCA phantom point on contralateral side with missing MCA.
    # Of all nodes along SP from ICA to MCA_landmark on missing MCA side, select one with min euclidean distance to the phantom as the MCA
    contralateral_MCA_phantom_pos = []
    if MCAL_node != None and MCAR_node == None:
        MCAL_pos = nx_g.nodes[MCAL_node]['pos']
        MCAL_to_centroid_x_distance = abs(MCAL_pos[0] - centroid_ICAL_ICAR[0])
        contralateral_MCA_phantom_pos = [centroid_ICAL_ICAR[0] - MCAL_to_centroid_x_distance, MCAL_pos[1], MCAL_pos[2]]
        # try:
        #     sp_ICAR_MCAR_landmark = nx.shortest_path(nx_g, source=ICAR_node, target=MCAR_landmark)
        # except:
        #     # Handle other exceptions that might occur
        #     raise ValueError("No path found from ICAR to MCAR landmark!")
        assert len(sp_ICAR_MCAR_landmark) > 2, 'xxx'
        min_distance_to_phantom = float('inf')
        MCAR_node = None
        for MCAR_candidate in sp_ICAR_MCAR_landmark[1:]:
            MCAR_candidate_pos = nx_g.nodes[MCAR_candidate]['pos']
            curr_distance_to_phantom = distance(contralateral_MCA_phantom_pos, MCAR_candidate_pos)
            if curr_distance_to_phantom < min_distance_to_phantom:
                min_distance_to_phantom = curr_distance_to_phantom
                MCAR_node = MCAR_candidate
        if do_verbose_errors:
            print('\t Notification: Found an MCAR using symmetrical phantom.')
    elif MCAR_node != None and MCAL_node == None:
        MCAR_pos = nx_g.nodes[MCAR_node]['pos']
        MCAR_to_centroid_x_distance = abs(MCAR_pos[0] - centroid_ICAL_ICAR[0])
        contralateral_MCA_phantom_pos = [centroid_ICAL_ICAR[0] + MCAR_to_centroid_x_distance, MCAR_pos[1], MCAR_pos[2]]
        # try:
        #     sp_ICAL_MCAL_landmark = nx.shortest_path(nx_g, source=ICAL_node, target=MCAL_landmark)
        # except:
        #     # Handle other exceptions that might occur
        #     raise ValueError("No path found from ICAL to MCAL landmark!")
        assert len(sp_ICAL_MCAL_landmark) > 2, 'xxx'
        min_distance_to_phantom = float('inf')
        MCAL_node = None
        for MCAL_candidate in sp_ICAL_MCAL_landmark[1:]:
            MCAL_candidate_pos = nx_g.nodes[MCAL_candidate]['pos']
            curr_distance_to_phantom = distance(contralateral_MCA_phantom_pos, MCAL_candidate_pos)
            if curr_distance_to_phantom < min_distance_to_phantom:
                min_distance_to_phantom = curr_distance_to_phantom
                MCAL_node = MCAL_candidate
        if do_verbose_errors:
            print('\t Notification: Found an MCAL using symmetrical phantom.')
    # else:
    #     raise ValueError('xxx')

    do_continue = False
    if MCAR_node == None or MCAL_node == None:
        # if cohort == 'UNC' and c == 30:
        #     node_labelling_predictions[3] = 1
        #     node_labelling_predictions[4] = 90
        #     node_labelling_predictions[5] = 31
        #     node_labelling_predictions[8] = 12
        #     node_labelling_predictions[9] = 13
        #     node_labelling_predictions[11] = 30
        output_curr_node_labelling_predictions(tag, c, node_labelling_predictions, node_labelling_output_path, do_output_node_labelling_predictions)
        if not do_suppress_skips:
            print('\t SKIPPING, Either MCA Node Not Found XXXXXXXXXXXXXXXXXXXXXXXXXXXX')
            print('\t SKIPPING, Either MCA Node Not Found XXXXXXXXXXXXXXXXXXXXXXXXXXXX')
            print('\t SKIPPING, Either MCA Node Not Found XXXXXXXXXXXXXXXXXXXXXXXXXXXX')
            print('\t SKIPPING, Either MCA Node Not Found XXXXXXXXXXXXXXXXXXXXXXXXXXXX')
            print('\t SKIPPING, Either MCA Node Not Found XXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        # raise ValueError('There must be at least one MCA node')
        c += 1
        do_continue = True
        return MCAL_node, MCAR_node, contralateral_MCA_phantom_pos, do_continue, c

    return MCAL_node, MCAR_node, contralateral_MCA_phantom_pos, do_continue, c

# XAXA done Check for MCA cycles and adjsut MCA solution to be the most peripheral node in cycle, store cycle for later nx_g_BAB construction
def adjust_for_MCA_cycles(nx_g, MCAL_node, MCAR_node, do_verbose_errors):
    cycles = nx.cycle_basis(nx_g)
    MCA_cycle_nodes = []
    for cycle in cycles:
        if len(cycle) < 6:
            if MCAL_node in cycle:
                max_x_node = max(cycle, key=lambda node: nx_g.nodes[node]['pos'][0])
                MCAL_node = max_x_node
                MCA_cycle_nodes += cycle
                if do_verbose_errors:
                    print('\t Warning: MCAL lies on a (likely non-CoW) cycle')
            if MCAR_node in cycle:
                min_x_node = min(cycle, key=lambda node: nx_g.nodes[node]['pos'][0])
                MCAR_node = min_x_node
                MCA_cycle_nodes += cycle
                if do_verbose_errors:
                    print('\t Warning: MCAR lies on a (likely non-CoW) cycle')
    return MCAL_node, MCAR_node, MCA_cycle_nodes

def get_MCA_node_params(nx_g, MCAL_node, MCAR_node):
    MCA_nodes_pos_attributes = [nx_g.nodes[node]['pos'] for node in [MCAL_node, MCAR_node] if node != None]
    MCAL_pos = None
    MCAR_pos = None
    if MCAL_node != None:
        MCAL_pos = nx_g.nodes[MCAL_node]['pos']
        # assert ICAL_pos[0] > MCAL_pos[0], f'{f}\n{MCA_nodes_pos_attributes[0][0]}\n{MCAL_pos[0]}'
    else:
        raise ValueError('No MCAL found !!!')
        # output_curr_node_labelling_predictions(tag, c, node_labelling_predictions, node_labelling_output_path,
        #                                        do_output_node_labelling_predictions)
        # print('\t Warning: No MCAL found !!! Skipping')
        # c += 1
        # continue

    if MCAR_node != None:
        MCAR_pos = nx_g.nodes[MCAR_node]['pos']
        # assert ICAR_pos[0] < MCAR_pos[0], f'{f}\n{MCA_nodes_pos_attributes[1][0]}\n{MCAR_pos[0]}'
    else:
        raise ValueError('No MCAL found !!!')
        # output_curr_node_labelling_predictions(tag, c, node_labelling_predictions, node_labelling_output_path,
        #                                        do_output_node_labelling_predictions)
        # print('\t Warning: No MCAR found !!! Skipping')
        # c += 1
        # continue
    return MCAL_pos, MCAR_pos, MCA_nodes_pos_attributes
    # MCAL_GT = [node for node, data in nx_g.nodes(data=True) if data.get('boitype') == 3]
    # MCAR_GT = [node for node, data in nx_g.nodes(data=True) if data.get('boitype') == 4]
    # if len(MCAL_GT) == 0:
    #     print(tag, 'No MCAL in GT !!!')
    # elif len(MCAL_GT) > 1:
    #     print(tag, 'Multiple MCAL in GT !!!')
    #
    # if len(MCAR_GT) == 0:
    #     print(tag, 'No MCAR in GT !!!')
    # elif len(MCAR_GT) > 1:
    #     print(tag, 'Multiple MCAR in GT !!!')

def legacy_find_MCA_cone_nodes(nx_g_ACA, MCAL_node, MCAR_node, MCAL_pos, MCAR_pos, ICAL_pos, ICAR_pos):

    if MCAL_node != None:
        MCAL_pos = np.array(MCAL_pos)
    if MCAR_node != None:
        MCAR_pos = np.array(MCAR_pos)

    if (MCAL_node != None) and (MCAR_node != None):

        # Cone radius
        MCAL_MCAR_distance = np.linalg.norm(MCAR_pos - MCAL_pos)
        cone_radius = MCAL_MCAR_distance / 2

        # Calculate the direction vector from tip to base
        MCAL_MCAR_midpoint = (MCAL_pos + MCAR_pos) / 2
        MCAL_to_MCA_center_vector = MCAL_MCAR_midpoint - MCAL_pos
        MCAR_to_MCA_center_vector = MCAL_MCAR_midpoint - MCAR_pos

        # Iterate through the nodes and check if they are within the cone
        MCA_cone_nodes = []
        for MCA_cone_node_cand, data in nx_g_ACA.nodes(data=True):
            node_x_pos = data['pos'][0]
            node_is_between_MCAs = node_x_pos < MCAL_pos[0] and node_x_pos > MCAR_pos[0]
            if node_is_between_MCAs:
                node_position = data['pos']
                vector_MCAL_to_node = node_position - MCAL_pos
                vector_MCAR_to_node = node_position - MCAR_pos

                # Calculate the cosine of the angle between the direction vector and the vector to each node
                angle_cosine_to_MCAL = np.dot(vector_MCAL_to_node, MCAL_to_MCA_center_vector) / (
                        np.linalg.norm(vector_MCAL_to_node) * np.linalg.norm(MCAL_to_MCA_center_vector))
                angle_cosine_to_MCAR = np.dot(vector_MCAR_to_node, MCAR_to_MCA_center_vector) / (
                        np.linalg.norm(vector_MCAR_to_node) * np.linalg.norm(MCAR_to_MCA_center_vector))

                # Calculate the angle in radians and compare it to the cone angle
                angle_MCAL = math.acos(angle_cosine_to_MCAL)
                angle_MCAR = math.acos(angle_cosine_to_MCAR)
                if angle_MCAL <= math.atan(cone_radius / np.linalg.norm(MCAL_to_MCA_center_vector)) \
                        or angle_MCAR <= math.atan(cone_radius / np.linalg.norm(MCAR_to_MCA_center_vector)):
                    MCA_cone_nodes.append(MCA_cone_node_cand)

    else:
        kn1_kn2_distance = np.linalg.norm(ICAR_pos - ICAL_pos)
        print('XXXXXXXXXXXX')

    MCA_cone_nodes_pos_attributes = [nx_g_ACA.nodes[node]['pos'] for node in MCA_cone_nodes]

    return MCA_cone_nodes, MCA_cone_nodes_pos_attributes

# ACA functions #####################################################################
def find_ACA_cone_nodes(nx_g, ICAL_pos, ICAR_pos, centroid_ICAL_ICAR, centroid_ICAL_ICAR_at_centroid_glob_zpos, cone_radius_multiplier):
    min_ypos = min(data['pos'][1] for _, data in nx_g.nodes(data=True) if 'pos' in data)
    max_zpos = max(data['pos'][2] for _, data in nx_g.nodes(data=True) if 'pos' in data)
    ACA_cone_base = [centroid_ICAL_ICAR[0], min_ypos, max_zpos]

    # Cone radius
    kn1_kn2_x_distance = abs(ICAL_pos[0] - ICAR_pos[0])
    ACA_cone_radius = kn1_kn2_x_distance * cone_radius_multiplier
    # print(ACA_cone_radius)

    # Calculate the direction vector from tip to base, and the angle spanned by the cone
    com_anchor_to_ACA_cone_base_vector = ACA_cone_base - centroid_ICAL_ICAR_at_centroid_glob_zpos
    cone_angle = math.atan(ACA_cone_radius / np.linalg.norm(com_anchor_to_ACA_cone_base_vector))

    # Iterate through the nodes and check if they are within the cone
    ACA_cone_nodes = []
    for node, data in nx_g.nodes(data=True):
        if 'pos' in data:
            node_position = data['pos']
            vector_to_node = node_position - centroid_ICAL_ICAR_at_centroid_glob_zpos

            # Calculate the cosine of the angle between the direction vector and the vector to each node
            angle_cosine = np.dot(vector_to_node, com_anchor_to_ACA_cone_base_vector) / (
                    np.linalg.norm(vector_to_node) * np.linalg.norm(com_anchor_to_ACA_cone_base_vector))

            # Calculate the angle of the cone node candidate in radians and compare it to the cone angle
            angle = math.acos(angle_cosine)
            if angle <= cone_angle:
                ACA_cone_nodes.append(node)

    ACA_cone_nodes_pos_attributes = [nx_g.nodes[node]['pos'] for node in ACA_cone_nodes if
                                     'pos' in nx_g.nodes[node]]

    return ACA_cone_nodes, ACA_cone_nodes_pos_attributes, ACA_cone_base, ACA_cone_radius

def at_least_one_ACA_cone_node_reachable_from_MCA_landmarks(nx_g, ACA_cone_nodes, MCAL_landmark, MCAR_landmark):
    for ACA_cone_node in ACA_cone_nodes:
        if nx.has_path(nx_g, ACA_cone_node, MCAL_landmark) or nx.has_path(nx_g, ACA_cone_node, MCAR_landmark):
            return True
    return False

def check_and_update_ACA_cone_nodes(nx_g, ACA_cone_nodes, ACA_cone_nodes_pos_attributes, ACA_cone_base, ACA_cone_radius, cone_radius_multiplier, exists_valid_ACA_cone_node, do_verbose_errors, c, tag,
                                    node_labelling_predictions, node_labelling_output_path, do_output_node_labelling_predictions,
                                    do_suppress_skips, ICAL_pos, ICAR_pos, centroid_ICAL_ICAR, centroid_ICAL_ICAR_at_centroid_glob_zpos, MCAL_landmark, MCAR_landmark):
    do_continue = False
    if ACA_cone_nodes != [] and not exists_valid_ACA_cone_node:
        print(f'\t WARNING: Found ACA cone nodes, but none are connected to either MCA landmark')
    if ACA_cone_nodes == [] or not exists_valid_ACA_cone_node:
        if not exists_valid_ACA_cone_node:
            if do_verbose_errors:
                print('\t Notification: No ACA landmarks found !!! Initiating cone expansion phase')
                # print(tag, '11111111111111111111111111111111')
                # print(tag, '11111111111111111111111111111111')
                # print(tag, '11111111111111111111111111111111')
                # print(tag, '11111111111111111111111111111111')
                # print(tag, '11111111111111111111111111111111')
        # cone expansion phase
        # cone_radius_expansion_factors = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
        cone_radius_expansion_factors = np.arange(1.1, 5.1, 0.1)
        for cone_radius_expansion_factor in cone_radius_expansion_factors:
            if do_verbose_errors:
                print(f'\t Notification: Current expansion factor {cone_radius_expansion_factor}')
            cone_radius_multiplier_expanded = cone_radius_multiplier * cone_radius_expansion_factor
            ACA_cone_nodes, ACA_cone_nodes_pos_attributes, ACA_cone_base, ACA_cone_radius = find_ACA_cone_nodes(nx_g, ICAL_pos, ICAR_pos, centroid_ICAL_ICAR, centroid_ICAL_ICAR_at_centroid_glob_zpos, cone_radius_multiplier_expanded)
            contains_valid_node = at_least_one_ACA_cone_node_reachable_from_MCA_landmarks(ACA_cone_nodes, MCAL_landmark, MCAR_landmark)
            if ACA_cone_nodes != [] and contains_valid_node:
                break
        if ACA_cone_nodes == [] or not contains_valid_node:
            print('WARNING: no ACA cone nodes found. GRAPH SKIPPED')
            # plot aca cone on raw html
            # if do_save_network_graph:
            #     edges = np.zeros((nx_g.number_of_edges(), 12), dtype=np.float32)
            #     # edges = np.zeros((nx_g_BAT.number_of_edges(), 12), dtype=np.float32)
            #     # edges = np.zeros((nx_g_ACA.number_of_edges(), 12), dtype=np.float32)
            #     # edges = np.zeros((nx_g_PCA.number_of_edges(), 12), dtype=np.float32)
            #     lookup_coord_class = {}
            #     for i, edge in enumerate(nx_g.edges):
            #         # for i, edge in enumerate(nx_g_BAT.edges):
            #         # for i, edge in enumerate(nx_g_ACA.edges):
            #         # for i, edge in enumerate(nx_g_PCA.edges):
            #         u = nx_g.nodes[edge[0]]
            #         v = nx_g.nodes[edge[1]]
            #         # u = nx_g_BAT.nodes[edge[0]]
            #         # v = nx_g_BAT.nodes[edge[1]]
            #         # u = nx_g_ACA.nodes[edge[0]]
            #         # v = nx_g_ACA.nodes[edge[1]]
            #         # u = nx_g_PCA.nodes[edge[0]]
            #         # v = nx_g_PCA.nodes[edge[1]]
            #
            #         if html_type == "coloured_nodes_only":
            #             u_coord = (u['pos'][0], u['pos'][1], u['pos'][2])
            #             v_coord = (v['pos'][0], v['pos'][1], v['pos'][2])
            #         else:
            #             u_coord = (int(u['pos'][0]), int(u['pos'][1]), int(u['pos'][2]))
            #             v_coord = (int(v['pos'][0]), int(v['pos'][1]), int(v['pos'][2]))
            #
            #         if u_coord not in lookup_coord_class:
            #             lookup_coord_class[u_coord] = [int(u['boitype']), edge[0]]
            #             # raise ValueError('class should be from indices not from GT labels...')
            #         if v_coord not in lookup_coord_class:
            #             lookup_coord_class[v_coord] = [int(v['boitype']), edge[1]]
            #
            #         edges[i, 3] = u_coord[0]
            #         edges[i, 4] = u_coord[1]
            #         edges[i, 5] = u_coord[2]
            #         edges[i, 6] = v_coord[0]
            #         edges[i, 7] = v_coord[1]
            #         edges[i, 8] = v_coord[2]
            #         edges[i, 9] = nx_g.edges[edge[0], edge[1]]['dist']
            #         edges[i, 10] = nx_g.edges[edge[0], edge[1]]['rad']
            #         # print(u_coord, v_coord)
            # new_tag = f'{c}_{tag}'
            #
            # PCAL_edge_landmark_pos = []
            # PCAR_edge_landmark_pos = []
            # BAB_candidates_pos_attributes = []
            # MCA_landmark_pos_attributes = []
            # MCA_nodes_pos_attributes = []
            # ACA_nodes_pos_attributes = []
            # PComA_anterior_nodes_pos_attributes = []
            # PComA_posterior_nodes_pos_attributes = []
            # PCAL_landmarks_pos_attributes = []
            # PCAR_landmarks_pos_attributes = []
            # PCAL_landmarks_sphere = []
            # PCAR_landmarks_sphere = []
            # PCAL_landmarks_ellipsoid = [0, 0, 0, 0, 0, 0]
            # PCAR_landmarks_ellipsoid = [0, 0, 0, 0, 0, 0]
            # PCAL_anchor_pos_attributes = []
            # PCAR_anchor_pos_attributes = []
            # BAT_node_pos_attributes = []
            # BAB_node_pos_attributes = []
            # BAB_cylinder_params = [[0, 0, 0], 0, 0, 0, [0, 0, 0], [0, 0, 0]]
            # contralateral_MCA_phantom_pos = []
            # PCAL_edge_pos = []
            # PCAR_edge_pos = []
            # html_path = f'/hpc/jshe690/jshe690/Desktop/Jiantao/tensorflow-tutorial/GNNART/graph/htmls/{cohort}/Chen et al UNC graphs/{c}_{tag}.html'
            # save_network_graph(edges, 23, lookup_coord_class,
            #                    html_path, new_tag, html_type, fiducial_nodes=[centroid_glob],
            #                    fiducial_nodes2=[centroid_ICAL_ICAR, centroid_ICAL_ICAR_at_centroid_glob_zpos, ACA_cone_base,
            #                                     PCAL_edge_landmark_pos, PCAR_edge_landmark_pos],
            #                    fiducial_nodes3=BAB_candidates_pos_attributes + ACA_cone_nodes_pos_attributes,
            #                    MCA_landmarks=MCA_landmark_pos_attributes, MCA=MCA_nodes_pos_attributes,
            #                    ACA=ACA_nodes_pos_attributes, PComA_anterior=PComA_anterior_nodes_pos_attributes,
            #                    PComA_posterior=PComA_posterior_nodes_pos_attributes,
            #                    PCAL_landmarks=PCAL_landmarks_pos_attributes, PCAR_landmarks=PCAR_landmarks_pos_attributes,
            #                    PCAL_landmarks_sphere=PCAL_landmarks_sphere, PCAR_landmarks_sphere=PCAR_landmarks_sphere,
            #                    PCAL_landmarks_ellipsoid=PCAL_landmarks_ellipsoid,
            #                    PCAR_landmarks_ellipsoid=PCAR_landmarks_ellipsoid,
            #                    PCAL_anchor=PCAL_anchor_pos_attributes, PCAR_anchor=PCAR_anchor_pos_attributes,
            #                    BAT_node=BAT_node_pos_attributes, BAB_node=BAB_node_pos_attributes,
            #                    BAB_cylinder=BAB_cylinder_params,
            #                    ACA_cone=[centroid_ICAL_ICAR_at_centroid_glob_zpos, ACA_cone_base, ACA_cone_radius],
            #                    MCA_phantom=[contralateral_MCA_phantom_pos], PCAL_edge=PCAL_edge_pos,
            #                    PCAR_edge=PCAR_edge_pos,
            #                    PCAL_edge_landmark=PCAL_edge_landmark_pos, PCAR_edge_landmark=PCAR_edge_landmark_pos,
            #                    ICA=[ICAL_pos, ICAR_pos])
            output_curr_node_labelling_predictions(tag, c, node_labelling_predictions, node_labelling_output_path, do_output_node_labelling_predictions)
            if not do_suppress_skips:
                # if cohort == 'UNC' and c == 10:
                #     node_labelling_predictions[3] = 2
                #     node_labelling_predictions[4] = 8
                #     node_labelling_predictions[5] = 26
                #     node_labelling_predictions[6] = 39
                #     node_labelling_predictions[8] = 131
                #     # node_labelling_predictions[9] = 13
                #     # node_labelling_predictions[11] = 30
                # if cohort == 'UNC' and c == 12:
                #     node_labelling_predictions[3] = 10
                #     node_labelling_predictions[4] = 25
                #     # node_labelling_predictions[5] =
                #     # node_labelling_predictions[6] =
                #     node_labelling_predictions[8] = 15
                #     node_labelling_predictions[9] = 16
                #     node_labelling_predictions[10] = 32
                #     node_labelling_predictions[11] = 10
                #     node_labelling_predictions[12] = 24
                # if cohort == 'UNC' and c == 14:
                #     node_labelling_predictions[3] = 80
                #     node_labelling_predictions[4] = 2
                #     node_labelling_predictions[5] = 93
                #     # node_labelling_predictions[6] =
                #     node_labelling_predictions[8] = 142
                #     # node_labelling_predictions[9] = 16
                #     # node_labelling_predictions[10] = 32
                #     # node_labelling_predictions[11] = 10
                #     # node_labelling_predictions[12] = 24
                print('\t Warning: No ACA landmarks found even after cone exansion phase!!! Skipping')
            c += 1
            do_continue = True
            return [], [], [], [], do_continue, c
        # else:
        #     print(f'\t Warning: Found ACA landmarks found via expansion phase with radius_expansion_factor {radius_expansion_factor}.')
    # print("Nodes within the cone:", ACA_cone_nodes_pos_attributes)
    do_continue = False
    return ACA_cone_nodes, ACA_cone_nodes_pos_attributes, ACA_cone_base, ACA_cone_radius, do_continue, c

def gen_nx_g_ACA(nx_g, SP_ICAL_MCAL_landmark, SP_ICAR_MCAR_landmark, MCAL_node, MCAR_node):
    # Slice the network to disconnect MCA to MCA connection via posterior circulation
    #     MCA_max_y = max(MCAL_pos[1], MCAR_pos[1])
    # elif MCAR_node != None:
    #     MCA_max_y = MCAR_pos[1]
    # elif MCAL_node != None:
    #     MCA_max_y = MCAL_pos[1]
    # else:
    #     raise ValueError('There must be at least one MCA node')
    # MCA_max_y = (MCA_max_y + centroid_glob[1]) / 2 # Anzhen35 kn5 kn6 is posterior to MCA... but better not do this as some posterior loops might slip through.
    # Disconnect posterior cirulation from anterior by deleting all nodes along ICA-MCA path except for the MCA node !!! fails if PComA_anterior is the MCA node! potential fix is to

    nx_g_ACA = copy.deepcopy(nx_g)
    ICA_MCA_landmark_SP_minus_MCA = [n for n in SP_ICAL_MCAL_landmark + SP_ICAR_MCAR_landmark if n not in [MCAL_node, MCAR_node]]
    nodes_to_remove = ICA_MCA_landmark_SP_minus_MCA
    # nodes_to_remove = []
    # for node in list(nx_g_ACA.nodes()):
        # pos = nx_g_ACA.nodes[node]['pos']
        # if pos[1] > MCA_max_y:
        #     nodes_to_remove += [node]
    nx_g_ACA.remove_nodes_from(nodes_to_remove)
    return nx_g_ACA

def find_ACAL_candidates(nx_g, nx_g_ACA, ACA_cone_nodes, MCAL_node, MCAL_pos, SP_ICAR_MCAR_landmark):

    # No ACA node if no ipsi MCA node
    if MCAL_node == None:
        ACAL_candidates3 = []
    else:
        # ACA is an immediate neighbour of MCA. Using nx_g_ACA here so no ICA-MCA nodes will be candidates
        MCAL_neighbours = list(nx.neighbors(nx_g_ACA, MCAL_node))

        # ACA necessarily between two MCAs in x direction
        MCAL_x_pos = MCAL_pos[0]
        ACAL_candidates = [n for n in MCAL_neighbours if nx_g_ACA.nodes[n]['pos'][0] < MCAL_x_pos]

        # ACA necessarily on at least one path from MCA to ACA cone nodes
        # The path cannot be via posterior circulation
        nx_g_copy = copy.deepcopy(nx_g)
        nx_g_copy.remove_nodes_from(SP_ICAR_MCAR_landmark)
        ACAL_candidates2 = []
        for ACAL_cand in ACAL_candidates:
            for ACA_cone_node in ACA_cone_nodes:
                try:
                    sp_MCAL_ACA_cone_node = nx.shortest_path(nx_g_copy, source=MCAL_node, target=ACA_cone_node)
                    if ACAL_cand in sp_MCAL_ACA_cone_node:
                        ACAL_candidates2 += [ACAL_cand]
                        break
                except:
                    pass

        if len(ACAL_candidates2) == 0:
            ACAL_candidates3 = []
        elif len(ACAL_candidates2) == 1:
            ACAL_candidates3 = ACAL_candidates2
        elif len(ACAL_candidates2) >= 2:
            min_x = np.inf
            ACAL_candidates3 = []
            for ACAL_cand in ACAL_candidates2:
                x_pos = nx_g_ACA.nodes[ACAL_cand]['pos'][0]
                if x_pos < min_x:
                    ACAL_candidates3 = [ACAL_cand]
                    min_x = x_pos

    ACAL_candidates_pos_attributes = [nx_g_ACA.nodes[node]['pos'] for node in ACAL_candidates3]

    return ACAL_candidates3, ACAL_candidates_pos_attributes

def find_ACAR_candidates(nx_g, nx_g_ACA, ACA_cone_nodes, MCAR_node, MCAR_pos, SP_ICAL_MCAL_landmark):

    if MCAR_node == None:
        ACAR_candidates3 = []
    else:
        # ACA is an immediate neighbour of MCA. Using nx_g_ACA here so no ICA-MCA nodes will be candidates
        MCAR_neighbours = list(nx.neighbors(nx_g_ACA, MCAR_node))

        # ACA necessarily between two MCAs
        MCAR_x_pos = MCAR_pos[0]
        ACAR_candidates = [n for n in MCAR_neighbours if nx_g_ACA.nodes[n]['pos'][0] > MCAR_x_pos]

        # ACA necessarily on at least one path from MCA to ACA cone nodes
        # The path cannot be via posterior circulation
        nx_g_copy = copy.deepcopy(nx_g)
        nx_g_copy.remove_nodes_from(SP_ICAL_MCAL_landmark)
        ACAR_candidates2 = []
        for ACAR_cand in ACAR_candidates:
            for ACA_cone_node in ACA_cone_nodes:
                try:
                    sp_MCAR_ACA_cone_node = nx.shortest_path(nx_g_copy, source=MCAR_node, target=ACA_cone_node)
                    if ACAR_cand in sp_MCAR_ACA_cone_node:
                        ACAR_candidates2 += [ACAR_cand]
                        break
                except:
                    pass

        # If multiple ACA cands, potentially due to recurrent artery of Huebner, which generally falls short of the ACA1/ACA2 junction
        if len(ACAR_candidates2) == 0:
            ACAR_candidates3 = []
        elif len(ACAR_candidates2) == 1:
            ACAR_candidates3 = ACAR_candidates2
        elif len(ACAR_candidates2) >= 2:
            max_x = 0
            ACAR_candidates3 = []
            for ACAR_cand in ACAR_candidates2:
                x_pos = nx_g_ACA.nodes[ACAR_cand]['pos'][0]
                if x_pos > max_x:
                    ACAR_candidates3 = [ACAR_cand]
                    max_x = x_pos

    ACAR_candidates_pos_attributes = [nx_g_ACA.nodes[node]['pos'] for node in ACAR_candidates3]

    return ACAR_candidates3, ACAR_candidates_pos_attributes

def find_ACA_node_from_MCA(nx_g, nx_g_ACA, ACA_cone_nodes, MCAL_node, MCAL_pos, MCAR_node, MCAR_pos, dist_thresholds_dict, shortest_path_ICAL_MCAL_landmark, shortest_path_ICAR_MCAR_landmark, direction, do_verbose_errors):
    # NO AComA connection!
    if direction == 'left':
        ACA_candidates, ACA_candidates_pos_attributes = find_ACAL_candidates(nx_g, nx_g_ACA, ACA_cone_nodes, MCAL_node, MCAL_pos, shortest_path_ICAR_MCAR_landmark)
    elif direction == 'right':
        ACA_candidates, ACA_candidates_pos_attributes = find_ACAR_candidates(nx_g, nx_g_ACA, ACA_cone_nodes, MCAR_node, MCAR_pos, shortest_path_ICAL_MCAL_landmark)
    else:
        raise ValueError('xxx')

    assert len(ACA_candidates) in [0, 1], 'xxx'
    if ACA_candidates:
        ACA_candidate = ACA_candidates[0]
        if direction == 'left':
            ACA_candidate_dist = calc_dist_along_SP(nx_g_ACA, MCAL_node, ACA_candidate)
            if do_verbose_errors:
                if ACA_candidate_dist <= dist_thresholds_dict['ACAL1_dist_thresh_lower']:
                    print(f'\t WARNING: The ACA candidate is less than dist thresh lower!')
            if ACA_candidate_dist >= dist_thresholds_dict['ACAL1_dist_thresh_lower'] and ACA_candidate_dist <= dist_thresholds_dict['ACAL1_dist_thresh_upper']:
                return ACA_candidate, True
            else:
                return None, True
        elif direction == 'right':
            ACA_candidate_dist = calc_dist_along_SP(nx_g_ACA, MCAR_node, ACA_candidate)
            if do_verbose_errors:
                if ACA_candidate_dist <= dist_thresholds_dict['ACAR1_dist_thresh_lower']:
                    print(f'\t WARNING: The ACA candidate is less than dist thresh lower!')
            if ACA_candidate_dist >= dist_thresholds_dict['ACAR1_dist_thresh_lower'] and ACA_candidate_dist <= dist_thresholds_dict['ACAR1_dist_thresh_upper']:
                return ACA_candidate, True
            else:
                return None, True
        else:
            raise ValueError('xxx')
    else:
        # No ACA node found, no candidates exist
        return None, False

def find_ACA_node_across_potential_hanging_AComA(nx_g_ACA, MCAL_node, MCAR_node, ACAL_node, ACAR_node, AComA_dist_thresh_upper, direction):
    # Potential hanging AComA connection!
    if direction == 'left':
        ACA_node = ACAL_node
    elif direction == 'right':
        ACA_node = ACAR_node
    else:
        raise ValueError('xxx')

    contralateral_ACA_candidates = []
    for neighbor in nx_g_ACA.neighbors(ACA_node):
        if neighbor not in [MCAL_node, MCAR_node, ACA_node]: # XAXA done (this function not used anymore) ACA_node added (self edge)
            dist = nx_g_ACA[ACA_node][neighbor]['dist']
            if dist < AComA_dist_thresh_upper:
                if len(list(nx_g_ACA.neighbors(neighbor))) > 1: # XAXA done (this function not used anymore) add this to paper
                    contralateral_ACA_candidates += [neighbor]

    if contralateral_ACA_candidates:
        print('\t Warning: Potential hanging AComA situation')
    # Of the contralateral ACA candidates on the hanging AComA connection, select the one consistent with L/R direction !!! potential edge case for weird reversed kn5/6 positioning
    # if contralateral_ACA_candidates:
    #     chosen_contralateral_ACA = None
    #     if direction == 'left':
    #         min_x_pos = np.inf
    #         for contralateral_ACA_candidate in contralateral_ACA_candidates:
    #             curr_x_pos = nx_g_ACA.nodes[contralateral_ACA_candidate]['pos'][0]
    #             if curr_x_pos < min_x_pos:
    #                 chosen_contralateral_ACA = contralateral_ACA_candidate
    #                 min_x_pos = curr_x_pos
    #     elif direction == 'right':
    #         max_x_pos = 0
    #         for contralateral_ACA_candidate in contralateral_ACA_candidates:
    #             curr_x_pos = nx_g_ACA.nodes[contralateral_ACA_candidate]['pos'][0]
    #             if curr_x_pos > max_x_pos:
    #                 chosen_contralateral_ACA = contralateral_ACA_candidate
    #                 max_x_pos = curr_x_pos
    #     assert chosen_contralateral_ACA != None, 'xxx'
    #     return chosen_contralateral_ACA
    # else:
    #     return None

def find_ACA_nodes(nx_g, nx_g_ACA, ACA_cone_nodes, MCAL_node, MCAR_node, MCAL_pos, MCAR_pos, dist_thresholds_dict, ICAL_pos, ICAR_pos, shortest_path_ICAL_MCAL_landmark, shortest_path_ICAR_MCAR_landmark, do_verbose_errors):

    ICAL_ICAR_distance = np.linalg.norm(np.array(ICAL_pos) - np.array(ICAR_pos))
    # ACA_candidate_neighbour_y_threshold = ICAL_ICAR_distance * ACA_coronal_validity_multiplier
    MCAL_MCAR_centroid = np.mean([MCAL_pos, MCAR_pos], axis=0)
    dist_along_SP_MCAL_MCAR_upper = dist_thresholds_dict['ACAL1_dist_thresh_upper'] + dist_thresholds_dict['ACAR1_dist_thresh_upper'] + dist_thresholds_dict['AComA_dist_thresh_upper']
    SP_MCAL_MCAR = []
    ACAL_node = None
    ACAR_node = None
    # Both MCA nodes exist, test for SP existence, otherwise run nonSP ACA module
    if MCAL_node != None and MCAR_node != None:
        # SP from MCAL_node to MCAR_node
        try:
            SP_MCAL_MCAR = nx.shortest_path(nx_g_ACA, source=MCAL_node, target=MCAR_node, weight='dist')
            dist_along_SP_MCAL_MCAR = 0
            for i in range(len(SP_MCAL_MCAR) - 1):
                u, v = SP_MCAL_MCAR[i], SP_MCAL_MCAR[i + 1]
                curr_dist = nx_g_ACA[u][v]['dist']
                dist_along_SP_MCAL_MCAR += curr_dist
            found_SP_MCAL_MCAR = True
        except:
            found_SP_MCAL_MCAR = False
        # try:
        #     SP_MCAL_MCAR2 = nx.shortest_path(nx_g_ACA, source=MCAL_node, target=MCAR_node, weight='dist')
        #     if SP_MCAL_MCAR != SP_MCAL_MCAR2:
        #         print('GXGXGXGXGXGXGXGXGXGXGX')
        #         print('GXGXGXGXGXGXGXGXGXGXGX')
        #         print('GXGXGXGXGXGXGXGXGXGXGX')
        #         print('GXGXGXGXGXGXGXGXGXGXGX')
        #         print('GXGXGXGXGXGXGXGXGXGXGX')
        # except:
        #     pass

        # If exist MCA SP, perform mass comparison to isolate ACA nodes
        if found_SP_MCAL_MCAR:
            exist_ACAL_cand = True
            exist_ACAR_cand = True
            assert len(SP_MCAL_MCAR) >= 3, 'ccc'
            # If the path between MCAL MCAR is too long, then there is either a False distal ACA connection or a true distal AComA
            if dist_along_SP_MCAL_MCAR > dist_along_SP_MCAL_MCAR_upper:
                ACAL_node, exist_ACAL_cand = find_ACA_node_from_MCA(nx_g, nx_g_ACA, ACA_cone_nodes, MCAL_node, MCAL_pos, MCAR_node, MCAR_pos, dist_thresholds_dict, shortest_path_ICAL_MCAL_landmark, shortest_path_ICAR_MCAR_landmark, 'left', do_verbose_errors)
                ACAR_node, exist_ACAR_cand = find_ACA_node_from_MCA(nx_g, nx_g_ACA, ACA_cone_nodes, MCAL_node, MCAL_pos, MCAR_node, MCAR_pos, dist_thresholds_dict, shortest_path_ICAL_MCAL_landmark, shortest_path_ICAR_MCAR_landmark, 'right', do_verbose_errors)
            else:
                if len(SP_MCAL_MCAR) == 3:  # only one node between MCAs. merged ACA node. ACAL = ACAR
                    if do_verbose_errors:
                        print('\t Notification: Found a merged ACA node')
                    return SP_MCAL_MCAR[1], SP_MCAL_MCAR[1], SP_MCAL_MCAR

                elif len(SP_MCAL_MCAR) == 4:  # two middle nodes defined as ACAs
                    return SP_MCAL_MCAR[1], SP_MCAL_MCAR[2], SP_MCAL_MCAR

                else:
                    if do_verbose_errors:
                        print('\t Warning: Multiple ACA L/R candidates along SP_MCAL_MCAR')
                    # node mass comparison
                    node_mass_dict = {}
                    # For each ACA_cand along the MCA sp exclusive of MCA nodes, calculate its mass
                    for ACA_cand in SP_MCAL_MCAR[1:-1]:
                        neighbours = list(nx.neighbors(nx_g_ACA, ACA_cand))
                        neighbours_not_on_sp = [n for n in neighbours if n not in SP_MCAL_MCAR]

                        # Bare ACA cand (node of degree 2 due to ACA nx_g
                        if len(neighbours_not_on_sp) == 0:
                            assert len(neighbours_not_on_sp) == 0, 'xxx'
                            assert ACA_cand not in node_mass_dict, 'xxx'
                            node_mass_dict[ACA_cand] = 0
                        else:
                            node_mass = 0
                            valid_nei = False
                            for nei in neighbours_not_on_sp:
                                nei_pos = nx_g_ACA.nodes[nei]['pos']
                                nei_mass = nx_g_ACA.edges[ACA_cand, nei]['rad']
                                # valid_nei = nei_pos[1] < (MCAL_MCAR_centroid[1] + ACA_candidate_neighbour_y_threshold) # this tries to exclude cases when the PComA attaches to A1 segment
                                valid_nei = nei_pos[1] < (MCAL_MCAR_centroid[1]) # this tries to exclude cases when the PComA attaches to A1 segment
                                if valid_nei and nei_mass > node_mass:
                                    node_mass = nei_mass
                                # nx_g_mass_calc = copy.deepcopy(nx_g_ACA)
                            # assert node_mass != 0, 'xxx'
                            assert ACA_cand not in node_mass_dict, 'xxx'
                            node_mass_dict[ACA_cand] = node_mass

                    assert len(node_mass_dict) >= 3, 'xxx'
                    node_mass_dict_sorted = dict(sorted(node_mass_dict.items(), key=lambda item: item[1], reverse=True))
                    eligible_ACA_nodes = list(node_mass_dict_sorted.keys())[:2]
                    # step along the SP from MCAL to MCAR again. first node encountered that falls in eligible nodes is MCAL, next is MCAR
                    ACAL_found = False
                    ACAR_found = False
                    for ACA_cand in SP_MCAL_MCAR[1:-1]:
                        if ACA_cand in eligible_ACA_nodes:
                            if ACAL_found:
                                ACAR_node = ACA_cand
                                ACAR_found = True
                            else:
                                ACAL_node = ACA_cand
                                ACAL_found = True
                    assert ACAL_found, 'xxx'
                    assert ACAR_found, 'xxx'

        else:
            # pass
            ACAL_node, exist_ACAL_cand = find_ACA_node_from_MCA(nx_g, nx_g_ACA, ACA_cone_nodes, MCAL_node, MCAL_pos, MCAR_node, MCAR_pos, dist_thresholds_dict, shortest_path_ICAL_MCAL_landmark, shortest_path_ICAR_MCAR_landmark, 'left', do_verbose_errors)
            ACAR_node, exist_ACAR_cand = find_ACA_node_from_MCA(nx_g, nx_g_ACA, ACA_cone_nodes, MCAL_node, MCAL_pos, MCAR_node, MCAR_pos, dist_thresholds_dict, shortest_path_ICAL_MCAL_landmark, shortest_path_ICAR_MCAR_landmark, 'right', do_verbose_errors)

    elif True: # always need both mca nodes. functions below not updated. not expected to be used
        raise ValueError('xxx')
    # if either of the MCA nodes are None, run nonSP ACA module
    elif MCAL_node != None:
        # pass
        exist_ACAR_cand = False
        ACAL_node, exist_ACAL_cand = find_ACA_node_from_MCA(nx_g_ACA, ACA_cone_nodes, MCAL_node, MCAL_pos, MCAR_node, MCAR_pos, ACAL1_dist_thresh_upper, ACAR1_dist_thresh_upper, shortest_path_ICAL_MCAL_landmark, shortest_path_ICAR_MCAR_landmark, 'left', do_verbose_errors)
    elif MCAR_node != None:
        # pass
        exist_ACAL_cand = False
        ACAR_node, exist_ACAR_cand = find_ACA_node_from_MCA(nx_g_ACA, ACA_cone_nodes, MCAL_node, MCAL_pos, MCAR_node, MCAR_pos, ACAL1_dist_thresh_upper, ACAR1_dist_thresh_upper, shortest_path_ICAL_MCAL_landmark, shortest_path_ICAR_MCAR_landmark, 'right', do_verbose_errors)
    else:
        raise ValueError('xxx')

    if do_verbose_errors:
        if not exist_ACAL_cand:
            print('\t KKKKK Warning: No feasible ACAL candidates found')
        if not exist_ACAR_cand:
            print('\t KKKKK Warning: No feasible ACAR candidates found')

    # ACA candidates can exist yet no ACA node is labelling due to A1/A2 continuation.
    # However, if contra ACA from nonCandExist side exists, try finding dual kn across Acoma
    # if not exist_ACAL_cand and not exist_ACAR_cand:  # !!! should this check existence of ACA_nodes instead?
    # something wrong with nx_g_ACA network slicing or truly no ACAs
    #   pass
    # elif ACAL_node != None and not exist_ACAR_cand:
    #     # missing ACAR coming from MCAR. check if dual kn5/6 situation at ACAL
    #     assert ACAR_node == None, 'xxx'
    #     ACAR_node = find_ACA_node_across_potential_hanging_AComA(nx_g_ACA, MCAL_node, MCAR_node, ACAL_node, ACAR_node, AComA_dist_thresh_upper, 'left')
    # elif ACAR_node != None and not exist_ACAL_cand:
    #     # missing ACAL coming from MCAL. check if dual kn5/6 situation at ACAR
    #     assert ACAL_node == None, 'xxx'
    #     ACAL_node = find_ACA_node_across_potential_hanging_AComA(nx_g_ACA, MCAL_node, MCAR_node, ACAL_node, ACAR_node, AComA_dist_thresh_upper, 'right')

    return ACAL_node, ACAR_node, SP_MCAL_MCAR

    # # Find divergence nodes of SPs from ICAL to each ACA cone node
    # MCAL_node = None
    # divergence_nodes = {}
    # for ACA_cone_node in ACA_cone_nodes:
    #     try:
    #         sp_ICAL_MCA_cone_node = nx.shortest_path(nx_g, source=ICAL_node, target=ACA_cone_node)
    #         divergence_node, divergence_type = find_divergence_node(sp_ICAL_MCAL_landmark, sp_ICAL_ACA_cone_node)
    #         assert divergence_type not in ['straggler', 'identical'], 'xxx'
    #         if divergence_node not in divergence_nodes:
    #             divergence_nodes[divergence_node] = 1
    #         else:
    #             divergence_nodes[divergence_node] += 1
    #     except:
    #         pass
    #         # raise ValueError("No path found from ICAL to ACA cone node!")
    # if len(divergence_nodes) > 0:
    #     # if len(divergence_nodes) > 1:
    #     #     print('Multiple divergence node candidates found. Most frequent one is selected.')
    #     MCAL_node = max(divergence_nodes, key=divergence_nodes.get)
    #
    # #######################################
    #
    # # SP from ICAR to MCAR landmark
    # try:
    #     sp_ICAR_MCAR_landmark = nx.shortest_path(nx_g, source=ICAR_node, target=MCAR_landmark)
    # except:
    #     # Handle other exceptions that might occur
    #     raise ValueError("No path found from ICAR to MCAR landmark!")
    #
    # # Find divergence nodes of SPs from ICAR to each ACA cone node
    # MCAR_node = None
    # divergence_nodes = {}
    # for ACA_cone_node in ACA_cone_nodes:
    #     try:
    #         sp_ICAR_ACA_cone_node = nx.shortest_path(nx_g, source=ICAR_node, target=ACA_cone_node)
    #         divergence_node, divergence_type = find_divergence_node(sp_ICAR_MCAR_landmark, sp_ICAR_ACA_cone_node)
    #         assert divergence_type not in ['straggler', 'identical'], 'xxx'
    #         if divergence_node not in divergence_nodes:
    #             divergence_nodes[divergence_node] = 1
    #         else:
    #             divergence_nodes[divergence_node] += 1
    #     except:
    #         pass
    #         # raise ValueError("No path found from ICAR to ACA cone node!")
    # if len(divergence_nodes) > 0:
    #     # if len(divergence_nodes) > 1:
    #     #     print('Multiple divergence node candidates found. Most frequent one is selected.')
    #     MCAR_node = max(divergence_nodes, key=divergence_nodes.get)
    #
    # return ACAL_node, ACAR_node

def get_ACA_node_params(nx_g, ACAL_node, ACAR_node):
    ACA_nodes_pos_attributes = [nx_g.nodes[node]['pos'] for node in [ACAL_node, ACAR_node] if node != None]
    ACAL_pos = None
    ACAR_pos = None
    if ACAL_node:
        ACAL_pos = nx_g.nodes[ACAL_node]['pos']
        # assert MCAL_pos[0] > ACAL_pos[0], f'{f}\n{ACA_nodes_pos_attributes[0][0]}\n{ACAL_pos[0]}'
    if ACAR_node:
        ACAR_pos = nx_g.nodes[ACAR_node]['pos']
        # assert MCAR_pos[0] < ACAR_pos[0], f'{f}\n{ACA_nodes_pos_attributes[1][0]}\n{ACAR_pos[0]}'
    return ACAL_pos, ACAR_pos, ACA_nodes_pos_attributes

def legacy_ACA_functions():
    pass
    # ACAL_candidates, ACAL_candidates_pos_attributes = find_ACAL_candidates(nx_g_ACA, ACA_cone_nodes, MCAL_node, MCAL_pos)
    # ACAR_candidates, ACAR_candidates_pos_attributes = find_ACAR_candidates(nx_g_ACA, ACA_cone_nodes, MCAR_node, MCAR_pos)

    # ACAL_GT = [node for node, data in nx_g.nodes(data=True) if data.get('boitype') == 5]
    # ACAR_GT = [node for node, data in nx_g.nodes(data=True) if data.get('boitype') == 6]
    #
    # if len(ACAL_GT) == 0:
    #     print(tag, 'No ACAL in GT !!!')
    # elif len(ACAL_GT) > 1:
    #     print(tag, 'Multiple ACAL in GT !!!')
    #
    # if len(ACAR_GT) == 0:
    #     print(tag, 'No ACAR in GT !!!')
    # elif len(ACAR_GT) > 1:
    #     print(tag, 'Multiple ACAR in GT !!!')

# BAB functions #####################################################################
def find_BAB_candidates_circular_cylinder(nx_g, ICAL_node, ICAL_pos, ICAR_node, ICAR_pos):
    def distance(point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    # cast ICA nodes to base plane
    ICAL_pos[2] = 0
    ICAR_pos[2] = 0

    vector_ICAL_ICAR = np.array(ICAR_pos) - np.array(ICAL_pos)
    vector_ICAL_ICAR = vector_ICAL_ICAR / np.linalg.norm(vector_ICAL_ICAR)

    perpendicular_vector = [vector_ICAL_ICAR[1], -vector_ICAL_ICAR[0], 0]
    perpendicular_length = np.linalg.norm(perpendicular_vector)
    perpendicular_vector = [perpendicular_vector[0] / perpendicular_length,
                            perpendicular_vector[1] / perpendicular_length]

    M = [(ICAL_pos[0] + ICAR_pos[0]) / 2, (ICAL_pos[1] + ICAR_pos[1]) / 2, 0]
    D = distance(ICAL_pos, ICAR_pos)
    cylinder_radius = D / 2 # Can be arbitrarily large and wont capture ICAs
    cylinder_height = D / 8 # D/4 too high; low z posterior circulation might get captured
    cylinder_center = [M[0] + cylinder_radius * perpendicular_vector[0],
                       M[1] + cylinder_radius * perpendicular_vector[1], 0]

    # Iterate through the nodes in the graph and check if they fall within the circle
    cylinder_nodes = []
    for node, attributes in nx_g.nodes(data=True):
        node_pos = attributes['pos']
        # Calculate the distance from the node to the axis of the cylinder (ignoring z-axis)
        distance_to_axis = distance([node_pos[0], node_pos[1]], [cylinder_center[0], cylinder_center[1]])
        # Check if the node is within the cylinder's radius and height
        if distance_to_axis <= cylinder_radius and abs(node_pos[2] - cylinder_center[2]) <= cylinder_height:
            cylinder_nodes.append(node)
    if len(cylinder_nodes) == 0:
        if do_verbose_errors:
            print('\t LLLLL No BAB cylinder nodes were found.')

    cylinder_nodes_pos_attributes = [nx_g.nodes[node]['pos'] for node in cylinder_nodes]
    cylinder_params = [cylinder_center, cylinder_height, cylinder_radius1, cylinder_radius2, cylinder_vector1, cylinder_vector2]

    return cylinder_nodes, cylinder_nodes_pos_attributes, cylinder_params

def find_BAB_candidates(nx_g, params_dict, ICAL_node, ICAL_pos, ICAR_node, ICAR_pos, MCAL_node, MCAR_node, MCA_cycle_nodes, ACAL_node, ACAR_node, MCAL_landmark, MCAR_landmark, SP_ICAL_MCAL_landmark, SP_ICAR_MCAR_landmark, do_verbose_errors):
    def distance(point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))
    def BAB_candidate_is_in_anterior_circulation(nx_g, BAB_candidate, ICAL_node, ICAR_node, MCAL_node, MCAR_node, MCA_cycle_nodes, ACAL_node, ACAR_node, MCAL_landmark, MCAR_landmark, SP_ICAL_MCAL_landmark, SP_ICAR_MCAR_landmark):
        # delete all nodes from ICA to MCA, excluding MCA. hoping to cut off posterior circulation. (!!!  if ever a bridge remains
        # between anterior and posterior circulations after this, the bridge should be touching A1 segment. so delete ICA-ACA paths then ICA-MCA paths)
        # if BAB candidate reachable from MCA still, then node is in anterior circulation
        nx_g_BAB = copy.deepcopy(nx_g)
        SP_ICAL_MCAL = find_shortest_path(nx_g, ICAL_node, MCAL_node)
        SP_ICAR_MCAR = find_shortest_path(nx_g, ICAR_node, MCAR_node)

        try:
            if ACAL_node != None:
                # must prevent SP from routing through contralateral circulation
                nx_g_test = copy.deepcopy(nx_g)
                nx_g_test.remove_nodes_from(SP_ICAR_MCAR_landmark + [ACAR_node])
                SP_MCAL_ACAL = nx.shortest_path(nx_g_test, source=MCAL_node, target=ACAL_node)
            else:
                SP_MCAL_ACAL = []
        except:
            SP_MCAL_ACAL = []

        try:
            if ACAR_node != None:
                # must prevent SP from routing through contralateral circulation
                nx_g_test = copy.deepcopy(nx_g)
                nx_g_test.remove_nodes_from(SP_ICAL_MCAL_landmark + [ACAL_node])
                SP_MCAR_ACAR = nx.shortest_path(nx_g_test, source=MCAR_node, target=ACAR_node)
            else:
                SP_MCAR_ACAR = []
        except:
            SP_MCAR_ACAR = []
        # nodes_to_remove = SP_ICAL_MCAL[:-1] + SP_ICAR_MCAR[:-1] + SP_MCAL_ACAL[1:-1] + SP_MCAR_ACAR[1:-1]
        nodes_to_remove = SP_ICAL_MCAL[:] + SP_ICAR_MCAR[:] + SP_MCAL_ACAL[:] + SP_MCAR_ACAR[:] + MCA_cycle_nodes # XAXA done changed to resolve MCA=poma ant situations!!!
        nx_g_BAB.remove_nodes_from(nodes_to_remove)

        try:
            # If any path is found between the BAB candidate and (hopefully) isolated anterior circulation, then BAB candidate is in anterior circulation
            SP_BAB_candidate_MCAL = nx.shortest_path(nx_g_BAB, source=BAB_candidate, target=MCAL_landmark)
            assert len(SP_BAB_candidate_MCAL) > 1, 'xxx'
            return True
        except:
            # no path found, possibly a posterior circulation onde
            pass

        try:
            # If any path is found between the BAB candidate and (hopefully) isolated anterior circulation, then BAB candidate is in anterior circulation
            SP_BAB_candidate_MCAR = nx.shortest_path(nx_g_BAB, source=BAB_candidate, target=MCAR_landmark)
            assert len(SP_BAB_candidate_MCAR) > 1, 'xxx'
            return True
        except:
            # no path found, possibly a posterior circulation onde
            pass

        return False
    def BAB_candidate_is_on_ICA(nx_g, BAB_candidate, ICAL_node, ICAR_node, MCAL_node, MCAR_node, ACAL_node, ACAR_node):
        # BAB cannot be an immediate neighbour of ICA or a node on SP ICA-MCA

        SP_ICAL_MCAL = find_shortest_path(nx_g, ICAL_node, MCAL_node)
        SP_ICAR_MCAR = find_shortest_path(nx_g, ICAR_node, MCAR_node)
        ICAL_neighbours = list(nx.neighbors(nx_g, ICAL_node))
        ICAR_neighbours = list(nx.neighbors(nx_g, ICAR_node))
        ICA_nodes = SP_ICAL_MCAL + SP_ICAR_MCAR + ICAL_neighbours + ICAR_neighbours
        if BAB_candidate in ICA_nodes:
            return True

        return False

    min_z = float('inf')
    for node, data in nx_g.nodes(data=True):
        if data['pos'][2] < min_z:
            min_z = data['pos'][2]

    # epsilon for radius comparison
    eps = 1e-4
    # cast ICA nodes to base plane
    ICAL_pos_copy = copy.deepcopy(ICAL_pos)
    ICAR_pos_copy = copy.deepcopy(ICAR_pos)
    ICAL_pos_copy[2] = min_z
    ICAR_pos_copy[2] = min_z

    vector_ICAL_ICAR = np.array(ICAR_pos_copy) - np.array(ICAL_pos_copy)
    vector_ICAL_ICAR = vector_ICAL_ICAR / np.linalg.norm(vector_ICAL_ICAR)

    perpendicular_vector = [vector_ICAL_ICAR[1], -vector_ICAL_ICAR[0], 0]
    perpendicular_length = np.linalg.norm(perpendicular_vector)
    perpendicular_vector = np.array([perpendicular_vector[0] / perpendicular_length,
                            perpendicular_vector[1] / perpendicular_length, 0])

    M = [(ICAL_pos_copy[0] + ICAR_pos_copy[0]) / 2, (ICAL_pos_copy[1] + ICAR_pos_copy[1]) / 2, min_z] # is this right? !!!
    D = distance(ICAL_pos_copy, ICAR_pos_copy)
    cylinder_radius1 = D * params_dict['cylinder_radius1_multiplier'] # Can be arbitrarily large and wont capture ICAs
    cylinder_radius2 = D * params_dict['cylinder_radius2_multiplier'] # Can be arbitrarily large and wont capture ICAs
    cylinder_height =  D * params_dict['cylinder_height_multiplier'] # D/4 too high; low z posterior circulation might get captured. is cylinder center right? !!!
    cylinder_center = np.array([M[0] + cylinder_radius1 * perpendicular_vector[0],
                       M[1] + cylinder_radius1 * perpendicular_vector[1], min_z])


    # Define the vectors v1 (minor axis) and v2 (major axis)
    v1 = perpendicular_vector
    v2 = vector_ICAL_ICAR

    # Calculate the transformation matrix to align the cylinder with the x-y plane
    G_normalized = np.cross(v1, v2)
    G_normalized = G_normalized / np.linalg.norm(G_normalized)
    transformation_matrix = np.array([v1, v2, G_normalized])

    # Find nodes in G that fall within the aligned elliptic cylinder
    cylinder_nodes = []
    for node, data in nx_g.nodes(data=True):
        # Transform the node's position to the aligned coordinate system
        pos = np.array(data['pos'])
        pos_transformed = np.dot(transformation_matrix, pos - cylinder_center)

        # Check if the transformed position is within the bounds of the aligned elliptic cylinder
        x, y, z = pos_transformed
        ellipse_check = (x ** 2 / (cylinder_radius1 ** 2)) + (y ** 2 / (cylinder_radius2 ** 2)) <= 1

        if ellipse_check and (min_z <= z <= cylinder_height):
        # if (-(cylinder_radius1 - eps) < x < (cylinder_radius1 - eps) and
        #     -(cylinder_radius2 - eps) < y < (cylinder_radius2 - eps) and
        #     min_z <= z <= cylinder_height):
            BAB_in_anterior = BAB_candidate_is_in_anterior_circulation(nx_g, node, ICAL_node, ICAR_node, MCAL_node, MCAR_node, MCA_cycle_nodes, ACAL_node, ACAR_node, MCAL_landmark, MCAR_landmark, SP_ICAL_MCAL_landmark, SP_ICAR_MCAR_landmark)
            BAB_on_ICA = BAB_candidate_is_on_ICA(nx_g, node, ICAL_node, ICAR_node, MCAL_node, MCAR_node, ACAL_node, ACAR_node)
            if not BAB_in_anterior and not BAB_on_ICA:
                cylinder_nodes.append(node)

    if len(cylinder_nodes) == 0:
        if do_verbose_errors:
            print('\t XXXXXXXXXXXXXXXXXX no BAB cylinder nodes found!')
            # print('\t XXXXXXXXXXXXXXXXXX no BAB cylinder nodes found!')
            # print('\t XXXXXXXXXXXXXXXXXX no BAB cylinder nodes found!')
            # print('\t XXXXXXXXXXXXXXXXXX no BAB cylinder nodes found!')
            # print('\t XXXXXXXXXXXXXXXXXX no BAB cylinder nodes found!')

    cylinder_nodes_pos_attributes = [nx_g.nodes[node]['pos'] for node in cylinder_nodes]
    cylinder_params = [cylinder_center, cylinder_height, cylinder_radius1, cylinder_radius2, v1, v2]

    # BAB_candidates = [14, 15]
    # BAB_candidates_pos_attributes = []
    # BAB_cylinder_params = [[50,50,50], 5, 5, 5, [0,0,1], [0,1,0]]

    return cylinder_nodes, cylinder_nodes_pos_attributes, cylinder_params

def find_BAB_node_using_divergence(nx_g, BAB_candidates, BAT_node):
    # Find SP from BAT to all BAB candidates. Of the subpath in common among all candidates, choose the furtherst one away from BAT
    shortest_SP_len = float('inf')
    BAT_to_BAB_SPs = []
    for BAB_candidate in BAB_candidates:
        try:
            SP_ICA_BAB_candidate = nx.shortest_path(nx_g, source=BAT_node, target=BAB_candidate)
            BAT_to_BAB_SPs += [SP_ICA_BAB_candidate]
            if len(SP_ICA_BAB_candidate) < shortest_SP_len:
                shortest_SP_len = len(SP_ICA_BAB_candidate)
        except:
            pass

    assert len(BAT_to_BAB_SPs) > 0, 'xxx'
    for SP in BAT_to_BAB_SPs:
        assert len(SP) > 1, 'Found a shortest path between BAT and BAB candidate that is only one node long (aka BAT is the BAB cnadidate?)'
        assert SP[0] == BAT_node, 'xxx'

    BAB_node = BAT_node
    for i in range(1, shortest_SP_len):
        BAB_candidate = None
        found_first_node_not_in_common = False
        for SP in BAT_to_BAB_SPs:
            if BAB_candidate == None:
                BAB_candidate = SP[i]
            else:
                if SP[i] == BAB_candidate:
                    pass
                else:
                    found_first_node_not_in_common = True
        if not found_first_node_not_in_common:
            BAB_node = BAB_candidate

    assert BAB_node != BAT_node, 'xxx'
    BAB_node_pos_attributes = [nx_g.nodes[node]['pos'] for node in [BAB_node] if node != None]

    return BAB_node, BAB_node_pos_attributes

def find_BAB_node_as_immediate_neighbour_of_BAT_node(nx_g, BAB_candidates, BAT_node, centroid_ICAL_ICAR, do_verbose_errors):
    # Find SP from BAT to one arbitrary BAB candidate. The first node away from the BAT node along this path is the BAB node
    chosen_BAB_candidate = None
    min_distance_to_centroid_ICAL_ICAR = float('inf')
    for BAB_candidate in BAB_candidates:
        if BAB_candidate in nx_g.nodes():
            pos = nx_g.nodes[BAB_candidate]['pos']
            # curr_x_distance = abs(pos[0] - centroid_ICAL_ICAR[0])
            curr_distance_from_centroid = np.linalg.norm(np.array(centroid_ICAL_ICAR) - np.array(pos))
            if curr_distance_from_centroid < min_distance_to_centroid_ICAL_ICAR:
                min_distance_to_centroid_ICAL_ICAR = curr_distance_from_centroid
                chosen_BAB_candidate = BAB_candidate
    BAB_node = None
    if chosen_BAB_candidate == None:
        if do_verbose_errors:
            print(f'\t No BAB candidate even!')
        return None, []
        # assert chosen_BAB_candidate != None, 'xxx'
    else:
        try:
            SP_BAT_BAB_candidate = nx.shortest_path(nx_g, source=BAT_node, target=chosen_BAB_candidate)
        except:
            raise ValueError(f'BAT {BAT_node} chosen_BAB_candidate {chosen_BAB_candidate}')

        if len(SP_BAT_BAB_candidate) <= 1:
            BAB_node = BAT_node
        else:
            # assert len(SP_BAT_BAB_candidate) > 1, f'BAT {BAT_node} chosen_BAB_candidate {chosen_BAB_candidate} SP_BAT_BAB_candidate {SP_BAT_BAB_candidate}'
            BAB_node = SP_BAT_BAB_candidate[1]
            assert BAB_node != BAT_node, f'BAT {BAT_node} chosen_BAB_candidate {chosen_BAB_candidate} SP_BAT_BAB_candidate {SP_BAT_BAB_candidate}'
        # print([BAB_node])
        BAB_node_pos_attributes = [nx_g.nodes[node]['pos'] for node in [BAB_node] if node != None]

    return BAB_node, BAB_node_pos_attributes

def find_BAB_node(nx_g_BAT, BAB_candidates, BAT_node, centroid_ICAL_ICAR, do_verbose_errors):
    # Find SP from BAT to all BAB candidates. Of the subpath in common among all candidates, choose the furtherst one away from BAT
    BAB_node = None
    BAB_node_pos_attributes = []
    if BAT_node != None:
        # BAB_node, BAB_node_pos_attributes = find_BAB_node_using_divergence(nx_g_BAT, BAB_candidates, BAT_node)
        BAB_node, BAB_node_pos_attributes = find_BAB_node_as_immediate_neighbour_of_BAT_node(nx_g_BAT, BAB_candidates, BAT_node, centroid_ICAL_ICAR, do_verbose_errors)
        # assert BAB_node != None, 'xxx'
    return BAB_node, BAB_node_pos_attributes

# PComA PCA functions #####################################################################
def gen_nx_g_PComA(nx_g, nodes_to_remove):
    nx_g_PComA = copy.deepcopy(nx_g)
    nx_g_PComA.remove_nodes_from(nodes_to_remove)
    return nx_g_PComA

def find_PComA_nodes(nx_g, ICA_node, MCA_landmark, BAB_candidates, params_dict, do_verbose_errors):
    # !!! ensure that the PComA anterior is not <=2 hops away from BAB for ALL bab candidates!!! dont break the for loop! move PComA posterior search OUT of the anterior for loop

    # Find SP from ICA to MCA landmark (should always exist)
    try:
        SP_ICA_MCA_landmark = nx.shortest_path(nx_g, source=ICA_node, target=MCA_landmark)
    except:
        # pass
        raise ValueError('No connection between ICA and MCA landmark!')

    # Find SP from ICA to BAB candidates (if exist, then PComA exists)
    divergence_nodes = {}
    for BAB_candidate in BAB_candidates:
        try:
            SP_ICA_BAB_candidate = nx.shortest_path(nx_g, source=ICA_node, target=BAB_candidate)
            divergence_node, divergence_type = find_divergence_node(SP_ICA_MCA_landmark, SP_ICA_BAB_candidate, do_verbose_errors)
            assert divergence_type not in ['straggler', 'identical'], 'xxx'
            if divergence_node not in divergence_nodes:
                divergence_nodes[divergence_node] = 1
            else:
                divergence_nodes[divergence_node] += 1
        except:
            pass

    if len(divergence_nodes) > 0:
        # if len(divergence_nodes) > 1:
        #     print('Multiple divergence node candidates found. Most frequent one is selected.')
        PComA_anterior = max(divergence_nodes, key=divergence_nodes.get)
        PComA_posterior = None

        # Check validity of the PComA anterior node, invalid if direct neighbour of a BAP landmark! #!!!XAXA done
        for BAB_candidate in BAB_candidates:
            try:
                SP_PComA_anterior_BAB_candidate = nx.shortest_path(nx_g, source=PComA_anterior, target=BAB_candidate)
            except:
                # print('\t Warning: Found a BAB candidate unreachable from a PComA_anterior') # turned off XAXA done
                continue
                # SP_PComA_anterior_BAB_candidate = []
                # raise ValueError('xxx')

            if len(SP_PComA_anterior_BAB_candidate) <= 2: # !!! instead of detecting BAB candidate-conected-to-ICA-situation this way, robustly use paths to prune BAB_candidates above
                # physiologically impossible to have a PComA_anterior to BAB_candidate connection of one edge. !!! this fails if theres two edges between a False BAB and ICA...
                PComA_anterior = None
                PComA_posterior = None
                # print('\t Warning: The selected PComA anterior node is immediately adjacent to a Basilar Base node! This is invalid, so no PComA anterior nodes are defined.')
                return PComA_anterior, PComA_posterior

        # By this stage PComA anterior node should be solid. Search for nearby PComA posterior node
        for BAB_candidate in BAB_candidates:
            try:
                SP_PComA_anterior_BAB_candidate = nx.shortest_path(nx_g, source=PComA_anterior, target=BAB_candidate)
                # (A)
                # divergence_node, divergence_type = find_divergence_node(SP_ICA_MCA_landmark, SP_PComA_anterior_BAB_candidate)
                # assert divergence_type not in ['straggler', 'identical'], 'xxx'
                # assert SP_PComA_anterior_BAB_candidate[0] == PComA_anterior, 'xxx'ff

                # (B)
                # PComA_posterior = SP_PComA_anterior_BAB_candidate[1]
            except:
                # print('\t Warning: Found a BAB candidate unreachable from a PComA_anterior') # turned off XAXA done
                continue # added XAXA done
                SP_PComA_anterior_BAB_candidate = []
                # raise ValueError('xxx')

            # (C)
            # assert len(SP_PComA_anterior_BAB_candidate) > 2, f'{SP_PComA_anterior_BAB_candidate}'
            # XAXA turned off
            # if len(SP_PComA_anterior_BAB_candidate) <= 2: # !!! instead of detecting BAB candidate-conected-to-ICA-situation this way, robustly use paths to prune BAB_candidates above
            #     # physiologically impossible to have a PComA_anterior to BAB_candidate connection of one edge. !!! this fails if theres two edges between a False BAB and ICA...
            #     PComA_anterior = None
            #     PComA_posterior = None
            #     break # !!!XAXA should be continue? move this if block outside this for loop, and create another for loop through all BAB candidates with tis check!!!

            for PComA_posterior_candidate in SP_PComA_anterior_BAB_candidate[1:]:
                neighbours = list(nx.neighbors(nx_g, PComA_posterior_candidate))
                # assert PComA_posterior_candidate not in neighbours, 'self loops not allowed'
                neighbours = [n for n in neighbours if n != PComA_posterior_candidate] #'self loops not allowed'

                # condition A - is candidate a deg 2 node
                candidate_is_deg_2 = len(neighbours) == 2

                # condition B - does candidate have a (short) leaf
                candidate_has_leaf = False
                neighbours_not_on_SP_PComA_anterior_BAB_candidate = [n for n in neighbours if n not in SP_PComA_anterior_BAB_candidate]
                for nei in neighbours_not_on_SP_PComA_anterior_BAB_candidate:
                    nei_neighbours = list(nx.neighbors(nx_g, nei))
                    PComA_posterior_candidate_nei_dist = nx_g[PComA_posterior_candidate][nei]['dist']
                    if len(nei_neighbours) == 1 and PComA_posterior_candidate_nei_dist < params_dict['PComA_leaf_thresh']:
                        # print('LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL', PComA_posterior_candidate_nei_dist)
                        # print('LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL')
                        # print('LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL')
                        # print('LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL')
                        # print('LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL')
                        candidate_has_leaf = True
                        break

                if candidate_is_deg_2 or candidate_has_leaf:
                    continue
                else:
                    PComA_posterior = PComA_posterior_candidate
                    break

            break

    else:
        # no valid PComA found when using SP from ICA to MCAlandmarks/BABcandidates
        PComA_anterior = None
        PComA_posterior = None

    # if cohort == 'CROPCheck' and train_or_test == 'train' and c == 140:
    #     print('O X O X O X forcing a special case')
    #     PComAR_posterior_node = None
    #     PComAR_anterior_node = None
    # assert (PComAL_posterior_node == None and PComAL_anterior_node == None) or (PComAL_posterior_node != None and PComAL_anterior_node != None), 'xxx'
    # assert (PComAR_posterior_node == None and PComAR_anterior_node == None) or (PComAR_posterior_node != None and PComAR_anterior_node != None), 'xxx'

    # if (PComAL_posterior_node == None and PComAL_anterior_node != None) or (PComAL_posterior_node != None and PComAL_anterior_node == None): #XAXA done commented out 13.06.24
    #     print('\t Warning: Found only one of the PComAL nodes.')
    #     PComAL_posterior_node = None
    #     PComAL_anterior_node = None
    # if (PComAR_posterior_node == None and PComAR_anterior_node != None) or (PComAR_posterior_node != None and PComAR_anterior_node == None):
    #     print('\t Warning: Found only one of the PComAR nodes.')
    #     PComAR_posterior_node = None
    #     PComAR_anterior_node = None
    
    return PComA_anterior, PComA_posterior

# XAXA done
def find_PComA_nodes_round_B(nx_g, ICA_node, MCA_node, MCA_landmark, PCA_landmarks, params_dict, PComA_dist_thresh_lower, PComA_dist_thresh_upper, do_verbose_errors):
    def check_if_PComA_might_exist_despite_not_defining_it(nx_g, PComA_anterior, MCA_node, PCA_landmarks):
        if PComA_anterior == None:
            PCA_landmarks_tmp = copy.deepcopy(PCA_landmarks)
            PCA_landmarks_tmp = [n for n in PCA_landmarks_tmp if n in nx_g.nodes()]
            top_three_ypos_nodes = sorted(PCA_landmarks_tmp, key=lambda node: nx_g.nodes[node]['pos'][1], reverse=True)[:3]
            for PCA_landmark in top_three_ypos_nodes:
                try:
                    SP_MCA_PCA_landmark = nx.shortest_path(nx_g, source=MCA_node, target=PCA_landmark)
                    if do_verbose_errors:
                        print('\t PComA_anterior not found, but possibly exists due to MCA connection to posterior PCA region.') #XAXA done
                    # print('.......................', 'PComA_anterior not found, but possibly exists due to MCA connection to posterior PCA region.')
                    # print('.......................', 'PComA_anterior not found, but possibly exists due to MCA connection to posterior PCA region.')
                    # print('.......................', 'PComA_anterior not found, but possibly exists due to MCA connection to posterior PCA region.')
                    # print('.......................', 'PComA_anterior not found, but possibly exists due to MCA connection to posterior PCA region.')
                    # print('.......................', 'PComA_anterior not found, but possibly exists due to MCA connection to posterior PCA region.')
                    break
                except:
                    pass

    # Find SP from ICA to MCA landmark (should always exist)
    try:
        SP_ICA_MCA_landmark = nx.shortest_path(nx_g, source=ICA_node, target=MCA_landmark)
    except:
        # pass
        raise ValueError('No connection between ICA and MCA landmark!')

    # Find posterior-most PCA landmark to use as target for ICA-PCA landmark path
    PCA_landmark_max_y = None
    PCA_landmark_max_y_pos = -float('inf')
    for node in PCA_landmarks:
        if node in nx_g.nodes(): # some may be in MCA
            pos = nx_g.nodes[node]['pos']
            if pos[1] > PCA_landmark_max_y_pos:
                PCA_landmark_max_y = node
                PCA_landmark_max_y_pos = pos[1]

    if PCA_landmark_max_y == None:
        PComA_anterior = None
        PComA_posterior = None
        check_if_PComA_might_exist_despite_not_defining_it(nx_g, PComA_anterior, MCA_node, PCA_landmarks)
        return PComA_anterior, PComA_posterior

    # Find SP from ICA to PCA_landmark_max_y
    try:
        SP_ICA_PCA_landmark = nx.shortest_path(nx_g, source=ICA_node, target=PCA_landmark_max_y)
    except:
        # If not exist, then no PComA nodes are labelled (if its a False Negative, santity check later on conenctivity between ICA and top 3 max y ipso PCA landmarks should catch it)
        PComA_anterior = None
        PComA_posterior = None
        check_if_PComA_might_exist_despite_not_defining_it(nx_g, PComA_anterior, MCA_node, PCA_landmarks)
        return PComA_anterior, PComA_posterior

    # Find divergence_node
    divergence_node, divergence_type = find_divergence_node(SP_ICA_MCA_landmark, SP_ICA_PCA_landmark, do_verbose_errors)
    if divergence_type in ['straggler', 'identical']:   # for BRAVE train 124
        divergence_node = None

    if divergence_node != None:
        # if len(divergence_nodes) > 1:
        #     print('Multiple divergence node candidates found. Most frequent one is selected.')
        # PComA_anterior = max(divergence_nodes, key=divergence_nodes.get)
        PComA_anterior = divergence_node
        PComA_posterior = None

        # print('XXXXXXXXXXXXXXXXXXXXXXXXXXXX', 'found a pcoma anterior using ROUND B')
        # print('XXXXXXXXXXXXXXXXXXXXXXXXXXXX', 'found a pcoma anterior using ROUND B')
        # print('XXXXXXXXXXXXXXXXXXXXXXXXXXXX', 'found a pcoma anterior using ROUND B')
        # print('XXXXXXXXXXXXXXXXXXXXXXXXXXXX', 'found a pcoma anterior using ROUND B')
        # print('XXXXXXXXXXXXXXXXXXXXXXXXXXXX', 'found a pcoma anterior using ROUND B')

        # Find SP from PComA_anterior to PCA_landmark_max_y
        try:
            SP_PComA_anterior_PCA_landmark = nx.shortest_path(nx_g, source=PComA_anterior, target=PCA_landmark_max_y)
        except:
            raise ValueError('This path cannot not exist!')

        # By this stage PComA anterior node should be solid. Search for nearby PComA posterior node
        for PComA_posterior_candidate in SP_PComA_anterior_PCA_landmark[1:]:

            neighbours = list(nx.neighbors(nx_g, PComA_posterior_candidate))
            # assert PComA_posterior_candidate not in neighbours, 'self loops not allowed'
            neighbours = [n for n in neighbours if n != PComA_posterior_candidate] #'self loops not allowed'

            # condition A - is candidate a deg 2 node
            candidate_is_deg_2 = len(neighbours) == 2

            # condition B - does candidate have a (short) leaf
            candidate_has_leaf = False
            neighbours_not_on_SP_PComA_anterior_PCA_landmark = [n for n in neighbours if n not in SP_PComA_anterior_PCA_landmark]
            for nei in neighbours_not_on_SP_PComA_anterior_PCA_landmark:
                nei_neighbours = list(nx.neighbors(nx_g, nei))
                PComA_posterior_candidate_nei_dist = nx_g[PComA_posterior_candidate][nei]['dist']
                if len(nei_neighbours) == 1 and PComA_posterior_candidate_nei_dist < params_dict['PComA_leaf_thresh']:
                    candidate_has_leaf = True
                    break

            if candidate_is_deg_2:
            # if candidate_is_deg_2 or candidate_has_leaf:
                continue
            else:
                dist_PComA_anterior_PComA_Posterior = calc_dist_along_SP(nx_g, PComA_anterior, PComA_posterior_candidate)
                if dist_PComA_anterior_PComA_Posterior < PComA_dist_thresh_upper:
                    if dist_PComA_anterior_PComA_Posterior > PComA_dist_thresh_lower:

                        # raise ValueError('asfkjbia bieu h', dist_PComA_anterior_PComA_Posterior)
                        PComA_posterior = PComA_posterior_candidate
                        # print('OOOOOOOOOOOOOOOOOO', 'found a pcoma posterior using ROUND B')
                        # print('OOOOOOOOOOOOOOOOOO', 'found a pcoma posterior using ROUND B')
                        # print('OOOOOOOOOOOOOOOOOO', 'found a pcoma posterior using ROUND B')
                        # print('OOOOOOOOOOOOOOOOOO', 'found a pcoma posterior using ROUND B')
                        # print('OOOOOOOOOOOOOOOOOO', 'found a pcoma posterior using ROUND B')
                        break
                    else:
                        pass
                        # if do_verbose_errors:
                        #     print(f'\t OOOOOOOOOOOOOOOOOOOOOOO', 'found a short pcoma posterior')
                        #     print(f'\t OOOOOOOOOOOOOOOOOOOOOOO', 'found a short pcoma posterior')
                        #     print(f'\t OOOOOOOOOOOOOOOOOOOOOOO', 'found a short pcoma posterior')
                        #     print(f'\t OOOOOOOOOOOOOOOOOOOOOOO', 'found a short pcoma posterior')
                        #     print(f'\t OOOOOOOOOOOOOOOOOOOOOOO', 'found a short pcoma posterior')

    else:
        # no valid PComA found when using SP from ICA to MCAlandmarks/BABcandidates
        PComA_anterior = None
        PComA_posterior = None

    check_if_PComA_might_exist_despite_not_defining_it(nx_g, PComA_anterior, MCA_node, PCA_landmarks)

    return PComA_anterior, PComA_posterior

def get_PComA_node_params(nx_g, PComAL_anterior_node, PComAR_anterior_node, PComAL_posterior_node, PComAR_posterior_node):
    PComA_anterior_nodes_pos_attributes = [nx_g.nodes[node]['pos'] for node in [PComAL_anterior_node, PComAR_anterior_node] if node != None]
    PComA_posterior_nodes_pos_attributes = [nx_g.nodes[node]['pos'] for node in [PComAL_posterior_node, PComAR_posterior_node] if node != None]
    return PComA_anterior_nodes_pos_attributes, PComA_posterior_nodes_pos_attributes

def check_for_double_PComA(nx_g, PComAL_anterior_node, PComAL_posterior_node, PComAR_anterior_node, PComAR_posterior_node, ICAL_node, MCAL_node, ACAL_node, ICAR_node, MCAR_node, ACAR_node, BAT_node, do_verbose_errors):
    # Check for double PComA situation, initialisation. Find the actual edges between PComA anterior and posterior, potentially containing intermediary nodes
    try:
        SP_PComAL_edges = []
        SP_PComAR_edges = []
        if PComAL_anterior_node != None and PComAL_posterior_node != None:
            SP_PComAL = nx.shortest_path(nx_g, source=PComAL_anterior_node, target=PComAL_posterior_node)
            SP_PComAL_edges = [(SP_PComAL[i], SP_PComAL[i + 1]) for i in range(len(SP_PComAL) - 1)]
        if PComAR_anterior_node != None and PComAR_posterior_node != None:
            SP_PComAR = nx.shortest_path(nx_g, source=PComAR_anterior_node, target=PComAR_posterior_node)
            SP_PComAR_edges = [(SP_PComAR[i], SP_PComAR[i + 1]) for i in range(len(SP_PComAR) - 1)]

        edges_to_remove_PComA = []
        if SP_PComAL_edges:
            edges_to_remove_PComA += SP_PComAL_edges
        if SP_PComAR_edges:
            edges_to_remove_PComA += SP_PComAR_edges
    except:
        raise ValueError('xxx')

    # After deleting the existing PComA edges, is there still a connection from psoterior to anterior circulation?
    anterior_nodes = [n for n in [ICAL_node, MCAL_node, ACAL_node, ICAR_node, MCAR_node, ACAR_node] if n != None]
    # edges_to_remove = [(PComAL_anterior_node, PComAL_posterior_node), (PComAR_anterior_node, PComAR_posterior_node)] # not general enough, may be intermediary nodes between the PComA end poitns!
    nx_g_double_PComA_test = copy.deepcopy(nx_g)
    nx_g_double_PComA_test.remove_edges_from(edges_to_remove_PComA)
    found_double_PComA = False
    for anterior_node in anterior_nodes:
        try:
            # If any SP is found, throw a Double PComA situation
            shortest_path = nx.shortest_path(nx_g_double_PComA_test, source=BAT_node, target=anterior_node)
            found_double_PComA = True
        except:
            pass
    if found_double_PComA:
        if do_verbose_errors:
            print('\t There is possibly a double ipsilateral PComA situation.')

def legacy_find_PComA_posterior_node(nx_g, PComA_anterior_node):
    # If this function is called, then guaranteed to exist a PCOMA posterior node
    neighbors = list(nx.neighbors(nx_g, PComA_anterior_node))
    assert len(neighbors) >= 3, 'xxx'

    # Find the neighbor with the max y axis 'pos' attribute
    max_y_neighbor = max(neighbors, key=lambda neighbor: nx_g.nodes[neighbor]['pos'][1])
    PComA_posterior_node = max_y_neighbor

    return PComA_posterior_node

def find_PCA_landmarks(nx_g, params_dict, centroid_ICAL_ICAR, ICAL_pos, ICAR_pos, MCAL_pos, MCAR_pos, do_verbose_errors):
    def distance(point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def find_nodes_in_sphere(nx_g, sphere_center, sphere_radius):

        x = sphere_center[0]
        y = sphere_center[1]
        z = sphere_center[2]
        nodes_in_sphere = []
        for node in nx_g.nodes():
            node_pos = nx_g.nodes[node]['pos']

            # Calculate the distance between the sphere center and the node
            distance = ((x - node_pos[0]) ** 2 + (y - node_pos[1]) ** 2 + (z - node_pos[2]) ** 2) ** 0.5
            if distance <= sphere_radius:
                nodes_in_sphere.append(node)

        return nodes_in_sphere

    def check_edge_sphere_intersection(nx_g, edge, sphere_center, sphere_radius):
        x = sphere_center[0]
        y = sphere_center[1]
        z = sphere_center[2]
        node1, node2 = edge
        node1_pos = np.array(nx_g.nodes[node1]['pos'])
        node2_pos = np.array(nx_g.nodes[node2]['pos'])

        node1_node2_vector = node2_pos - node1_pos
        node1_center_vector = np.array(sphere_center) - node1_pos
        dot = np.dot(node1_center_vector, node1_node2_vector)
        len_node1_node2_vector_squared = np.dot(node1_node2_vector, node1_node2_vector)

        t = dot / len_node1_node2_vector_squared
        if t < 0:
            closest_point = node1_pos
        elif t > 1:
            closest_point = node2_pos
        else:
            closest_point = node1_pos + t * node1_node2_vector

        # Calculate the distance between sphere center and the closest point
        distance = np.linalg.norm(sphere_center - closest_point)

        return distance <= sphere_radius

    def find_nodes_intersecting_sphere(nx_g, sphere_center, sphere_radius):
        intersecting_nodes = set()
        for edge in nx_g.edges():
            if check_edge_sphere_intersection(nx_g, edge, sphere_center, sphere_radius):
                intersecting_nodes.add(edge[0])
                intersecting_nodes.add(edge[1])
        return list(intersecting_nodes)

    def check_edge_ellipsoid_intersection(nx_g, edge, ellipsoid_center, ellipsoid_radii):
        # https://math.stackexchange.com/questions/3722553/find-intersection-between-line-and-ellipsoid

        node1, node2 = edge
        p1 = np.array(nx_g.nodes[node1]['pos'])
        p2 = np.array(nx_g.nodes[node2]['pos'])
        a = ellipsoid_radii[0]
        b = ellipsoid_radii[1]
        c = ellipsoid_radii[2]

        # Translate line and ellipsoid so that ellipsoid is centered at the origin
        p1t = np.array(p1) - np.array(ellipsoid_center)
        p2t = np.array(p2) - np.array(ellipsoid_center)

        # First test whether either node lies within ellipsoid
        if ((p1t[0] / a) ** 2 + (p1t[1] / b) ** 2 + (p1t[2] / c) ** 2) <= 1:
            return True
        if ((p2t[0] / a) ** 2 + (p2t[1] / b) ** 2 + (p2t[2] / c) ** 2) <= 1:
            return True

        # Continue testing whether any lines intersect with ellipsoid envelope
        # Calculate coefficients for the quadratic equation
        A = ((p2[0] - p1[0]) / a) ** 2 + ((p2[1] - p1[1]) / b) ** 2 + ((p2[2] - p1[2]) / c) ** 2
        B = 2 * p1t[0] * (p2[0] - p1[0]) / a ** 2 + 2 * p1t[1] * (p2[1] - p1[1]) / b ** 2 + 2 * p1t[2]* (p2[2] - p1[2]) / c ** 2
        C = (p1t[0] / a) ** 2 + (p1t[1] / b) ** 2 + (p1t[2] / c) ** 2 - 1

        discriminant = B ** 2 - 4 * A * C
        if discriminant < 0 or A == 0: # A==0 added 23.05.24 XAXA
            # No real solutions, no intersection
            return False
        else:
            # Check if there are real solutions within the line segment [0, 1]
            t1 = (-B + np.sqrt(discriminant)) / (2 * A)
            t2 = (-B - np.sqrt(discriminant)) / (2 * A)
            # print(discriminant, A, t1, t2)

            # Check if any of the solutions are within the line segment [0, 1]
            if 0 <= t1 <= 1 or 0 <= t2 <= 1:
                return True
            else:
                return False

    def find_nodes_intersecting_ellipsoid(nx_g, ellipsoid_center, ellipsoid_radii):
        intersecting_nodes = set()
        for edge in nx_g.edges():
            if edge in [(34, 35), (35,34)]:
                gg = 5
            if check_edge_ellipsoid_intersection(nx_g, edge, ellipsoid_center, ellipsoid_radii):
                intersecting_nodes.add(edge[0])
                intersecting_nodes.add(edge[1])
        return list(intersecting_nodes)

    def find_all_pairs_shortest_paths():
        return

    ICA_distance = distance(ICAL_pos, ICAR_pos)

    # When both MCAs exist, use centroid_MCA
    if MCAL_pos != None and MCAR_pos != None:
        MCA_x_distance = np.abs(MCAL_pos[0] - MCAR_pos[0])
        MCA_z_pos = (MCAL_pos[2] + MCAR_pos[2]) / 2
        # centroid = centroid_ICAL_ICAR #(np.array(MCAL_pos) + np.array(MCAR_pos)) / 2
        centroid = (np.array(MCAL_pos) + np.array(MCAR_pos)) / 2
        x_offset = MCA_x_distance * params_dict['ellipsoid_x_offset_multiplier']
        y_offset = MCA_x_distance * params_dict['ellipsoid_y_offset_multiplier']
        z_offset = MCA_x_distance * params_dict['ellipsoid_z_offset_multiplier']
        PCA_sphere_radius = MCA_x_distance / 2
        PCA_ellipsoid_radii = [MCA_x_distance * params_dict['ellipsoid_x_radius_multiplier'], MCA_x_distance * params_dict['ellipsoid_y_radius_multiplier'], MCA_x_distance * params_dict['ellipsoid_z_radius_multiplier']]
        # MCAL_x_offset_from_centroid = np.abs(MCAL_pos[0] - centroid_ICAL_ICAR[0])
        # MCAR_x_offset_from_centroid = np.abs(MCAR_pos[0] - centroid_ICAL_ICAR[0])
        # MCA_x_offset_from_centroid = (MCAL_x_offset_from_centroid + MCAR_x_offset_from_centroid) / 2
    # Otherwise use centroid ICA as approximation (the if statemtns belwo should never be used: if an MCA was not found, entire scan flagged, wont reach this point!!!)
    elif MCAL_pos != None:
        MCA_z_pos = MCAL_pos[2]
        centroid = centroid_ICAL_ICAR
        x_offset = ICA_distance / 4
        y_offset = ICA_distance / 3
        z_offset = ICA_distance / 4
        PCA_sphere_radius = ICA_distance / 4
        PCA_ellipsoid_radii = [ICA_distance / 4, ICA_distance / 3, ICA_distance / 2]
        # MCA_x_offset_from_centroid = np.abs(MCAL_pos[0] - centroid_ICAL_ICAR[0])
    elif MCAR_pos != None:
        MCA_z_pos = MCAL_pos[2]
        centroid = centroid_ICAL_ICAR
        x_offset = ICA_distance / 4
        y_offset = ICA_distance / 3
        z_offset = ICA_distance / 4
        PCA_sphere_radius = ICA_distance / 4
        PCA_ellipsoid_radii = [ICA_distance / 4, ICA_distance / 3, ICA_distance / 2]
        # MCA_x_offset_from_centroid = np.abs(MCAR_pos[0] - centroid_ICAL_ICAR[0])
    else:
        raise ValueError('xxx')
    # Define PCA landmark search spheres
    PCAL_sphere_center = [centroid[0] + x_offset,
                            centroid[1] + y_offset,
                            MCA_z_pos + z_offset]
    PCAR_sphere_center = [centroid[0] - x_offset,
                            centroid[1] + y_offset,
                            MCA_z_pos + z_offset]

    # PCAL_landmarks = find_nodes_intersecting_sphere(nx_g, PCAL_sphere_center, PCA_sphere_radius)
    # PCAR_landmarks = find_nodes_intersecting_sphere(nx_g, PCAR_sphere_center, PCA_sphere_radius)
    PCAL_landmarks = find_nodes_intersecting_ellipsoid(nx_g, PCAL_sphere_center, PCA_ellipsoid_radii)
    PCAR_landmarks = find_nodes_intersecting_ellipsoid(nx_g, PCAR_sphere_center, PCA_ellipsoid_radii)

    if do_verbose_errors:
        if PCAL_landmarks == []:
            print('\t Warning: no PCAL landmarks')
        if PCAR_landmarks == []:
            print('\t Warning: no PCAR landmarks')
    PCAL_landmarks_pos_attributes = [nx_g.nodes[node]['pos'] for node in PCAL_landmarks]
    PCAR_landmarks_pos_attributes = [nx_g.nodes[node]['pos'] for node in PCAR_landmarks]
    PCAL_landmarks_sphere = PCAL_sphere_center + [PCA_sphere_radius]
    PCAR_landmarks_sphere = PCAR_sphere_center + [PCA_sphere_radius]
    PCAL_landmarks_ellipsoid = PCAL_sphere_center + PCA_ellipsoid_radii
    PCAR_landmarks_ellipsoid = PCAR_sphere_center + PCA_ellipsoid_radii

    return PCAL_landmarks, PCAR_landmarks, PCAL_landmarks_pos_attributes, PCAR_landmarks_pos_attributes, \
           PCAL_landmarks_sphere, PCAR_landmarks_sphere, PCAL_landmarks_ellipsoid, PCAR_landmarks_ellipsoid

def find_PCA_anchors_orig(nx_g, centroid_ICAL_ICAR, PCAL_landmarks, PCAR_landmarks):

    # slice nx_g_BAT again such that all nodes further back in y direction than the max y PCA landmark are pruned
    # This reduces chance of errors in Triple Node detection below thats meant to find a BAB, and two PCA nodes
    nx_g_PCA_pruned = copy.deepcopy(nx_g)
    PCA_landmark_max_y = -float('inf')
    for node in PCAL_landmarks + PCAR_landmarks:
        if node in nx_g_PCA_pruned.nodes(): # some may be in MCA
            pos = nx_g_PCA_pruned.nodes[node]['pos']
            if pos[1] > PCA_landmark_max_y:
                PCA_landmark_max_y = pos[1]

    nodes_to_remove = []
    for node in list(nx_g_PCA_pruned.nodes()):
        if node in nx_g_PCA_pruned.nodes(): # some may be in MCA
            pos = nx_g_PCA_pruned.nodes[node]['pos']
            if pos[1] > PCA_landmark_max_y:
                nodes_to_remove += [node]
    nx_g_PCA_pruned.remove_nodes_from(nodes_to_remove)

    # One or both ends of the longest path in the isolated posterior tree is guaranteed by evolution to be in one of the PCA subtrees (edge case deep vertebral subtree with shallow PCA subtrees)
    all_pairs_shortest_paths = dict(nx.all_pairs_dijkstra(nx_g_PCA_pruned, cutoff=None, weight='dist'))
    longest_path = None
    longest_path_length = -1
    for source_node, shortest_paths_tuple in all_pairs_shortest_paths.items():
        shortest_paths = shortest_paths_tuple[1]
        for target_node, path in shortest_paths.items():
            path_length = sum(nx_g_PCA_pruned[path[i]][path[i + 1]]['dist'] for i in range(len(path) - 1))
            # print(source_node, target_node, path, path_length)
            if path_length > longest_path_length:
                longest_path = path
                longest_path_length = path_length

    # A third node, being maximally separated from both the nodes discovered above, is potentially in the contralateral PCA subtree, or a BAB node, but also could be in ipsilateral PCA subtree if asymetrical posterior subtrees!
    if longest_path:
        assert len(longest_path) > 2, 'xxx'
        node_A = longest_path[0]
        node_B = longest_path[-1]

        furthest_node = None
        max_sum_distance = -1  # Initialize with a negative value

        for node, shortest_paths_tuple in all_pairs_shortest_paths.items():
            shortest_paths_dists = shortest_paths_tuple[0]
            distance_to_A = shortest_paths_dists[node_A]
            distance_to_B = shortest_paths_dists[node_B]
            sum_distance_to_AB = distance_to_A + distance_to_B
            if sum_distance_to_AB > max_sum_distance:
                furthest_node = node
                max_sum_distance = sum_distance_to_AB

        # print(f"Longest path: {longest_path}")
        # print(f"Longest path length: {longest_path_length}")
        # print(f"{node_A}, {node_B}, {furthest_node}")
    else:
        raise ValueError('xxx')

    assert node_A != None and node_B != None and furthest_node != None, 'xxx'

    # If there doesnt exist landmarks for either PCA subtree, there is an aberrant posterior circulation, return Nones for manual intervention
    # Otherwise, of the three discovered nodes of maximal separation, the two with highest z pos are chosen as potential PCA anchors
    posterior_nodes_reachable_from_BAB_candidates_contains_PCAL_landmarks = any(node in PCAL_landmarks for node in nx_g.nodes)
    posterior_nodes_reachable_from_BAB_candidates_contains_PCAR_landmarks = any(node in PCAR_landmarks for node in nx_g.nodes)
    PCAL_anchor = None
    PCAR_anchor = None
    # if PCAL_landmarks != None and PCAR_landmarks != None:
    if posterior_nodes_reachable_from_BAB_candidates_contains_PCAL_landmarks and posterior_nodes_reachable_from_BAB_candidates_contains_PCAR_landmarks:
        min_z_node = None
        min_z_pos = float('inf')
        for node in [node_A, node_B, furthest_node]:
            z_coord = nx_g.nodes[node]['pos'][2]
            if z_coord < min_z_pos:
                min_z_node = node
                min_z_pos = z_coord
        PCA_anchors = [node_A, node_B, furthest_node]
        PCA_anchors.remove(min_z_node)

        x_coords = [nx_g.nodes[node]['pos'][0] for node in PCA_anchors]
        PCAL_anchor = PCA_anchors[x_coords.index(max(x_coords))]
        PCAR_anchor = PCA_anchors[x_coords.index(min(x_coords))]
    else:
        # No path found from BAB candidates to either PCAL or PCAR trees!
        pass

    # The two anchors at this point if they exist at all are either in contralateral PCA subtrees, or in ipsilateral subtree if PCA subtrees are asymmetrical.
    # Using x component of centroid_ICAL_ICAR to define a midsagital plane. If the two PCA anchors fall on same side of plane,
    # consider it a "same-sided PCA anchor" situation (this is either a True case or False case)
    # Whether true or not, select the most lateral node in L/R side to be the PCA anchor for that direction. Evolution hopefully guaantees the truth of this but edge case can always be created
    # (In actual BAT calculation, can include sanity checks to detect these edge cases resulting in ipsilateral PCA anchors despite all this logic so far)
    # Set the contralateral anchor to be the most posterior contralateral PCA landmark node from the contralateral y-region
    if PCAL_anchor != None and PCAR_anchor != None:
        PCAL_anchor_x_pos = nx_g.nodes[PCAL_anchor]['pos'][0]
        PCAR_anchor_x_pos = nx_g.nodes[PCAR_anchor]['pos'][0]
        mid_sagittal_x = centroid_ICAL_ICAR[0]
        if PCAL_anchor_x_pos > mid_sagittal_x and PCAR_anchor_x_pos > mid_sagittal_x:
            # determine adjusted PCAL_anchor
            if PCAL_anchor_x_pos > PCAR_anchor_x_pos:
                PCAL_anchor = PCAL_anchor
            elif PCAL_anchor_x_pos <= PCAR_anchor_x_pos:
                PCAL_anchor = PCAR_anchor

            # determine adjusted PCAR_anchor
            PCAR_landmarks_right_of_mid_saggital_x = [node for node in PCAR_landmarks if (node in nx_g.nodes) and (nx_g.nodes[node]['pos'][0] < mid_sagittal_x)]
            # Find the node with the min x (NOT maximum 'y pos'!) attribute among the PCAR_landmarks_right_of_mid_saggital_x nodes
            if PCAR_landmarks_right_of_mid_saggital_x:
                PCAR_anchor = max(PCAR_landmarks_right_of_mid_saggital_x, key=lambda node: nx_g.nodes[node]['pos'][1])
                # PCAR_anchor = min(PCAR_landmarks_right_of_mid_saggital_x, key=lambda node: nx_g.nodes[node]['pos'][1])
                # print("Node with max 'y pos' among those with 'x pos' > 5:", PCAR_anchor)
                # print("Adjusted a PCAR anchor", max_y_node)
            else:
                raise ValueError('xxx')
        elif PCAL_anchor_x_pos <= mid_sagittal_x and PCAR_anchor_x_pos <= mid_sagittal_x:
            # determine adjusted PCAR_anchor
            if PCAL_anchor_x_pos > PCAR_anchor_x_pos:
                PCAR_anchor = PCAR_anchor
            elif PCAL_anchor_x_pos <= PCAR_anchor_x_pos:
                PCAR_anchor = PCAL_anchor

            # determine adjusted PCAL_anchor
            PCAL_landmarks_left_of_mid_saggital_x = [node for node in PCAL_landmarks if (node in nx_g.nodes) and (nx_g.nodes[node]['pos'][0] > mid_sagittal_x)]
            # Find the node with the maximum 'y pos' attribute among the PCAL_landmarks_left_of_mid_saggital_x nodes
            if PCAL_landmarks_left_of_mid_saggital_x:
                PCAL_anchor = max(PCAL_landmarks_left_of_mid_saggital_x, key=lambda node: nx_g.nodes[node]['pos'][1])
                # print("Node with max 'y pos' among those with 'x pos' > 5:", PCAL_anchor)
                # print("Adjusted a PCAL anchor", max_y_node)
            else:
                raise ValueError('xxx')
        else:
            # both PCA anchors are hopefully correct
            pass

    PCAL_anchor_pos_attributes = [nx_g.nodes[node]['pos'] for node in [PCAL_anchor] if node != None]
    PCAR_anchor_pos_attributes = [nx_g.nodes[node]['pos'] for node in [PCAR_anchor] if node != None]

    return PCAL_anchor, PCAR_anchor, PCAL_anchor_pos_attributes, PCAR_anchor_pos_attributes

def find_PCA_anchors(nx_g, centroid_ICAL_ICAR, PCAL_landmarks, PCAR_landmarks, MCAL_pos, MCAR_pos):
    def does_exist_path_between_PCA_anchors_after_landmark_deletion(nx_g, PCA_anchor, PCA_landmarks, PCAL_anchor, PCAR_anchor):
        # Final check to see whether PCA anchors are indeed in Left and Right PCA subtrees, and not on same side, by deletion of landmarks (except anchor) then testing SP

        nx_g_PCA_anchor_test = copy.deepcopy(nx_g)
        PCA_landmarks_except_PCA_anchor = [node for node in PCA_landmarks if node != PCA_anchor]
        nx_g_PCA_anchor_test.remove_nodes_from(PCA_landmarks_except_PCA_anchor)
        try:
            # if path between PCA anchors remains after deleting PCA landmarks on one side, means PCA anchors wrong.
            SP_PCA_anchor_PCA_anchor = nx.shortest_path(nx_g_PCA_anchor_test, source=PCAL_anchor, target=PCAR_anchor)
            # print('xxx\nxxx\nxxx\nxxx\nxxx Incorrect PCAR')
            return True
        except:
            # good. there shouldnt be a path from the PCAL anchor to the PCAR anchor if all PCAL landmarks on left or right are deleted
            return False
    # slice nx_g_BAT again such that all nodes further back in y direction than the max y PCA landmark are pruned
    # This reduces chance of errors in Triple Node detection below thats meant to find a BAB, and two PCA nodes
    nx_g_PCA_pruned = copy.deepcopy(nx_g)
    PCA_landmark_max_y = -float('inf')
    for node in PCAL_landmarks + PCAR_landmarks:
        if node in nx_g_PCA_pruned.nodes(): # some may be in MCA
            pos = nx_g_PCA_pruned.nodes[node]['pos']
            if pos[1] > PCA_landmark_max_y:
                PCA_landmark_max_y = pos[1]

    nodes_to_remove = []
    for node in list(nx_g_PCA_pruned.nodes()):
        if node in nx_g_PCA_pruned.nodes(): # some may be in MCA
            pos = nx_g_PCA_pruned.nodes[node]['pos']
            if pos[1] > PCA_landmark_max_y:
                nodes_to_remove += [node]
    nx_g_PCA_pruned.remove_nodes_from(nodes_to_remove)

    # One or both ends of the longest path in the isolated posterior tree is guaranteed by evolution to be in one of the PCA subtrees (edge case deep vertebral subtree with shallow PCA subtrees)
    all_pairs_shortest_paths = dict(nx.all_pairs_dijkstra(nx_g_PCA_pruned, cutoff=None, weight='dist'))
    longest_path = None
    longest_path_length = -1
    for source_node, shortest_paths_tuple in all_pairs_shortest_paths.items():
        shortest_paths = shortest_paths_tuple[1]
        for target_node, path in shortest_paths.items():
            path_length = sum(nx_g_PCA_pruned[path[i]][path[i + 1]]['dist'] for i in range(len(path) - 1))
            # print(source_node, target_node, path, path_length)
            if path_length > longest_path_length:
                longest_path = path
                longest_path_length = path_length

    # A third node, being maximally separated from both the nodes discovered above, is potentially in the contralateral PCA subtree, or a BAB node, but also could be in ipsilateral PCA subtree if asymetrical posterior subtrees!
    if longest_path:
        assert len(longest_path) > 2, 'xxx'
        node_A = longest_path[0]
        node_B = longest_path[-1]

        furthest_node = None
        max_sum_distance = -1  # Initialize with a negative value

        for node, shortest_paths_tuple in all_pairs_shortest_paths.items():
            # print(node, shortest_paths_tuple, node_A, node_B)
            shortest_paths_dists = shortest_paths_tuple[0]
            if node_A in shortest_paths_dists and node_B in shortest_paths_dists:
                distance_to_A = shortest_paths_dists[node_A]
                distance_to_B = shortest_paths_dists[node_B]
                sum_distance_to_AB = distance_to_A + distance_to_B
                if sum_distance_to_AB > max_sum_distance:
                    furthest_node = node
                    max_sum_distance = sum_distance_to_AB

        # print(f"Longest path: {longest_path}")
        # print(f"Longest path length: {longest_path_length}")
        # print(f"{node_A}, {node_B}, {furthest_node}")
    else:
        return None, None, [], []
        raise ValueError('xxx')
        # print('XOXO XOXO XOXO No longest tripartite path found!!!')
        # return None, None, [], []

    assert node_A != None and node_B != None and furthest_node != None, 'xxx'

    # If there doesnt exist landmarks for either PCA subtree, there is an aberrant posterior circulation, return Nones for manual intervention
    # Otherwise, of the three discovered nodes of maximal separation, the two with highest z pos are chosen as potential PCA anchors
    posterior_nodes_reachable_from_BAB_candidates_contains_PCAL_landmarks = any(node in PCAL_landmarks for node in nx_g.nodes)
    posterior_nodes_reachable_from_BAB_candidates_contains_PCAR_landmarks = any(node in PCAR_landmarks for node in nx_g.nodes)
    PCAL_anchor = None
    PCAR_anchor = None
    # if PCAL_landmarks != None and PCAR_landmarks != None:
    if posterior_nodes_reachable_from_BAB_candidates_contains_PCAL_landmarks and posterior_nodes_reachable_from_BAB_candidates_contains_PCAR_landmarks:
        min_z_node = None
        min_z_pos = float('inf')
        for node in [node_A, node_B, furthest_node]:
            z_coord = nx_g.nodes[node]['pos'][2]
            if z_coord < min_z_pos:
                min_z_node = node
                min_z_pos = z_coord
        PCA_anchors = [node_A, node_B, furthest_node]
        PCA_anchors.remove(min_z_node)

        x_coords = [nx_g.nodes[node]['pos'][0] for node in PCA_anchors]
        PCAL_anchor = PCA_anchors[x_coords.index(max(x_coords))]
        PCAR_anchor = PCA_anchors[x_coords.index(min(x_coords))]
    else:
        # No path found from BAB candidates to either PCAL or PCAR trees!
        pass

    # The two anchors at this point if they exist at all are either in contralateral PCA subtrees, or in ipsilateral subtree if PCA subtrees are asymmetrical.
    # Using x component of centroid_ICAL_ICAR to define a midsagital plane. If the two PCA anchors fall on same side of plane,
    # consider it a "same-sided PCA anchor" situation (this is either a True case or False case)
    # Whether true or not, select the most lateral node in L/R side to be the PCA anchor for that direction. Evolution hopefully guaantees the truth of this but edge case can always be created
    # (In actual BAT calculation, can include sanity checks to detect these edge cases resulting in ipsilateral PCA anchors despite all this logic so far)
    # Set the contralateral anchor to be the most posterior contralateral PCA landmark node from the contralateral y-region
    if PCAL_anchor != None and PCAR_anchor != None:

        # Condition A: if both PCA anchors are on same side relative to the mid-saggital-x, must correct PCA anchor
        PCAL_anchor_x_pos = nx_g.nodes[PCAL_anchor]['pos'][0]
        PCAR_anchor_x_pos = nx_g.nodes[PCAR_anchor]['pos'][0]
        # mid_sagittal_x = centroid_ICAL_ICAR[0] # inaccurate when ICAs skewed! use MCA centroid
        mid_sagittal_x = (MCAL_pos[0] + MCAR_pos[0]) / 2

        # Condition  B: if after deletion of one sided PCA landmarks, there still exists path between anchors, must correct anchor
        PCAL_anchor_needs_correction = does_exist_path_between_PCA_anchors_after_landmark_deletion(nx_g, PCAL_anchor, PCAL_landmarks, PCAL_anchor, PCAR_anchor)
        PCAR_anchor_needs_correction = does_exist_path_between_PCA_anchors_after_landmark_deletion(nx_g, PCAR_anchor, PCAR_landmarks, PCAL_anchor, PCAR_anchor)

        if (PCAL_anchor_x_pos > mid_sagittal_x and PCAR_anchor_x_pos > mid_sagittal_x) or PCAR_anchor_needs_correction:
            # determine adjusted PCAL_anchor
            if PCAL_anchor_x_pos > PCAR_anchor_x_pos:
                PCAL_anchor = PCAL_anchor
            elif PCAL_anchor_x_pos <= PCAR_anchor_x_pos:
                PCAL_anchor = PCAR_anchor

            # determine adjusted PCAR_anchor
            PCAR_landmarks_right_of_mid_saggital_x = [node for node in PCAR_landmarks if (node in nx_g.nodes) and (nx_g.nodes[node]['pos'][0] < mid_sagittal_x)]
            # Find the node with the min x (NOT maximum 'y pos'!) attribute among the PCAR_landmarks_right_of_mid_saggital_x nodes
            if PCAR_landmarks_right_of_mid_saggital_x:
                # PCAR_anchor = max(PCAR_landmarks_right_of_mid_saggital_x, key=lambda node: nx_g.nodes[node]['pos'][1])
                PCAR_anchor = min(PCAR_landmarks_right_of_mid_saggital_x, key=lambda node: nx_g.nodes[node]['pos'][0])
                # print("Node with max 'y pos' among those with 'x pos' > 5:", PCAR_anchor)
                # print("Adjusted a PCAR anchor", max_y_node)
            else:
                # raise ValueError('xxx')
                return PCAL_anchor, PCAR_anchor, [], []
        elif (PCAL_anchor_x_pos <= mid_sagittal_x and PCAR_anchor_x_pos <= mid_sagittal_x) or PCAL_anchor_needs_correction:
            # determine adjusted PCAR_anchor
            if PCAL_anchor_x_pos > PCAR_anchor_x_pos:
                PCAR_anchor = PCAR_anchor
            elif PCAL_anchor_x_pos <= PCAR_anchor_x_pos:
                PCAR_anchor = PCAL_anchor

            # determine adjusted PCAL_anchor
            PCAL_landmarks_left_of_mid_saggital_x = [node for node in PCAL_landmarks if (node in nx_g.nodes) and (nx_g.nodes[node]['pos'][0] > mid_sagittal_x)]
            # Find the node with the max x (NOT maximum 'y pos'!) attribute among the PCAL_landmarks_left_of_mid_saggital_x nodes
            if PCAL_landmarks_left_of_mid_saggital_x:
                # PCAL_anchor = max(PCAL_landmarks_left_of_mid_saggital_x, key=lambda node: nx_g.nodes[node]['pos'][1])
                PCAL_anchor = max(PCAL_landmarks_left_of_mid_saggital_x, key=lambda node: nx_g.nodes[node]['pos'][0])
                # print("Node with max 'y pos' among those with 'x pos' > 5:", PCAL_anchor)
                # print("Adjusted a PCAL anchor", max_y_node)
            else:
                # raise ValueError('xxx')
                return PCAL_anchor, PCAR_anchor, [], []
        else:
            # both PCA anchors are hopefully correct
            pass


    PCAL_anchor_pos_attributes = [nx_g.nodes[node]['pos'] for node in [PCAL_anchor] if node != None]
    PCAR_anchor_pos_attributes = [nx_g.nodes[node]['pos'] for node in [PCAR_anchor] if node != None]

    return PCAL_anchor, PCAR_anchor, PCAL_anchor_pos_attributes, PCAR_anchor_pos_attributes

def find_PCA_anchors_most_peripheral_in_x(nx_g, centroid_ICAL_ICAR, PCAL_landmarks, PCAR_landmarks):

    # slice nx_g_BAT again such that all nodes further back in y direction than the max y PCA landmark are pruned
    # This reduces chance of errors in Triple Node detection below thats meant to find a BAB, and two PCA nodes
    # nx_g_PCA_pruned = copy.deepcopy(nx_g)
    # PCA_landmark_max_y = -float('inf')
    # for node in PCAL_landmarks + PCAR_landmarks:
    #     if node in nx_g_PCA_pruned.nodes(): # some may be in MCA
    #         pos = nx_g_PCA_pruned.nodes[node]['pos']
    #         if pos[1] > PCA_landmark_max_y:
    #             PCA_landmark_max_y = pos[1]
    #
    # nodes_to_remove = []
    # for node in list(nx_g_PCA_pruned.nodes()):
    #     if node in nx_g_PCA_pruned.nodes(): # some may be in MCA
    #         pos = nx_g_PCA_pruned.nodes[node]['pos']
    #         if pos[1] > PCA_landmark_max_y:
    #             nodes_to_remove += [node]
    # nx_g_PCA_pruned.remove_nodes_from(nodes_to_remove)

    # If there doesnt exist landmarks for either PCA subtree, there is an aberrant posterior circulation, return Nones for manual intervention
    # Otherwise, of the three discovered nodes of maximal separation, the two with highest z pos are chosen as potential PCA anchors
    posterior_nodes_reachable_from_BAB_candidates_contains_PCAL_landmarks = any(node in PCAL_landmarks for node in nx_g.nodes)
    posterior_nodes_reachable_from_BAB_candidates_contains_PCAR_landmarks = any(node in PCAR_landmarks for node in nx_g.nodes)
    PCAL_anchor = None
    PCAR_anchor = None
    # if PCAL_landmarks != None and PCAR_landmarks != None:
    if posterior_nodes_reachable_from_BAB_candidates_contains_PCAL_landmarks and posterior_nodes_reachable_from_BAB_candidates_contains_PCAR_landmarks:
        # min_z_node = None
        # min_z_pos = float('inf')
        # for node in [node_A, node_B, furthest_node]:
        #     z_coord = nx_g.nodes[node]['pos'][2]
        #     if z_coord < min_z_pos:
        #         min_z_node = node
        #         min_z_pos = z_coord
        # PCA_anchors = [node_A, node_B, furthest_node]
        # PCA_anchors.remove(min_z_node)

        PCAL_landmarks_x_coords = [nx_g.nodes[node]['pos'][0] if node in nx_g.nodes else -float('inf') for node in
                                   PCAL_landmarks]
        PCAR_landmarks_x_coords = [nx_g.nodes[node]['pos'][0] if node in nx_g.nodes else float('inf') for node in
                                   PCAR_landmarks]
        PCAL_anchor = PCAL_landmarks[PCAL_landmarks_x_coords.index(max(PCAL_landmarks_x_coords))]
        PCAR_anchor = PCAR_landmarks[PCAR_landmarks_x_coords.index(min(PCAR_landmarks_x_coords))]
    else:
        # No path found from BAB candidates to either PCAL or PCAR trees!
        pass

    # The two anchors at this point if they exist at all are either in contralateral PCA subtrees, or in ipsilateral subtree if PCA subtrees are asymmetrical.
    # Using x component of centroid_ICAL_ICAR to define a midsagital plane. If the two PCA anchors fall on same side of plane,
    # consider it a "same-sided PCA anchor" situation (this is either a True case or False case)
    # Whether true or not, select the most lateral node in L/R side to be the PCA anchor for that direction. Evolution hopefully guaantees the truth of this but edge case can always be created
    # (In actual BAT calculation, can include sanity checks to detect these edge cases resulting in ipsilateral PCA anchors despite all this logic so far)
    # Set the contralateral anchor to be the most posterior contralateral PCA landmark node from the contralateral y-region
    # if PCAL_anchor != None and PCAR_anchor != None:
    #     PCAL_anchor_x_pos = nx_g.nodes[PCAL_anchor]['pos'][0]
    #     PCAR_anchor_x_pos = nx_g.nodes[PCAR_anchor]['pos'][0]
    #     mid_sagittal_x = centroid_ICAL_ICAR[0]
    #     if PCAL_anchor_x_pos > mid_sagittal_x and PCAR_anchor_x_pos > mid_sagittal_x:
    #         # determine adjusted PCAL_anchor
    #         if PCAL_anchor_x_pos > PCAR_anchor_x_pos:
    #             PCAL_anchor = PCAL_anchor
    #         elif PCAL_anchor_x_pos <= PCAR_anchor_x_pos:
    #             PCAL_anchor = PCAR_anchor
    #
    #         # determine adjusted PCAR_anchor
    #         PCAR_landmarks_right_of_mid_saggital_x = [node for node in PCAR_landmarks if (node in nx_g.nodes) and (nx_g.nodes[node]['pos'][0] < mid_sagittal_x)]
    #         # Find the node with the maximum 'y pos' attribute among the PCAR_landmarks_right_of_mid_saggital_x nodes
    #         if PCAR_landmarks_right_of_mid_saggital_x:
    #             PCAR_anchor = max(PCAR_landmarks_right_of_mid_saggital_x, key=lambda node: nx_g.nodes[node]['pos'][1])
    #             # print("Node with max 'y pos' among those with 'x pos' > 5:", PCAR_anchor)
    #             # print("Adjusted a PCAR anchor", max_y_node)
    #         else:
    #             raise ValueError('xxx')
    #     elif PCAL_anchor_x_pos <= mid_sagittal_x and PCAR_anchor_x_pos <= mid_sagittal_x:
    #         # determine adjusted PCAR_anchor
    #         if PCAL_anchor_x_pos > PCAR_anchor_x_pos:
    #             PCAR_anchor = PCAR_anchor
    #         elif PCAL_anchor_x_pos <= PCAR_anchor_x_pos:
    #             PCAR_anchor = PCAL_anchor
    #
    #         # determine adjusted PCAL_anchor
    #         PCAL_landmarks_left_of_mid_saggital_x = [node for node in PCAL_landmarks if (node in nx_g.nodes) and (nx_g.nodes[node]['pos'][0] > mid_sagittal_x)]
    #         # Find the node with the maximum 'y pos' attribute among the PCAL_landmarks_left_of_mid_saggital_x nodes
    #         if PCAL_landmarks_left_of_mid_saggital_x:
    #             PCAL_anchor = max(PCAL_landmarks_left_of_mid_saggital_x, key=lambda node: nx_g.nodes[node]['pos'][1])
    #             # print("Node with max 'y pos' among those with 'x pos' > 5:", PCAL_anchor)
    #             # print("Adjusted a PCAL anchor", max_y_node)
    #         else:
    #             raise ValueError('xxx')
    #     else:
    #         # both PCA anchors are hopefully correct
    #         pass

    PCAL_anchor_pos_attributes = [nx_g.nodes[node]['pos'] for node in [PCAL_anchor] if node != None]
    PCAR_anchor_pos_attributes = [nx_g.nodes[node]['pos'] for node in [PCAR_anchor] if node != None]

    return PCAL_anchor, PCAR_anchor, PCAL_anchor_pos_attributes, PCAR_anchor_pos_attributes

def find_PCA_edges(nx_g, PComAL_posterior_node, PComAR_posterior_node, BAT_node):
    # Given a PComA_posterior solution or a BAT solution, define the root edge of the PCAs

    def find_PCA_edge_using_PComA(nx_g, direction, PComA_posterior_node):
        PComA_posterior_node_pos = nx_g.nodes[PComA_posterior_node]['pos']
        if direction == 'left':
            max_xpos = max(data['pos'][0] for _, data in nx_g.nodes(data=True))
            max_ypos = max(data['pos'][1] for _, data in nx_g.nodes(data=True))
            PCA_edge_landmark = [max_xpos, max_ypos, PComA_posterior_node_pos[2]]
        elif direction == 'right':
            min_xpos = min(data['pos'][0] for _, data in nx_g.nodes(data=True))
            max_ypos = max(data['pos'][1] for _, data in nx_g.nodes(data=True))
            PCA_edge_landmark = [min_xpos, max_ypos, PComA_posterior_node_pos[2]]
        else:
            raise ValueError('direction must be either left or right')

        PCA_edge_anterior_node = PComA_posterior_node
        PCA_edge_posterior_node = None
        PCA_edge_posterior_node_candidates = list(nx_g.neighbors(PComA_posterior_node))
        min_distance = float('inf')
        for neighbor in PCA_edge_posterior_node_candidates:
            neighbor_pos = nx_g.nodes[neighbor]['pos']
            distance = np.linalg.norm(np.array(neighbor_pos) - np.array(PCA_edge_landmark))
            if distance < min_distance:
                min_distance = distance
                PCA_edge_posterior_node = neighbor

        PCA_edge = (PCA_edge_anterior_node, PCA_edge_posterior_node)
        PCA_edge_pos = (nx_g.nodes[PCA_edge_anterior_node]['pos'], nx_g.nodes[PCA_edge_posterior_node]['pos'])

        return PCA_edge, PCA_edge_pos, PCA_edge_landmark
    def find_PCA_edge_using_BAT(nx_g, direction, BAT_node):

        BAT_node_pos = np.array(nx_g.nodes[BAT_node]['pos'])
        PCA_edge_anterior_node = BAT_node
        PCA_edge_posterior_node = None
        PCA_edge_posterior_node_candidates = list(nx_g.neighbors(BAT_node))
        # Of the candidates on the appropriate side to the sagittal plane at the BAT node, calculate their angle to the XY plane. most positive angle is PCA edge
        if direction == 'left':
            PCA_edge_posterior_node_candidates_ipsi = [node for node in PCA_edge_posterior_node_candidates if
                                                       nx_g.nodes[node]['pos'][0] > BAT_node_pos[0]]
        elif direction == 'right':
            PCA_edge_posterior_node_candidates_ipsi = [node for node in PCA_edge_posterior_node_candidates if
                                                       nx_g.nodes[node]['pos'][0] <= BAT_node_pos[0]]
        else:
            raise ValueError('direction must be left or right')

        # assert len(PCA_edge_posterior_node_candidates_ipsi) in [1, 2], 'xxx'
        max_angle = -float('inf')
        for PCA_edge_candidate in PCA_edge_posterior_node_candidates_ipsi:
            # Define the normal vector of the XY plane
            PCA_edge_candidate_pos = np.array(nx_g.nodes[PCA_edge_candidate]['pos'])
            vector = PCA_edge_candidate_pos - BAT_node_pos
            normal_xy_plane = np.array([0, 0, 1])
            angle_rad = np.pi / 2 - np.arccos(np.dot(vector, normal_xy_plane) / np.linalg.norm(vector))
            angle_deg = np.degrees(angle_rad)
            if angle_deg > max_angle:
                max_angle = angle_deg
                PCA_edge_posterior_node = PCA_edge_candidate

        # usually when a patients missing entire hemisphere
        if PCA_edge_posterior_node == None:
            return None, [], None

        PCA_edge = (PCA_edge_anterior_node, PCA_edge_posterior_node)
        PCA_edge_pos = (nx_g.nodes[PCA_edge_anterior_node]['pos'], nx_g.nodes[PCA_edge_posterior_node]['pos'])

        return PCA_edge, PCA_edge_pos, None

    PCAL_edge = None
    PCAL_edge_landmark_pos = None
    PCAL_edge_pos = []

    if PComAL_posterior_node != None:
        PCAL_edge, PCAL_edge_pos, PCAL_edge_landmark_pos = find_PCA_edge_using_PComA(nx_g, 'left', PComAL_posterior_node)
    elif BAT_node != None:
        PCAL_edge, PCAL_edge_pos, PCAL_edge_landmark_pos = find_PCA_edge_using_BAT(nx_g, 'left', BAT_node)
    else:
        pass
        # if do_verbose_errors:
        #     print('\t xxx No PCAL edge found')

    PCAR_edge = None
    PCAR_edge_landmark_pos = None
    PCAR_edge_pos = []
    if PComAR_posterior_node != None:
        PCAR_edge, PCAR_edge_pos, PCAR_edge_landmark_pos = find_PCA_edge_using_PComA(nx_g, 'right', PComAR_posterior_node)
    elif BAT_node != None:
        PCAR_edge, PCAR_edge_pos, PCAR_edge_landmark_pos = find_PCA_edge_using_BAT(nx_g, 'right', BAT_node)
    else:
        pass
        # if do_verbose_errors:
        #     print('\t xxx No PCAR edge found')

    PCAL_edge_eix = None
    PCAR_edge_eix = None
    if PCAL_edge != None:
        PCAL_edge_eix = int(nx_g.get_edge_data(PCAL_edge[0], PCAL_edge[1])['eix'])
    if PCAR_edge != None:
        PCAR_edge_eix = int(nx_g.get_edge_data(PCAR_edge[0], PCAR_edge[1])['eix'])

    return PCAL_edge, PCAL_edge_eix, PCAL_edge_pos, PCAL_edge_landmark_pos, PCAR_edge, PCAR_edge_eix, PCAR_edge_pos, PCAR_edge_landmark_pos

# BAT functions #####################################################################

def find_BAT_round_A(nx_g, BAB_candidates, PCAL_landmarks, PCAR_landmarks, centroid_ICAL_ICAR, MCAL_pos, MCAR_pos,
                     PComAL_anterior_node, PComAR_anterior_node, do_verbose_errors):

    def is_BAB_candidate_in_posterior_circulaltion(nx_g, BAB_candidate, PCAL_landmarks, PCAR_landmarks):
        PCA_landmarks = PCAL_landmarks + PCAR_landmarks
        for PCA_landmark in PCA_landmarks:
            try:
                SP = nx.shortest_path(nx_g, source=BAB_candidate, target=PCA_landmark)
                return True
            except:
                pass
        return False
    def find_BAT_node_using_PCA_anchors(nx_g, chosen_BAB_candidate, PCAL_anchor, PCAR_anchor, do_verbose_errors):
        # Find SP from chosen_BAB_candidate to PCA anchors
        try:
            # Find the shortest path
            SP_BAB_PCAL_anchor = nx.shortest_path(nx_g, source=chosen_BAB_candidate, target=PCAL_anchor,
                                                  weight='dist')  # liz 1133'3'33 ww.qp.wr
            SP_BAB_PCAR_anchor = nx.shortest_path(nx_g, source=chosen_BAB_candidate, target=PCAR_anchor, weight='dist')
        except:
            raise ValueError('No connection between chosen_BAB_candidate and one or both PCA anchors!')

        # Find SP from ICA to BAB candidates (if exist, then PComA exists)
        divergence_node, divergence_type = find_divergence_node(SP_BAB_PCAL_anchor, SP_BAB_PCAR_anchor,
                                                                do_verbose_errors)
        if divergence_type == 'straggler':
            if do_verbose_errors:
                print('\t Warning: No valid BAT node. Found a straggler when finding divergence node for BAT node.')
            return None, []
        assert divergence_type not in ['identical'], 'xxx'
        assert divergence_node != None, 'xxx'

        BAT_node = divergence_node
        BAT_node_pos_attributes = [nx_g.nodes[node]['pos'] for node in [BAT_node] if node != None]

        return BAT_node, BAT_node_pos_attributes

    BAT_node = None
    PCAL_anchor = None
    PCAR_anchor = None
    BAT_node_pos_attributes = []
    PCAL_anchor_pos_attributes = []
    PCAR_anchor_pos_attributes = []
    chosen_BAB_candidate = None
    BAB_candidates_connected_to_PCA_landmarks = []

    if BAB_candidates:
        # some BAB candidates might not be in BA. might be in MCA tree. or in disconnected component from posterior circulation.
        # chosen_BAB_candidate must be connected to one of PCAL/PCAR landmarks; must be in BA.
        # of eligible nodes, take the one with min distance to centroid_ICAL_ICAR.
        # min_x_distance_to_centroid_ICAL_ICAR = float('inf')
        min_distance_to_centroid_ICAL_ICAR = float('inf')
        BAB_candidates_connected_to_PCA_landmarks = [n for n in BAB_candidates if
                                                     is_BAB_candidate_in_posterior_circulaltion(nx_g, n, PCAL_landmarks, PCAR_landmarks)]
        if BAB_candidates_connected_to_PCA_landmarks:
            for BAB_candidate in BAB_candidates_connected_to_PCA_landmarks:
                pos = nx_g.nodes[BAB_candidate]['pos']
                # curr_x_distance = abs(pos[0] - centroid_ICAL_ICAR[0])
                curr_distance_from_centroid = np.linalg.norm(np.array(centroid_ICAL_ICAR) - np.array(pos))
                if curr_distance_from_centroid < min_distance_to_centroid_ICAL_ICAR:
                    min_distance_to_centroid_ICAL_ICAR = curr_distance_from_centroid
                    chosen_BAB_candidate = BAB_candidate

            # Slice the network to isolate posterior circulation. ???must also slice out MCA-MCA SP to avoid situations where PComA attaches to A1 segment??? # XAXAXA! Slice the network to isolate posterior circulation BEFORE finding the PCA landmarks!!!
            nx_g_BAT = copy.deepcopy(nx_g)
            nodes_to_remove_BAT = [PComAL_anterior_node, PComAR_anterior_node]
            nx_g_BAT.remove_nodes_from(nodes_to_remove_BAT)

            bfs_edges_BAB_candidate = list(nx.edge_bfs(nx_g_BAT, chosen_BAB_candidate))
            bfs_nodes_BAB_candidate = [item for sublist in bfs_edges_BAB_candidate for item in sublist[0:2]]
            reachable_nodes = list(set(bfs_nodes_BAB_candidate))
            unreachable_nodes = [n for n in nx_g_BAT.nodes() if n not in reachable_nodes]
            assert len(reachable_nodes) + len(unreachable_nodes) == len(nx_g_BAT.nodes), 'f3r3'
            nx_g_BAT.remove_nodes_from(unreachable_nodes)

            PCAL_anchor, PCAR_anchor, PCAL_anchor_pos_attributes, PCAR_anchor_pos_attributes = \
                find_PCA_anchors(nx_g_BAT, centroid_ICAL_ICAR, PCAL_landmarks, PCAR_landmarks, MCAL_pos, MCAR_pos)
            assert (PCAL_anchor == None and PCAL_anchor == None) or (
                        PCAL_anchor != None and PCAL_anchor != None), 'xxx'
            if do_verbose_errors:
                if PCAL_anchor == None:
                    print(f'\t LLLLL No PCAL anchor was found.')
                if PCAR_anchor == None:
                    print(f'\t LLLLL No PCAR anchor was found.')
            if PCAL_anchor != None:
                BAT_node, BAT_node_pos_attributes = find_BAT_node_using_PCA_anchors(nx_g_BAT, chosen_BAB_candidate, PCAL_anchor, PCAR_anchor, do_verbose_errors)
                # BAT_node = None
                # BAT_node_pos_attributes = []
        else:
            # None of the BAB candidates are connected to PCA landmarks
            nx_g_BAT = copy.deepcopy(nx_g)
            if do_verbose_errors:
                print('\t LLLLL No BAB candidates were found A.')
    else:
        # WHen no BAB candidates, throw error for user to manually correct. rare. Otherwise, find PCA anchors then find BAT
        nx_g_BAT = copy.deepcopy(nx_g)
        if do_verbose_errors:
            print('\t LLLLL No BAB candidates were found B.')

    return BAT_node, BAT_node_pos_attributes, PCAL_anchor, PCAR_anchor, PCAL_anchor_pos_attributes, PCAR_anchor_pos_attributes, chosen_BAB_candidate, BAB_candidates_connected_to_PCA_landmarks, nx_g_BAT

def find_BAT_round_B(nx_g, nx_g_BAT, PComAL_posterior_node, PComAR_posterior_node, BAT_node, BAT_node_pos_attributes):
    # XAXA NOT YET IN PAPER try use the PComA posterior node ruloe
    # if BAT_node != None:
    if PComAL_posterior_node != None and PComAR_posterior_node != None:
        try:
            SP_PComAL_posterior_PComAR_posterior = nx.shortest_path(nx_g_BAT, source=PComAL_posterior_node,
                                                                    target=PComAR_posterior_node)
            if len(SP_PComAL_posterior_PComAR_posterior) == 3 and SP_PComAL_posterior_PComAR_posterior[
                1] != BAT_node:
                BAT_node = SP_PComAL_posterior_PComAR_posterior[1]
                BAT_node_pos_attributes = [nx_g.nodes[node]['pos'] for node in [BAT_node] if node != None]
                # replace bad with new solution and then udpate NOTES!!!!!!!!!!
                # print(BAT_node, 'BAT UPDATED ' * 50)
        except:
            pass
    return BAT_node, BAT_node_pos_attributes

def find_BAT_round_C(nx_g, nx_g_BAT, chosen_BAB_candidate, MCAL_node, MCAR_node, ACAL_node, ACAR_node,
                     PComAL_anterior_node, PComAR_anterior_node, PComAL_posterior_node, PComAR_posterior_node,
                     PCAL_landmarks, PCAR_landmarks, BAB_candidates, BAB_candidates_connected_to_PCA_landmarks,
                     MCAL_pos, MCAR_pos, BAT_node, BAT_node_pos_attributes, do_verbose_errors):
    def does_BAB_connect_to_anterior(nx_g, chosen_BAB_candidate, MCAL_node, MCAR_node):
        try:
            SP_chosen_BAB_candidate_MCAL = nx.shortest_path(nx_g, source=chosen_BAB_candidate, target=MCAL_node)
            return True
        except:
            pass
        try:
            SP_chosen_BAB_candidate_MCAR = nx.shortest_path(nx_g, source=chosen_BAB_candidate, target=MCAR_node)
            return True
        except:
            pass
        return False

    def find_PCA_landmarks_connected_to_BAB_candidate(nx_g_BAT, chosen_BAB_candidate, PCAL_landmarks,
                                                      PCAR_landmarks):
        # check which side the BAD supplies by seeing which PCAL landmarks are connected to it, after deleting all PComAs
        PCAL_landmarks_reachable_from_chosen_BAB_candidate = [node for node in nx_g_BAT.nodes if
                                                              node in PCAL_landmarks]
        PCAR_landmarks_reachable_from_chosen_BAB_candidate = [node for node in nx_g_BAT.nodes if
                                                              node in PCAR_landmarks]
        for node in PCAL_landmarks_reachable_from_chosen_BAB_candidate + PCAR_landmarks_reachable_from_chosen_BAB_candidate:
            assert nx.has_path(nx_g_BAT, chosen_BAB_candidate, node)
        return PCAL_landmarks_reachable_from_chosen_BAB_candidate, PCAR_landmarks_reachable_from_chosen_BAB_candidate

    def find_side_supplied_by_BAB(nx_g, chosen_BAB_candidate, MCAL_node, MCAR_node, ACAL_node, ACAR_node,
                                  PComAL_anterior_node, PComAR_anterior_node, PComAL_posterior_node,
                                  PComAR_posterior_node):

        nx_g_tmp = copy.deepcopy(nx_g)
        nodes_to_remove_tmp = [PComAL_anterior_node, PComAL_posterior_node, MCAL_node, ACAL_node, ACAR_node]
        nx_g_tmp.remove_nodes_from(nodes_to_remove_tmp)
        try:
            SP_BAB_MCA = nx.shortest_path(nx_g_tmp, source=chosen_BAB_candidate, target=MCAR_node, weight='dist')
            supplies_MCAR = True
        except:
            supplies_MCAR = False

        nx_g_tmp = copy.deepcopy(nx_g)
        nodes_to_remove_tmp = [PComAR_anterior_node, PComAR_posterior_node, MCAR_node, ACAL_node, ACAR_node]
        nx_g_tmp.remove_nodes_from(nodes_to_remove_tmp)
        try:
            SP_BAB_MCA = nx.shortest_path(nx_g_tmp, source=chosen_BAB_candidate, target=MCAL_node, weight='dist')
            supplies_MCAL = True
        except:
            supplies_MCAL = False

        if supplies_MCAL and supplies_MCAR:
            return 'both'
        elif supplies_MCAL and not supplies_MCAR:
            return 'left'
        elif not supplies_MCAL and supplies_MCAR:
            return 'right'
        elif supplies_MCAL and supplies_MCAR:
            return 'none'

    def find_BAT_when_PCA_P1_missing_PComA_present(nx_g, chosen_BAB_candidate, PComAL_anterior_node,
                                                   PComAR_anterior_node, PCA_landmarks, side_supplied_by_BAB):
        if side_supplied_by_BAB == 'left':
            chosen_PComA_anterior_node = PComAL_anterior_node
        elif side_supplied_by_BAB == 'right':
            chosen_PComA_anterior_node = PComAR_anterior_node
        elif side_supplied_by_BAB == 'both':
            return None, []
        else:
            raise ValueError('xxx')

        # Find SP_BAB_MCA
        try:
            SP_BAB_MCA = nx.shortest_path(nx_g, source=chosen_BAB_candidate, target=chosen_PComA_anterior_node,
                                          weight='dist')
        except nx.NetworkXNoPath:
            return None, []

        # Find SP_BAB_PCA_landmark
        SP_BAB_PCA_landmark = None
        longest_shortest_path_length = float('-inf')
        for PCA_landmark in PCA_landmarks:
            try:
                shortest_path_length = nx.shortest_path_length(nx_g, source=chosen_BAB_candidate,
                                                               target=PCA_landmark, weight='dist')
                shortest_path = nx.shortest_path(nx_g, source=chosen_BAB_candidate, target=PCA_landmark,
                                                 weight='dist')
                if shortest_path_length > longest_shortest_path_length:
                    longest_shortest_path_length = shortest_path_length
                    SP_BAB_PCA_landmark = shortest_path
            except nx.NetworkXNoPath:
                continue
        if SP_BAB_PCA_landmark == None:
            return None, []

        # BAT divergence node
        divergence_node, divergence_type = find_divergence_node(SP_BAB_PCA_landmark, SP_BAB_MCA, do_verbose_errors)
        if divergence_type == 'straggler':
            if do_verbose_errors:
                print('\t Warning: No valid BAT node. Found a straggler when finding divergence node for BAT node.')
            return None, []
        assert divergence_type not in ['identical'], 'xxx'
        assert divergence_node != None, 'xxx'

        BAT_node = divergence_node
        BAT_node_pos_attributes = [nx_g.nodes[node]['pos'] for node in [BAT_node] if node != None]

        return BAT_node, BAT_node_pos_attributes

    def find_BAT_when_PCA_P1_missing_PComA_missing(nx_g, chosen_BAB_candidate, MCA_centroid, PCA_landmarks):
        # Find SP_BAB_PCA_landmark
        SP_BAB_PCA_landmark = None
        longest_shortest_path_length = float('-inf')
        for PCA_landmark in PCA_landmarks:
            try:
                # Find the shortest path using the 'r' metric
                shortest_path_length = nx.shortest_path_length(nx_g, source=chosen_BAB_candidate,
                                                               target=PCA_landmark, weight='dist')
                shortest_path = nx.shortest_path(nx_g, source=chosen_BAB_candidate, target=PCA_landmark,
                                                 weight='dist')
                if shortest_path_length > longest_shortest_path_length:
                    longest_shortest_path_length = shortest_path_length
                    SP_BAB_PCA_landmark = shortest_path
            except nx.NetworkXNoPath:
                continue

        # assert len(SP_BAB_PCA_landmark) > 3, 'xxx'
        if len(SP_BAB_PCA_landmark) < 3:
            return None, []

        BAT_node_candidate = None
        min_distance = float('inf')
        for node in SP_BAB_PCA_landmark[
                    1:-1]:  # BAT top should not be a BAB candidate nor the furthest PCA landmark from the BAB candidate
            node_pos = nx_g.nodes[node]['pos']
            distance = np.linalg.norm(node_pos - MCA_centroid)
            if distance < min_distance:
                min_distance = distance
                BAT_node_candidate = node
        if BAT_node_candidate != None:
            BAT_node = BAT_node_candidate
            BAT_node_pos_attributes = [nx_g.nodes[node]['pos'] for node in [BAT_node] if node != None]
            return BAT_node, BAT_node_pos_attributes
        else:
            return None, []

    # Deal with unilateral missing PCA P1 variant (unilateral fetal-type PCA)
    if BAT_node == None:
        if BAB_candidates and BAB_candidates_connected_to_PCA_landmarks:
            chosen_BAB_candidate_connects_to_anterior_circulation = does_BAB_connect_to_anterior(nx_g,
                                                                                                 chosen_BAB_candidate,
                                                                                                 MCAL_node,
                                                                                                 MCAR_node)
            PCAL_landmarks_reachable_from_chosen_BAB_candidate, PCAR_landmarks_reachable_from_chosen_BAB_candidate = find_PCA_landmarks_connected_to_BAB_candidate(
                nx_g_BAT, chosen_BAB_candidate, PCAL_landmarks, PCAR_landmarks)
            PCA_landmarks = PCAL_landmarks_reachable_from_chosen_BAB_candidate + PCAR_landmarks_reachable_from_chosen_BAB_candidate

            # PComA present: BAB connected to anterior, divergence between bab-mca and bab-posteriormost-pcalandmark
            if chosen_BAB_candidate_connects_to_anterior_circulation:
                side_supplied_by_BAB = find_side_supplied_by_BAB(nx_g, chosen_BAB_candidate, MCAL_node, MCAR_node,
                                                                 ACAL_node, ACAR_node, PComAL_anterior_node,
                                                                 PComAR_anterior_node, PComAL_posterior_node,
                                                                 PComAR_posterior_node)
                BAT_node, BAT_node_pos_attributes = find_BAT_when_PCA_P1_missing_PComA_present(nx_g,
                                                                                               chosen_BAB_candidate,
                                                                                               PComAL_anterior_node,
                                                                                               PComAR_anterior_node,
                                                                                               PCA_landmarks,
                                                                                               side_supplied_by_BAB)

            # PComA absent: find longest path from BAB to a PCA landmark, and closest point on that path to MCA_centroid
            else:
                MCA_centroid = (np.array(MCAL_pos) + np.array(MCAR_pos)) / 2
                BAT_node, BAT_node_pos_attributes = find_BAT_when_PCA_P1_missing_PComA_missing(nx_g,
                                                                                               chosen_BAB_candidate,
                                                                                               MCA_centroid,
                                                                                               PCA_landmarks)

    if BAT_node == None:
        if do_verbose_errors:
            print('\t No BAT was found')

    return BAT_node, BAT_node_pos_attributes

# Utilities #####################################################################

def sanity_check_BAT(nx_g, BAT_node, BAB_node, PComAL_anterior_node, PComAR_anterior_node, PComAL_posterior_node, PComAR_posterior_node, MCAL_pos, MCAR_pos, BAT_node_pos_attributes, params_dict, do_verbose_errors):
    # BAT node cannot be same as PComA node

    if BAT_node != None:
        if BAT_node in [PComAL_anterior_node, PComAR_anterior_node, PComAL_posterior_node, PComAR_posterior_node]:
            print('\t Error in Graph: one of the PComA nodes is the same as the BAT node')
        if BAT_node == BAB_node:
            print('GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG')
            print('GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG')
            print('GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG')
            print('\t Error in Graph: BAB node is the same as the BAT node')

        # Is BAT node on a cycle of length 5 or less?
        cycles = nx.cycle_basis(nx_g)
        for cycle in cycles:
            if len(cycle) < 6:
                if BAT_node in cycle:
                    if len(cycle) == 1:
                        if do_verbose_errors:
                            print(f'\t Warning: BAT node lies on a (likely non-CoW) SELF cycle of length {len(cycle)}')
                    else:
                        if do_verbose_errors:
                            print(f'\t Warning: BAT node lies on a (likely non-CoW) cycle of length {len(cycle)}')

        MCA_distance = np.linalg.norm(np.array(MCAL_pos) - np.array(MCAR_pos))
        MCA_centroid = (np.array(MCAL_pos) + np.array(MCAR_pos)) / 2
        neighbours_BAT = nx_g.neighbors(BAT_node)

        for neighbour in neighbours_BAT:
            curr_dist = nx_g[BAT_node][neighbour]['dist']
            BAT_pos = nx_g.nodes[BAT_node]['pos']
            neighbour_pos = nx_g.nodes[neighbour]['pos']
            if neighbour_pos[2] > BAT_pos[2]:
                if curr_dist < params_dict['BAT_superior_neighbour_thresh']:
                    print(f"\t Warning: BAT node has a very close neighbour above it. At a distance of {curr_dist}, when the upper threshold is {params_dict['BAT_superior_neighbour_thresh']}")

        # only when BAT not is NOT a neighbour of either pcoma posterior node, check for positional validity with respect to MCA nodes
        if (PComAL_posterior_node not in neighbours_BAT) and (PComAR_posterior_node not in neighbours_BAT):
            # Is BAT node anterior to intermca centroid offset posteriorly by inter_mca distance?
            if BAT_node_pos_attributes[0][1] > (MCA_centroid[1] + MCA_distance):
                # print('XXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                # print('XXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                # print('XXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                # print('XXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                # print('XXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                print('\t Warning: BAT node is further posterior than the inter-MCA centroid offset posteriorly by the inter-MCA distance.')

            # Is BAT node superior to intermca centroid offset inferiorly by inter_mca distance
            if BAT_node_pos_attributes[0][2] < (MCA_centroid[2] - MCA_distance/2):
                # print('XXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                # print('XXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                # print('XXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                # print('XXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                # print('XXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                print('\t Warning: BAT node is further inferior than the inter-MCA centroid offset inferior by the inter-MCA distance.')

            if BAT_node_pos_attributes[0][0] > MCAL_pos[0]:
                # print('XXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                # print('XXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                # print('XXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                # print('XXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                # print('XXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                print('\t Warning: BAT node is further left than MCAL.')

            if BAT_node_pos_attributes[0][0] < MCAR_pos[0]:
                print('XXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                print('XXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                print('XXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                print('XXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                print('XXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                print('\t Warning: BAT node is further right than MCAR.')

        # BAB node must reach PComA post through BAT
        if PComAL_posterior_node != None:
            try:
                shortest_path = nx.shortest_path(nx_g, source=BAB_node, target=PComAL_posterior_node)
                if BAT_node not in shortest_path:
                    print('\t Warning: SP BAB to PComA Posterior bypasses BAT')
            except:
                # No path from BAB to PComA Posterior
                print('\t No path from BAB to PComA Posterior')
                # print('XOXO XOXO XOXO !!!!!!! No path from BAB to PComA Posterior')
                pass

def sanity_check_ACA(nx_g, nx_g_ACA, dist_thresholds_dict, ICAL_node, ICAR_node, MCAL_node, MCAR_node, ACAL_node, ACAR_node, BAT_node, PComAL_anterior_node, PComAR_anterior_node, PComAL_posterior_node, PComAR_posterior_node):
    # ACA node cannot be connected to ICA after deleting MCA and PComAs

    # ACAL to ICAL
    if ACAL_node != None and ICAL_node != None:
        nx_g_sanity = copy.deepcopy(nx_g)
        nodes_to_remove = [ACAR_node, MCAL_node, MCAR_node, BAT_node, PComAL_anterior_node, PComAR_anterior_node,
                           PComAL_posterior_node, PComAR_posterior_node]
        nx_g_sanity.remove_nodes_from(nodes_to_remove)
        try:
            # If any SP is found, throw a strange ACA connectivity error
            shortest_path = nx.shortest_path(nx_g_sanity, source=ACAL_node, target=ICAL_node)
            print('\t Error in Graph: An ACA node connects to the ICA node even after deleting the MCA node')
        except:
            pass

    # ACAR to ICAR
    if ACAR_node != None and ICAR_node != None:
        nx_g_sanity = copy.deepcopy(nx_g)
        nodes_to_remove = [ACAL_node, MCAL_node, MCAR_node, BAT_node, PComAL_anterior_node, PComAR_anterior_node,
                           PComAL_posterior_node, PComAR_posterior_node]
        nx_g_sanity.remove_nodes_from(nodes_to_remove)
        try:
            # If any SP is found, throw a strange ACA connectivity error
            shortest_path = nx.shortest_path(nx_g_sanity, source=ACAR_node, target=ICAR_node)
            print('\t Error in Graph: An ACA node connects to the ICA node even after deleting the MCA node')
        except:
            pass

    # ACAL to BAT
    if ACAL_node != None and BAT_node != None:
        nx_g_sanity = copy.deepcopy(nx_g)
        nodes_to_remove = [ACAR_node, MCAL_node, MCAR_node, PComAL_anterior_node, PComAR_anterior_node,
                           PComAL_posterior_node, PComAR_posterior_node]
        nx_g_sanity.remove_nodes_from(nodes_to_remove)
        try:
            # If any SP is found, throw a strange ACA connectivity error
            shortest_path = nx.shortest_path(nx_g_sanity, source=ACAL_node, target=BAT_node)
            print('\t Error in Graph: An ACA node connects to the BAT node even after deleting the MCA node')
        except:
            pass

    # ACAR to BAT
    if ACAR_node != None and BAT_node != None:
        nx_g_sanity = copy.deepcopy(nx_g)
        nodes_to_remove = [ACAL_node, MCAL_node, MCAR_node, PComAL_anterior_node, PComAR_anterior_node,
                           PComAL_posterior_node, PComAR_posterior_node]
        nx_g_sanity.remove_nodes_from(nodes_to_remove)
        try:
            # If any SP is found, throw a strange ACA connectivity error
            shortest_path = nx.shortest_path(nx_g_sanity, source=ACAR_node, target=BAT_node)
            print('\t Error in Graph: An ACA node connects to the BAT node even after deleting the MCA node')
        except:
            pass

    # flag hanging AComA
    if ACAL_node != None and ACAR_node == None:
        find_ACA_node_across_potential_hanging_AComA(nx_g_ACA, MCAL_node, MCAR_node, ACAL_node, ACAR_node, dist_thresholds_dict['AComA_dist_thresh_upper'], 'left')
    if ACAL_node == None and ACAR_node != None:
        find_ACA_node_across_potential_hanging_AComA(nx_g_ACA, MCAL_node, MCAR_node, ACAL_node, ACAR_node, dist_thresholds_dict['AComA_dist_thresh_upper'], 'right')

def sanity_check_PCA(nx_g, nx_g_BAT, PComAL_posterior_node, PComAR_posterior_node, BAT_node, PCAL_edge, PCAR_edge, PCAL_landmarks, PCAR_landmarks, BAB_candidates):
    PCAL_landmarks_in_posterior_circulation = [node for node in PCAL_landmarks if node in nx_g.nodes]
    PCAR_landmarks_in_posterior_circulation = [node for node in PCAR_landmarks if node in nx_g.nodes]

    ###########################################################################
    # Exist dense subtree rooted between PComA Posterior and BAT. might mean ambiguous superior cerebellar existence
    for PComA_posterior_node in [PComAL_posterior_node, PComAR_posterior_node]:
        if PComA_posterior_node != None and BAT_node != None:
            try:
                SP_PComA_posterior_BAT = nx.shortest_path(nx_g, source=PComA_posterior_node, target=BAT_node)
                if len(SP_PComA_posterior_BAT) > 2:
                    for node in SP_PComA_posterior_BAT[1:-1]:
                        nx_g_sanity = copy.deepcopy(nx_g)
                        nodes_to_remove = [n for n in SP_PComA_posterior_BAT if n != node]
                        nx_g_sanity.remove_nodes_from(nodes_to_remove)
                        n_descendants = len(nx.descendants(nx_g_sanity, node))
                        if n_descendants >= 2:
                            if PComA_posterior_node == PComAL_posterior_node:
                                print('\t Warning: Confounding subtree of two or more nodes exists between PComAL posterior node and BAT node')
                            else:
                                print('\t Warning: Confounding subtree of two or more nodes exists between PComAR posterior node and BAT node')

            except:
                pass
    ###########################################################################
    # PCA edge is a short leaf
    # if PCAL_edge:
    #     if (nx_g.degree(PCAL_edge[0]) == 1 or nx_g.degree(PCAL_edge[1]) == 1):
    #         # print('\t Warning: PCAL edge is a leaf.')
    #         if nx_g[PCAL_edge[0]][PCAL_edge[1]]['dist'] < PCA_dist_thresh:
    #             print('\t Warning: PCAL edge is a short leaf.')
    #
    # if PCAR_edge:
    #     if (nx_g.degree(PCAR_edge[0]) == 1 or nx_g.degree(PCAR_edge[1]) == 1):
    #         # print('\t Warning: PCAR edge is a leaf.')
    #         if nx_g[PCAR_edge[0]][PCAR_edge[1]]['dist'] < PCA_dist_thresh:
    #             print('\t Warning: PCAR edge is a short leaf.')

    ###########################################################################
    # PCA edge extends beyond ellipsoids laterally

    # MCA_x_distance = np.abs(MCAL_pos[0] - MCAR_pos[0])
    # MCA_centroid = (np.array(MCAL_pos) + np.array(MCAR_pos)) / 2
    # x_offset = MCA_x_distance
    #
    # if PCAL_edge != None:
    #     if (nx_g.degree(PCAL_edge[0]) == 1 or nx_g.degree(PCAL_edge[1]) == 1):
    #         PCAL_posterior_pos = nx_g.nodes[PCAL_edge[1]]['pos']
    #         if PCAL_posterior_pos[0] > MCA_centroid[0] + x_offset:
    #             print('\t Warning: PCAL edge is a leaf and extends beyond ellipsoids')
    #
    # if PCAR_edge != None:
    #     if (nx_g.degree(PCAR_edge[0]) == 1 or nx_g.degree(PCAR_edge[1]) == 1):
    #         PCAR_posterior_pos = nx_g.nodes[PCAR_edge[1]]['pos']
    #         if PCAR_posterior_pos[0] < MCA_centroid[0] - x_offset:
    #             print('\t Warning: PCAR edge is a leaf and extends beyond ellipsoids')

    ################################################################
    # Once a PCA edge is deleted, there cannot remain a connection from the PCA edge posterior node to a BAB candidate!
    # Nor can there remain a connection from the PCA posterior to the contralateral PCA edge

    # if PCAL_edge != None:
    #     nx_g_sanity_PCAL_edge = copy.deepcopy(nx_g)
    #     nx_g_sanity_PCAL_edge.remove_edge(PCAL_edge[0], PCAL_edge[1])
    #     SP_PCAL_posterior_BAB_candidate = []
    #     for BAB_candidate in BAB_candidates:
    #         try:
    #             SP_PCAL_posterior_BAB_candidate = nx.shortest_path(nx_g_sanity_PCAL_edge, source=PCAL_edge[1], target=BAB_candidate)
    #             break
    #         except:
    #             pass
    #     if len(SP_PCAL_posterior_BAB_candidate) != 0:
    #         print('\t Warning: There exists a path from PCAL posterior to a BAB despite deleting PCAL edge. Possible PCA/Superior Cerebellar connection')
    #
    #     ############
    #
    #     if PCAR_edge != None:
    #         SP_PCAL_posterior_contralateral_PCA = []
    #         for contralateral_PCA_node in [PCAR_edge[0], PCAR_edge[1]]:
    #             try:
    #                 SP_PCAL_posterior_contralateral_PCA = nx.shortest_path(nx_g_sanity_PCAL_edge, source=PCAL_edge[1], target=contralateral_PCA_node)
    #                 break
    #             except:
    #                 pass
    #         if len(SP_PCAL_posterior_contralateral_PCA) != 0:
    #             print('\t Warning: There exists a path from PCAL posterior to the PCAR edge despite deleting PCAL edge.')
    #
    # if PCAR_edge != None:
    #     nx_g_sanity_PCAR_edge = copy.deepcopy(nx_g)
    #     nx_g_sanity_PCAR_edge.remove_edge(PCAR_edge[0], PCAR_edge[1])
    #     SP_PCAR_posterior_BAB_candidate = []
    #     for BAB_candidate in BAB_candidates:
    #         try:
    #             SP_PCAR_posterior_BAB_candidate = nx.shortest_path(nx_g_sanity_PCAR_edge, source=PCAR_edge[1], target=BAB_candidate)
    #             break
    #         except:
    #             pass
    #     if len(SP_PCAR_posterior_BAB_candidate) != 0:
    #         print('\t Warning: There exists a path from PCAR posterior to a BAB despite deleting PCAR edge. Possible PCA/Superior Cerebellar connection')
    #
    #     ############
    #
    #     if PCAL_edge != None:
    #         SP_PCAR_posterior_contralateral_PCA = []
    #         for contralateral_PCA_node in [PCAL_edge[0], PCAL_edge[1]]:
    #             try:
    #                 SP_PCAR_posterior_contralateral_PCA = nx.shortest_path(nx_g_sanity_PCAR_edge, source=PCAR_edge[1], target=contralateral_PCA_node)
    #                 break
    #             except:
    #                 pass
    #         if len(SP_PCAR_posterior_contralateral_PCA) != 0:
    #             print('\t Warning: There exists a path from PCAR posterior to the PCAL edge despite deleting the PCAR edge.')

    ################################################################
    # angle between PCA edge posterior nodes should not be more than 45 degrees
    # angle between PCA edges when both BAT rooted cannot be more than 45 degrees
    # if PCAL_edge != None and PCAR_edge != None:
    #     PCAL_posterior_pos = np.array(nx_g.nodes[PCAL_edge[1]]['pos'])
    #     PCAR_posterior_pos = np.array(nx_g.nodes[PCAR_edge[1]]['pos'])
    #     PCA_edge_posterior_angle_radians = np.arcsin(np.abs(PCAL_posterior_pos[2] - PCAR_posterior_pos[2]) / np.linalg.norm(PCAL_posterior_pos - PCAR_posterior_pos))
    #     PCA_edge_posterior_angle_degrees = np.degrees(PCA_edge_posterior_angle_radians)
    #     if PCA_edge_posterior_angle_degrees > 45:
    #         print('\t Warning: Large >45 deg angle between PCA edge posterior nodes')

        #######
        # When a PCA edge is BAT rooted, check for verticalitry
        # PCAL_anterior_pos = np.array(nx_g.nodes[PCAL_edge[0]]['pos'])
        # PCAR_anterior_pos = np.array(nx_g.nodes[PCAR_edge[0]]['pos'])
        #
        # if PCAL_edge[0] == BAT_node:
        #     vector_BAT_PCAL = PCAL_posterior_pos - PCAL_anterior_pos
        #     vector_BAT_PCAL_norm = vector_BAT_PCAL / np.linalg.norm(vector_BAT_PCAL)
        #     dot_product = np.dot(vector_BAT_PCAL_norm, np.array([0, 0, 1]))
        #     angle_radians = np.arccos(dot_product)
        #     angle_degrees = np.degrees(angle_radians)
        #     if angle_degrees < 20 and nx_g.edges[PCAL_edge[0], PCAL_edge[1]]['dist'] > 10:
        #         print('\t Warning: BAT-rooted PCAL edge is highly vertical and not very short', angle_degrees, nx_g.edges[PCAL_edge[0], PCAL_edge[1]]['dist'])
        #
        # if PCAR_edge[0] == BAT_node:
        #     vector_BAT_PCAR = PCAR_posterior_pos - PCAR_anterior_pos
        #     vector_BAT_PCAR_norm = vector_BAT_PCAR / np.linalg.norm(vector_BAT_PCAR)
        #     dot_product = np.dot(vector_BAT_PCAR_norm, np.array([0, 0, 1]))
        #     angle_radians = np.arccos(dot_product)
        #     angle_degrees = np.degrees(angle_radians)
        #     if angle_degrees < 20 and nx_g.edges[PCAR_edge[0], PCAR_edge[1]]['dist'] > 10:
        #         print('\t Warning: BAT-rooted PCAR edge is highly vertical and not very short', angle_degrees, nx_g.edges[PCAR_edge[0], PCAR_edge[1]]['dist'])


        #######
        # Both PCAs are BAT rooted
        # if PCAL_edge[0] == PCAR_edge[0]:
        #     assert PCAL_edge[0] == BAT_node and PCAR_edge[0] == BAT_node, 'xxx'
        #     PCAL_anterior_pos = np.array(nx_g.nodes[PCAL_edge[0]]['pos'])
        #     PCAR_anterior_pos = np.array(nx_g.nodes[PCAR_edge[0]]['pos'])
        #
        #     PCAL_posterior_pos[0] = 0
        #     PCAR_posterior_pos[0] = 0
        #     PCAL_anterior_pos[0] = 0
        #     PCAR_anterior_pos[0] = 0
        #
        #     vector_PCAL = PCAL_posterior_pos - PCAL_anterior_pos
        #     vector_PCAR = PCAR_posterior_pos - PCAR_anterior_pos
        #     dot_product = np.dot(vector_PCAL, vector_PCAR)
        #     magnitude_AB = np.linalg.norm(vector_PCAL)
        #     magnitude_AC = np.linalg.norm(vector_PCAR)
        #
        #     angle_radians = np.arccos(dot_product / (magnitude_AB * magnitude_AC))
        #     angle_degrees = np.degrees(angle_radians)
        #     print('xoxo xoxo xoxo', angle_degrees)


    # if len(PCAL_landmarks_in_posterior_circulation) <= 2:
    #     print('\t 2 or less PCAL landmarks detected. Check the PCAL subtree')
    # if len(PCAR_landmarks_in_posterior_circulation) <= 2:
    #     print('\t 2 or less PCAR landmarks detected. Check the PCAR subtree')

def sanity_check_MCA(nx_g, ICAL_node, ICAR_node, MCAL_node, MCAR_node, ACA_cone_nodes, MCAL_landmark, MCAR_landmark, PCAL_landmarks, PCAR_landmarks, PComAL_anterior_node, PComAR_anterior_node):
    PCAL_landmarks_in_posterior_circulation = [node for node in PCAL_landmarks if node in nx_g.nodes]
    PCAR_landmarks_in_posterior_circulation = [node for node in PCAR_landmarks if node in nx_g.nodes]

    ######################################################################################################
    #

    ######################################################################################################
    # SP ICA to MCA Landmark should not contain an ACA cone node. If it does probably an aberrant edge from anterior ciruclation to MCA circulation
    for ICA_node, MCA_landmark in [(ICAL_node, MCAL_landmark), (ICAR_node, MCAR_landmark)]:
        try:
            SP_ICA_MCA_landmark = nx.shortest_path(nx_g, source=ICA_node, target=MCA_landmark)
            for node in SP_ICA_MCA_landmark:
                if node in ACA_cone_nodes:
                    if ICA_node == ICAL_node:
                        print('\t Warning: SP from ICAL to MCAL Landmark contains an ACA landmark')
                        break
                    else:
                        print('\t Warning: SP from ICAR to MCAR Landmark contains an ACA landmark')
                        break
        except:
            raise ValueError('xxx')

    ######################################################################################################
    # Check for Posterior extension PCA landmark originating from SP MCAL MCAL landmark path
    try:
        SP_MCAL_MCAL_landmark = nx.shortest_path(nx_g, source=MCAL_node, target=MCAL_landmark)
    except:
        raise ValueError('xxx')
    try:
        SP_MCAR_MCAR_landmark = nx.shortest_path(nx_g, source=MCAR_node, target=MCAR_landmark)
    except:
        raise ValueError('xxx')

    # # Posterior extension PCA landmark found on SP MCAR MCAR landmark
    # MCA_x_distance = np.abs(MCAL_pos[0] - MCAR_pos[0])
    # MCA_centroid = (np.array(MCAL_pos) + np.array(MCAR_pos)) / 2
    # y_threshold = MCA_centroid[1] + MCA_x_distance
    # x_offset = MCA_x_distance
    #
    # for node in SP_MCAL_MCAL_landmark[:2]: # only need to check nodes in vicinity of MCA
    #     if node in PCAL_landmarks + PCAR_landmarks:
    #         neighbours = nx_g.neighbors(node)
    #         for nei in neighbours:
    #             pos = nx_g.nodes[nei]['pos']
    #             if pos[1] > y_threshold and pos[0] < MCA_centroid[0] + x_offset:
    #                 print('\t Warning: Posterior extension PCA landmark found on SP MCAL MCAL landmark')
    #
    # for node in SP_MCAR_MCAR_landmark[:2]:
    #     if node in PCAL_landmarks + PCAR_landmarks:
    #         neighbours = nx_g.neighbors(node)
    #         for nei in neighbours:
    #             pos = nx_g.nodes[nei]['pos']
    #             if pos[1] > y_threshold and pos[0] > MCA_centroid[0] - x_offset:
    #                 print('\t Warning: Posterior extension PCA landmark found on SP MCAR MCAR landmark')

    ######################################################################################################
    # Check if MCA lies on a NON COW cycle. Cow cycle minimum 6 nodes (allowing merged ACA situation)

    ######################################################################################################
    # Check if MCA node is same as a PComA anterior node

    if PComAL_anterior_node == MCAL_node:
        print('\t Warning: PComAL anterior node is the same as MCAL')

    if PComAR_anterior_node == MCAR_node:
        print('\t Warning: PComAR anterior node is the same as MCAR')

# XAXA done
def sanity_check_PComA(nx_g, nx_g_PComAL, nx_g_PComAR, dist_thresholds_dict, PComAL_anterior_node, PComAL_posterior_node, PComAR_anterior_node, PComAR_posterior_node, BAT_node, BAB_node, MCAL_node, MCAR_node, ICAL_node, ICAR_node, ACAL_node, ACAR_node, do_verbose_errors):

    check_for_double_PComA(nx_g, PComAL_anterior_node, PComAL_posterior_node, PComAR_anterior_node, PComAR_posterior_node, ICAL_node, MCAL_node, ACAL_node, ICAR_node, MCAR_node, ACAR_node, BAT_node, do_verbose_errors)

    # Check for PComA attached to A1 segment situation. if PComA_anterior NOT on SP ICA-MCA, current PComA path-based logic fails so throw error
    if PComAL_anterior_node != None:
        if MCAL_node != None and ACAL_node != None:
            SP_ACAL_MCAL = find_shortest_path_dont_throw_errors(nx_g, ACAL_node, MCAL_node, [PComAL_posterior_node, PComAR_anterior_node, PComAR_posterior_node, MCAR_node])
            if PComAL_anterior_node in SP_ACAL_MCAL:
                print('\t Warning: PComAL_anterior_node on SP_ACAL_MCAL! Potential PComA attached to A1 segment situation.')

    if PComAR_anterior_node != None:
        if MCAR_node != None and ACAR_node != None:
            SP_ACAR_MCAR = find_shortest_path_dont_throw_errors(nx_g, ACAR_node, MCAR_node, [PComAR_posterior_node, PComAL_anterior_node, PComAL_posterior_node, MCAL_node])
            if PComAR_anterior_node in SP_ACAR_MCAR:
                print('\t Warning: PComAR_anterior_node on SP_ACAR_MCAR! Potential PComA attached to A1 segment situation.')

    if PComAL_anterior_node != None and PComAL_posterior_node != None:
        dist_x = calc_dist_along_SP(nx_g, PComAL_anterior_node, PComAL_posterior_node)
        if dist_x >= dist_thresholds_dict['PComAL_dist_thresh_upper']:
            print(f"\t PComAL anterior to posterior distance exceeds {dist_thresholds_dict['PComAL_dist_thresh_upper']}")
            # print('PPPPLLLLLLLLLLLLLLLLLLLL')
            # print('PPPPLLLLLLLLLLLLLLLLLLLL')
            # print('PPPPLLLLLLLLLLLLLLLLLLLL')
            # print('PPPPLLLLLLLLLLLLLLLLLLLL')
            # print('PPPPLLLLLLLLLLLLLLLLLLLL')

    if PComAR_anterior_node != None and PComAR_posterior_node != None:
        dist_x = calc_dist_along_SP(nx_g, PComAR_anterior_node, PComAR_posterior_node)
        if dist_x >= dist_thresholds_dict['PComAR_dist_thresh_upper']:
            print(f"\t PComAR anterior to posterior distance exceeds {dist_thresholds_dict['PComAR_dist_thresh_upper']}")
            # print('PPPRRRRRRRRRRRRRRRRRR')
            # print('PPPRRRRRRRRRRRRRRRRRR')
            # print('PPPRRRRRRRRRRRRRRRRRR')
            # print('PPPRRRRRRRRRRRRRRRRRR')
            # print('PPPRRRRRRRRRRRRRRRRRR')

    if PComAL_anterior_node != None and MCAL_node != None:
        try:
            sp_ICAL_MCAL = nx.shortest_path(nx_g, source=ICAL_node, target=MCAL_node)
            if PComAL_anterior_node not in sp_ICAL_MCAL:
                print(f'\t PComAL_anterior_node not on sp_ICAL_MCAL!') # XAXA done
                # print(f'\t ~o~o~o~o~o~o~o~o~o~o~o~o~o~o~o~o~o~o PComAL_anterior_node not on sp_ICAL_MCAL!')
                # print(f'\t ~o~o~o~o~o~o~o~o~o~o~o~o~o~o~o~o~o~o PComAL_anterior_node not on sp_ICAL_MCAL!')
                # print(f'\t ~o~o~o~o~o~o~o~o~o~o~o~o~o~o~o~o~o~o PComAL_anterior_node not on sp_ICAL_MCAL!')
        except:
            raise ValueError("No path found from ICAR to MCAR!")

    if PComAR_anterior_node != None and MCAR_node != None:
        try:
            sp_ICAR_MCAR = nx.shortest_path(nx_g, source=ICAR_node, target=MCAR_node)
            if PComAR_anterior_node not in sp_ICAR_MCAR:
                print(f'\t PComAR_anterior_node not on sp_ICAR_MCAR!') # XAXA done
                # print(f'\t ~o~o~o~o~o~o~o~o~o~o~o~o~o~o~o~o~o~o PComAR_anterior_node not on sp_ICAR_MCAR!')
                # print(f'\t ~o~o~o~o~o~o~o~o~o~o~o~o~o~o~o~o~o~o PComAR_anterior_node not on sp_ICAR_MCAR!')
                # print(f'\t ~o~o~o~o~o~o~o~o~o~o~o~o~o~o~o~o~o~o PComAR_anterior_node not on sp_ICAR_MCAR!')
        except:
            raise ValueError("No path found from ICAR to MCAR!")

    if PComAL_posterior_node != None and BAT_node != None:
        if PComAL_posterior_node == BAT_node:
            print('\t Warning: PComAL posterior node is the same as BAT')

        nx_g_PComAL_posterior = copy.deepcopy(nx_g)
        nodes_to_remove_PComAL = [ACAL_node, ACAR_node, MCAL_node, MCAR_node, PComAL_anterior_node, PComAR_anterior_node, PComAR_posterior_node]
        nx_g_PComAL_posterior.remove_nodes_from(nodes_to_remove_PComAL)
        try:
            a_path = calc_dist_along_SP(nx_g_PComAL, PComAL_posterior_node, BAT_node)
        except:
            print('\t Warning: PComAL posterior and BAT node both exist, but not connected via PCA P1 segment.')
            # print('7' * 50)
            # print('7' * 50)
            # print('7' * 50)
            # print('7' * 50)
            # print('7' * 50)

    if PComAL_posterior_node != None and BAB_node != None:
        if PComAL_posterior_node == BAB_node:
            print('\t Warning: PComAL posterior node is the same as BAB')

    if PComAR_posterior_node != None and BAT_node != None:
        if PComAR_posterior_node == BAT_node:
            print('\t Warning: PComAR posterior node is the same as BAT')

        nx_g_PComAR_posterior = copy.deepcopy(nx_g)
        nodes_to_remove_PComAR = [ACAL_node, ACAR_node, MCAL_node, MCAR_node, PComAR_anterior_node, PComAL_anterior_node, PComAL_posterior_node]
        nx_g_PComAR_posterior.remove_nodes_from(nodes_to_remove_PComAR)
        try:
            a_path = calc_dist_along_SP(nx_g_PComAR, PComAR_posterior_node, BAT_node)
        except:
            print('\t Warning: PComAR posterior and BAT node both exist, but not connected via PCA P1 segment.')
            # print('8' * 50)
            # print('8' * 50)
            # print('8' * 50)
            # print('8' * 50)
            # print('8' * 50)

    if PComAR_posterior_node != None and BAB_node != None:
        if PComAR_posterior_node == BAB_node:
            print('\t Warning: PComAR posterior node is the same as BAB')

def find_divergence_node(path_A, path_B, do_verbose_errors):
    # Iterate through nodes in both paths pairwise, stopping when the shorter path ends
    # assert len(path_A) > 2, 'xxx'
    # assert len(path_B) > 2, 'xxx'
    if do_verbose_errors:
        if len(path_A) <= 2:
            print('\t Warning: Invalid paths when finding divergence node')
        if len(path_B) <= 2:
            print('\t Warning: Invalid paths when finding divergence node')

    divergence_node = None
    divergence_type = None
    prev_node = None
    for node_A, node_B in zip(path_A, path_B):
        if node_A != node_B:
            divergence_node = prev_node  # or node_B, both paths diverge at this point
            divergence_type = 'normal'
            break
        else:
            prev_node = node_A
    else:
        # Handle the case where the shorter path ends but the longer one continues
        if len(path_A) == len(path_B):
            divergence_type = 'identical'
        elif len(path_A) < len(path_B):
            divergence_node = path_A[-1]
            divergence_type = 'straggler' # 1,2,3,4,5 // 1,2,3,4
        elif len(path_A) > len(path_B):
            divergence_node = path_B[-1]
            divergence_type = 'straggler' # 1,2,3,4,5 // 1,2,3,4

    return divergence_node, divergence_type

def find_shortest_path(nx_g, source_node, target_node, nodes_to_remove=[]):
    if source_node == None or target_node == None:
        return []

    nx_g_copy = copy.deepcopy(nx_g)
    nx_g_copy.remove_nodes_from(nodes_to_remove)

    try:
        # Find the shortest path
        shortest_path = nx.shortest_path(nx_g_copy, source=source_node, target=target_node)
        # Print the shortest path
        # print("Shortest path from node {} to node {}: {}".format(source_node, target_node, shortest_path))
        # return shortest_path

    except nx.NetworkXNoPath:
        # print("There is no path from node {} to node {} in the graph.".format(source_node, target_node))
        raise ValueError('xxx')
    except nx.NodeNotFound as e:
        # Handle the case where one or both of the specified nodes do not exist in the graph
        print("One of either source_node or target_node does not exist.")
        raise ValueError('xxx')
    except Exception as e:
        # Handle other exceptions that might occur
        print("An error occurred during shortest path calculation:", e)
        raise ValueError('xxx')

    return shortest_path

def find_shortest_path_dont_throw_errors(nx_g, source_node, target_node, nodes_to_remove=[]):
    shortest_path = []

    if source_node == None or target_node == None:
        return shortest_path

    nx_g_copy = copy.deepcopy(nx_g)
    nx_g_copy.remove_nodes_from(nodes_to_remove)

    try:
        # Find the shortest path
        shortest_path = nx.shortest_path(nx_g_copy, source=source_node, target=target_node)
        # Print the shortest path
        # print("Shortest path from node {} to node {}: {}".format(source_node, target_node, shortest_path))
        # return shortest_path
    except:
        pass

    return shortest_path

def calc_dist_along_SP(nx_g, source_node, target_node):
    try:
        # Find the shortest path
        shortest_path = nx.shortest_path(nx_g, source=source_node, target=target_node, weight='dist')
        # Print the shortest path
        # print("Shortest path from node {} to node {}: {}".format(source_node, target_node, shortest_path))
        # return shortest_path

    except nx.NetworkXNoPath:
        # print("There is no path from node {} to node {} in the graph.".format(source_node, target_node))
        raise ValueError('xxx')
    except nx.NodeNotFound as e:
        # Handle the case where one or both of the specified nodes do not exist in the graph
        print("One of either source_node or target_node does not exist.")
        raise ValueError('xxx')
    except Exception as e:
        # Handle other exceptions that might occur
        print("An error occurred during shortest path calculation:", e)
        raise ValueError('xxx')

    dist_along_SP = 0
    for i in range(len(shortest_path) - 1):
        u, v = shortest_path[i], shortest_path[i + 1]
        curr_dist = nx_g[u][v]['dist']
        dist_along_SP += curr_dist

    return dist_along_SP

def plot_raw_network(c, tag, cohort, nx_g, html_path, html_type):
        # html_path = f'/hpc/jshe690/jshe690/Desktop/Jiantao/tensorflow-tutorial/GNNART/graph/htmls/{cohort}/skipped/{c}_{tag}.html'
        new_tag = f'{c}_{tag}'

        edges = np.zeros((nx_g.number_of_edges(), 12), dtype=np.float32)
        # edges = np.zeros((nx_g_BAT.number_of_edges(), 12), dtype=np.float32)
        # edges = np.zeros((nx_g_ACA.number_of_edges(), 12), dtype=np.float32)
        # edges = np.zeros((nx_g_PCA.number_of_edges(), 12), dtype=np.float32)
        lookup_coord_class = {}
        for i, edge in enumerate(nx_g.edges):
            # for i, edge in enumerate(nx_g_BAT.edges):
            # for i, edge in enumerate(nx_g_ACA.edges):
            # for i, edge in enumerate(nx_g_PCA.edges):
            u = nx_g.nodes[edge[0]]
            v = nx_g.nodes[edge[1]]
            # u = nx_g_BAT.nodes[edge[0]]
            # v = nx_g_BAT.nodes[edge[1]]
            # u = nx_g_ACA.nodes[edge[0]]
            # v = nx_g_ACA.nodes[edge[1]]
            # u = nx_g_PCA.nodes[edge[0]]
            # v = nx_g_PCA.nodes[edge[1]]

            u_coord = (int(u['pos'][0]), int(u['pos'][1]), int(u['pos'][2]))
            v_coord = (int(v['pos'][0]), int(v['pos'][1]), int(v['pos'][2]))
            if u_coord not in lookup_coord_class:
                lookup_coord_class[u_coord] = [int(u['boitype']), edge[0]]
                # raise ValueError('class should be from indices not from GT labels...')
            if v_coord not in lookup_coord_class:
                lookup_coord_class[v_coord] = [int(v['boitype']), edge[1]]

            edges[i, 3] = u_coord[0]
            edges[i, 4] = u_coord[1]
            edges[i, 5] = u_coord[2]
            edges[i, 6] = v_coord[0]
            edges[i, 7] = v_coord[1]
            edges[i, 8] = v_coord[2]
            edges[i, 9] = nx_g.edges[edge[0], edge[1]]['dist']
            edges[i, 10] = nx_g.edges[edge[0], edge[1]]['rad']
            # print(u_coord, v_coord)

        save_network_graph(edges, 23, lookup_coord_class,
                           html_path, new_tag, html_type, fiducial_nodes=[[0, 0, 0]],
                           fiducial_nodes2=[[0, 0, 0], [0, 0, 0], [0, 0, 0],
                                            [0, 0, 0], [0, 0, 0]],
                           fiducial_nodes3=[[0, 0, 0]],
                           MCA_landmarks=[[0, 0, 0]], MCA=[[0, 0, 0]],
                           ACA=[[0, 0, 0]], PComA_anterior=[[0, 0, 0], [0, 0, 0]],
                           PComA_posterior=[[0, 0, 0], [0, 0, 0]],
                           PCAL_landmarks=[[0, 0, 0]], PCAR_landmarks=[[0, 0, 0]],
                           PCAL_landmarks_sphere=[0, 0, 0, 0], PCAR_landmarks_sphere=[0, 0, 0, 0],
                           PCAL_landmarks_ellipsoid=[0, 0, 0, 0, 0, 0], PCAR_landmarks_ellipsoid=[0, 0, 0, 0, 0, 0],
                           PCAL_anchor=[[0, 0, 0]], PCAR_anchor=[[0, 0, 0]],
                           BAT_node=[[0, 0, 0]], BAB_node=[[0, 0, 0]],
                           BAB_cylinder=[[0, 0, 0], 0, 0, 0, [0, 0, 0], [0, 0, 0]],
                           ACA_cone=[[0, 0, 0], [0, 0, 0], 0],
                           MCA_phantom=[[0, 0, 0]], PCAL_edge=[[0, 0, 0], [0, 0, 0]], PCAR_edge=[[0, 0, 0], [0, 0, 0]],
                           PCAL_edge_landmark=[0, 0, 0], PCAR_edge_landmark=[0, 0, 0],
                           ICA=[[0, 0, 0], [0, 0, 0]])

def get_rescaling_factor(cohort):
    if cohort == 'CROPCheck':
        res = 0.3515625
    elif cohort == 'ArizonaCheck':
        res = 0.3906
    elif cohort == 'BRAVE':
        res = 0.4297
    elif cohort == 'Parkinson2TPCheck':
        res = 0.399120
    elif cohort == 'UNC':
        res = 0.51339286
    elif cohort == 'Anzhen':
        res = 0.469
    else:
        raise ValueError('Invalid cohort name')

    return res

def load_ACA_PComA_dist_thresholds(node_labelling_dist_thresholds_root, cohort, xval_split='', do_verbose_progress=True):
    if cohort in ['Anzhen', 'ArizonaCheck', 'BRAVE', 'CROPCheck', 'Parkinson2TPCheck', 'IXI-IOP', 'HITH_all', 'Any']:
        dataset_tags = ['Anzhen', 'ArizonaCheck', 'BRAVE', 'CROPCheck', 'Parkinson2TPCheck']
        extra = '_distsRescaled'
    elif cohort in ['UNC']:
        dataset_tags = ['Anzhen', 'ArizonaCheck', 'BRAVE', 'CROPCheck', 'Parkinson2TPCheck']
        extra = '_distsRescaled'
    else:
        raise ValueError('xxx')

    ACAL1_dists = np.array([])
    ACAR1_dists = np.array([])
    AComA_dists = np.array([])
    PComAL_dists = np.array([])
    PComAR_dists = np.array([])

    for tag in dataset_tags:
        # res = get_rescaling_factor(cohort)
        # res = 1.0
        with open(f'{node_labelling_dist_thresholds_root}/{tag}_A1L{xval_split}{extra}.json', 'r') as f:
            ACAL1_dists = np.append(ACAL1_dists, np.array(json.load(f)))
        with open(f'{node_labelling_dist_thresholds_root}/{tag}_A1R{xval_split}{extra}.json', 'r') as f:
            ACAR1_dists = np.append(ACAR1_dists, np.array(json.load(f)))
        with open(f'{node_labelling_dist_thresholds_root}/{tag}_ACom{xval_split}{extra}.json', 'r') as f:
            AComA_dists = np.append(AComA_dists, np.array(json.load(f)))
        with open(f'{node_labelling_dist_thresholds_root}/{tag}_PComAL{xval_split}{extra}.json', 'r') as f:
            PComAL_dists = np.append(PComAL_dists, np.array(json.load(f)))
        with open(f'{node_labelling_dist_thresholds_root}/{tag}_PComAR{xval_split}{extra}.json', 'r') as f:
            PComAR_dists = np.append(PComAR_dists, np.array(json.load(f)))
        # PComAL_dists = [0,0,0]
        # PComAR_dists = [0,0,0]

    std_factor = 2.0

    ACAL1_mean = np.array(ACAL1_dists).mean()
    ACAL1_std = np.array(ACAL1_dists).std()
    ACAL1_dist_thresh_upper = ACAL1_mean + std_factor * ACAL1_std
    ACAL1_dist_thresh_lower = ACAL1_mean - std_factor * ACAL1_std

    ACAR1_mean = np.array(ACAR1_dists).mean()
    ACAR1_std = np.array(ACAR1_dists).std()
    ACAR1_dist_thresh_upper = ACAR1_mean + std_factor * ACAR1_std
    ACAR1_dist_thresh_lower = ACAR1_mean - std_factor * ACAR1_std

    AComA_mean = np.array(AComA_dists).mean()
    AComA_std = np.array(AComA_dists).std()
    AComA_dist_thresh_upper = AComA_mean + std_factor * AComA_std
    AComA_dist_thresh_lower = AComA_mean - std_factor * AComA_std

    PComAL_mean = np.array(PComAL_dists).mean()
    PComAL_std = np.array(PComAL_dists).std()
    PComAL_dist_thresh_upper = PComAL_mean + std_factor * PComAL_std
    PComAL_dist_thresh_lower = PComAL_mean - std_factor * PComAL_std

    PComAR_mean = np.array(PComAR_dists).mean()
    PComAR_std = np.array(PComAR_dists).std()
    PComAR_dist_thresh_upper = PComAR_mean + std_factor * PComAR_std
    PComAR_dist_thresh_lower = PComAR_mean - std_factor * PComAR_std

    if do_verbose_progress:
        print(f'\t   ACA dist thresholds, L:[{ACAL1_dist_thresh_lower:.2f}-{ACAL1_dist_thresh_upper:.2f}] R:[{ACAR1_dist_thresh_lower:.2f}-{ACAR1_dist_thresh_upper:.2f}]')
        print(f'\t AComA dist thresholds,   [{AComA_dist_thresh_lower:.2f}-{AComA_dist_thresh_upper:.2f}]')
        print(f'\t PComA dist thresholds, L:[{PComAL_dist_thresh_lower:.2f}-{PComAL_dist_thresh_upper:.2f}] R:[{PComAR_dist_thresh_lower:.2f}-{PComAR_dist_thresh_upper:.2f}]')

    # if cohort in ['midas', 'IXI-IOP']:
    #     ACAL1_dist_thresh_upper /= 3
    #     ACAL1_dist_thresh_lower /= 3
    #     ACAR1_dist_thresh_upper /= 3
    #     ACAR1_dist_thresh_lower /= 3
    #     AComA_dist_thresh_upper /= 3
    #     AComA_dist_thresh_lower /= 3
    #     PComAL_dist_thresh_upper /= 3
    #     PComAL_dist_thresh_lower /= 3
    #     PComAR_dist_thresh_upper /= 3
    #     PComAR_dist_thresh_lower /= 3
    #     print('Certain params modified!!!')
    # return ACAL1_dist_thresh_upper, ACAL1_dist_thresh_lower, ACAR1_dist_thresh_upper, ACAR1_dist_thresh_lower, AComA_dist_thresh_upper, AComA_dist_thresh_lower
    # return ACAL1_dist_thresh_upper, ACAL1_dist_thresh_lower, ACAR1_dist_thresh_upper, ACAR1_dist_thresh_lower, AComA_dist_thresh_upper, AComA_dist_thresh_lower, PComAL_dist_thresh_upper, PComAL_dist_thresh_lower, PComAR_dist_thresh_upper, PComAR_dist_thresh_lower
    return {
        "ACAL1_dist_thresh_upper": ACAL1_dist_thresh_upper,
        "ACAL1_dist_thresh_lower": ACAL1_dist_thresh_lower,
        "ACAR1_dist_thresh_upper": ACAR1_dist_thresh_upper,
        "ACAR1_dist_thresh_lower": ACAR1_dist_thresh_lower,
        "AComA_dist_thresh_upper": AComA_dist_thresh_upper,
        "AComA_dist_thresh_lower": AComA_dist_thresh_lower,
        "PComAL_dist_thresh_upper": PComAL_dist_thresh_upper,
        "PComAL_dist_thresh_lower": PComAL_dist_thresh_lower,
        "PComAR_dist_thresh_upper": PComAR_dist_thresh_upper,
        "PComAR_dist_thresh_lower": PComAR_dist_thresh_lower
    }

def load_node_labelling_gt_csv(node_labelling_gt_path):
    L = []
    with open(node_labelling_gt_path, mode='r') as file:
        reader = csv.reader(file)
        i = 0
        for row in reader:
            if i > 0:
                x = row[0]
                y = int(row[1])  # Column 1 (second column, 0-based index))
                z = int(row[2])  # Column 2 (third column, 0-based index))
                L.append([x, y, z])
            i += 1
    return L

def initialise_node_labelling_output_file(node_labelling_output_path, do_output_node_labelling_predictions):
    if do_output_node_labelling_predictions:
        header = f'Name,ICAL,ICAR,MCAL,MCAR,ACAL,ACAR,BAP,BAD,9,10,11,12\n'
        with open(node_labelling_output_path, 'w', newline='\n') as file:
            file.write(header)

def output_curr_node_labelling_predictions(tag, c, node_labelling_predictions, node_labelling_output_path, do_output_node_labelling_predictions):
    if do_output_node_labelling_predictions:
        node_labelling_predictions_formatted = [f'{c}_{tag}']
        node_labelling_predictions_formatted += [value if value is not None else '' for key, value in node_labelling_predictions.items()]
        with open(node_labelling_output_path, 'a', newline='\n') as file:
            writer = csv.writer(file)
            writer.writerow(node_labelling_predictions_formatted)

def generate_plotting_data(nx_g):
    edges = np.zeros((nx_g.number_of_edges(), 12), dtype=np.float32)
    # edges = np.zeros((nx_g_BAT.number_of_edges(), 12), dtype=np.float32)
    # edges = np.zeros((nx_g_ACA.number_of_edges(), 12), dtype=np.float32)
    # edges = np.zeros((nx_g_PCA.number_of_edges(), 12), dtype=np.float32)
    lookup_coord_class = {}
    for i, edge in enumerate(nx_g.edges):
        # for i, edge in enumerate(nx_g_BAT.edges):
        # for i, edge in enumerate(nx_g_ACA.edges):
        # for i, edge in enumerate(nx_g_PCA.edges):
        u = nx_g.nodes[edge[0]]
        v = nx_g.nodes[edge[1]]
        # u = nx_g_BAT.nodes[edge[0]]
        # v = nx_g_BAT.nodes[edge[1]]
        # u = nx_g_ACA.nodes[edge[0]]
        # v = nx_g_ACA.nodes[edge[1]]
        # u = nx_g_PCA.nodes[edge[0]]
        # v = nx_g_PCA.nodes[edge[1]]

        u_coord = (u['pos'][0], u['pos'][1], u['pos'][2])
        v_coord = (v['pos'][0], v['pos'][1], v['pos'][2])
        # print(u)
        # print(v)
        if u_coord not in lookup_coord_class:
            lookup_coord_class[u_coord] = [int(u['boitype']), edge[0]]
            # raise ValueError('class should be from indices not from GT labels...')
        if v_coord not in lookup_coord_class:
            lookup_coord_class[v_coord] = [int(v['boitype']), edge[1]]

        edges[i, 3] = u_coord[0]
        edges[i, 4] = u_coord[1]
        edges[i, 5] = u_coord[2]
        edges[i, 6] = v_coord[0]
        edges[i, 7] = v_coord[1]
        edges[i, 8] = v_coord[2]
        edges[i, 9] = nx_g.edges[edge[0], edge[1]]['dist']
        edges[i, 10] = nx_g.edges[edge[0], edge[1]]['rad']
        # print(u_coord, v_coord)
    return edges, lookup_coord_class

    # save_network_graph(edges, 23, lookup_coord_class,
    #                    f'{html_path}/{tag}.html', f'{tag}', fiducial_nodes=[centroid_glob],
    #                    fiducial_nodes2=[centroid_ICAL_ICAR, centroid_ICAL_ICAR_at_centroid_glob_zpos, ACA_cone_base],
    #                    fiducial_nodes3=ACA_cone_nodes_pos_attributes, MCA_landmarks=MCA_landmark_pos_attributes, MCA=MCA_nodes_pos_attributes,
    #                    ACA=ACA_nodes_pos_attributes)
    # save_network_graph(edges, 23, lookup_coord_class,
    #                    f'{html_path}/{tag}.html', f'{tag}', fiducial_nodes=[centroid_glob],
    #                    fiducial_nodes2=[centroid_ICAL_ICAR, centroid_ICAL_ICAR_at_centroid_glob_zpos, ACA_cone_base],
    #                    fiducial_nodes3=ACAL_candidates_pos_attributes + ACAR_candidates_pos_attributes, MCA_landmarks=MCA_landmark_pos_attributes, MCA=MCA_nodes_pos_attributes,
    #                    ACA=ACA_nodes_pos_attributes)

def generate_html_path(html_type, train_or_test, graph_output_path_root, cohort, c, tag, do_save_network_graph):
    if html_type == "detailed":
        if train_or_test == 'train':
            # !!!xval
            html_path = f'{graph_output_path_root}/htmls_detailed/{cohort}/train_val/{c}_{tag}.html'
            # html_path = f'{graph_output_path_root}/htmls_detailed/{cohort}/train_val cv {xval_split[1:]}/{c}_{tag}.html'
        elif train_or_test == 'test':
            html_path = f'{graph_output_path_root}/htmls_detailed/{cohort}/test/{c}_{tag}.html'
            # html_path = f'{graph_output_path_root}/htmls_detailed/{cohort}/test cv {xval_split[1:]}/{c}_{tag}.html'
        elif train_or_test == 'test_postMVC':
            html_path = f'{graph_output_path_root}/htmls_detailed/{cohort}/test_postMVC/{c}_{tag}.html'
            # html_path = f'{graph_output_path_root}/htmls_detailed/{cohort}/test cv {xval_split[1:]}/{c}_{tag}.html'
        elif train_or_test == 'all':
            do_save_network_graph = False
        else:
            raise ValueError('xxx')
        new_tag = f'{c}_{tag}'
    elif html_type == "bare":
        if train_or_test == 'train':
            html_path = f'{graph_output_path_root}/htmls/{cohort}/train_val/{c}_{tag}.html'
        elif train_or_test == 'test':
            html_path = f'{graph_output_path_root}/htmls/{cohort}/test/{c}_{tag}.html'
        elif train_or_test == 'all':
            do_save_network_graph = False
        else:
            raise ValueError('xxx')
        new_tag = f'{c}_{tag}'
    elif html_type == "coloured_nodes_only":
        if train_or_test == 'train':
            html_path = f'{graph_output_path_root}/htmls_coloured_nodes_only/{cohort}/train_val/{c}_{tag}.html'
        elif train_or_test == 'test':
            html_path = f'{graph_output_path_root}/htmls_coloured_nodes_only/{cohort}/test/{c}_{tag}.html'
        elif train_or_test == 'all':
            do_save_network_graph = False
        else:
            raise ValueError('xxx')
        new_tag = f'{c}_{tag}'

    return html_path, do_save_network_graph, new_tag

def add_artificials(nx_g):
    def add_a_leaf(nx_g, x, y, leaf_dist_from_x, leaf_len, leaf_direction=None):

        # Get positions and distance
        pos_x = np.array(nx_g.nodes[x]['pos'])
        pos_y = np.array(nx_g.nodes[y]['pos'])
        dist = nx_g.edges[x, y]['dist']
        rad_xy = nx_g.edges[x, y]['rad']

        # Calculate position of node z (0.3 * dist along x to y)
        direction = pos_y - pos_x
        direction_norm = np.linalg.norm(direction)
        direction_unit = direction / direction_norm
        pos_z = pos_x + leaf_dist_from_x * direction

        # Determine the new node ID for 'z'
        existing_nodes = list(nx_g.nodes())
        new_node_id = max(existing_nodes) + 1 if existing_nodes else 1

        # Add node z
        nx_g.add_node(new_node_id, pos=pos_z)
        nx_g.nodes[new_node_id]['boitype'] = 0

        dist_xz = np.linalg.norm(pos_z - pos_x)
        dist_zy = np.linalg.norm(pos_z - pos_y)

        # Update edges: Remove x-y, add x-z and z-y
        nx_g.remove_edge(x, y)
        nx_g.add_edge(x, new_node_id, dist=dist_xz, rad=rad_xy)
        nx_g.add_edge(new_node_id, y, dist=dist_zy, rad=rad_xy)

        # Add node w (0.5 units away from z in a random direction)
        if leaf_direction:
            leaf_direction /= np.linalg.norm(leaf_direction)  # Normalize
            pos_w = pos_z + leaf_len * leaf_direction
        else:
            random_direction = np.random.randn(3)  # Random direction
            random_direction /= np.linalg.norm(random_direction)  # Normalize
            pos_w = pos_z + leaf_len * random_direction

        # Determine the new node ID for 'w'
        w_node_id = new_node_id + 1

        dist_zw = np.linalg.norm(pos_w - pos_z)

        # Add node w and connect it to z
        nx_g.add_node(w_node_id, pos=pos_w)
        nx_g.nodes[w_node_id]['boitype'] = 0
        nx_g.add_edge(new_node_id, w_node_id, dist=dist_zw, rad=0.1)

        return nx_g, new_node_id, [pos_z, pos_w]
    def add_a_bridge(nx_g, x, y, bridge_dist_from_x, a, b, bridge_dist_from_a):

        # Get positions and distance
        pos_x = np.array(nx_g.nodes[x]['pos'])
        pos_y = np.array(nx_g.nodes[y]['pos'])
        rad_xy = nx_g.edges[x, y]['rad']

        # Calculate position of node z (0.3 * dist along x to y)
        direction = pos_y - pos_x
        direction_norm = np.linalg.norm(direction)
        direction_unit = direction / direction_norm
        pos_z = pos_x + bridge_dist_from_x * direction

        # Determine the new node ID for 'z'
        existing_nodes = list(nx_g.nodes())
        new_node_id = max(existing_nodes) + 1 if existing_nodes else 1

        # Add node z
        nx_g.add_node(new_node_id, pos=pos_z)
        nx_g.nodes[new_node_id]['boitype'] = 0

        dist_xz = np.linalg.norm(pos_z - pos_x)
        dist_zy = np.linalg.norm(pos_z - pos_y)

        # Update edges: Remove x-y, add x-z and z-y
        nx_g.remove_edge(x, y)
        nx_g.add_edge(x, new_node_id, dist=dist_xz, rad=rad_xy)
        nx_g.add_edge(new_node_id, y, dist=dist_zy, rad=rad_xy)

        ########################################

        # Get positions and distance
        pos_a = np.array(nx_g.nodes[a]['pos'])
        pos_b = np.array(nx_g.nodes[b]['pos'])
        rad_ab = nx_g.edges[a, b]['rad']

        # Calculate position of node z (0.3 * dist along x to y)
        direction = pos_b - pos_a
        direction_norm = np.linalg.norm(direction)
        direction_unit = direction / direction_norm
        pos_c = pos_a + bridge_dist_from_a * direction

        c_node_id = new_node_id + 1

        # Add node z
        nx_g.add_node(c_node_id, pos=pos_c)
        nx_g.nodes[c_node_id]['boitype'] = 0

        dist_ac = np.linalg.norm(pos_c - pos_a)
        dist_cb = np.linalg.norm(pos_c - pos_b)

        # Update edges: Remove x-y, add x-z and z-y
        nx_g.remove_edge(a, b)
        nx_g.add_edge(a, c_node_id, dist=dist_ac, rad=rad_ab)
        nx_g.add_edge(c_node_id, b, dist=dist_cb, rad=rad_ab)

        return nx_g, new_node_id, c_node_id, [pos_z, pos_c]
    # # remove MCAL subtree AnTV0
    # # nodes_to_remove = [node for node, data in nx_g.nodes(data=True) if data['pos'][0] > nx_g.nodes[7]['pos'][0]]
    # # nx_g.remove_nodes_from(nodes_to_remove)
    #
    # # add artificial lenticulostriates AnTV0
    artificial_leaves = []
    # nx_g, new_leaf_rootx, artificial_leaf = add_a_leaf(nx_g, 17, 18, 0.2, 0.03, [-0.9633591, -0.85740653, 0.50554272])
    # artificial_leaves += [copy.deepcopy(artificial_leaf)]
    # nx_g, new_leaf_root, artificial_leaf = add_a_leaf(nx_g, 17, 38, 0.15, 0.015, [-0.09633591, 0.85740653, -0.50554272])
    # artificial_leaves += [copy.deepcopy(artificial_leaf)]
    # nx_g, new_leaf_root, artificial_leaf = add_a_leaf(nx_g, new_leaf_root, 38, 0.25, 0.025, [-0.09633591, 0.85740653, -0.50554272])
    # artificial_leaves += [copy.deepcopy(artificial_leaf)]
    #
    # # add artificial anterior choroidal AnTV0
    artificial_leaves2 = []
    # nx_g, new_leaf_root, artificial_leaf = add_a_leaf(nx_g, 17, bridge_endpoint_2x, 0.5, 0.02, [-0.09633591, 0.85740653, -0.50554272])
    # artificial_leaves2 += [copy.deepcopy(artificial_leaf)]
    #
    # # add artificial bridge AnTV0
    artificial_bridges = []
    # nx_g, bridge_endpoint_1, bridge_endpoint_2x, artificial_bridge = add_a_bridge(nx_g, new_leaf_rootx, 18, 0.4, 17, 16, 0.7)
    # artificial_bridges += [copy.deepcopy(artificial_bridge)]
    # nx_g, bridge_endpoint_1, bridge_endpoint_2, artificial_bridge = add_a_bridge(nx_g, 15, 16, 0.5, 15, 14, 0.1)
    # artificial_bridges += [copy.deepcopy(artificial_bridge)]
    return nx_g, artificial_leaves, artificial_leaves2, artificial_bridges

def load_params():
    PComA_leaf_thresh = 30 # XAXAXA done
    BAT_superior_neighbour_thresh = 2 # XAXAXA done
    # ACA_coronal_validity_multiplier = 0 # 1 / 8 XAXA  done

    ### OG empirical params
    # cone_radius_multiplier = 3 / 4
    #
    # cylinder_radius1_multiplier = 1 / 2
    # cylinder_radius2_multiplier = 1.0
    # cylinder_height_multiplier = 1 / 8
    #
    # ellipsoid_x_radius_multiplier = 1 / 2
    # ellipsoid_y_radius_multiplier = 1.0
    # ellipsoid_z_radius_multiplier = 1.0
    #
    # ellipsoid_x_offset_multiplier = 1 / 2
    # ellipsoid_y_offset_multiplier = 1.0
    # ellipsoid_z_offset_multiplier = 1 / 3

    ### sequentially optimised on knowledge dataset
    cone_radius_multiplier = 0.8

    cylinder_radius1_multiplier = 0.6
    cylinder_radius2_multiplier = 1.0
    cylinder_height_multiplier = 0.3

    ellipsoid_x_radius_multiplier = 0.4 # 0.5 1 / 2
    ellipsoid_y_radius_multiplier = 1.0 # 1.0
    ellipsoid_z_radius_multiplier = 0.8 # 1.0

    ellipsoid_x_offset_multiplier = 0.4 # 1 / 2
    ellipsoid_y_offset_multiplier = 1.0 # 1.0
    ellipsoid_z_offset_multiplier = 0.2 # 1 / 3

    # if cohort in ['midas', 'IXI-IOP']:
    #     # PCA_dist_thresh /= 3
    #     PComA_leaf_thresh /= 3
    #     BAT_superior_neighbour_thresh /= 3
    #     print('Certain params modified!!!')

    return {
        'PComA_leaf_thresh': PComA_leaf_thresh,
        'BAT_superior_neighbour_thresh': BAT_superior_neighbour_thresh,
        'cone_radius_multiplier': cone_radius_multiplier,
        'cylinder_radius1_multiplier': cylinder_radius1_multiplier,
        'cylinder_radius2_multiplier': cylinder_radius2_multiplier,
        'cylinder_height_multiplier': cylinder_height_multiplier,
        'ellipsoid_x_radius_multiplier': ellipsoid_x_radius_multiplier,
        'ellipsoid_y_radius_multiplier': ellipsoid_y_radius_multiplier,
        'ellipsoid_z_radius_multiplier': ellipsoid_z_radius_multiplier,
        'ellipsoid_x_offset_multiplier': ellipsoid_x_offset_multiplier,
        'ellipsoid_y_offset_multiplier': ellipsoid_y_offset_multiplier,
        'ellipsoid_z_offset_multiplier': ellipsoid_z_offset_multiplier
    }

def output_graphsim_gt_labelling(files, to_process, cohort, train_or_test, dir_path):
    node_labelling_output_path_root = f'/hpc/jshe690/jshe690/Desktop/Jiantao/tensorflow-tutorial/GNNART/graph/htmls_detailed/'
    node_labelling_output_path = f'{node_labelling_output_path_root}/{cohort}_node_labelling_gt_GNNART_testSet_raw.csv'
    initialise_node_labelling_output_file(node_labelling_output_path, do_output_node_labelling_predictions)

    c = 0
    for f in files:
        if f not in to_process:
            continue
        if cohort in ['Anzhen', 'ArizonaCheck', 'BRAVE', 'CROPCheck', 'Parkinson2TPCheck']:
            tag = f.split(".")[0]
            print(f'{c}, {dir_path}/{f}')
            with open(f'{dir_path}/{f}', 'rb') as pickle_file:
                nx_g = pickle.load(pickle_file)

            mean_val_off_L = np.array([0.55649313, 0.40892808, 0.1999954])
            mean_val_off_R = np.array([0.40247367, 0.41225129, 0.20614317])
            mean_val_off_LR = (mean_val_off_L + mean_val_off_R) / 2
            res = get_rescaling_factor(cohort)

            if train_or_test == 'test':
                mpos = mean_val_off_LR
            else:
                lapos_L = [nx_g.nodes[i]['pos'] for i in nx_g.nodes() if nx_g.nodes[i]['boitype'] in [3]]
                if len(lapos_L) == 0:
                    mpos_L = mean_val_off_L
                    # print("ICAL missing", tag)
                else:
                    mpos_L = np.mean(lapos_L, axis=0) * res / 200
                lapos_R = [nx_g.nodes[i]['pos'] for i in nx_g.nodes() if nx_g.nodes[i]['boitype'] in [4]]
                if len(lapos_R) == 0:
                    mpos_R = mean_val_off_R
                    # print("ICAR missing", tag)
                else:
                    mpos_R = np.mean(lapos_R, axis=0) * res / 200
                mpos = (mpos_L + mpos_R) / 2
                # print('mpos',mpos)

            for i in nx_g.nodes():
                cpos = copy.copy(nx_g.nodes[i]['pos'])
                npos = (np.array(cpos)) * res / 200 - np.array(mpos)
                if cohort == 'UNC':
                    if nx_g.nodes[i]['pos'][2] < 10:
                        npos -= [0, 0, 50 * res / 200]
                        # print('low 10', npos)
                    offset = [-0.09209063, -0.07075882, 0.11279931]
                    npos += offset
                nx_g.nodes[i]['pos'] = npos.tolist()
                nx_g.nodes[i]['rad'] = nx_g.nodes[i]['rad'] * res
            for i in nx_g.edges():
                nx_g.edges[i]['rad'] *= res
                nx_g.edges[i]['dist'] *= res

            ICAL_candidates = [node for node, data in nx_g.nodes(data=True) if 'boitype' in data and data['boitype'] == 1]
            ICAR_candidates = [node for node, data in nx_g.nodes(data=True) if 'boitype' in data and data['boitype'] == 2]
            if len(ICAL_candidates) == 0:
                if not do_suppress_skips:
                    print('\t Missing ICAL node. Skipping this subject')
                c += 1
                continue
            elif len(ICAL_candidates) > 1:
                if do_verbose_errors:
                    print(f'\t Multiple ICAL candidates found: {ICAL_candidates}. Correct one has been hard coded.')

            if len(ICAR_candidates) == 0:
                if not do_suppress_skips:
                    print('\t Missing ICAR node. Skipping this subject')
                c += 1
                continue
            elif len(ICAR_candidates) > 1:
                if do_verbose_errors:
                    print(
                        f'\t WARNING! Multiple ICAR candidates found: {ICAR_candidates}. Correct one has been hard coded.')
                if cohort == 'CROPCheck' and c == 32:
                    ICAR_candidates = [81]

            assert len(ICAL_candidates) == 1, '111'
            assert len(ICAR_candidates) == 1, '222'
            ICAL_node = ICAL_candidates[0]
            ICAR_node = ICAR_candidates[0]

            node_labelling_predictions = {i: None for i in range(1, 13)}
            node_labelling_predictions[1] = ICAL_node
            node_labelling_predictions[2] = ICAR_node
            for knix in [3,4,5,6,17,18,19,20,21,22]:
                kn_candidates = [node for node, data in nx_g.nodes(data=True) if 'boitype' in data and data['boitype'] == knix]
                if len(kn_candidates) != 0:
                    if knix > 6:
                        knix -= 10
                    node_labelling_predictions[knix] = kn_candidates[0]

            output_curr_node_labelling_predictions(tag, c, node_labelling_predictions, node_labelling_output_path, do_output_node_labelling_predictions)
            c += 1

def calc_PComA_segment_dists_here():
    pass
    #######################################################################################################
    # Calculate PComA segment dists using data from subject with SP along PComA anterior and Posterior
    # Paste the code below to the location of where 'calc_PComA_segment_dists_here' is called
    # def calc_PComA_segment_dists(nx_g, PComAL_dists, PComAR_dists, c):
    #     ### if PComAL_anterior_node != None and PComAL_anterior_node != None and len(SP_PComAL_anterior_PComAL_posterior) == 2:
    #     if PComAL_anterior_node != None and PComAL_posterior_node != None:
    #         try:         # have to use SP calculation because sometimes spurious nodes (recurrent huebner) along ACA
    #             PComAL_dist = calc_dist_along_SP(nx_g, PComAL_anterior_node, PComAL_posterior_node)
    #             PComAL_dists += [PComAL_dist]
    #         except:
    #             pass
    #     if PComAR_anterior_node != None and PComAR_posterior_node != None:
    #         try:
    #             PComAR_dist = calc_dist_along_SP(nx_g, PComAR_anterior_node, PComAR_posterior_node)
    #             PComAR_dists += [PComAR_dist]
    #         except:
    #             pass
    #     c += 1
    #     return c
    # PComAL_dists, PComAR_dists, c = calc_PComA_segment_dists(nx_g, PComAL_dists, PComAR_dists, c)
    # continue
    ########################################################################################################
    # with open('/hpc/jshe690/jshe690/Desktop/Jiantao/tensorflow-tutorial/scripts/Parkinson2TPCheck_PComAL.json', 'w') as f:
    #     json.dump(PComAL_dists, f)
    # with open('/hpc/jshe690/jshe690/Desktop/Jiantao/tensorflow-tutorial/scripts/Parkinson2TPCheck_PComAR.json', 'w') as f:
    #     json.dump(PComAR_dists, f)
    ########################################################################################################

def calc_ACA_segment_dists_here():
    pass
    ########################################################################################################################################
    # Calculate ACA1 segment dists using data from subject with SP along ACA between MCAs ONLY WHEN there are exactly two nodes between the MCAs (the two nodes for sure being ACA L/R)
    # WARNING!!! Comment out find_ACA_nodes function above and LOAD ACA DISTSS function at very top above! put breakpoint at gg=0 at the very bottom then run json dump commands
    # Paste the code below to the location of where 'calc_ACA_segment_dists_here' is called
    # def calc_ACA_segment_dists(nx_g, MCAL_node, MCAR_node, SP_MCAL_MCAR, ACAL1_dists, ACAL1_dists_tags, ACAR1_dists, ACAR1_dists_tags, AComA_dists, c):
    #     if MCAL_node != None and MCAR_node != None:
    #         try:
    #             SP_MCAL_MCAR = nx.shortest_path(nx_g_ACA, source=MCAL_node, target=MCAR_node)
    #             found_SP_MCAL_MCAR = True
    #         except:
    #             found_SP_MCAL_MCAR = False
    #
    #         if found_SP_MCAL_MCAR and len(SP_MCAL_MCAR) == 4:
    #             node_L = SP_MCAL_MCAR[1]
    #             node_R = SP_MCAL_MCAR[2]
    #             # have to use SP calculation because sometimes spurious nodes (recurrent huebner) along ACA
    #             try:
    #                 ACAL1_dist = calc_dist_along_SP(nx_g_ACA, MCAL_node, node_L)
    #                 ACAL1_dists += [ACAL1_dist]
    #                 ACAL1_dists_tags += [tag]
    #             except:
    #                 pass
    #             try:
    #                 ACAR1_dist = calc_dist_along_SP(nx_g_ACA, MCAR_node, node_R)
    #                 ACAR1_dists += [ACAR1_dist]
    #                 ACAR1_dists_tags += [tag]
    #             except:
    #                 pass
    #             try:
    #                 AComA_dist = calc_dist_along_SP(nx_g_ACA, node_L, node_R)
    #                 AComA_dists += [AComA_dist]
    #             except:
    #                 pass
    #
    #     c += 1
    #     return ACAL1_dists, ACAL1_dists_tags, ACAR1_dists, ACAR1_dists_tags, AComA_dists
    # ACAL1_dists, ACAL1_dists_tags, ACAR1_dists, ACAR1_dists_tags, AComA_dists, c = calc_ACA_segment_dists(nx_g, MCAL_node, MCAR_node, SP_MCAL_MCAR, ACAL1_dists, ACAL1_dists_tags, ACAR1_dists, ACAR1_dists_tags, AComA_dists, c)
    # continue
    ########################################################################################################################################
    # import json
    # with open('/hpc/jshe690/jshe690/Desktop/Jiantao/tensorflow-tutorial/scripts/Anzhen_A1L_2.json', 'w') as f:
    #     json.dump(ACAL1_dists, f)
    # with open('/hpc/jshe690/jshe690/Desktop/Jiantao/tensorflow-tutorial/scripts/Anzhen_A1R_2.json', 'w') as f:
    #     json.dump(ACAR1_dists, f)
    # with open('/hpc/jshe690/jshe690/Desktop/Jiantao/tensorflow-tutorial/scripts/Anzhen_ACom_2.json', 'w') as f:
    #    json.dump(AComA_dists, f)
    ########################################################################################################################################
    # Old version:
    # if ACAR_node != None and ACAL_node != None and len(SP_MCAL_MCAR) == 4:
    #     # have to use SP calculation because sometimes spurious nodes (recurrent huebner) along ACA
    #     try:
    #         ACAL1_dist = calc_dist_along_SP(nx_g_ACA, MCAL_node, ACAL_node)
    #         ACAL1_dists += [ACAL1_dist]
    #         ACAL1_dists_tags += [tag]
    #     except:
    #         pass
    #
    #     try:
    #         ACAR1_dist = calc_dist_along_SP(nx_g_ACA, MCAR_node, ACAR_node)
    #         ACAR1_dists += [ACAR1_dist]
    #         ACAR1_dists_tags += [tag]
    #     except:
    #         pass
    #
    #     try:
    #         AComA_dist = calc_dist_along_SP(nx_g_ACA, ACAL_node, ACAR_node)
    #         AComA_dists += [AComA_dist]
    #     except:
    #         pass
    ########################################################################################################################################

# Cycle Breaking #####################################################################
def is_edge_intersecting_plane(nx_g, u, v, x, y):
    # Convert points to numpy arrays for vector calculations
    u = np.array(u)
    v = np.array(v)
    x = np.array(x)
    y = np.array(y)

    # Define the normal vector of the plane
    normal = np.array([y[0] - x[0], y[1] - x[1], 0])  # Only x and y components are used

    # Vector from x to u and from x to v
    vector_u = u - x
    vector_v = v - x

    # Compute the cross products
    cross_product_u = np.cross(normal, vector_u)
    cross_product_v = np.cross(normal, vector_v)

    # Check the z-component of the cross products
    return np.sign(cross_product_u[2]) != np.sign(cross_product_v[2])

def find_CoW_edges(nx_g, ICAL_pos, ICAR_pos):
    CoW_edges = []
    cycles = nx.cycle_basis(nx_g)
    for cycle in cycles:
        intersects_plane = False
        for i in range(len(cycle)):
            u_pos = nx_g.nodes[cycle[i]].get('pos')
            v_pos = nx_g.nodes[cycle[(i + 1) % len(cycle)]].get('pos')
            if is_edge_intersecting_plane(nx_g, u_pos, v_pos, ICAL_pos, ICAR_pos):
                intersects_plane = True

        if intersects_plane:
            for i in range(len(cycle)):
                CoW_edges += [sorted([cycle[i], cycle[(i + 1) % len(cycle)]])]
    return CoW_edges

def does_removing_this_edge_disconnect_the_graph(nx_g, u, v):
    initial_components = nx.number_connected_components(nx_g)
    nx_g_copy = copy.deepcopy(nx_g)
    nx_g_copy.remove_edge(u, v)
    new_components = nx.number_connected_components(nx_g_copy)
    if new_components > initial_components:
        return True
    else:
        return False

def break_cycles(nx_g, ICAL_pos, ICAR_pos):
    # !!! try allowing to delete zero rad edges???
    prev_cycles_to_break = None
    centroid = np.mean([ICAL_pos, ICAR_pos], axis=0)
    CoW_edges = find_CoW_edges(nx_g, ICAL_pos, ICAR_pos)
    # print(CoW_edges)
    while True:
        cycles = nx.cycle_basis(nx_g)
        if not cycles:
            break
        if [3,2,4,9,8,6] in [sorted(c) for c in cycles]:
            gg = 0
        cycles_to_break = []

        for cycle in cycles:
            if 36 in cycle:
                gg = 4

            intersects_plane = False
            for i in range(len(cycle)):
                u_pos = nx_g.nodes[cycle[i]].get('pos')
                v_pos = nx_g.nodes[cycle[(i + 1) % len(cycle)]].get('pos')
                # print(4)
                # print(is_edge_intersecting_plane(nx_g, u_pos, v_pos, ICAL_pos, ICAR_pos))
                if is_edge_intersecting_plane(nx_g, u_pos, v_pos, ICAL_pos, ICAR_pos):
                    intersects_plane = True

            if not intersects_plane:
                min_distance = float('inf')

                for node in cycle:
                    node_position = np.array(nx_g.nodes[node].get('pos'))
                    distance = np.linalg.norm(node_position[0:2] - np.array(centroid)[0:2])
                    if distance < min_distance:
                        min_distance = distance

                cycles_to_break.append((cycle, min_distance))

        # Sort cycles to break based on distance to centroid in descending order
        cycles_to_break.sort(key=lambda x: x[1], reverse=True)

        if not cycles_to_break:
            break
        if cycles_to_break == prev_cycles_to_break:
            print('Avoid infinite loop. No more cycles can be broken.')
            break
        prev_cycles_to_break = copy.deepcopy(cycles_to_break)
        for cycle, _ in cycles_to_break:
            if 36 in cycle:
                gg = 4
            max_distance = -float('inf')
            furthest_node = None

            for node in cycle:
                node_position = np.array(nx_g.nodes[node].get('pos'))
                distance = np.linalg.norm(node_position[0:2] - np.array(centroid)[0:2])
                if distance > max_distance:
                    max_distance = distance
                    furthest_node = node

            idx = cycle.index(furthest_node)
            prev_node = cycle[idx - 1]
            next_node = cycle[(idx + 1) % len(cycle)]
            # if sorted([furthest_node, prev_node]) == [2, 12]:
            #     print('GGGG')
            if isinstance(nx_g, nx.MultiGraph):
                edge_1 = (furthest_node, prev_node)
                edge_2 = (furthest_node, next_node)

                if nx_g.has_edge(*edge_1) and nx_g.has_edge(*edge_2):
                    min_radius_1 = min([nx_g[edge_1[0]][edge_1[1]][key].get('rad', float('inf')) for key in
                                        nx_g[edge_1[0]][edge_1[1]]])
                    min_radius_2 = min([nx_g[edge_2[0]][edge_2[1]][key].get('rad', float('inf')) for key in
                                        nx_g[edge_2[0]][edge_2[1]]])

                    if min_radius_1 < min_radius_2:
                        key_to_remove = min(nx_g[edge_1[0]][edge_1[1]],
                                            key=lambda k: nx_g[edge_1[0]][edge_1[1]][k].get('rad', float('inf')))
                        nx_g.remove_edge(edge_1[0], edge_1[1], key=key_to_remove)
                    else:
                        key_to_remove = min(nx_g[edge_2[0]][edge_2[1]],
                                            key=lambda k: nx_g[edge_2[0]][edge_2[1]][k].get('rad', float('inf')))
                        nx_g.remove_edge(edge_2[0], edge_2[1], key=key_to_remove)
                elif nx_g.has_edge(*edge_1):
                    key_to_remove = min(nx_g[edge_1[0]][edge_1[1]],
                                        key=lambda k: nx_g[edge_1[0]][edge_1[1]][k].get('rad', float('inf')))
                    nx_g.remove_edge(edge_1[0], edge_1[1], key=key_to_remove)
                elif nx_g.has_edge(*edge_2):
                    key_to_remove = min(nx_g[edge_2[0]][edge_2[1]],
                                        key=lambda k: nx_g[edge_2[0]][edge_2[1]][k].get('rad', float('inf')))
                    nx_g.remove_edge(edge_2[0], edge_2[1], key=key_to_remove)

            else:
                if nx_g.has_edge(furthest_node, prev_node) and nx_g.has_edge(furthest_node, next_node):
                    radius_1 = nx_g[furthest_node][prev_node].get('rad', float('inf'))
                    radius_2 = nx_g[furthest_node][next_node].get('rad', float('inf'))
                    if radius_1 < radius_2:
                        # print([furthest_node, prev_node])
                        if sorted([furthest_node, prev_node]) not in CoW_edges:
                            if does_removing_this_edge_disconnect_the_graph(nx_g, furthest_node, prev_node):
                                if sorted([furthest_node, next_node]) not in CoW_edges:
                                    if not does_removing_this_edge_disconnect_the_graph(nx_g, furthest_node, next_node):
                                        nx_g.remove_edge(furthest_node, next_node)
                            else:
                                nx_g.remove_edge(furthest_node, prev_node)
                    else:
                        if sorted([furthest_node, next_node]) not in CoW_edges:
                            if does_removing_this_edge_disconnect_the_graph(nx_g, furthest_node, next_node):
                                if sorted([furthest_node, prev_node]) not in CoW_edges:
                                    if not does_removing_this_edge_disconnect_the_graph(nx_g, furthest_node, prev_node):
                                        nx_g.remove_edge(furthest_node, prev_node)
                            else:
                                nx_g.remove_edge(furthest_node, next_node)
                    # if radius_1 < eps:
                    #     if radius_2 < eps:
                    #         continue
                    #         # raise ValueError('ccc', furthest_node, next_node, prev_node, nx_g.nodes[prev_node]['pos'])
                    #     else:
                    #         nx_g.remove_edge(furthest_node, next_node)
                    # elif radius_2 < eps:
                    #     nx_g.remove_edge(furthest_node, prev_node)
                    # elif radius_1 < radius_2:
                    #     nx_g.remove_edge(furthest_node, prev_node)
                    # else:
                    #     nx_g.remove_edge(furthest_node, next_node)
                elif nx_g.has_edge(furthest_node, prev_node):
                    # radius_1 = nx_g[furthest_node][prev_node].get('rad', float('inf'))
                    # if radius_1 < eps:
                    #     continue
                        # raise ValueError('xxx222')
                    if sorted([furthest_node, prev_node]) not in CoW_edges:
                        if does_removing_this_edge_disconnect_the_graph(nx_g, furthest_node, prev_node):
                            if nx_g.has_edge(furthest_node, next_node):
                                if sorted([furthest_node, next_node]) not in CoW_edges:
                                    if not does_removing_this_edge_disconnect_the_graph(nx_g, furthest_node, next_node):
                                        nx_g.remove_edge(furthest_node, next_node)
                        else:
                            nx_g.remove_edge(furthest_node, prev_node)
                elif nx_g.has_edge(furthest_node, next_node):
                    # radius_1 = nx_g[furthest_node][next_node].get('rad', float('inf'))
                    # if radius_1 < eps:
                    #     continue
                        # raise ValueError('xxx333')
                    if sorted([furthest_node, next_node]) not in CoW_edges:
                        if does_removing_this_edge_disconnect_the_graph(nx_g, furthest_node, next_node):
                            pass # dont need to check furthest_node, prev_node again since already checked above
                        else:
                            nx_g.remove_edge(furthest_node, next_node)

    return nx_g

def find_CoW_edges_using_MCA_key_nodes(nx_g, MCAL_nix, MCAR_nix):
    def find_all_cycles_centered_on_CoW(nx_g, MCAL_nix, MCAR_nix):
        x_pos_mcal = nx_g.nodes[MCAL_nix]['pos'][0]
        x_pos_mcar = nx_g.nodes[MCAR_nix]['pos'][0]
        y_pos_mcal = nx_g.nodes[MCAL_nix]['pos'][1]
        y_pos_mcar = nx_g.nodes[MCAR_nix]['pos'][1]

        gg1 = np.array(copy.deepcopy(nx_g.nodes[MCAL_nix]['pos']))
        gg1[2] = 0
        gg2 = np.array(copy.deepcopy(nx_g.nodes[MCAR_nix]['pos']))
        gg2[2] = 0
        MCAL_MCAR_distance = np.linalg.norm(gg1 - gg2)
        min_x = min(x_pos_mcal, x_pos_mcar)
        max_x = max(x_pos_mcal, x_pos_mcar)
        min_y = min(y_pos_mcal, y_pos_mcar)
        max_y = max(y_pos_mcal, y_pos_mcar)
        # must prune otherwise simple cycle enumerative search takes too long
        nodes_to_remove = [node for node, data in nx_g.nodes(data=True) if
                           data['pos'][0] < min_x - MCAL_MCAR_distance
                           or data['pos'][0] > max_x + MCAL_MCAR_distance
                           or data['pos'][1] < min_y - MCAL_MCAR_distance
                           or data['pos'][1] > min_y + MCAL_MCAR_distance]

        nx_g2 = copy.deepcopy(nx_g)
        nx_g2.remove_nodes_from(nodes_to_remove)
        cycles = nx.simple_cycles(nx_g2.to_directed())
        # chordless_cycles = [cycle for cycle in cycles if is_chordless_cycle(cycle, nx_g2)]
        # cycles2 = [] # dont materialise cycles by list(cycles) during debug since gnerator object only gives it once... when back to main code cycles will be empty!
        # for c in cycles:
        #     if len(c) > 5:
        #         cycles2.append(c)
        cycles = [cycle for cycle in cycles if len(cycle) > 5]
        return cycles

    CoW_edges = []
    CoW_nodes = []
    cycles = find_all_cycles_centered_on_CoW(nx_g, MCAL_nix, MCAR_nix)
    cycles_with_MCAL_nix = [cycle for cycle in cycles if MCAL_nix in cycle]
    cycles_with_MCAR_nix = [cycle for cycle in cycles if MCAR_nix in cycle]

    smallest_cycle_with_MCAL_nix = []
    smallest_cycle_with_MCAR_nix = []
    if cycles_with_MCAL_nix:
        smallest_cycle_with_MCAL_nix = min(cycles_with_MCAL_nix, key=len)
    if cycles_with_MCAR_nix:
        smallest_cycle_with_MCAR_nix = min(cycles_with_MCAR_nix, key=len)

    for cycle in [smallest_cycle_with_MCAL_nix, smallest_cycle_with_MCAR_nix]:
        if MCAL_nix in cycle or MCAR_nix in cycle:  # or use AND condition here??? but sometimes mca solution gets pushed laterally beyond CoW
            for i in range(len(cycle)):
                CoW_edges += [sorted([cycle[i], cycle[(i + 1) % len(cycle)]])]
                CoW_nodes += [cycle[i]]
        # intersects_plane = False
        # for i in range(len(cycle)):
        #     u_pos = nx_g.nodes[cycle[i]].get('pos')
        #     v_pos = nx_g.nodes[cycle[(i + 1) % len(cycle)]].get('pos')
        #     if is_edge_intersecting_plane(nx_g, u_pos, v_pos, ICAL_pos, ICAR_pos):
        #         intersects_plane = True
    return CoW_edges, CoW_nodes

def break_cycles_using_MCA_key_nodes(nx_g, ICAL_pos, ICAR_pos, MCAL_nix, MCAR_nix, MCA_posterior_buffer):
    # !!! try allowing to delete zero rad edges???
    prev_cycles_to_break = None
    centroid = np.mean([ICAL_pos, ICAR_pos], axis=0)
    CoW_edges, CoW_nodes = find_CoW_edges_using_MCA_key_nodes(nx_g, MCAL_nix, MCAR_nix)
    # print(CoW_edges)
    while True:
        cycles = nx.cycle_basis(nx_g)
        if not cycles:
            break

        cycles_to_break = []
        for cycle in cycles:
            if MCAL_nix not in cycle and MCAR_nix not in cycle:
                min_distance = float('inf')
                for node in cycle:
                    node_position = np.array(nx_g.nodes[node].get('pos'))
                    distance = np.linalg.norm(node_position[0:2] - np.array(centroid)[0:2])
                    if distance < min_distance:
                        min_distance = distance
                cycles_to_break.append((cycle, min_distance))

        # Sort cycles to break based on distance to centroid in descending order
        cycles_to_break.sort(key=lambda x: x[1], reverse=True)
        if not cycles_to_break:
            break
        if cycles_to_break == prev_cycles_to_break:
            print('Avoid infinite loop. No more cycles can be broken.')
            break

        MCA_max_y = max(nx_g.nodes[MCAL_nix]['pos'][1], nx_g.nodes[MCAR_nix]['pos'][1])
        MCA_post_thresh = MCA_max_y + MCA_posterior_buffer
        prev_cycles_to_break = copy.deepcopy(cycles_to_break)
        for cycle, _ in cycles_to_break:
            if 36 in cycle:
                gg = 4

            def find_furthest_node(nx_g, cycle, centroid):
                min_x = float('inf')
                max_x = -float('inf')
                for node in cycle:
                    node_position = np.array(nx_g.nodes[node].get('pos'))
                    min_x = min(min_x, node_position[0])
                    max_x = max(max_x, node_position[0])

                centroid_x = centroid[0]
                min_x_distance = abs(min_x - centroid_x)
                max_x_distance = abs(max_x - centroid_x)

                if min_x_distance > max_x_distance:
                    cycle_type = 'minxR' #
                else:
                    cycle_type = 'maxxL'

                # Step 4: Find bounding box coordinates
                bounding_box_min = np.array([float('inf'), float('inf'), float('inf')])
                bounding_box_max = np.array([-float('inf'), -float('inf'), -float('inf')])
                for node in cycle:
                    node_position = np.array(nx_g.nodes[node].get('pos'))
                    bounding_box_min = np.minimum(bounding_box_min, node_position)
                    bounding_box_max = np.maximum(bounding_box_max, node_position)

                # Step 5: Determine anchor point
                if cycle_type == 'minxR':
                    anchor = np.array([bounding_box_min[0], bounding_box_max[1], bounding_box_min[2]])
                else:
                    anchor = np.array([bounding_box_max[0], bounding_box_max[1], bounding_box_min[2]])

                # Step 6: Find closest node to the anchor point
                closest_node = None
                min_distance = float('inf')
                for node in cycle:
                    node_position = np.array(nx_g.nodes[node].get('pos'))
                    distance = np.linalg.norm(node_position - anchor)
                    if distance < min_distance:
                        min_distance = distance
                        closest_node = node

                # Furthest with respect to CoW core
                furthest_node = closest_node
                return furthest_node
            # basic version
            # max_distance = -float('inf')
            # furthest_node = None
            # for node in cycle:
            #     node_position = np.array(nx_g.nodes[node].get('pos'))
            #     distance = np.linalg.norm(node_position[0:2] - np.array(centroid)[0:2])
            #     if distance > max_distance:
            #         max_distance = distance
            #         furthest_node = node
            furthest_node = find_furthest_node(nx_g, cycle, centroid)
            idx = cycle.index(furthest_node)
            prev_node = cycle[idx - 1]
            next_node = cycle[(idx + 1) % len(cycle)]

            prev_node_y = nx_g.nodes[prev_node]['pos'][1]
            next_node_y = nx_g.nodes[next_node]['pos'][1]
            # if sorted([furthest_node, prev_node]) == [2, 12]:
            #     print('GGGG')
            if isinstance(nx_g, nx.MultiGraph):
                edge_1 = (furthest_node, prev_node)
                edge_2 = (furthest_node, next_node)

                if nx_g.has_edge(*edge_1) and nx_g.has_edge(*edge_2):
                    min_radius_1 = min([nx_g[edge_1[0]][edge_1[1]][key].get('rad', float('inf')) for key in
                                        nx_g[edge_1[0]][edge_1[1]]])
                    min_radius_2 = min([nx_g[edge_2[0]][edge_2[1]][key].get('rad', float('inf')) for key in
                                        nx_g[edge_2[0]][edge_2[1]]])

                    if min_radius_1 < min_radius_2:
                        key_to_remove = min(nx_g[edge_1[0]][edge_1[1]],
                                            key=lambda k: nx_g[edge_1[0]][edge_1[1]][k].get('rad', float('inf')))
                        nx_g.remove_edge(edge_1[0], edge_1[1], key=key_to_remove)
                    else:
                        key_to_remove = min(nx_g[edge_2[0]][edge_2[1]],
                                            key=lambda k: nx_g[edge_2[0]][edge_2[1]][k].get('rad', float('inf')))
                        nx_g.remove_edge(edge_2[0], edge_2[1], key=key_to_remove)
                elif nx_g.has_edge(*edge_1):
                    key_to_remove = min(nx_g[edge_1[0]][edge_1[1]],
                                        key=lambda k: nx_g[edge_1[0]][edge_1[1]][k].get('rad', float('inf')))
                    nx_g.remove_edge(edge_1[0], edge_1[1], key=key_to_remove)
                elif nx_g.has_edge(*edge_2):
                    key_to_remove = min(nx_g[edge_2[0]][edge_2[1]],
                                        key=lambda k: nx_g[edge_2[0]][edge_2[1]][k].get('rad', float('inf')))
                    nx_g.remove_edge(edge_2[0], edge_2[1], key=key_to_remove)

            else:
                if nx_g.has_edge(furthest_node, prev_node) and nx_g.has_edge(furthest_node, next_node):
                    radius_1 = nx_g[furthest_node][prev_node].get('rad', float('inf'))
                    radius_2 = nx_g[furthest_node][next_node].get('rad', float('inf'))
                    if radius_1 < radius_2:
                        # print([furthest_node, prev_node])
                        if sorted([furthest_node, prev_node]) not in CoW_edges and prev_node_y > MCA_post_thresh:
                            if does_removing_this_edge_disconnect_the_graph(nx_g, furthest_node, prev_node):
                                if sorted([furthest_node, next_node]) not in CoW_edges and next_node_y > MCA_post_thresh:
                                    if not does_removing_this_edge_disconnect_the_graph(nx_g, furthest_node, next_node):
                                        nx_g.remove_edge(furthest_node, next_node)
                            else:
                                nx_g.remove_edge(furthest_node, prev_node)
                    else:
                        if sorted([furthest_node, next_node]) not in CoW_edges and next_node_y > MCA_post_thresh:
                            if does_removing_this_edge_disconnect_the_graph(nx_g, furthest_node, next_node):
                                if sorted([furthest_node, prev_node]) not in CoW_edges and prev_node_y > MCA_post_thresh:
                                    if not does_removing_this_edge_disconnect_the_graph(nx_g, furthest_node, prev_node):
                                        nx_g.remove_edge(furthest_node, prev_node)
                            else:
                                    nx_g.remove_edge(furthest_node, next_node)
                    # if radius_1 < eps:
                    #     if radius_2 < eps:
                    #         continue
                    #         # raise ValueError('ccc', furthest_node, next_node, prev_node, nx_g.nodes[prev_node]['pos'])
                    #     else:
                    #         nx_g.remove_edge(furthest_node, next_node)
                    # elif radius_2 < eps:
                    #     nx_g.remove_edge(furthest_node, prev_node)
                    # elif radius_1 < radius_2:
                    #     nx_g.remove_edge(furthest_node, prev_node)
                    # else:
                    #     nx_g.remove_edge(furthest_node, next_node)
                elif nx_g.has_edge(furthest_node, prev_node):
                    # radius_1 = nx_g[furthest_node][prev_node].get('rad', float('inf'))
                    # if radius_1 < eps:
                    #     continue
                        # raise ValueError('xxx222')
                    if sorted([furthest_node, prev_node]) not in CoW_edges and prev_node_y > MCA_post_thresh:
                        if does_removing_this_edge_disconnect_the_graph(nx_g, furthest_node, prev_node):
                            if nx_g.has_edge(furthest_node, next_node):
                                if sorted([furthest_node, next_node]) not in CoW_edges and next_node_y > MCA_post_thresh:
                                    if not does_removing_this_edge_disconnect_the_graph(nx_g, furthest_node, next_node):
                                        nx_g.remove_edge(furthest_node, next_node)
                        else:
                                nx_g.remove_edge(furthest_node, prev_node)
                elif nx_g.has_edge(furthest_node, next_node):
                    # radius_1 = nx_g[furthest_node][next_node].get('rad', float('inf'))
                    # if radius_1 < eps:
                    #     continue
                        # raise ValueError('xxx333')
                    if sorted([furthest_node, next_node]) not in CoW_edges and next_node_y > MCA_post_thresh:
                        if does_removing_this_edge_disconnect_the_graph(nx_g, furthest_node, next_node):
                            pass # dont need to check furthest_node, prev_node again since already checked above
                        else:
                                nx_g.remove_edge(furthest_node, next_node)

    return nx_g

