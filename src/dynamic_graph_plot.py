import pandas as pd
import numpy as np
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import time
import json

pd.set_option('display.width', None)


def show_dynamic_graph_by_batch(data, root, step=50):
    g = nx.DiGraph()
    batches = data['Batch'].drop_duplicates().values.tolist()

    batch_list = [batches[i] for i in range(0, len(batches), step)]
    batch_list.append(batches[-1]+1)

    exist_nodes = set()

    #graph_dict = dict()

    #plt.figure()
    #plt.ion()

    for batch in range(len(batch_list)-1):
        records = data[(data['Batch'] >= batch_list[batch]) & ((data['Batch'] < batch_list[batch+1]))]
        node_set = records[records['Inst'] == 'CREATE']['Node1'].drop_duplicates().values.tolist()
        node_set = set(node_set) - exist_nodes
        exist_nodes = exist_nodes | node_set
        edge_set = records[records['Inst'] == 'EDGE']
        remove_edge_set = records[records['Inst'] == 'REMOVE']

        g.add_nodes_from(node_set)
        g.add_edges_from([(n['Node1'], n['Node2']) for _, n in edge_set.iterrows()])
        g.remove_edges_from([(n['Node1'], n['Node2']) for _, n in remove_edge_set.iterrows()])

        plt.clf()
        pos = nx.spring_layout(g)
        nx.draw(g, pos)
        plt.savefig('../figure/{}_batch_{}.png'.format(root, batch))
        # plt.show()
        #plt.pause(0.1)

        #graph_dict[batch] = g.copy()

        print('batch{}-{}'.format(batch_list[batch], batch_list[batch+1]))

        nx.write_gpickle(g, "graph_{}_batch_{}".format(root, batch_list[batch]))

def select_k(spectrum, minimum_energy = 0.9):
    running_total = 0.0
    total = sum(spectrum)
    if total == 0.0:
        return len(spectrum)
    for i in range(len(spectrum)):
        running_total += spectrum[i]
        if running_total / total >= minimum_energy:
            return i + 1
    return len(spectrum)


def calculate_GED(video_id, data, step=50):
    batches = data['Batch'].drop_duplicates().values.tolist()
    batch_list = [batches[i] for i in range(0, len(batches), step)]

    batch_n = len(batch_list)
    graphs = []
    for i in batch_list:
        g = nx.read_gpickle("../figure/graph_{}_batch_{}".format(video_id, i))
        graphs.append(g)

    edges_nodes = []
    results = np.zeros(batch_n)

    for i in range(batch_n):
        edges_nodes.append((nx.number_of_edges(graphs[i]), nx.number_of_nodes(graphs[i])))
        #print(i)
    print(edges_nodes)

    for i in range(batch_n-1):
        laplacian1 = nx.spectrum.laplacian_spectrum(graphs[i].to_undirected())
        laplacian2 = nx.spectrum.laplacian_spectrum(graphs[i+1].to_undirected())

        k1 = select_k(laplacian1)
        k2 = select_k(laplacian2)
        k = min(k1, k2)

        results[i] = sum((laplacian1[:k] - laplacian2[:k]) ** 2)

    print(results)
    np.savetxt('../figure/{}_results.txt'.format(video_id), results, delimiter=',')

def show_dynamic_graph(records):
    g = nx.DiGraph()
    node_list = records[records['Inst'] == 'CREATE']['Node1'].drop_duplicates().values.tolist()
    node_set = set(node_list)
    node_dict = dict(zip(node_set, set(range(len(node_set)))))
    g.add_nodes_from(node_set)
    pos = nx.spring_layout(g)
    node_value = [0] * len(node_set)

    print(node_dict)

    plt.figure()
    plt.ion()

    # nx.draw(g, pos, node_color=node_value, cmap=plt.cm.Blues)
    # plt.show()

    t = 15
    for _, row in records.iterrows():
        inst = row['Inst']
        node1 = row['Node1']
        node2 = row['Node2']
        val = row['Value']
        bat = row['Batch']
        offset = row['Offset']
        weight = row['Weight']
        while offset > t:
            t += 1
            print(t)
            plt.clf()
            nx.draw(g, pos, node_color=node_value, cmap=plt.cm.Blues)
            plt.savefig('figure/matplot_{}.png'.format(t))
            # plt.show()
            plt.pause(0.1)

        if inst == 'CREATE':
            node_value[node_dict[node1]] += 1
            # TODO
        elif inst == 'UPDATE':
            node_value[node_dict[node1]] += 1
            # TODO
        elif inst == 'EDGE':
            g.add_weighted_edges_from([(node1, node2, weight)])
            # TODO
        elif inst == 'REMOVE':
            g.remove_edge(node1, node2)

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    root = 'admiralbulldog_w30_t0.8660254037844387'
    video_id = '97DWg8tqo4M'
    data = pd.read_csv('../dynamicGraph/{}_dynamic_graph.csv'.format(video_id))
    records = data
    print(records)
    #show_dynamic_graph(records)

    show_dynamic_graph_by_batch(data, video_id, 50)
    calculate_GED(video_id, records)
