import argparse

import src.create_dynamic_graph
import src.data_concat
import src.read_data
import src.sentence_vector

parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('--channel', type=str, help="Channel ID (refer to dir '/channel/')",
                    default='UC1opHUrw8rvnsadT-iGp7Cg')
parser.add_argument('--type', type=str, help='Length of the dynamic graph', choices=[
    "short", "mid", "long"], default='mid')
parser.add_argument('--graph', type=bool, help='Whether to create dynamic graph', action='store_true')
parser.add_argument('--initial', type=bool, help='Whether to initialize', action='store_true')

args = parser.parse_args()

if args.initial:
    src.data_concat.dataConcat()  # concat all the data
    src.read_data.splitData()  # split the data by channels and videos
    src.sentence_vector.travelEmbed()  # embed the chat messages into embedding vectors

if args.graph:
    src.create_dynamic_graph.data_process(args.channel, args.type)  # create dynamic graph for a specific channel