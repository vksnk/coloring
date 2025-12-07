import argparse
import os
import sys


def checkpoint_name(args):
    return f"checkpoints/checkpoint_{args.num_of_gcns}_{args.input_dim}_{args.hidden_dim}.pth"


def best_checkpoint_name(args):
    return f"checkpoints/best_checkpoint_{args.num_of_gcns}_{args.input_dim}_{args.hidden_dim}.pth"


def get_args():
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Configuration for GCN model")

    # Add arguments
    parser.add_argument(
        "--num_of_gcns",
        type=int,
        required=False,
        default=3,
        help="The number of GCN layers to use",
    )

    parser.add_argument(
        "--input_dim",
        type=int,
        required=False,
        default=16,
        help="The dimension size of the input layer",
    )

    parser.add_argument(
        "--hidden_dim",
        type=int,
        required=False,
        default=128,
        help="The dimension size of the hidden layers",
    )

    # Parse the arguments
    return parser.parse_args()
