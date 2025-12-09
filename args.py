import argparse
import os
import sys


def checkpoint_name(args):
    return f"checkpoints/checkpoint_{args.conv_type}_{args.num_of_gcns}_{args.input_dim}_{args.hidden_dim}_{args.num_classes}.pth"


def best_checkpoint_name(args):
    return f"checkpoints/best_checkpoint_{args.conv_type}_{args.num_of_gcns}_{args.input_dim}_{args.hidden_dim}_{args.num_classes}.pth"


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

    parser.add_argument(
        "--num_classes",
        type=int,
        required=False,
        default=8,
        help="The maximum number of colors/the size of the output layer",
    )

    parser.add_argument(
        "--conv_type",
        type=str,
        required=False,
        default="sage",
        help="The type of the graph convolution layer to use",
    )

    parser.add_argument(
        "--draw_model",
        type=bool,
        required=False,
        default=False,
        help="Save the drawing of the model architecture into file",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        help="The path to checkpoint for evaluation",
    )

    # Parse the arguments
    return parser.parse_args()
