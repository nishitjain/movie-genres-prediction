import argparse

def training_parser():
    parser = argparse.ArgumentParser(description='List of arguments to train the model.')
    parser.add_argument('--seed', required=False, default=42, type=int, help='seed value')
    parser.add_argument('--images', required=True, type=str, help='the path to images directory')
    parser.add_argument('--csv', required=True, type=str, help='the path to csv file')
    parser.add_argument('--ckpt', required=True, type=str, help='the path to model checkpoints directory')
    parser.add_argument('--logs', required=True, type=str, help='the path to logs directory')
    return parser.parse_args()

def prediction_parser():
    parser = argparse.ArgumentParser(description='List of arguments to make prediction.')
    parser.add_argument('--model', required=True, type=str, help='the path to model file')
    parser.add_argument('--image', required=True, type=str, help='the path to image file')
    parser.add_argument('--n', required=False, default=3, type=int, help='Top N genres')
    return parser.parse_args()

