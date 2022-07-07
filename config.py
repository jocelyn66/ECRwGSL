import argparse

# data configuration#####################
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "--d", type=str, default="ecb+",
                    help="Knowledge Graph dataset")
parser.add_argument("--model", default="ecrwgsl",
                    help="ECR models")
parser.add_argument("--gpu", type=int, default=0,
                    help="gpu")
parser.add_argument("--debug", action="store_true",
                    help="Only use 1000 examples for debugging")
parser.add_argument("--double-precision", action="store_true",
                    help="Machine precision")

# training configuration###############
parser.add_argument("--regularizer", choices=[None, "N3", "F2"], default=None,
                    help="Regularizer")
parser.add_argument("--reg", default=0, type=float,
                    help="Regularization weight")   # 0: no reg

parser.add_argument("--optimizer", choices=["Adagrad", "Adam"], default="Adam",
                    help="Optimizer")
parser.add_argument("--max-epochs", default=100, type=int,
                    help="Maximum number of epochs to train for")
parser.add_argument("--patience", default=50, type=int,
                    help="Number of epochs before early stopping")
parser.add_argument("--valid-freq", default=1, type=int,
                    help="Number of epochs before validation")
parser.add_argument("--batch-size", default=512, type=int,
                    help="Batch size")

parser.add_argument("--init-size", default=1e-3, type=float,
                    help="Initial embeddings' scale")
parser.add_argument("--learning-rate", default=1e-3, type=float,
                    help="Learning rate")

# configuration for optimal parameters
parser.add_argument("--rand-search", "--rs", action='store_true',
                    help="perform random search for best configuration")  # !

# gsl config#####################
parser.add_argument("--gsl", type=str,
                    help="gsl model")
parser.add_argument("--dropout", "--en-dropout", type=float, default=0.,
                    help="dropout probability")
parser.add_argument("--n-layers", type=int, default=2,
                    help="number of propagation rounds")


parser = parser.parse_args()
