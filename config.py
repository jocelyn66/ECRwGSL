import argparse
from email.policy import default

# data configuration#####################
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "--d", type=str, default="ecb+",
                    help="Knowledge Graph dataset")
parser.add_argument("--dataset_root_path", type=str, default="./data/ECB+")
parser.add_argument('--train_file', type=str, default='./data/ECB+/processed_ECB+/ECB_Train_processed_data.json')
parser.add_argument('--dev_file', type=str, default='./data/ECB+/processed_ECB+/ECB_Dev_processed_data.json')
parser.add_argument('--test_file', type=str, default='./data/ECB+/processed_ECB+/ECB_Test_processed_data.json')
parser.add_argument('--schema_path', type=str, default="./data/ECB+/processed_ECB+/ECB_schema.json")
parser.add_argument('--ckp_dir', type=str, default=None)
parser.add_argument('--train_cache_file', type=str, default='./data/ECB+/dataset_cache/ECB_train.arrow')
parser.add_argument('--dev_cache_file', type=str, default='./data/ECB+/dataset_cache/ECB_dev.arrow')
parser.add_argument('--test_cache_file', type=str, default='./data/ECB+/dataset_cache/ECB_test.arrow')
parser.add_argument('--log_dir', type=str, default=None)              

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
parser.add_argument("--max-epochs", default=200, type=int,
                    help="Maximum number of epochs to train for")
parser.add_argument("--patience", default=500, type=int,
                    help="Number of epochs before early stopping")
parser.add_argument("--valid-freq", default=1, type=int,
                    help="Number of epochs before validation")
parser.add_argument("--batch-size", default=512, type=int,
                    help="Batch size")

parser.add_argument("--init-size", default=1e-3, type=float,
                    help="Initial embeddings' scale")
parser.add_argument("--learning-rate", default=1e-5, type=float,
                    help="Learning rate")
parser.add_argument('--weight_decay', type=float, default=1e-2)

# configuration for optimal parameters
parser.add_argument("--rand-search", "--rs", action='store_true',
                    help="perform random search for best configuration")
# mdoel config
parser.add_argument("--model", default="ecr-gsl",
                    help="ECR models")
parser.add_argument('--tokenizer_name', type=str, default='bert-base-uncased')
parser.add_argument('--plm_name', type=str, default='bert-base-uncased')
parser.add_argument('--feat_dim', type=int, default=768, 
                    help='size of features, i.e. bert embedding dim')
# encoder config#####################
parser.add_argument('--rand_node_rate', type=int, default=0.2)
parser.add_argument("--encoder", type=str, choices = ['gae', 'gvae'], default='gae',
                    help="gsl model")
parser.add_argument("--dropout", "--en-dropout", type=float, default=0.,
                    help="dropout probability")
parser.add_argument("--n-layers", type=int, default=2,
                    help="number of propagation rounds")
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')

# decoder
parser.add_argument("--decoder", type=str, default='')
parser = parser.parse_args()
