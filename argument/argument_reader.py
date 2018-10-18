import argparse


def create_argparse(args=None):
    parser = argparse.ArgumentParser()

    # Dataset and data related arguments
    parser.add_argument("-dp",
                        "--data_path",
                        required=False,
                        default="D:/PyTorchNLP/data/TWNERTC_TC_Coarse Grained NER_No_NoiseReduction.DUMP",
                        help="Absolute path of dataset.")
    parser.add_argument("-ev",
                        "--embedding_vector",
                        required=False,
                        default="fasttext.tr.300d",
                        help="Embedding vector type for Torchtext. Built-in pretrained embedding aliases can be found "
                             "under vocab.py file of Torchtext lib.")
    parser.add_argument("-vc",
                        "--vector_cache",
                        required=False,
                        default="D:/PyTorchNLP/data/fasttext",
                        help="If pretrained embeddings have been downloaded already, provide its folder path.")
    parser.add_argument("-swp",
                        "--stop_word_path",
                        required=False,
                        default="D:/Anaconda3/nltk_data/corpora/stopwords/turkish",
                        help="Absolute path of stop words")
    parser.add_argument("-pep",
                        "--pretrained_embedding_path",
                        required=False,
                        default="D:/PyTorchNLP/data/fasttext/wiki.tr",
                        help="This is only required if you want to use Fasttext embeddings to create OOV embeddings"
                             "on the fly.")
    parser.add_argument("-oet",
                        "--oov_embedding_type",
                        required=False,
                        default="zeros",
                        help="Action to handle OOV tokens. Possible choices are (1) zeros, (2) ones, (3) random,"
                             "(4) uniform and (5) fasttext_oov.")
    parser.add_argument("-bs",
                        "--batch_size",
                        required=False,
                        default=128,
                        help="Batch size for dataset iterators. Training, validation and test sets will use same size!")
    # Model related arguments
    parser.add_argument("-use",
                        "--use_pretrained_embed",
                        required=False,
                        type=bool,
                        default=True,
                        help="Choose whether using pretrained or random embeddings.")
    parser.add_argument("-ett",
                        "--embed_train_type",
                        required=False,
                        default="static",
                        help="Choose whether embeddings are static, nonstatic or multichannel.")
    parser.add_argument("-upc",
                        "--use_padded_conv",
                        required=False,
                        type=bool,
                        default=True,
                        help="Choose whether convolution layers should apply padding or not.")
    parser.add_argument("-kp",
                        "--keep_prob",
                        required=False,
                        type=float,
                        default=0.5,
                        help="Dropout probability.")
    parser.add_argument("-ubn",
                        "--use_batch_norm",
                        required=False,
                        type=bool,
                        default=True,
                        help="Choose whether applying batch normalization or not.")
    parser.add_argument("-bnm",
                        "--batch_norm_momentum",
                        required=False,
                        type=float,
                        default=0.1,
                        help="Batch normalization's momentum parameter.")
    parser.add_argument("-bna",
                        "--batch_norm_affine",
                        required=False,
                        type=bool,
                        default=False,
                        help="Batch normalization's affine parameter.")
    parser.add_argument("-fc",
                        "--filter_count",
                        required=False,
                        type=int,
                        default=128,
                        help="Convolution filter count number.")
    parser.add_argument("-fs",
                        "--filter_sizes",
                        required=False,
                        nargs="+",
                        type=int,
                        default=[3, 4, 5],
                        help="Convolution filter sizes.")
    parser.add_argument("-rm",
                        "--run_mode",
                        required=False,
                        default="train",
                        help="Select run mode: train or eval_interactive")
    # Training related arguments
    parser.add_argument("-o",
                        "--optimizer",
                        required=False,
                        default="Adam",
                        help="Choose optimizer type: Adam or SGD.")
    parser.add_argument("-lr",
                        "--learning_rate",
                        required=False,
                        type=float,
                        default=0.01,
                        help="Learning rate.")
    parser.add_argument("-wd",
                        "--weight_decay",
                        required=False,
                        type=float,
                        default=0,
                        help="Weight decay or L2 regularization term.")
    parser.add_argument("-m",
                        "--momentum",
                        required=False,
                        type=float,
                        default=0.9,
                        help="Momentum.")
    parser.add_argument("-nr",
                        "--norm_ratio",
                        required=False,
                        type=int,
                        default=10,
                        help="Gradient clipping ratio.")
    parser.add_argument("-e",
                        "--epoch",
                        required=False,
                        type=int,
                        default=10,
                        help="Number of epochs to train")
    parser.add_argument("-peb",
                        "--print_every_batch_step",
                        required=False,
                        type=int,
                        default=250,
                        help="Number of batch step to print (not epoch!)")
    parser.add_argument("-see",
                        "--save_every_epoch",
                        required=False,
                        type=int,
                        default=1,
                        help="Number of epoch to save the model")
    parser.add_argument("-ee",
                        "--eval_every",
                        required=False,
                        type=int,
                        default=1,
                        help="Number of epoch to evaluate the model while training")
    # Evaluation arguments
    parser.add_argument("-mp",
                        "--model_path",
                        required=False,
                        default="D:/PyTorchNLP/saved/2018-10-18/",
                        help="Folder path for the saved model")
    parser.add_argument("-sv",
                        "--sentence_vocab",
                        required=False,
                        default="D:/PyTorchNLP/saved/2018-10-18/sentence_vocab.dat",
                        help="Saved word vocabulary file path")
    parser.add_argument("-cv",
                        "--category_vocab",
                        required=False,
                        default="D:/PyTorchNLP/saved/2018-10-18/category_vocab.dat",
                        help="Saved category vocabulary file path (for human readable prediction printing =))")

    return parser.parse_args(args)
