"""

Model Configurations

"""

SEMEVAL_2017 = {
    "name": "SEMEVAL_2017",
    "batch_train": 32,
    "batch_eval": 32,
    "epochs": 100,
    "encoder_size": 300,
    "encoder_dropout": 0.3,
    "encoder_layers": 1,
    "encoder_bidirectional": False,
    "embed_finetune": False,
    "embed_noise": 0.1,
    "embed_dropout": 0.1,
    "base": 0.68,
    "patience": 10,
    "weight_decay": 0.0,
    "clip_norm": 1,
}
WASSA_2018 = {
    "name": "WASSA_2018",
    "token_type": "word",
    "batch_train": 32,
    "batch_eval": 32,
    "epochs": 50,
    "embeddings_file": "ntua_twitter_300",
    "embed_dim": 300,
    "embed_finetune": False,
    "embed_noise": 0.2,
    "embed_dropout": 0.3,
    "encoder_dropout": 0.4,
    "encoder_size": 300,
    "encoder_layers": 1,
    "encoder_bidirectional": True,
    "attention": True,
    "attention_layers": 1,
    "attention_context": False,
    "attention_activation": "tanh",
    "attention_dropout": 0.0,
    "base": 0.58,
    "patience": 20,
    "weight_decay": 0.0,
    "clip_norm": 1,
    "tie_weights": False

}
