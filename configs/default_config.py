import ml_collections

def get_config():

    config = ml_collections.ConfigDict({
        'actor_lr': 3e-4,
        'value_lr': 3e-4,
        'critic_lr': 3e-4,
        'hidden_dims': (256, 256),
        'discount': 0.99,
        'expectile': 0.7,
        'temperature': 3.0,
        'dropout_rate': ml_collections.config_dict.placeholder(float),
        'tau': 0.005,
    })
    return config