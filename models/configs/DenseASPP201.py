#
#  models/configs/DenseASPP201.py
#
#  Created by Clément Malonda on 13/07/2021.

Model_CFG = {
    'bn_size': 4,
    'drop_rate': 0,
    'growth_rate': 32,
    'num_init_features': 64,
    'block_config': (6, 12, 48, 32),

    'dropout0': 0.1,
    'dropout1': 0.1,
    'd_feature0': 480,
    'd_feature1': 240
}
