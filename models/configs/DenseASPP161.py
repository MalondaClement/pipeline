#
#  models/configs/DenseASPP161.py
#
#  Created by Clément Malonda on 13/07/2021.

Model_CFG = {
    'bn_size': 4,
    'drop_rate': 0,
    'growth_rate': 48,
    'num_init_features': 96,
    'block_config': (6, 12, 36, 24),

    'dropout0': 0.1,
    'dropout1': 0.1,
    'd_feature0': 512,
    'd_feature1': 128
}
