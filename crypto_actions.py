import pickle


actions = {
    1: [0.6, 0.4],
    2: [0.57, 0.43],
    3: [0.65, 0.35],
    4: [0.75, 0.25],
    5: [0.48, 0.52],
    6: [0.6, 0.4],
    8: [0, 1],
    9: [0.7, 0.3],
    10: [0.53, 0.47],
    13: [0.76, 0.24],
    14: [0.76, 0.24],
    15: [0.65, 0.35],
    17: [0.64, 0.36],
    18: [0.59, 0.41],
    19: [1, 0],
    22: [0.14, 0.86],
    23: [0.67, 0.33],
    25: [0, 1],
    26: [0.7, 0.3],
    27: [0.71, 0.29],
    28: [0.27, 0.44],
    29: [0.23, 0.4],
    30: [0.33, 0.33],
    32: [0.4, 0.4],
    33: [0.26, 0.53],
    36: [0.6, 0.3],
    37: [0.19, 0.55],
    38: [0, 0.6],
    40: [0.3, 0.41],
    41: [0.27, 0.5],
    42: [0.21, 0.46],
    43: [1, 0],
    44: [0.29, 0.57],
    45: [0.29, 0.47],
    46: [0, 0.67],
    49: [0.19, 0.38],
    50: [0.15, 0.62],
    52: [0, 0.5],
    53: [0.42, 0.46],
    54: [0.25, 0.44],
    55: [0, 0.3],
    56: [0, 0.33],
    57: [0, 0],
    59: [0, 0],
    60: [0, 0.25],
    63: [0, 1],
    64: [0, 0.4],
    65: [0, 0.33],
    67: [0, 0.38],
    68: [0, 0.35],
    69: [0, 0.35],
    72: [0, 0.44],
    73: [0, 0.32],
    74: [0, 0.5],
    76: [0, 0.36],
    77: [0, 0.38],
    78: [0, 0.64],
    79: [0, 0.48],
    80: [0, 0.36],
    81: [0, 0.42],
}


with open("crypto_actions.pkl", "wb") as pklfile:
    pickle.dump(actions, pklfile)
