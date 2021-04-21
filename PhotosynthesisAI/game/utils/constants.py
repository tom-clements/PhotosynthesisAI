BOARD_RADIUS = 3
PLANT_LP_COST = 1
COLLECT_LP_COST = 4
MAX_SUN_ROTATIONS = 3

# defining all the trees in the game per player
TREES = {
    "seed": {"size": 0, "total": 6, "starting": 2, "cost": [1, 1, 2, 2], "score": 0,},
    "small": {"size": 1, "total": 8, "starting": 4, "cost": [2, 2, 3, 3], "score": 1,},
    "medium": {"size": 2, "total": 4, "starting": 1, "cost": [3, 3, 4], "score": 2,},
    "large": {"size": 3, "total": 2, "starting": 0, "cost": [4, 5], "score": 3,},
}

# defining score of tokens based off richness
# {score: List[value]}
TOKENS = {
    4: [22, 21, 20],
    3: [19, 18, 18, 17, 17],
    2: [17, 16, 16, 14, 14, 13, 13],
    1: [14, 14, 13, 13, 13, 12, 12, 12, 12],
}

# defining how rich the soil is based on the radius
# {radius: richness}
RICHNESS = {0: 4, 1: 3, 2: 2, 3: 1}
