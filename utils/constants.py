import matplotlib.pyplot as plt
import random
import numpy as np
coco_categories_mapping = {
    56: 0,  # chair
    57: 1,  # couch
    58: 2,  # potted plant
    59: 3,  # bed
    61: 4,  # toilet
    62: 5,  # tv
    60: 6,  # dining-table
    69: 7,  # oven
    71: 8,  # sink
    72: 9,  # refrigerator
    73: 10,  # book
    74: 11,  # clock
    75: 12,  # vase
    41: 13,  # cup
    39: 14,  # bottle
}

color_palette = [
    1.0, 1.0, 1.0,
    0.6, 0.6, 0.6,
    0.95, 0.95, 0.95,
    0.96, 0.36, 0.26,
    0.12156862745098039, 0.47058823529411764, 0.7058823529411765, # NOTE if reach goal
    0.9400000000000001, 0.7818, 0.66,
    0.9400000000000001, 0.8868, 0.66,
    0.8882000000000001, 0.9400000000000001, 0.66,
    0.7832000000000001, 0.9400000000000001, 0.66,
    0.6782000000000001, 0.9400000000000001, 0.66,
    0.66, 0.9400000000000001, 0.7468000000000001,
    0.66, 0.9400000000000001, 0.8518000000000001,
    0.66, 0.9232, 0.9400000000000001,
    0.66, 0.8182, 0.9400000000000001,
    0.66, 0.7132, 0.9400000000000001,
    0.7117999999999999, 0.66, 0.9400000000000001,
    0.8168, 0.66, 0.9400000000000001,
    0.9218, 0.66, 0.9400000000000001,
    0.9400000000000001, 0.66, 0.8531999999999998,
    0.9400000000000001, 0.66, 0.748199999999999
]

from habitat_sim.utils.common import d3_40_colors_rgb

color_palette_40cls = [
    1.0, 1.0, 1.0,
    0.6, 0.6, 0.6,
    0.95, 0.95, 0.95,
    0.96, 0.36, 0.26,
    0.12156862745098039, 0.47058823529411764, 0.7058823529411765, # NOTE if reach goal
]
d3_40_colors_rgb[13] = [102, 180, 71] # green - plant
d3_40_colors_rgb[14] = [178, 255, 255] # celeste - sink 
d3_40_colors_rgb[15] = [114, 160, 193] # blue stairs
color_palette_40cls.extend([i/255. for i in d3_40_colors_rgb.flatten()])


semantic_sensor_40cat = {
    0: "wall",
    1: "floor",
    2: "chair",
    3: "door",
    4: "table",
    5: "picture",
    6: "cabinet",
    7: "cushion",
    8: "window",
    9: "sofa",
    10:"bed",
    11:"curtain",
    12:"chest_of_drawers",
    13:"plant",
    14:"sink",
    15:"stairs",
    16:"ceiling",
    17:"toilet",
    18:"stool",
    19:"towel",
    20:"mirror",
    21:"tv_monitor",
    22:"shower",
    23:"column",
    24:"bathtub",
    25:"counter",
    26:"fireplace",
    27:"lighting",
    28:"beam",
    29:"railing",
    30:"shelving",
    31:"blinds",
    32:"gym_equipment",
    33:"seating",
    34:"board_panel",
    35:"furniture",
    36:"appliances",
    37:"clothes",
    38:"objects",
    39:"misc"
}

ROOM_EXEPTION = {
    "entryway/foyer/lobby",
    "hallway",
    "porch/terrace/deck/driveway",
    "stairs",
    ""
}

OBJ_EXEPTION = {
    "wall",
    "floor",
    "ceiling",
    "misc",
    "railing"
}

roomname2idx = {
    'bar': 0, 
    'classroom': 1, 
    'dining booth': 2, 
    'spa/sauna': 3, 
    'junk': 4, 
    'bathroom': 5, 
    'bedroom': 6, 
    'closet': 7, 
    'dining room': 8, 
    'entryway/foyer/lobby': 9, 
    'familyroom/lounge': 10, 
    'garage': 11, 
    'hallway': 12, 
    'library': 13, 
    'laundryroom/mudroom': 14, 
    'kitchen': 15, 
    'living room': 16, 
    'meetingroom/conferenceroom': 17, 
    'lounge': 18, 
    'office': 19, 
    'porch/terrace/deck': 20, 
    'rec/game': 21, 
    'stairs': 22, 
    'toilet': 23, 
    'utilityroom/toolroom': 24, 
    'tv': 25, 
    'workout/gym/exercise': 26, 
    'outdoor': 27,
    'balcony': 28, 
    'other room': 29    
}
roomidx2name = {
    17: 'meetingroom/conferenceroom', 
    19: 'office', 
    12: 'hallway',
    15: 'kitchen', 
    29: 'other room', 
    1: 'classroom', 
    18: 'lounge', 
    13: 'library', 
    2: 'dining booth', 
    21: 'rec/game', 
    3: 'spa/sauna', 
    22: 'stairs', 
    5: 'bathroom', 
    8: 'dining room', 
    10: 'familyroom/lounge', 
    16: 'living room', 
    9: 'entryway/foyer/lobby', 
    6: 'bedroom', 
    14: 'laundryroom/mudroom', 
    7: 'closet', 
    23: 'toilet', 
    20: 'porch/terrace/deck', 
    28: 'balcony', 
    24: 'utilityroom/toolroom', 
    4: 'junk', 
    26: 'workout/gym/exercise', 
    25: 'tv', 
    11: 'garage', 
    0: 'bar', 
    27: 'outdoor'
}

room_30color_rgba = plt.get_cmap('Pastel1_r')(np.linspace(0, 1, 30)) * 255.


obj2wordidx = {
    "wall": 2392,
    "floor": 883,
    "chair": 424,
    "door": 701,
    "table": 2159,
    "picture": 1634,
    "cabinet": 375,
    "cushion": 606,
    "window": 2449,
    "sofa": 2020,
    "bed": 243,
    "curtain": 598,
    "chest_of_drawers": 728,
    "plant": 1667,
    "sink": 1972,
    "stairs": 2058,
    "ceiling": 414,
    "toilet": 2248,
    "stool": 2101,
    "towel": 2261,
    "mirror": 1390,
    "tv_monitor": 2306,
    "shower": 1951,
    "column": 501,
    "bathtub": 224,
    "counter": 553,
    "fireplace": 867,
    "lighting": 1270,
    "beam": 231,
    "railing": 1766,
    "shelving": 1941,
    "blinds": 290,
    "gym_equipment": 803,
    "seating": 1895,
    "board_panel": 1561,
    "furniture": 945,
    "appliances": 119,
    "clothes": 486,
    "objects": 1469,
    "misc": 1517,
}


room2wordidx = {'meetingroom/conferenceroom': 1370, 'office': 1474, 'hallway': 1036, 'kitchen': 1205, 'other room': 1517, 'classroom': 469, 'lounge': 1311, 'library': 1265, 'dining booth': 309, 'rec/game': 953, 'spa/sauna': 2029, 'stairs': 2058, 'bathroom': 222, 'dining room': 660, 'familyroom/lounge': 1311, 'living room': 1291, 'entryway/foyer/lobby': 1294, 'bedroom': 246, 'laundryroom/mudroom': 1235, 'closet': 482, 'toilet': 2248, 'porch/terrace/deck': 1706, 'balcony': 183, 'utilityroom/toolroom': 2344, 'junk': 956, 'workout/gym/exercise': 1027, 'tv': 2306, 'garage': 955, 'bar': 197, 'outdoor': 1524}

objs_joint_order = [
    "wall", "floor", "stairs", "furniture", "gym_equipment", 
    "counter", "seating", "toilet", "shower", "bathtub", 
    "bed", "table", "shelving", "cabinet", "chest_of_drawers", 
     
    "sofa", "chair", "stool", "cushion", "appliances", "clothes", 
    "window", "blinds", "curtain", "sink", "mirror", "towel", "tv_monitor",
    "picture", "plant", "lighting", "fireplace", "railing", "column",  "door"
]

rooms_joint_order = [
    'bar', 'classroom', 'dining booth', 'spa/sauna',   
    'bathroom', 'bedroom', 'dining room', 'entryway/foyer/lobby', 
    'familyroom/lounge', 'garage', 'hallway', 
    'library', 'laundryroom/mudroom', 'kitchen', 'living room', 
    'meetingroom/conferenceroom', 'lounge', 'office', 
    'porch/terrace/deck',  'stairs', 
    'toilet', 'utilityroom/toolroom', 
    'workout/gym/exercise', 'outdoor', 'balcony', 'other room',
    'rec/game', 'junk', 'closet', 'tv', 
]

nav2wordidx = {
    "path": 1594,
    "current": 597, 
    "full": 942,
    "empty": 778
}

# introduce "merge": 1375 as special token
action_phrase2wordidx = [
    [982, 2122, 1375],  # go straight
    [2300, 1251, 1375], # turn left
    [2300, 1891, 1375], # turn right
    [2104, 1084, 1375], # stop here
]