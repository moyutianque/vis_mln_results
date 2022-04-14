from tqdm import tqdm
import os
import os.path as osp
import h5py
import numpy as np
from utils.constants import roomname2idx, roomidx2name, room_30color_rgba, d3_40_colors_rgb, semantic_sensor_40cat
from PIL import Image, ImageDraw, ImageFont
font = ImageFont.truetype("./arial.ttf", 8)
import jsonlines
from habitat.utils.visualizations import maps
from habitat.utils.visualizations import utils
import scipy
import cv2
from typing import Tuple, Sequence
from .map_tools import simloc2maploc
import math
import quaternion

GMAP_ROOT='./data/mln_v1/maps/gmap_floor1_mpp_0.05_channel_last_with_bounds'
DEFAULT_GMAP_PATH_PREFIX= osp.join(GMAP_ROOT, "{scene_id}_gmap.h5")
OFFSET=5

DUMP_ROOT = "./data"

def draw_path(np_map, gt_annt, grid_dimensions, upper_bound, lower_bound, is_grid=False, color=(0,128,128,255), thickness=1):
    if isinstance(gt_annt, dict):
        locations = gt_annt['locations']
    elif isinstance(gt_annt, list):
        locations = gt_annt
    else:
        raise NotImplemented()

    for i in range(1, len(locations)):

        if not is_grid:
            start_grid_pos = simloc2maploc(
                locations[i-1], grid_dimensions, upper_bound, lower_bound
            )
            end_grid_pos = simloc2maploc(
                locations[i], grid_dimensions, upper_bound, lower_bound
            )
        else:
            start_grid_pos = locations[i-1]
            end_grid_pos = locations[i]
        cv2.line(
            np_map,
            (start_grid_pos[1], start_grid_pos[0]), # use x,y coord order
            (end_grid_pos[1], end_grid_pos[0]),
            color=color,
            thickness=thickness,
        )

def draw_point(pil_img, x, y, point_size, color, text=None):
    drawer = ImageDraw.Draw(pil_img, 'RGBA')
    drawer.ellipse((x-point_size, y-point_size, x+point_size, y+point_size), fill=color)
    if text is not None:
        drawer.text((x+point_size, y-point_size), text, font=font, fill=color)


def draw_agent(aloc, arot, np_map, is_radian=False, color=(0,0,255,255), scale=1):
    if is_radian:
        agent_orientation = arot
    else:
        agent_orientation = get_agent_orientation(arot)
    agent_arrow = get_contour_points( (aloc[1]*scale, aloc[0]*scale, agent_orientation), size=15*scale)
    cv2.drawContours(np_map, [agent_arrow], 0, color, -1)

def draw_legend(cmap, id2name, save_path, ncol=2):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import rgb2hex
    patch_handles = []
    for (name, color) in zip(list(id2name.values()), cmap):
        patch_handles.append( mpatches.Patch(color=rgb2hex(np.array(color)/255.), label=name) )
    plt.legend(handles=patch_handles, ncol=ncol)
    plt.gca().set_axis_off()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
        
def draw_topdown(nav_map):
    # nav_map = cv2.medianBlur(nav_map, 5)
    recolor_map = np.array(
            [[255, 255, 255, 255], [128, 128, 128, 255], [0, 0, 0, 255]], dtype=np.uint8
        )
    nav_map_rgba = recolor_map[nav_map]
    return Image.fromarray(nav_map_rgba, 'RGBA')

def add_room_map(room_map, alpha=0.6):
    for i, room in enumerate(room_map):
        room_map[i] = cv2.medianBlur(room.astype(np.float32), 5)
    # extend additional dim for no object
    room_map = np.concatenate([-1.0 * np.ones((1,room_map.shape[1], room_map.shape[2])), room_map], axis=0)
    room_map = np.nanargmax(np.where( room_map != 0, room_map, np.nan), axis=0)
    room_regions = room_map > 0
    room_cmap = room_30color_rgba.copy()
    room_cmap[:, 3]= room_cmap[:, 3] * alpha
    if os.environ.get('DEBUG', False): 
        draw_legend(room_cmap, roomidx2name, osp.join(DUMP_ROOT, f'room_legend.png'), ncol=2)
    
    ori_room_map = (room_map-1).astype(int) # subtract the additional dim index
    room_rgba = room_cmap[ori_room_map] *  np.repeat(room_regions[...,None],4,axis=2)
    return Image.fromarray(room_rgba.astype(np.uint8), 'RGBA')


def add_objs(obj_maps, exemption_list=[], alpha=0.5):
    obj_cmap = np.concatenate(
        [
            d3_40_colors_rgb, 
            np.ones((d3_40_colors_rgb.shape[0], 1))*255*alpha
        ], axis=1
    )
    if os.environ.get('DEBUG', False): 
        draw_legend(obj_cmap, semantic_sensor_40cat, osp.join(DUMP_ROOT, f'objs_legend.png'), ncol=4)

    for i, obj_map in enumerate(obj_maps):
        obj_maps[i] = cv2.medianBlur(obj_map.astype(np.float32), 5)
        if i in exemption_list:
            obj_maps[i].fill(0)
            continue
    
    obj_maps = np.concatenate([-1.0 * np.ones((1,obj_maps.shape[1], obj_maps.shape[2])), obj_maps], axis=0)
    obj_maps = np.nanargmax(np.where( obj_maps != 0, obj_maps, np.nan), axis=0)
    obj_regions = obj_maps > 0
    obj_rgba = obj_cmap[(obj_maps-1).astype(int)] * np.repeat(obj_regions[...,None],4,axis=2)
    return Image.fromarray(obj_rgba.astype(np.uint8), 'RGBA')

def merge_and_display(scene_id, gen_path=True, show_room=False):
    gmap_path = DEFAULT_GMAP_PATH_PREFIX.format(scene_id=scene_id)
    with h5py.File(gmap_path, "r") as f:
        nav_map = f['nav_map'][()]
        room_map = np.transpose(f['room_map'][()], (2,0,1))
        obj_maps = f['obj_maps'][()]

        # Draw container
        obj_maps = np.transpose(obj_maps, (2,0,1))
        nav_map_rgba = draw_topdown(nav_map)
        # nav_map_rgba = Image.fromarray(colorize_valid_map(nav_map), 'RGBA')
        room_map_rgba = add_room_map(room_map)
        obj_map_rgba = add_objs(obj_maps, exemption_list=[16])
        
        if show_room:
            out_map = Image.alpha_composite(nav_map_rgba, room_map_rgba)
            out_map = Image.alpha_composite(out_map, obj_map_rgba)
        else:
            out_map = Image.alpha_composite(nav_map_rgba, obj_map_rgba)


        return out_map

def get_agent_orientation(arot):
    """
    Args:
        arot is quaternion x,y,z,w
    Return:
        angle in radian, rotate from (head to down) counter-clockwise
    """
    arot_q = quaternion.from_float_array(np.array([arot[3], *arot[:3]]) )
    agent_forward = quaternion.rotate_vectors(arot_q, np.array([0,0,-1.]))
    rot = math.atan2(agent_forward[0], agent_forward[2])
    return rot

def get_contour_points(pos, size):
    x, y, o = pos
    pt1 = (int(x),
           int(y))
    # pt2 = (int(x + size / 1.5 * np.cos(o + np.pi * 4 / 3)),
    #        int(y + size / 1.5 * np.sin(o + np.pi * 4 / 3)))
    # pt3 = (int(x + size * np.cos(o)),
    #        int(y + size * np.sin(o)))
    # pt4 = (int(x + size / 1.5 * np.cos(o - np.pi * 4 / 3)),
    #        int(y + size / 1.5 * np.sin(o - np.pi * 4 / 3)))
    
    pt2 = (int(x + size / 1.5 * np.sin(o + np.pi * 4 / 3)),
           int(y + size / 1.5 * np.cos(o + np.pi * 4 / 3)))
    pt3 = (int(x + size * np.sin(o)),
           int(y + size * np.cos(o)))
    pt4 = (int(x + size / 1.5 * np.sin(o - np.pi * 4 / 3)),
           int(y + size / 1.5 * np.cos(o - np.pi * 4 / 3)))

    return np.array([pt1, pt2, pt3, pt4])



if __name__ == "__main__":
    scenes = os.listdir(GMAP_ROOT)
    scenes = [ file_name.split('_')[0] for file_name in scenes if file_name.endswith('.h5')]

    cnt = 0
    for i, scene_id in enumerate(tqdm(scenes)):
        if scene_id != 'ac26ZMwG7aT':
            continue
        gmap_path = DEFAULT_GMAP_PATH_PREFIX.format(scene_id=scene_id)
        print(gmap_path)
        if not osp.exists(gmap_path):
            continue
        merge_and_display(scene_id)