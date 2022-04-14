import os
import os.path as osp
import h5py
from typing import Tuple

def get_gmaps(gmap_root):
    gmaps = dict()
    for scene_file in os.listdir(gmap_root):
        gmap_path = osp.join(gmap_root, scene_file)
        scene_id = scene_file.split('_')[0]
        with h5py.File(gmap_path, "r") as f:
            nav_map  = f['nav_map'][()]
            room_map = f['room_map'][()] 
            obj_maps = f['obj_maps'][()] 
            obj_maps[:,:,1] = ((obj_maps[:,:,1]>0) ^ (obj_maps[:,:,15]>0)) * obj_maps[:,:,15] + obj_maps[:,:,1] # merge stairs to floor
            bounds = f['bounds'][()]
        grid_dimensions = (nav_map.shape[0], nav_map.shape[1])
        gmaps[scene_id] = {
            "nav_map": nav_map,
            "room_map": room_map,
            "obj_maps": obj_maps,
            "bounds": bounds,
            "grid_dimensions": grid_dimensions
        }
    return gmaps


def simloc2maploc(aloc, grid_dimensions, upper_bound, lower_bound):
    agent_grid_pos = to_grid(
        aloc[2], aloc[0], grid_dimensions, lower_bound=lower_bound, upper_bound=upper_bound
    )
    return agent_grid_pos

def to_grid(
    realworld_x: float,
    realworld_y: float,
    grid_resolution: Tuple[int, int],
    lower_bound, upper_bound
) -> Tuple[int, int]:
    """
    single point implementation
    """
    grid_size = (
        abs(upper_bound[2] - lower_bound[2]) / grid_resolution[0],
        abs(upper_bound[0] - lower_bound[0]) / grid_resolution[1],
    )
    grid_x = int((realworld_x - lower_bound[2]) / grid_size[0])
    grid_y = int((realworld_y - lower_bound[0]) / grid_size[1])
    return grid_x, grid_y