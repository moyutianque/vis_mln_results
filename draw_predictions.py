""" This script is going to visualize all prediction results """
import os
import os.path as osp
import json
import gzip
from PIL import Image
from utils.draw_toolbox import merge_and_display, GMAP_ROOT, draw_point, draw_path, draw_agent
from utils.heatmap_tools import draw_heat_map
from utils.map_tools import get_gmaps, simloc2maploc
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, help="predictions all_traj_pred.json from inference stage, best_routes.json from eval stage, [optional] sim_results.json from simulator")
parser.add_argument("--dump_dir", type=str, help="where to save drawn graphs")
parser.add_argument("--split", type=str, help="where to save drawn graphs")
args = parser.parse_args()

original_annt_path= './data/mln_v1/annt/{split}/{split}.json.gz'
ori_annt = original_annt_path.format(split=args.split)
epid2meta = {}
with gzip.open(ori_annt, 'rt') as f:
    meta = json.load(f)['episodes']
    for ep in meta:
        epid2meta[ep['episode_id']] = ep

gmaps = get_gmaps(GMAP_ROOT)
def show_episode(preds, episode_id, sim_results=None):
    """
    Draw heat map on semantic map, instruction at bottom, best path, ground truth path
    file name is identified by whether the result is correct
    Args:
        datum: 
        sim_results: key-value pair indicating whether the episode id is predicted correct
    
    """
    is_success = None
    if sim_results is not None:
        is_success = sim_results[episode_id]['is_success']
    
    ep_id = preds['episode_id']
    ep_path_id = preds['ep_id']
    score = preds['pred']
    scene_name = preds['scene_name']
    path_raw = preds['path_continue']
    
    instruction = epid2meta[int(ep_id)]['instruction']['instruction_text']

    # 1. Draw the base map (navigable area with colored objects)
    colorized_map = merge_and_display(scene_name, show_room=False)

    # 2. Draw heat map
    overlay_heat = Image.new('RGBA', colorized_map.size, (0,0,0)+(0,))
    end_point_and_scores = []

    for d in preds['traj_pred']:
        d_path_end = int(d[-1][0]), int(d[-1][1])
        end_point_and_scores.append((d_path_end, d[0]))
    print(f"processing episode {ep_id}")
    overlay_heat = draw_heat_map(overlay_heat, end_point_and_scores)

    pil_img = Image.alpha_composite(colorized_map, overlay_heat) 
    
    # 3. Draw gt path 
    bounds = gmaps[scene_name]['bounds']
    upper_bound, lower_bound = bounds[0], bounds[1]
    grid_dimensions =gmaps[scene_name]['grid_dimensions']

    gt_path_raw = epid2meta[int(ep_id)]['reference_path']
    gt_path = [simloc2maploc(agent_stat, grid_dimensions, upper_bound, lower_bound) for agent_stat in gt_path_raw]

    empty_map = np.zeros((colorized_map.size[1], colorized_map.size[0], 4), dtype=np.uint8)
    overlay = Image.new('RGBA', (empty_map.shape[1], empty_map.shape[0]), (0,0,0)+(0,))

    gt_path = [(int(agent_stat[0]), int(agent_stat[1])) for agent_stat in gt_path]
    line_color = (0, 0, 255, 255) #  blue for GT
    draw_path(empty_map, gt_path, None, None, None, is_grid=True, color=line_color)
    # for i, (point, rot) in enumerate(zip(gt_path, gt_rots)):
    #     if i==0:
    #         continue
    #     if i==len(gt_path)-1:
    #         # draw_point(
    #         #     overlay, point[1], point[0], 60, color=(255, 0, 0, 50)
    #         # )
    #         pass
    #     else:
    #         draw_point(
    #             overlay, point[1], point[0], 2, color=(255, 0, 0, 128)
    #         )

    # 4. Draw current path
    line_color = (255,0, 0, 255) # red for current path
    path = [(int(agent_stat[0]), int(agent_stat[1])) for agent_stat in path_raw]
    draw_path(empty_map, path, None, None, None, is_grid=True, color=line_color) 
    # # for i, (point, rot) in enumerate(zip(path, rots)):
    # #     if i==0:
    # #         continue
    # #     if i==len(path)-1:
    # #         draw_point(
    # #             overlay, point[1], point[0], 2, color=(0,0,0,255),
    # #             text="%.2f"%(score)
    # #         )

    # 5. mark start and end point
    start_point = simloc2maploc(epid2meta[int(ep_id)]['start_position'], grid_dimensions, upper_bound, lower_bound)
    draw_agent(start_point, epid2meta[int(ep_id)]['start_rotation'], empty_map) 
    path_img = Image.fromarray(np.copy(empty_map))
    # draw_point(path_img, start_point[1], start_point[0], 4, color=(255,0,0,200))
    end_point = simloc2maploc(epid2meta[int(ep_id)]['goals'][0]['position'], grid_dimensions, upper_bound, lower_bound)
    draw_point(path_img, end_point[1], end_point[0], 4, color=(255, 0, 0,200))

    pil_img = Image.alpha_composite(colorized_map, overlay_heat)
    pil_img = Image.alpha_composite(pil_img, path_img)

    if is_success is not None:
        pil_img.save(osp.join(args.dump_dir, f"{is_success}-{ep_id}.png"))
    else:
        pil_img.save(osp.join(args.dump_dir, f"{ep_id}.png"))

def process_data(annot_root):
    sim_results = None
    if osp.exists(osp.join(annot_root, 'sim_results.json')):
        sim_results = json.load(open(osp.join(annot_root, 'sim_results.json')))
    
    assert osp.exists(osp.join(annot_root, 'all_traj_pred.json')), f"{osp.join(annot_root, 'all_traj_pred.json')} not found"
    assert osp.exists(osp.join(annot_root, 'best_routes.json')), f"{osp.join(annot_root, 'best_routes.json')} not found"

    traj_pred = json.load(open(osp.join(annot_root, 'all_traj_pred.json')))
    best_routes = {line['episode_id']:line for line in json.load(open(osp.join(annot_root, 'best_routes.json'))) }
    for k,v in best_routes.items():
        best_routes[k].update({'traj_pred': traj_pred[k]})
        show_episode(best_routes[k], k, sim_results)
    
    

if __name__ == "__main__":
    process_data(args.input_dir)
    


