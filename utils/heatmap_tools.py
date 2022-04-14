from typing import overload
from PIL import ImageDraw, ImageFilter, Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import random
import io
import cv2

num_levels = 10
# heatcm = plt.get_cmap('coolwarm')(np.linspace(0, 1, num_levels+1)) * 255.
heatcm = plt.get_cmap('viridis')(np.linspace(0, 1, num_levels+1))
heatcm = plt.get_cmap('jet')(np.linspace(0, 1, num_levels+1))
# heatcm = plt.get_cmap('rainbow')(np.linspace(0, 1, num_levels+1))
# heatcm = plt.get_cmap('RdBu_r')(np.linspace(0, 1, num_levels+1))
# heatcm = plt.get_cmap('plasma')(np.linspace(0, 1, num_levels+1))
# heatcm[:num_levels//3 * 2, 3]=0 #remove some colors
# heatcm[:num_levels//3 * 2, :3] = [1,1,1]
# heatcm[:3, :3] = [1,1,1]
heatcm = heatcm[:, :3]

idx_step=1.0/num_levels

def scaler(bound, target_range, value):
    value = np.clip(value, bound[0], bound[1])
    v_std = (value-bound[0]) / (bound[1]-bound[0])
    return v_std * (target_range[1] - target_range[0]) + target_range[0]

def get_surrounding_point_rel_pos_with_radius(radius, num_split, sorted_type='ascending'):
    """
    Args:
        radius: max radius of point relative to center
        num_split: number of splits on one axis
        sorted_type: from centor to bound use 'ascending'
    """
    x = np.linspace(-radius, radius, num_split)
    y = np.linspace(-radius, radius, num_split)
    xv, yv = np.meshgrid(x, y)

    coord_pairs = list(zip(xv.flatten(), yv.flatten()))
    
    if sorted_type == 'ascending':
        out_pairs = sorted(coord_pairs, key=lambda z: z[0]**2 + z[1]**2 )
        out_pairs = coord_pairs[:len(coord_pairs)//2] \
            + [pair for pair in coord_pairs[len(coord_pairs)//2:] if pair[1]**2 + pair[0]**2 <= radius**2 ]
    else:
        raise NotImplementedError()
    return out_pairs

# def draw_heat_map(pil_overlay,  end_point_and_scores, point_gap=1.2, meters_per_pixel=0.05, draw_shape='rect'):
#     score_min = min([d[1] for d in end_point_and_scores])
#     score_max = max([d[1] for d in end_point_and_scores])

#     drawer = ImageDraw.Draw(pil_overlay, 'RGBA')
#     region_radius = int(point_gap/meters_per_pixel)//2
#     for end_point, score in end_point_and_scores: 
#         scaled = scaler([score_min, score_max], [0,1], score)
#         scale_idx = round(scaled/idx_step)
#         color = ( 
#             int(heatcm[int(scale_idx)][0]), int(heatcm[int(scale_idx)][1]) , 
#             int(heatcm[int(scale_idx)][2]) , 120 
#         )
#         y, x = end_point
#         if draw_shape == 'rect':
#             drawer.rectangle((x-region_radius, y-region_radius, x+region_radius, y+region_radius),
#                 fill=color)
#         else:
#             drawer.ellipse(
#                 (x-region_radius, y-region_radius, x+region_radius, y+region_radius), 
#                 fill=color
#             )

#     pil_overlay = pil_overlay.filter(ImageFilter.BLUR)



# def draw_heat_map(pil_overlay, end_point_and_scores, point_gap=1.2, meters_per_pixel=0.05, draw_shape='rect', down_scale=2):
#     """ KDE version """
#     ori_w, ori_h = pil_overlay.size
#     score_min = min([d[1] for d in end_point_and_scores])
#     score_max = max([d[1] for d in end_point_and_scores])
    
#     region_radius = round(point_gap/(meters_per_pixel*down_scale))//2
#     surroundings = get_surrounding_point_rel_pos_with_radius(region_radius, region_radius*2+1,)
#     random.shuffle(surroundings)

#     w,h = pil_overlay.size
#     pil_overlay = pil_overlay.resize((w//down_scale, h//down_scale))
#     w,h = pil_overlay.size
#     print(f"map size: {w} {h}")

#     points = []
#     score = []
#     mask = np.zeros((h,w))>0
#     for d in end_point_and_scores:
#         scaled = scaler([score_min, score_max], [0, 1], d[1])
#         for sur in surroundings:

#             mask[
#                 (d[0][0]//down_scale-region_radius):(d[0][0]//down_scale+region_radius+1), 
#                 (d[0][1]//down_scale-region_radius):(d[0][1]//down_scale+region_radius+1)
#             ] = True
#             if random.random() < scaled:
#                 # cs, rs = np.clip(d[0][1]+sur[1], 0, w-1), np.clip(d[0][0]+sur[0], 0, h-1)
#                 cs, rs = d[0][1]/down_scale+sur[1], d[0][0]/down_scale+sur[0]
#                 points.append((rs, cs))
#                 # points.append((cs, h-rs))

#                 score.append(scaled)


#     points = np.array(points)


#     print(f"use {len(points)} points")
#     # kde = KernelDensity(bandwidth=5, metric='haversine',
#     #                     kernel='gaussian', algorithm='ball_tree').fit(points)

#     # plt.scatter( points[:,1].astype(int), h - points[:,0].astype(int) , s=0.1)
#     # plt.xlim(0, w)
#     # plt.ylim(0, h)
#     # plt.gca().set_aspect('equal', adjustable='box')
#     # plt.show()

#     kde = KernelDensity(bandwidth=5, kernel='gaussian').fit(points)
#     X, Y = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
#     # X, Y = np.meshgrid(np.arange(w), np.arange(h), indexing='ij')
#     xy = np.vstack([X.ravel(), Y.ravel()]).T

#     flat_mask = mask.ravel()
#     xy = xy[flat_mask]
#     Z = np.zeros(len(flat_mask))
#     # Z = np.full(flat_mask.shape[0], -9999, dtype='int')
#     Z[flat_mask] = np.exp(kde.score_samples(xy))

#     z_min = np.amin(Z[flat_mask])
#     c = Z < z_min
#     Z[c] = z_min

#     Z = Z.reshape(X.shape)
#     levels = np.linspace(Z.min(), Z.max(), 20)
    
#     # fig, ax = plt.subplots(figsize=(ori_w/10, ori_h/10))
#     fig, ax = plt.subplots()
    
#     # plt.contourf(X, Y, Z, levels=levels, cmap=plt.cm.RdBu_r)
#     ax.contourf(X, Y, Z, levels=levels, cmap=plt.cm.Reds)

#     # plt.xlim(0, w)
#     # plt.ylim(0, h)
#     # plt.gca().set_aspect('equal', adjustable='box')
#     ax.axis("off")
#     plt.tight_layout()

#     plt.show()
#     # plt.savefig('./tmp/tmp_heat.png', dpi=10)
#     # im = Image.open('./tmp/tmp_heat.png')
    

#     import ipdb;ipdb.set_trace() # breakpoint 148

#     print()
    

def draw_heat_map(pil_overlay, end_point_and_scores, point_gap=1.2, meters_per_pixel=0.05, draw_shape='rect', down_scale=2):
    """ KDE version """
    ori_w, ori_h = pil_overlay.size
    score_min = min([d[1] for d in end_point_and_scores])
    score_max = max([d[1] for d in end_point_and_scores])
    
    region_radius = round(point_gap/(meters_per_pixel*down_scale))//2
    surroundings = get_surrounding_point_rel_pos_with_radius(region_radius, region_radius*2+1,)
    random.shuffle(surroundings)

    w,h = pil_overlay.size
    pil_overlay = pil_overlay.resize((w//down_scale, h//down_scale))
    w,h = pil_overlay.size
    print(f"map size: {w} {h}")

    points = []
    score = []
    mask = np.zeros((h,w))>0
    for d in end_point_and_scores:
        scaled = scaler([score_min, score_max], [0, 1], d[1])
        for sur in surroundings:

            mask[
                (d[0][0]//down_scale-region_radius):(d[0][0]//down_scale+region_radius+1), 
                (d[0][1]//down_scale-region_radius):(d[0][1]//down_scale+region_radius+1)
            ] = True
            if random.random() < scaled**2:
                # cs, rs = np.clip(d[0][1]+sur[1], 0, w-1), np.clip(d[0][0]+sur[0], 0, h-1)
                cs, rs = d[0][1]/down_scale+sur[1], d[0][0]/down_scale+sur[0]
                points.append((rs, cs))

                score.append(scaled)


    points = np.array(points)

    kde = KernelDensity(bandwidth=5, kernel='gaussian').fit(points)
    X, Y = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    xy = np.vstack([X.ravel(), Y.ravel()]).T

    flat_mask = mask.ravel()
    # xy = xy[flat_mask]

    Z = np.zeros(len(flat_mask))
    # Z[flat_mask] = np.exp(kde.score_samples(xy))
    Z = np.exp(kde.score_samples(xy))

    z_min = np.amin(Z[flat_mask])
    c = Z < z_min
    Z[c] = z_min
    region_mask = Z<=z_min

    Z = Z.reshape(X.shape)
    levels = np.linspace(Z.min(), Z.max(), num_levels)
    
    fig, ax = plt.subplots(figsize=(ori_h/10, ori_w/10))
    
    # NOTE level controls which part will be shown. 
    #      from level 0 to level max => level lowest to level highest
    ax.contourf(X, Y, Z, levels=levels[1:], colors=heatcm)

    plt.gca().set_aspect('equal', adjustable='box')
    ax.axis("off")
    plt.tight_layout()

    # plt.show()
    plt.savefig('./out/tmp_heat.png', dpi=10)
    plt.close()
    im = Image.open('./out/tmp_heat.png').convert("RGBA")
    im = im.resize(( ori_h, ori_w))
    np_img = np.array(im)
    np_img = np.transpose(np_img, (1, 0, 2))
    np_img = np.flip(np_img, axis=1)

    white_pixel_mask = (np_img == 255).all(axis=2)
    np_img[white_pixel_mask] = [255,255,255, 0]
    np_img[~white_pixel_mask] = np.hstack([
        np_img[~white_pixel_mask][:,:3], np.ones((np_img[~white_pixel_mask].shape[0],1))*128]
    )
    
    pil_overlay = Image.fromarray(np_img)
    return pil_overlay