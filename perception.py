import numpy as np
import cv2


def perspect_transform(img):
    img_size = (img.shape[1], img.shape[0])
    dst_size = 5
    bottom_offset = 6
    src = np.float32([[14, 140], [301, 140], [200, 96], [118, 96]])
    dst = np.float32([[img_size[0] / 2 - dst_size, img_size[1] - bottom_offset],
                      [img_size[0] / 2 + dst_size, img_size[1] - bottom_offset],
                      [img_size[0] / 2 + dst_size, img_size[1] - 2 * dst_size - bottom_offset],
                      [img_size[0] / 2 - dst_size, img_size[1] - 2 * dst_size - bottom_offset]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size)
    return warped


def color_thresh(img, low_thresh=(0, 0, 0), high_thresh=(255, 255, 255)):
    color_select = np.zeros_like(img[:, :, 0])
    thresh_img = (img[:, :, 0] >= low_thresh[0]) \
                 & (img[:, :, 1] >= low_thresh[1]) \
                 & (img[:, :, 2] >= low_thresh[2]) \
                 & (img[:, :, 0] <= high_thresh[0]) \
                 & (img[:, :, 1] <= high_thresh[1]) \
                 & (img[:, :, 2] <= high_thresh[2])
    color_select[thresh_img] = 1
    return color_select


def rover_coords(binary_img, limit=80):
    ypos, xpos = binary_img.nonzero()

    if len(ypos) == 0 or len(xpos) == 0:
        return np.array([]), np.array([])

    y_pixel = -(xpos - binary_img.shape[1] / 2).astype(np.float32)
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float32)

    dist = np.sqrt(x_pixel ** 2 + y_pixel ** 2)
    mask = dist < limit

    return x_pixel[mask], y_pixel[mask]


def pix_to_world(xpix, ypix, x_rover, y_rover, yaw_rover):
    if len(xpix) == 0 or len(ypix) == 0:
        return np.array([]), np.array([])

    dist = np.sqrt(xpix ** 2 + ypix ** 2)
    angles = np.arctan2(ypix, xpix)

    pix_angles = angles + (yaw_rover * np.pi / 180)
    world_size = 200
    scale = 10

    x_pix_world = np.clip(np.int32((dist / scale * np.sin(pix_angles)) + x_rover), 0, world_size - 1)
    y_pix_world = np.clip(np.int32((dist / scale * np.cos(pix_angles)) + y_rover), 0, world_size - 1)

    return x_pix_world, y_pix_world


def to_polar_coords(xpix, ypix):
    if len(xpix) == 0 or len(ypix) == 0:
        return np.array([]), np.array([])

    dist = np.sqrt(xpix ** 2 + ypix ** 2)
    angles = np.arctan2(ypix, xpix)

    return dist, angles


def calc_forward_dist(path_dists, path_angles):
    if len(path_dists) == 0:
        return 0

    # Prendre la distance médiane des points devant (angles proches de 0)
    front_mask = np.abs(path_angles) < 0.3  # ±17 degrés
    if np.any(front_mask):
        front_dists = path_dists[front_mask]
        return np.median(front_dists)
    else:
        return np.median(path_dists)


def perception_step(Rover):
    # Transformation de perspective
    map_img = perspect_transform(Rover.img)

    # Seuillage de couleur
    path_thres_img = color_thresh(map_img, low_thresh=(160, 160, 160))
    rock_thres_img = color_thresh(map_img, low_thresh=(140, 120, 0), high_thresh=(255, 230, 80))
    obstacle_thres_img = color_thresh(map_img, high_thresh=(160, 160, 160))

    # Image de vision
    Rover.vision_image[:, :, 2] = path_thres_img * 255
    Rover.vision_image[:, :, 1] = rock_thres_img * 255
    Rover.vision_image[:, :, 0] = obstacle_thres_img * 255

    # Coordonnées rover
    path_xpix, path_ypix = rover_coords(path_thres_img)
    rock_xpix, rock_ypix = rover_coords(rock_thres_img)
    obst_xpix, obst_ypix = rover_coords(obstacle_thres_img)

    # Coordonnées polaires
    path_dists, path_angles = to_polar_coords(path_xpix, path_ypix)
    rock_dist, rock_angles = to_polar_coords(rock_xpix, rock_ypix)
    obs_dist, obs_angles = to_polar_coords(obst_xpix, obst_ypix)

    # Conversion monde
    navigable_x_world, navigable_y_world = pix_to_world(path_xpix, path_ypix, Rover.pos[0], Rover.pos[1], Rover.yaw)
    rock_x_world, rock_y_world = pix_to_world(rock_xpix, rock_ypix, Rover.pos[0], Rover.pos[1], Rover.yaw)
    obstacle_x_world, obstacle_y_world = pix_to_world(obst_xpix, obst_ypix, Rover.pos[0], Rover.pos[1], Rover.yaw)

    # Distance aux obstacles
    if len(path_angles) > 0:
        Rover.dist_to_obstacle = calc_forward_dist(path_dists, path_angles)
    else:
        Rover.dist_to_obstacle = 0

    # Mise à jour carte monde
    if (Rover.pitch < 1 or Rover.pitch > 359) and (Rover.roll < 1 or Rover.roll > 359):
        if len(obstacle_x_world) > 0:
            Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] = 255
        if len(rock_x_world) > 0:
            Rover.worldmap[rock_y_world, rock_x_world, 1] = 255
        if len(navigable_x_world) > 0:
            Rover.worldmap[navigable_y_world, navigable_x_world, 2] = 255

        # Nettoyage
        nav_pix = Rover.worldmap[:, :, 2] > 0
        Rover.worldmap[nav_pix, 0] = 0
        Rover.worldmap = np.clip(Rover.worldmap, 0, 255)

    # Stockage données navigation
    Rover.nav_dists = path_dists
    Rover.nav_angles = path_angles
    Rover.samples_dists = rock_dist
    Rover.samples_angles = rock_angles
    Rover.nav_area = path_thres_img.sum()

    # Debug
    print(f"👁️ Perception: Nav={len(path_angles)}, Rocks={len(rock_angles)}, Dist={Rover.dist_to_obstacle:.1f}")

    return Rover