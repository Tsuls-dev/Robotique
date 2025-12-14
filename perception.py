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
    return cv2.warpPerspective(img, M, img_size)


def color_thresh(img, low_thresh=(0, 0, 0), high_thresh=(255, 255, 255)):
    color_select = np.zeros_like(img[:, :, 0])
    thresh_img = (img[:, :, 0] >= low_thresh[0]) & (img[:, :, 1] >= low_thresh[1]) & (img[:, :, 2] >= low_thresh[2]) \
                 & (img[:, :, 0] <= high_thresh[0]) & (img[:, :, 1] <= high_thresh[1]) & (
                             img[:, :, 2] <= high_thresh[2])
    color_select[thresh_img] = 1
    return color_select


def rover_coords(binary_img, limit=80):
    ypos, xpos = binary_img.nonzero()
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1] / 2).astype(np.float)
    dist = np.sqrt(x_pixel ** 2 + y_pixel ** 2)
    return x_pixel[dist < limit], y_pixel[dist < limit]


# MODIFICATION ICI : Ajout de world_size et scale dans les arguments
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    dist, angles = to_polar_coords(xpix, ypix)
    yaw_rad = yaw * np.pi / 180
    pix_angles = angles + yaw_rad
    x_world = (dist / scale) * np.cos(pix_angles) + xpos
    y_world = (dist / scale) * np.sin(pix_angles) + ypos
    x_pix_world = np.clip(np.int_(x_world), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(y_world), 0, world_size - 1)
    return x_pix_world, y_pix_world


def to_polar_coords(xpix, ypix):
    dist = np.sqrt(xpix ** 2 + ypix ** 2)
    angles = np.arctan2(ypix, xpix)
    return dist, angles


def perception_step(Rover):
    warped = perspect_transform(Rover.img)

    # Seuillage
    navigable = color_thresh(warped, low_thresh=(160, 160, 160))
    obstacles = color_thresh(warped, high_thresh=(100, 100, 100))
    rocks = color_thresh(warped, low_thresh=(110, 110, 0), high_thresh=(255, 255, 80))

    Rover.vision_image[:, :, 0] = obstacles * 255
    Rover.vision_image[:, :, 1] = rocks * 255
    Rover.vision_image[:, :, 2] = navigable * 255

    xpix, ypix = rover_coords(navigable)
    obsx, obsy = rover_coords(obstacles)
    rockx, rocky = rover_coords(rocks)

    world_size = Rover.worldmap.shape[0]
    scale = 10

    # Appels corrigés avec le bon nombre d'arguments
    nav_x_world, nav_y_world = pix_to_world(xpix, ypix, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)
    obs_x_world, obs_y_world = pix_to_world(obsx, obsy, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)
    rock_x_world, rock_y_world = pix_to_world(rockx, rocky, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)

    # Mise à jour de la carte si le rover est stable
    if (Rover.pitch < 1.0 or Rover.pitch > 359.0) and (Rover.roll < 1.0 or Rover.roll > 359.0):
        Rover.worldmap[obs_y_world, obs_x_world, 0] += 1
        Rover.worldmap[rock_y_world, rock_x_world, 1] = 255
        Rover.worldmap[nav_y_world, nav_x_world, 2] += 1

    dist, angles = to_polar_coords(xpix, ypix)
    Rover.nav_dists = dist
    Rover.nav_angles = angles

    if rocks.any():
        rdist, rangles = to_polar_coords(rockx, rocky)
        Rover.samples_dists = rdist
        Rover.samples_angles = rangles
    else:
        Rover.samples_dists = np.asarray([])
        Rover.samples_angles = np.asarray([])

    return Rover
