import numpy as np
import cv2

# Définir une fonction pour effectuer une transformation de perspective
# J'ai utilisé l'image de grille d'exemple ci-dessus pour choisir les points source pour la
# cellule de la grille devant le rover (chaque cellule de la grille est de 1 mètre carré dans la simulation).
def perspect_transform(img):
    img_size = (img.shape[1], img.shape[0])
    # Définir une boîte de calibration en coordonnées source (réelles) et de destination (souhaitées)
    # Ces points source et de destination sont définis pour déformer l'image
    # sur une grille où chaque carré de 10x10 pixels représente 1 mètre carré
    dst_size = 5
    # Définir un décalage inférieur pour tenir compte du fait que le bas de l'image
    # n'est pas la position du rover mais un peu devant lui
    # c'est juste une estimation approximative, n'hésitez pas à la modifier !
    bottom_offset = 6
    src = np.float32([[14, 140], [301, 140], [200, 96], [118, 96]])
    dst = np.float32 ([[img_size[0] / 2 - dst_size, img_size[1] - bottom_offset],
                      [img_size[0] / 2 + dst_size, img_size[1] - bottom_offset],
                      [img_size[0] / 2 + dst_size, img_size[1] - 2 * dst_size - bottom_offset],
                      [img_size[0] / 2 - dst_size, img_size[1] - 2 * dst_size - bottom_offset],
                      ])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size)  # conserver la même taille que l'image d'entrée
    return warped

# Appliquer les fonctions ci-dessus successivement
# Identifier les pixels en dessous du seuil
# Un seuil de RVB > 160 fait un bon travail d'identification des pixels au sol uniquement
def color_thresh(img, low_thresh=(0, 0, 0), high_thresh=(255, 255, 255)):
    # Créer un tableau de zéros de la même taille xy que l'image, mais avec un seul canal
    color_select = np.zeros_like(img[:, :, 0])
    # Exiger que chaque pixel soit au-dessus de toutes les valeurs de seuil en RVB
    # thresh_img contiendra maintenant un tableau booléen avec "True"
    # où le seuil a été atteint
    thresh_img = (img[:, :, 0] >= low_thresh[0]) \
                 & (img[:, :, 1] >= low_thresh[1]) \
                 & (img[:, :, 2] >= low_thresh[2]) \
                 & (img[:, :, 0] <= high_thresh[0]) \
                 & (img[:, :, 1] <= high_thresh[1]) \
                 & (img[:, :, 2] <= high_thresh[2])
    # Indexer le tableau de zéros avec le tableau booléen et le définir à 1
    color_select[thresh_img] = 1
    # Retourner l'image binaire
    return color_select

# Définir une fonction pour convertir en coordonnées centrées sur le rover
def rover_coords(binary_img, limit=80):
    # Identifier les pixels non nuls
    ypos, xpos = binary_img.nonzero()
    # Calculer les positions des pixels par rapport à la position du rover étant au
    # centre en bas de l'image.
    y_pixel = -(xpos - binary_img.shape[1] / 2).astype(np.float)
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    dist = np.sqrt(x_pixel ** 2 + y_pixel ** 2)
    x_pixel, y_pixel = x_pixel[dist < limit], y_pixel[dist < limit]
    return x_pixel, y_pixel

# Définir une fonction pour mapper les pixels de l'espace rover à l'espace mondial
def pix_to_world(dist, angles, x_rover, y_rover, yaw_rover):
    # Mapper les pixels de l'espace rover aux coordonnées mondiales
    pix_angles = angles + (yaw_rover * np.pi / 180)
    # Supposons une taille de carte mondiale de 200 x 200
    world_size = 200
    # Supposons un facteur de changement d'échelle de 10 entre l'espace rover et l'espace mondial
    scale = 10
    x_pix_world = np.clip(np.int_((dist / scale * np.sin(pix_angles)) + x_rover), 0, world_size - 1)
    y_pix_world = np.clip(np.int_((dist / scale * np.cos(pix_angles)) + y_rover), 0, world_size - 1)
    return x_pix_world, y_pix_world

def to_polar_coords(xpix, ypix):
    # Calculer la distance de chaque pixel
    dist = np.sqrt(xpix ** 2 + ypix ** 2)
    # Calculer l'angle en utilisant la fonction arctangente
    angles = np.arctan2(ypix, xpix)
    return dist, angles

# Appliquer les fonctions ci-dessus successivement
def calc_forward_dist(path_dists, path_angles):
    abs_angles = np.absolute(path_angles / sum(path_angles))
    idx = np.abs(abs_angles).argmin()
    return path_dists[idx]

def perception_step(Rover):
    # Effectuer les étapes de perception pour mettre à jour Rover()

    map_img = perspect_transform(Rover.img)
    path_thres_img = color_thresh(map_img, low_thresh=(160, 160, 160))
    rock_thres_img = color_thresh(map_img, low_thresh=(140, 120, 0), high_thresh=(255, 230, 80))
    obstacle_thres_img = color_thresh(map_img, high_thresh=(160, 160, 160))

    Rover.vision_image[:, :, 2] = path_thres_img * 255
    Rover.vision_image[:, :, 1] = rock_thres_img * 255
    Rover.vision_image[:, :, 0] = obstacle_thres_img * 255

    path_xpix, path_ypix = rover_coords(path_thres_img)  # Convertir en coordonnées centrées sur le rover
    rock_xpix, rock_ypix = rover_coords(rock_thres_img)  # Convertir en coordonnées centrées sur le rover
    obst_xpix, obst_ypix = rover_coords(obstacle_thres_img)  # Convertir en coordonnées centrées sur le rover

    path_dists, path_angles = to_polar_coords(path_xpix, path_ypix)  # Convertir en coordonnées polaires
    rock_dist, rock_angles = to_polar_coords(rock_xpix, rock_ypix)  # Convertir en coordonnées polaires
    obs_dist, obs_dist = to_polar_coords(obst_xpix, obst_ypix)  # Convertir en coordonnées polaires

    navigable_x_world, navigable_y_world = pix_to_world(path_dists, path_angles, Rover.pos[0], Rover.pos[1], Rover.yaw)
    rock_x_world, rock_y_world = pix_to_world(rock_dist, rock_angles, Rover.pos[0], Rover.pos[1], Rover.yaw)
    obstacle_x_world, obstacle_y_world = pix_to_world(obs_dist, obs_dist, Rover.pos[0], Rover.pos[1], Rover.yaw)

    if len(path_angles) > 0:
        Rover.dist_to_obstacle = calc_forward_dist(path_dists, path_angles)

    if (Rover.pitch < 1 or Rover.pitch > 359) and (Rover.roll < 1 or Rover.roll > 359):
        Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] = 255
        Rover.worldmap[rock_y_world, rock_x_world, 1] = 255
        Rover.worldmap[navigable_y_world, navigable_x_world, 2] = 255
        # supprimer les mesures qui se chevauchent
        nav_pix = Rover.worldmap[:, :, 2] > 0
        Rover.worldmap[nav_pix, 0] = 0
        # limiter pour éviter le débordement
        Rover.worldmap = np.clip(Rover.worldmap, 0, 255)

    Rover.nav_dists = path_dists
    Rover.nav_angles = path_angles

    Rover.samples_dists = rock_dist
    Rover.samples_angles = rock_angles

    Rover.nav_area = path_thres_img.sum()

    return Rover
