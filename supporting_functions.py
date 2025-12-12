import numpy as np
import cv2
from PIL import Image
from io import BytesIO, StringIO
import base64
import time

# Définir une fonction pour convertir les chaînes de télémétrie en nombres à virgule flottante, indépendamment de la convention décimale
def convert_to_float(string_to_convert):
    if ',' in string_to_convert:
        float_value = np.float(string_to_convert.replace(',', '.'))
    else:
        float_value = np.float(string_to_convert)
    return float_value

def update_rover(Rover, data):
    # Initialiser le temps de départ et les positions des échantillons
    if Rover.start_time is None:
        Rover.start_time = time.time()
        Rover.total_time = 0
        samples_xpos = np.int_([convert_to_float(pos.strip()) for pos in data["samples_x"].split(';')])
        samples_ypos = np.int_([convert_to_float(pos.strip()) for pos in data["samples_y"].split(';')])
        Rover.samples_pos = (samples_xpos, samples_ypos)
        Rover.samples_to_find = np.int(data["sample_count"])
    # Ou simplement mettre à jour le temps écoulé
    else:
        tot_time = time.time() - Rover.start_time
        if np.isfinite(tot_time):
            Rover.total_time = tot_time
    # Afficher les champs du dictionnaire de données de télémétrie
    # La vitesse actuelle du rover en m/s
    Rover.vel = convert_to_float(data["speed"])
    # La position actuelle du rover
    Rover.pos = [convert_to_float(pos.strip()) for pos in data["position"].split(';')]
    # L'angle de lacet actuel du rover
    Rover.yaw = convert_to_float(data["yaw"])
    # L'angle de tangage actuel du rover
    Rover.pitch = convert_to_float(data["pitch"])
    # L'angle de roulis actuel du rover
    Rover.roll = convert_to_float(data["roll"])
    # Paramètres de gaz actuels
    Rover.throttle = convert_to_float(data["throttle"])
    # L'angle de direction actuel
    Rover.steer = convert_to_float(data["steering_angle"])
    # Indicateur de proximité d'échantillon
    Rover.near_sample = np.int(data["near_sample"])
    # Indicateur de ramassage
    Rover.picking_up = np.int(data["picking_up"])
    # Mettre à jour le nombre de rochers collectés
    Rover.samples_collected = Rover.samples_to_find - np.int(data["sample_count"])

    # Obtenir l'image actuelle de la caméra centrale du rover
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    Rover.img = np.asarray(image)

    # Renvoyer Rover mis à jour et une image séparée pour une éventuelle sauvegarde
    return Rover, image

# Définir une fonction pour créer une sortie d'affichage en fonction des résultats de la carte du monde
def create_output_images(Rover):
    # Créer une carte mise à l'échelle pour le tracé et nettoyer un peu les pixels obs/nav
    if np.max(Rover.worldmap[:, :, 2]) > 0:
        nav_pix = Rover.worldmap[:, :, 2] > 0
        navigable = Rover.worldmap[:, :, 2] * (255 / np.mean(Rover.worldmap[nav_pix, 2]))
    else:
        navigable = Rover.worldmap[:, :, 2]
    if np.max(Rover.worldmap[:, :, 0]) > 0:
        obs_pix = Rover.worldmap[:, :, 0] > 0
        obstacle = Rover.worldmap[:, :, 0] * (255 / np.mean(Rover.worldmap[obs_pix, 0]))
    else:
        obstacle = Rover.worldmap[:, :, 0]

    likely_nav = navigable >= obstacle
    obstacle[likely_nav] = 0
    plotmap = np.zeros_like(Rover.worldmap)
    plotmap[:, :, 0] = obstacle
    plotmap[:, :, 2] = navigable
    plotmap = plotmap.clip(0, 255)

    # Superposer la carte des obstacles et du terrain navigable avec la carte de vérité terrain
    map_add = cv2.addWeighted(plotmap, 1, Rover.ground_truth, 0.5, 0)

    # Vérifiez si des détections de rochers sont présentes dans la carte du monde
    rock_world_pos = Rover.worldmap[:, :, 1].nonzero()
    # S'il y en a, nous allons parcourir les positions d'échantillons connues
    # pour confirmer si les détections sont réelles
    samples_located = 0
    if rock_world_pos[0].any():

        rock_size = 2
        for idx in range(len(Rover.samples_pos[0])):
            test_rock_x = Rover.samples_pos[0][idx]
            test_rock_y = Rover.samples_pos[1][idx]
            rock_sample_dists = np.sqrt((test_rock_x - rock_world_pos[1]) ** 2 + \
                                        (test_rock_y - rock_world_pos[0]) ** 2)
            # Si des rochers ont été détectés à moins de 3 mètres des positions d'échantillons connues
            # considérez-le comme un succès et tracez l'emplacement de l'échantillon connu sur la carte
            if np.min(rock_sample_dists) < 3:
                samples_located += 1
                map_add[test_rock_y - rock_size:test_rock_y + rock_size,
                        test_rock_x - rock_size:test_rock_x + rock_size, :] = 255

    # Calculez certaines statistiques sur les résultats de la carte
    # Tout d'abord, obtenez le nombre total de pixels dans la carte du terrain navigable
    tot_nav_pix = np.float(len((plotmap[:, :, 2].nonzero()[0])))
    # Ensuite, déterminez combien de ces pixels correspondent aux pixels de vérité terrain
    good_nav_pix = np.float(len(((plotmap[:, :, 2] > 0) & (Rover.ground_truth[:, :, 1] > 0)).nonzero()[0]))
    # Ensuite, trouvez combien ne correspondent pas aux pixels de vérité terrain
    bad_nav_pix = np.float(len(((plotmap[:, :, 2] > 0) & (Rover.ground_truth[:, :, 1] == 0)).nonzero()[0]))
    # Obtenez le nombre total de pixels de carte
    tot_map_pix = np.float(len((Rover.ground_truth[:, :, 1].nonzero()[0])))
    # Calculez le pourcentage de la carte de vérité terrain qui a été trouvée avec succès
    perc_mapped = round(100 * good_nav_pix / tot_map_pix, 1)
    # Calculez le nombre de bonnes détections de pixels de carte divisé par le nombre total de pixels
    # trouvés pour être du terrain navigable
    if tot_nav_pix > 0:
        fidelity = round(100 * good_nav_pix / (tot_nav_pix), 1)
    else:
        fidelity = 0
    # Inverser la carte pour le tracé de sorte que l'axe y pointe vers le haut dans l'affichage
    map_add = np.flipud(map_add).astype(np.float32)
    # Ajouter du texte sur les résultats de la carte et de la détection d'échantillon de roche
    cv2.putText(map_add, "Temps : " + str(np.round(Rover.total_time, 1)) + ' s', (0, 10),
                cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(map_add, "Cartographiee : " + str(perc_mapped) + '%', (0, 25),
                cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(map_add, "Fidelitee : " + str(fidelity) + '%', (0, 40),
                cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(map_add, "Rochers", (0, 55),
                cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(map_add, "  Localises : " + str(samples_located), (0, 70),
                cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(map_add, "  Collectes : " + str(Rover.samples_collected), (0, 85),
                cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
    # Convertir la carte et l'image de vision en chaînes base64 pour les envoyer au serveur
    pil_img = Image.fromarray(map_add.astype(np.uint8))
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    encoded_string1 = base64.b64encode(buff.getvalue()).decode("utf-8")

    pil_img = Image.fromarray(Rover.vision_image.astype(np.uint8))
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    encoded_string2 = base64.b64encode(buff.getvalue()).decode("utf-8")

    return encoded_string1, encoded_string2
