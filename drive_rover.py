import argparse
import shutil
import base64
from datetime import datetime
import os
import cv2
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO, StringIO
import json
import pickle
import matplotlib.image as mpimg
import time

# Importez les fonctions pour la perception et la prise de décision
from perception import perception_step
from decision import decision_step
from supporting_functions import update_rover, create_output_images

# Initialisez le serveur socketio et l'application Flask
# (en savoir plus sur : https://python-socketio.readthedocs.io/en/latest/)
sio = socketio.Server()
app = Flask(__name__)

# Lisez la carte de vérité terrain et créez une version verte en 3 canaux pour la superposition
# REMARQUE : les images sont lues par défaut avec l'origine (0, 0) en haut à gauche
# et l'axe des y augmentant vers le bas.
ground_truth = mpimg.imread('../calibration_images/map_bw.jpg')
# Cette ligne suivante crée des tableaux de zéros dans les canaux rouge et bleu
# et place la carte dans le canal vert. C'est pourquoi la carte sous-jacente
# a l'air verte dans l'image d'affichage
ground_truth_3d = np.dstack((ground_truth * 0, ground_truth * 255, ground_truth * 0)).astype(np.float)


# Définissez la classe RoverState() pour conserver les paramètres d'état du rover
class RoverState():
    def __init__(self):
        self.start_time = None  # Pour enregistrer l'heure de début de la navigation
        self.total_time = None  # Pour enregistrer la durée totale de la navigation
        self.img = None  # Image de caméra actuelle
        self.pos = None  # Position actuelle (x, y)
        self.yaw = None  # Angle de lacet actuel
        self.pitch = None  # Angle de tangage actuel
        self.roll = None  # Angle de roulis actuel
        self.vel = None  # Vitesse actuelle
        self.steer = 0  # Angle de direction actuel
        self.throttle = 0  # Valeur actuelle de l'accélération
        self.brake = 0  # Valeur actuelle du frein
        self.nav_angles = None  # Angles des pixels de terrain navigable
        self.nav_dists = None  # Distances des pixels de terrain navigable
        self.ground_truth = ground_truth_3d  # Carte du monde de vérité terrain
        self.mode = 'forward'  # Mode actuel (peut être "forward" ou "stop")
        self.nav_area = 0
        self.stop_forward = 50  # Seuil pour initier l'arrêt
        self.go_forward = 500  # Seuil pour avancer à nouveau
        self.max_vel = 2  # Vitesse maximale (mètres/seconde)

        self.vision_image = np.zeros((160, 320, 3), dtype=np.float)

        self.worldmap = np.zeros((200, 200, 3), dtype=np.float)
        self.samples_pos = None  # Pour stocker les positions d'échantillons réelles
        self.samples_to_find = 0  # Pour stocker le nombre initial d'échantillons
        self.samples_located = 0  # Pour stocker le nombre d'échantillons situés sur la carte
        self.samples_collected = 0  # Pour compter le nombre d'échantillons collectés
        self.near_sample = 0  # Sera défini sur la valeur de télémétrie data["near_sample"]
        self.picking_up = 0  # Sera défini sur la valeur de télémétrie data["picking_up"]
        self.send_pickup = False  # Défini sur True pour déclencher la collecte de rochers
        self.samples_dists = np.asarray([])
        self.samples_angles = np.asarray([])
        self.dist_to_obstacle = 0

# Initialisez notre rover
Rover = RoverState()

# Variables pour suivre les images par seconde (FPS)
# Compteur de trames initial
frame_counter = 0
# Initialisation du compteur de secondes
second_counter = time.time()
fps = None

# Définissez la fonction de télémétrie pour ce que vous voulez faire avec les données entrantes
@sio.on('telemetry')
def telemetry(sid, data):
    global frame_counter, second_counter, fps
    frame_counter += 1
    # Faites un calcul approximatif des images par seconde (FPS)
    if (time.time() - second_counter) > 1:
        fps = frame_counter
        frame_counter = 0
        second_counter = time.time()

    if data:
        global Rover
        # Initialisez/Mettez à jour Rover avec la télémétrie actuelle
        Rover, image = update_rover(Rover, data)

        if np.isfinite(Rover.vel):

            # Exécutez les étapes de perception et de décision pour mettre à jour l'état du Rover
            Rover = perception_step(Rover)
            Rover = decision_step(Rover)

            # Créez des images de sortie à envoyer au serveur
            out_image_string1, out_image_string2 = create_output_images(Rover)

            # L'étape d'action ! Envoyez des commandes au rover !

            # Ne pas envoyer les deux, elles déclenchent toutes deux le simulateur
            # pour renvoyer de nouvelles données de télémétrie, nous devons donc en envoyer une seule
            # en réponse aux données de télémétrie actuelles.

            # Si vous êtes dans un état où vous voulez collecter un rocher, envoyez la commande de collecte
            if Rover.send_pickup and not Rover.picking_up:
                send_pickup()
                # Réinitialisez les indicateurs de Rover
                Rover.send_pickup = False
            else:
                # Envoyez des commandes au rover !
                commands = (Rover.throttle, Rover.brake, Rover.steer)
                send_control(commands, out_image_string1, out_image_string2)

        # En cas de télémétrie non valide, envoyez des commandes nulles
        else:

            # Envoyez des zéros pour l'accélération, le frein et la direction et des images vides
            send_control((0, 0, 0), '', '')

        # Si vous souhaitez enregistrer des images de la caméra lors de la conduite autonome, spécifiez un chemin
        # Exemple : $ python drive_rover.py image_folder_path
        # Condition pour enregistrer une image si un dossier a été spécifié
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))

    else:
        sio.emit('manual', data={}, skip_sid=True)

# Fonction pour envoyer les commandes de contrôle
def send_control(commands, image_string1, image_string2):
    # Définissez les commandes à envoyer au rover
    data = {
        'throttle': commands[0].__str__(),
        'brake': commands[1].__str__(),
        'steering_angle': commands[2].__str__(),
        'inset_image1': image_string1,
        'inset_image2': image_string2,
    }
    # Envoyez les commandes via le serveur socketIO
    sio.emit(
        "data",
        data,
        skip_sid=True)
    eventlet.sleep(0)

# Définissez une fonction pour envoyer la commande de "pickup"
def send_pickup():
    print("Ramassage en cours")
    pickup = {}
    sio.emit(
        "pickup",
        pickup,
        skip_sid=True)
    eventlet.sleep(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conduite à distance')
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help="Chemin du dossier d'images. C'est l'endroit où les images de la course seront enregistrées."
    )
    args = parser.parse_args()

    # os.system('rm -rf IMG_stream/*')
    if args.image_folder != '':
        print("Création du dossier d'images à l'adresse {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("Enregistrement de cette course...")
    else:
        print("PAS d'enregistrement de cette course...")

    # Enveloppez l'application Flask avec le middleware socketio
    app = socketio.Middleware(sio, app)

    # Déployez en tant que serveur WSGI eventlet
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
