import numpy as np

# Il s'agit essentiellement d'un arbre de décision pour déterminer les commandes d'accélération, de frein et de direction
# basées sur la sortie de la fonction perception_step()
def decision_step(Rover):
    # Vérifiez si nous avons des données de vision pour prendre des décisions
    if Rover.nav_angles is not None:
        if Rover.mode == 'sampling':
            if Rover.vel != 0:
                Rover.brake = 10
                if len(Rover.samples_dists) == 0:
                    Rover.mode = 'forward'
        elif Rover.mode == 'stuck' and len(Rover.samples_dists) == 0:
            if Rover.vel != 0:
                Rover.brake = 10
            Rover.steer = -15
            Rover.throttle = 0
            Rover.brake = 0
            if Rover.nav_area > 650:
                Rover.mode = 'forward'
        elif Rover.mode == 'forward':
            if 650 > Rover.nav_area > 10:
                Rover.throttle = 0
                Rover.brake = 10
                Rover.steer = 0
                Rover.mode = 'stuck'
            if len(Rover.samples_dists) > 0 and np.min(Rover.samples_dists) < (5 * Rover.vel):
                Rover.mode = 'sampling'
                Rover.throttle = 0
                Rover.brake = 10
                Rover.steer = 0
            elif len(Rover.nav_angles) >= Rover.stop_forward:
                # Si le mode est en avant, le terrain navigable semble bon
                # Sauf au départ, si arrêté signifie être coincé.
                # Alterne entre les modes "coincé" et "avant"
                if len(Rover.samples_angles) > 0:
                    drive_angles = Rover.samples_angles
                    drive_distance = np.min(Rover.samples_dists)
                else:
                    drive_angles = Rover.nav_angles
                    drive_distance = Rover.dist_to_obstacle
                # Réglez la valeur de l'accélérateur sur le réglage de l'accélérateur
                Rover.throttle = np.clip(drive_distance * 0.005 - Rover.vel * 0.2, 0, 2)
                Rover.brake = 0
                # Réglez la direction sur l'angle moyen limité à la plage +/- 15
                Rover.steer = np.clip(np.mean(drive_angles * 180 / np.pi), -15, 15)
            # S'il y a un manque de pixels de terrain navigable, passez au mode "arrêt"
            # Si nous sommes déjà dans le mode "coincé". Restez ici pendant 1 seconde
            else:
                Rover.throttle = 0
                # Relâchez le frein pour permettre la rotation
                Rover.brake = 0
                # La plage de rotation est de +/- 15 degrés, quand arrêté, la ligne suivante induira une rotation à 4 roues
                # Puisque la direction doit être légèrement à droite pour coller au mur gauche :
                Rover.steer = 0
                Rover.mode = 'stop'
        # Si nous sommes déjà dans le mode "arrêt", prenez des décisions différentes
        elif Rover.mode == 'stop':
            # Si nous sommes en mode arrêt mais que nous bougeons toujours, continuez à freiner
            if Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = 10
                Rover.steer = 0
            if len(Rover.nav_angles) > 0:
                Rover.mode = 'forward'
    # Juste pour faire bouger le rover, même si aucune modification n'a été apportée au code
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0
    # Si dans un état où nous voulons ramasser un échantillon, envoyez la commande de ramassage
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True
    print('State', Rover.mode)
    return Rover
