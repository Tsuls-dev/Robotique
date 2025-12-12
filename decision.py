import numpy as np
import math


def decision_step(Rover):
    """Système de décision autonome intelligent"""

    # Initialiser les attributs manquants
    if not hasattr(Rover, 'throttle_set'):
        Rover.throttle_set = 0.3
    if not hasattr(Rover, 'brake_set'):
        Rover.brake_set = 10
    if not hasattr(Rover, 'stuck_counter'):
        Rover.stuck_counter = 0
    if not hasattr(Rover, 'reverse_counter'):
        Rover.reverse_counter = 0

    # Vérifier si nous avons des données de vision
    if Rover.nav_angles is not None and len(Rover.nav_angles) > 0:

        # MODE: SAMPLING (collecte d'échantillon)
        if Rover.mode == 'sampling':
            print("🎯 MODE SAMPLING: Approche d'échantillon")

            # Si échantillon à portée, arrêter pour collecter
            if Rover.near_sample:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
                if Rover.vel == 0 and not Rover.picking_up:
                    Rover.send_pickup = True
                return Rover

            # Si échantillon visible, s'en approcher
            if len(Rover.samples_dists) > 0 and len(Rover.samples_angles) > 0:
                # Prendre l'échantillon le plus proche
                idx = np.argmin(Rover.samples_dists)
                target_angle = Rover.samples_angles[idx]
                distance = Rover.samples_dists[idx]

                # Contrôle direction
                Rover.steer = np.clip(target_angle * 180 / np.pi, -15, 15)

                # Contrôle vitesse (ralentir à l'approche)
                if distance < 5:
                    Rover.throttle = 0.05  # Très lent
                elif distance < 10:
                    Rover.throttle = 0.1  # Lent
                else:
                    Rover.throttle = 0.2  # Normal

                Rover.brake = 0

                # Si échantillon perdu de vue
                if distance > 30:
                    Rover.mode = 'forward'
            else:
                # Plus d'échantillon en vue
                Rover.mode = 'forward'

        # MODE: STUCK (coincé)
        elif Rover.mode == 'stuck':
            print("⚠️ MODE STUCK: Tentative de déblocage")

            Rover.stuck_counter += 1

            # Phase 1: Reculer (1-10 frames)
            if Rover.stuck_counter < 10:
                Rover.throttle = -0.3
                Rover.brake = 0
                Rover.steer = 0

            # Phase 2: Tourner (10-30 frames)
            elif Rover.stuck_counter < 30:
                Rover.throttle = 0
                Rover.brake = 0
                Rover.steer = 15  # Tourner à gauche

            # Phase 3: Avancer doucement (30-40 frames)
            elif Rover.stuck_counter < 40:
                Rover.throttle = 0.1
                Rover.brake = 0
                Rover.steer = 0

            # Retour à la normale
            else:
                Rover.mode = 'forward'
                Rover.stuck_counter = 0

        # MODE: FORWARD (avancer - mode principal)
        elif Rover.mode == 'forward':
            # Vérifier s'il y a un échantillon à collecter
            if len(Rover.samples_dists) > 0 and np.min(Rover.samples_dists) < 15:
                print(f"🎯 Échantillon détecté à {np.min(Rover.samples_dists):.1f}m")
                Rover.mode = 'sampling'
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                return Rover

            # Vérifier s'il y a assez de terrain navigable
            nav_pixels = len(Rover.nav_angles)

            if nav_pixels >= Rover.stop_forward:
                # STRATÉGIE DE NAVIGATION INTELLIGENTE

                # 1. Suivi de mur à GAUCHE (meilleure stratégie)
                # Privilégier les angles à GAUCHE (positifs)
                left_angles = Rover.nav_angles[Rover.nav_angles > 0]

                if len(left_angles) > 0:
                    # Moyenne des angles à gauche avec un biais supplémentaire à gauche
                    mean_angle = np.mean(left_angles)
                    bias = 0.2  # radians de biais à gauche
                    steer_angle = (mean_angle + bias) * 180 / np.pi
                else:
                    # Si pas d'angles à gauche, prendre la moyenne générale
                    mean_angle = np.mean(Rover.nav_angles)
                    steer_angle = mean_angle * 180 / np.pi

                # Limiter l'angle de braquage
                Rover.steer = np.clip(steer_angle, -15, 15)

                # 2. CONTRÔLE DE VITESSE INTELLIGENT
                if Rover.vel < Rover.max_vel:
                    # Ajuster la vitesse selon la largeur du chemin
                    if nav_pixels > 800:  # Chemin très large
                        Rover.throttle = 0.4
                    elif nav_pixels > 400:  # Chemin large
                        Rover.throttle = 0.3
                    elif nav_pixels > 200:  # Chelin moyen
                        Rover.throttle = 0.2
                    else:  # Chemin étroit
                        Rover.throttle = 0.1

                    Rover.brake = 0
                else:
                    # Vitesse maximale atteinte
                    Rover.throttle = 0
                    Rover.brake = 0

                # 3. DÉTECTION DE BLOCAGE PROACTIVE
                if Rover.vel < 0.1 and nav_pixels > 300:
                    # Si vitesse très basse mais chemin large -> peut-être coincé
                    Rover.stuck_counter += 1
                    if Rover.stuck_counter > 30:  # 3 secondes
                        Rover.mode = 'stuck'
                        Rover.stuck_counter = 0
                else:
                    Rover.stuck_counter = 0

                # 4. ÉVITEMENT D'OBSTACLES
                if Rover.dist_to_obstacle < 5 and Rover.vel > 0.5:
                    # Ralentir si obstacle proche
                    Rover.throttle = 0
                    Rover.brake = min(5, Rover.brake_set)

            else:
                # Pas assez de terrain navigable -> arrêter
                print("🚫 Pas assez de terrain navigable - Arrêt")
                Rover.mode = 'stop'
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0

        # MODE: STOP (arrêt)
        elif Rover.mode == 'stop':
            print("⏸️ MODE STOP: Recherche de chemin")

            # Si on bouge encore, freiner
            if Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0

            # Si arrêté, chercher un chemin
            elif Rover.vel <= 0.2:
                Rover.throttle = 0
                Rover.brake = 0

                # Tourner pour chercher un chemin
                Rover.steer = -15  # Tourner à droite

                # Vérifier si un chemin est trouvé
                if len(Rover.nav_angles) > Rover.go_forward // 2:
                    print("✅ Chemin trouvé - Reprise")
                    Rover.mode = 'forward'
                    Rover.throttle = 0.2
                    Rover.steer = 0

    else:
        # Pas de données de navigation
        print("📡 Pas de données de navigation")
        Rover.throttle = 0
        Rover.brake = Rover.brake_set
        Rover.steer = 0
        if Rover.mode != 'stop':
            Rover.mode = 'stop'

    # Gestion du ramassage d'échantillon
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        print("🎯 Échantillon à portée - Envoi commande pickup")
        Rover.send_pickup = True

    return Rover