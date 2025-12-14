import numpy as np
import time

def decision_step(Rover):
    # 1. Gestion du ramassage
    if Rover.near_sample:
        Rover.throttle = 0
        Rover.brake = 10
        if not Rover.picking_up:
            Rover.send_pickup = True
        return Rover

    # 2. Détection de blocage
    if Rover.mode == 'forward':
        # Si on veut avancer mais que la vitesse est quasi nulle
        if Rover.vel < 0.2 and Rover.throttle > 0.1:
            if not hasattr(Rover, 'stuck_time'):
                Rover.stuck_time = time.time()
            elif time.time() - Rover.stuck_time > 2: # Bloqué plus de 2 sec
                Rover.mode = 'stuck'
        else:
            Rover.stuck_time = time.time()

    # 3. Logique des modes
    if Rover.nav_angles is not None:
        if Rover.mode == 'stuck':
            Rover.throttle = 0
            Rover.brake = 0
            Rover.steer = -15 # Tourne pour se dégager
            if len(Rover.nav_angles) > Rover.go_forward:
                Rover.mode = 'forward'
                delattr(Rover, 'stuck_time')

        elif Rover.mode == 'forward':
            if len(Rover.nav_angles) >= Rover.stop_forward:
                # Priorité aux rochers
                if len(Rover.samples_angles) > 0:
                    Rover.steer = np.clip(np.mean(Rover.samples_angles * 180 / np.pi), -15, 15)
                else:
                    # Navigation normale (on ajoute un petit offset de 0.1 pour longer les murs)
                    Rover.steer = np.clip(np.mean((Rover.nav_angles + 0.1) * 180 / np.pi), -15, 15)

                if Rover.vel < Rover.max_vel:
                    Rover.throttle = 0.2
                else:
                    Rover.throttle = 0
                Rover.brake = 0
            else:
                Rover.mode = 'stop'

        elif Rover.mode == 'stop':
            if Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = 10
                Rover.steer = 0
            else:
                Rover.throttle = 0
                Rover.brake = 0
                Rover.steer = -15
                if len(Rover.nav_angles) >= Rover.go_forward:
                    Rover.mode = 'forward'

    return Rover
