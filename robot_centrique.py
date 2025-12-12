import cv2 as cv
import numpy as np
import glob  # Pour la lecture d'image dans un dossier


path = '../test_dataset/IMG/*'
img_list = glob.glob(path)

# Recuperrer une image aleatoire et l'afficher
idx = np.random.randint(0, len(img_list)-1)
image = cv.imread(img_list[idx])
# Afficher l'image
cv.imshow("Image Random", image)
cv.waitKey(0)

# Dans le simulateur, afficher la grille
# Vous pouvez aussi afficher une Sample avec la touche 0 (zero).
# Voici un exemple avec une grille et une Sample
example_grid = '../calibration_images/example_grid1.jpg'
example_rock = '../calibration_images/example_rock1.jpg'

grid_img = cv.imread(example_grid)
rock_img = cv.imread(example_rock)

cv.imshow("GRID EXEMPLE", grid_img)
cv.imshow("ROCK EXEMPLE", rock_img)
cv.waitKey(0)

# J'ai utilisé les images d'exemple ci dessus comme reference
# Le cellule de grille dans le simulateur est defini comme 1m x 1m
# Definir une fonction perspective transform


def perspect_transform(img, src, dst):

    M = cv.getPerspectiveTransform(src, dst)
    warped = cv.warpPerspective(img, M, (img.shape[1], img.shape[0]))

    return warped


dst_size = 5
bottom_offset = 6

# Definir le point source
source = np.float32([[14, 140], [301, 140], [200, 96], [118, 96]])
# Definir le point de destination (1m x 1m)
destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                          [image.shape[1]/2 + dst_size,
                              image.shape[0] - bottom_offset],
                          [image.shape[1]/2 + dst_size, image.shape[0] -
                              2*dst_size - bottom_offset],
                          [image.shape[1]/2 - dst_size, image.shape[0] -
                              2*dst_size - bottom_offset],
                          ])
warped = perspect_transform(grid_img, source, destination)

cv.imshow("TRANSFORMEE",warped)
cv.imwrite('../output/output_img.png',
            cv.cvtColor(warped, cv.COLOR_RGB2BGR))
cv.waitKey(0)

VALEUR_SEUIL = 160 
# Identifier les pixels au-dessus du seuil
# Un seuil de RGB > 160 permet d'identifier efficacement les pixels du sol uniquement
def seuil_couleur(img, rgb_seuil=(VALEUR_SEUIL, VALEUR_SEUIL, VALEUR_SEUIL)):
    # Créer un tableau de zéros de la même taille (xy) que l'image, mais avec un seul canal
    couleur_selection = np.zeros_like(img[:,:,0])
    # Exiger que chaque pixel soit au-dessus des trois valeurs de seuil en RGB
    # above_seuil contiendra maintenant un tableau booléen avec "True"
    # là où le seuil a été atteint
    above_seuil = (img[:,:,0] > rgb_seuil[0]) \
                & (img[:,:,1] > rgb_seuil[1]) \
                & (img[:,:,2] > rgb_seuil[2])
    # Indexer le tableau de zéros avec le tableau booléen et le mettre à 1
    couleur_selection[above_seuil] = 1
    # Retourner l'image binaire
    return couleur_selection

seuil = seuil_couleur(warped)
bw = seuil.copy()
bw[bw > 0] = 255
cv.imshow("Zone Navigable", bw)
cv.imwrite('../output/output_bw_img.png', bw)

i = 0
for img_path in img_list:
    img = cv.imread(img_path)
    w = perspect_transform(img,source,destination)
    s = seuil_couleur(w)
    bw = s.copy()
    bw[bw > 0] = 255
    cv.imshow(f"Zone {i}", bw)
    i = i+1

    if i == 10:
        break

cv.waitKey(0)
cv.destroyAllWindows()
