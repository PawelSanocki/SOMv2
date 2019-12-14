import numpy as np
from itertools import cycle
import cv2
def generator(dir, size, i, rotation_dir):
    if rotation_dir:
        list = [(0,1), (1,0), (0,-1), (-1,0)]
    else:
        list = [(-1,0), (0,-1), (1,0), (0,1)]
    pool = cycle(list)
    direction = next(pool)
    while direction != dir:
        direction = next(pool)
    direction = next(pool)
    yield i
    #print(i)
    for edge_length in range(size):
        i = (i[0] + direction[0], i[1] + direction[1])
        yield i
        #print(i)
    for edges in range(3):
        direction = next(pool)
        for edge_length in range(size * 2):
            i = (i[0] + direction[0], i[1] + direction[1])
            yield i
            #print(i)
    direction = next(pool)
    for edge_length in range(size-1):
        i = (i[0] + direction[0], i[1] + direction[1])
        yield i
        #print(i)
    

def towards_biggest_difference(img, i, _size):
    #i = (2,2)
    #_size = 5
    size = _size//2
    #img = cv2.imread("test.png")

    biggest_difference = np.linalg.norm(img[(size+i[0], size+i[1])] - img[(size+i[0], -size+i[1])])
    direction = (1,0)
    cornerA = np.linalg.norm(img[(size+i[0], size+i[1])])
    cornerB = np.linalg.norm(img[(size+i[0], -size+i[1])])
    if (biggest_difference < np.linalg.norm(img[(size+i[0], -size+i[1])] - img[(-size+i[0], -size+i[1])])):
        direction = (0,-1)
        biggest_difference = np.linalg.norm(img[(size+i[0], -size+i[1])] - img[(-size+i[0], -size+i[1])])
        cornerA = np.linalg.norm(img[(size+i[0], -size+i[1])])
        cornerB = np.linalg.norm(img[(-size+i[0], -size+i[1])])
    if (biggest_difference < np.linalg.norm(img[(-size+i[0], -size+i[1])] - img[(-size+i[0], size+i[1])])):
        direction = (-1,0)
        biggest_difference = np.linalg.norm(img[(-size+i[0], -size+i[1])] - img[(-size+i[0], size+i[1])])
        cornerA = np.linalg.norm(img[(size+i[0], -size+i[1])])
        cornerB = np.linalg.norm(img[(-size+i[0], -size+i[1])])
    if (biggest_difference < np.linalg.norm(img[(-size+i[0], size+i[1])] - img[(size+i[0], size+i[1])])):
        direction = (0,1)
        cornerA = np.linalg.norm(img[(size+i[0], -size+i[1])])
        cornerB = np.linalg.norm(img[(-size+i[0], -size+i[1])])
    rotation_dir = (cornerA < cornerB)
    input_vec = np.empty(0)
    input_vec = np.append(input_vec, img[i])
    for ring in range(size):
        i = (i[0] + direction[0], i[1] + direction[1])
        for pixel in generator(direction, ring+1, i, rotation_dir):
            input_vec = np.append(input_vec, img[pixel])
    return(input_vec)


#print(towards_biggest_difference(cv2.imread("test.png"), (2,2), 5))
