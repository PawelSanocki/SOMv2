import numpy as np
import tensorflow as tf
import os
from os.path import join
import cv2
import spectral.io.envi as envi
from spectral import open_image
from scipy.io import loadmat
import input_vec_rotational


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Som:
    def __init__(self, dim_x = 2, dim_y = 3, dim_z = 4, input_dim = 3, learn_iter = 10, learning_rate = 0.01, size = 1, sigma = None, time_const = None, quality = 1):
        '''
        Constructor of the SOM, initialising the tensorflow graph
        dim_x, dim_y, dim_z - size of the map in given dimension
        input_dim - number of spectral bands of the image
        learning_iter - number of learning iterations
        learning_rate - how big influence has the iteration, decays over iterations
        size - size of the ?cluster? as an input vector
        sigma - initial size of neighbourhood
        time_const - time constant
        quality - quality of the 
        '''
        self.quality = quality
        self.d_x = dim_x
        self.d_y = dim_y
        self.d_z = dim_z
        self._size = size
        if sigma == None:
            self.sigma = max(dim_x, dim_y, dim_z) / 2.0
        else:
            self.sigma = float(sigma)
        if time_const == None:
            self.time_constant = learn_iter / np.log(self.sigma)
        else:
            self.time_constant = time_constant
        self.input_dim = input_dim * (size**2)
        self._learn_iterations = learn_iter
        self.lr = learning_rate
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._weights = tf.Variable(tf.random.normal(shape = [dim_x, dim_y, dim_z, self.input_dim], mean = 400, stddev = 300,  name = "weights"))
            self._input_vec = tf.compat.v1.placeholder("float", [self.input_dim], name = "input_vec")
            self._learning_iteration = tf.compat.v1.placeholder("float", name = "learning_iteration")
            self.input_tensor = tf.stack([tf.stack([tf.stack([self._input_vec for i in range(dim_z)]) for j in range(dim_y)]) for k in range(dim_x)])

            distances_weights = tf.math.sqrt(tf.math.reduce_sum(((self._weights - self.input_tensor)**2.0), axis = 3))
            self.bmu_distance = tf.math.reduce_min(distances_weights)

            linear_location = tf.math.argmin(
                tf.reshape(
                    tf.reshape(distances_weights, (self.d_x, self.d_y * self.d_z)), (self.d_x * self.d_y * self.d_z, 1)))
            self.bmu_location = [(linear_location // self.d_z // self.d_y) % self.d_x, (linear_location // self.d_z) % self.d_y, linear_location % self.d_z]

            sigma_t = (self.sigma * tf.math.exp(-((self._learning_iteration / self.time_constant))))
            learning_rate_t = (learning_rate * tf.math.exp(-((self._learning_iteration / self.time_constant))))
            self.bmu_mask = tf.compat.v1.placeholder("float", shape = [dim_x, dim_y, dim_z])
            self.bmu_gaussian = tf.compat.v1.placeholder("float", shape = [dim_x, dim_y, dim_z])

            #bmu_influence = tf.stack([self.bmu_mask * tf.math.exp(-((self.bmu_gaussian ** 2) / ((sigma_t ** 2) * 2.0))) for i in range(self.input_dim)], axis = 3)
            bmu_influence = tf.stack([self.bmu_mask * self.bmu_gaussian for i in range(self.input_dim)], axis = 3)
            new_weights = self._weights + ((bmu_influence * learning_rate_t) * (self.input_tensor - self._weights))
            self._training_op = tf.compat.v1.assign(self._weights, new_weights)
            
            self._sess = tf.compat.v1.Session()
            init_op = tf.compat.v1.global_variables_initializer()

            self._sess.run(init_op)
            self.saver = tf.compat.v1.train.Saver([self._weights])
            print("som created")

    def train(self, input_vecs):
        '''
        Method for training SOm with set of input vectors
        input_vecs - set with input vectors
        '''
        for iter in range(self._learn_iterations):
            for input_vec in input_vecs:
                bmu_loc = self._sess.run(self.bmu_location, feed_dict={self._input_vec: input_vec})
                bmu_mask, bmu_distance = self.create_mask_for_updating_weights(bmu_loc, iter)
                self._sess.run(self._training_op, feed_dict={self._input_vec: np.reshape(input_vec, (self.input_dim)), self._learning_iteration: float(iter), \
                    self.bmu_mask: bmu_mask, self.bmu_gaussian: bmu_distance})
    def convert(self, input_vec):
        '''
        Method converting single input vector
        input_vec - numpy vector to be converted
        Returns coordinates of BMU
        '''
        return self._sess.run(self.bmu_location, feed_dict={self._input_vec: input_vec})
    def create_mask_for_updating_weights(self, bmu_location, iter):
        '''
        Method creating the mask defining the size of neighbourhhood and defining the influence of the BMU placement on neighbour nodes
        bmu_lacation - location of BMU
        iter - number of current iteration
        Returns mask marking the nodes taking part in updating process and the influence of BMU on them
        '''
        sigma_t = int(self.sigma * np.exp(-((iter / self.time_constant))))
        mask = np.zeros((self.d_x, self.d_y, self.d_z), dtype=np.float32)
        bmu_gaussian = np.zeros((self.d_x, self.d_y, self.d_z), dtype=np.float32)
        for i in range(int(bmu_location[0].item()) - sigma_t, int(bmu_location[0].item()) + sigma_t):
            for j in range(bmu_location[1].item() - sigma_t, bmu_location[1].item()//1 + sigma_t):
                for k in range(bmu_location[2].item() - sigma_t, bmu_location[2].item()//1 + sigma_t):
                    if i >= 0 and i < mask.shape[0] and j >= 0 and j < mask.shape[1] and k >= 0 and k < mask.shape[2]:
                        mask[i, j, k] = 1
                        bmu_gaussian[i, j, k] = np.exp(-( ((i-bmu_location[0])**2) + (j-bmu_location[1])**2 + (k-bmu_location[2])**2 ) / (sigma_t ** 2 * 2.0))
        return mask, bmu_gaussian
    def convert_image(self, img):
        '''
        Method that converts the image and returns the result
        img - image to be converted
        Returns numpy array with converted image
        '''
        converted_img = np.zeros((img.shape[0]//self.quality, img.shape[1]//self.quality, 3))
        for i in self.generator_image(img.shape[0], img.shape[1]):
            if (i[0]%100 == 0 and i[1] == self.quality):
                print("Converting: " + str(np.floor(100.0 * i[0] / img.shape[0])) + "%")
            input_vec = self.create_input_vec(img, i)
            a = self._sess.run(self.bmu_location, feed_dict={self._input_vec: input_vec}) 
            converted_img[i[0]//self.quality, i[1]//self.quality] = a
        print("converted") 
        return converted_img * (256 // max(self.d_x, self.d_y, self.d_z))
    def generator_image(self, cols, rows, quality = None):
        '''
        Method used to iterate over pixels of an image
        cols - number of columns
        rows - number of rows
        quality - size of the step
        '''
        if quality == None:
            quality = self.quality
        margin = self._size//2
        if quality > margin:
            margin = quality - 1
        for i in range(margin, cols-margin, quality):
            for j in range(margin, rows-margin, quality):
                yield i, j
    def finish(self):
        '''
        Method closing the tensorflow session
        '''
        tf.keras.backend.clear_session()
        self._sess.close()
        tf.compat.v1.reset_default_graph()        
    def train(self, threshold, path):
        '''
        Training SOM with all images from folder specified as parameter, which number of spatial bands fit the SOM
        threshold - threshold used in building training set
        path - path of folder containing images for training SOM
        '''
        def passThreshold(threshold, input_vec, centroids):
            if centroids.shape[0] == 0:
                return True
            for i in range(centroids.shape[0]):
                if np.linalg.norm(input_vec - centroids[i,:]) < threshold:
                    return False
            # if (np.min(np.linalg.norm(input_vec - centroids[range(centroids.shape[0])])) > threshold):
            #     return True
            return True

        filenames = [join(path, name) for name in os.listdir(path)]
        centroids = np.empty((0,self.input_dim))
        while centroids.shape[0] < 30:
            for filename in filenames:
                if filename.find(".lan") > 0:
                    img = open_image(filename)
                elif filename.find(".hdr") > 0:
                    img = envi.open(filename)
                elif filename.find(".mat") > 0:
                    img_mat = loadmat(filename)
                    for i in img_mat:
                        img = img_mat[i]
                elif filename.find(".png") > 0 or filename.find(".jpg") > 0:
                    img = cv2.imread(filename)
                else:
                    continue
                if img.shape[2] != self.input_dim / (self._size**2):
                    continue
                for i in self.generator_image(img.shape[0], img.shape[1], quality = 1):
                    input_vec = np.resize(self.create_input_vec(img, i), (1, self.input_dim))
                    if (passThreshold(threshold=threshold, input_vec=input_vec, centroids=centroids)):
                        centroids = np.append(centroids, input_vec, axis=0)
                    if i[1] == img.shape[1]-self._size and i[0]%100==0:
                        print("Creating training set: " + str(i[0] * 100 // img.shape[0] ) + "% Number of vectors: " + str(centroids.shape[0]))
            print("Threshold: " + str(threshold//self._size**2))
            print("Training vectors: " + str(centroids.shape[0]))
            threshold *= 0.95
        for iter in range(1, self._learn_iterations+1):
            if (iter%(self._learn_iterations//10) == 0):
                print("Training: " + str(100.0 * iter//self._learn_iterations) + "%")
            for input_vec in centroids:
                bmu_loc = self._sess.run(self.bmu_location, feed_dict={self._input_vec: input_vec})
                bmu_mask, bmu_distance = self.create_mask_for_updating_weights(bmu_loc, iter)
                self._sess.run(self._training_op, feed_dict={self._input_vec: np.reshape(input_vec, (self.input_dim)), self._learning_iteration: float(iter), \
                    self.bmu_mask: bmu_mask, self.bmu_gaussian: bmu_distance})
            np.random.shuffle(centroids)
        print("training done")

    def create_input_vec(self, img, i):
        '''
        Method for creating input vector of the SOM
        img - numpy array with hyperspectral image
        i - coordinates of current pixel to be converted
        Returns input vector
        '''
        # #simple method
        input_vec = np.empty(0)
        for it_r in range(i[0] - self._size//2, i[0] + self._size//2 + self._size%2):
            for it_c in range(i[1] - self._size//2, i[1] + self._size//2 + self._size%2):
                input_vec = np.append(input_vec, img[it_r, it_c])
        return (input_vec)
        
        # checking corners, rotational
        #return input_vec_rotational.towards_biggest_difference(img, i, self._size)

        

