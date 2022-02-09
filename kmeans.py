
import numpy as np
from numpy.core.fromnumeric import cumsum, reshape

#################################
# DO NOT IMPORT OHTER LIBRARIES
#################################

def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data - numpy array of points
    :param generator: random number generator. Use it in the same way as np.random.
            In grading, to obtain deterministic results, we will be using our own random number generator.


    :return: a list of length n_clusters with each entry being the *index* of a sample
             chosen as centroid.
    '''
    p=generator.randint(0, n) #this is the index of the first center
    #############################################################################
    # TODO: implement the rest of Kmeans++ initialization. To sample an example
	# according to some distribution, first generate a random number between 0 and
	# 1 using generator.rand(), then find the the smallest index n so that the 
	# cumulative probability from example 1 to example n is larger than r.
    #############################################################################
    centers=[]
    centers.append(p)
    for k in range(n_cluster-1):
        minimum_distance=[]
        for x_val in x:
            norm=np.linalg.norm(x[centers]-x_val,axis=1)**2
            dist=np.min(norm)
            minimum_distance.append(dist)
        minimum_distance=np.array(minimum_distance)
        prob_val=minimum_distance/np.sum(minimum_distance)
        new_p=[]
        for i in range(0,len(x)):
            new_p.append(i)
        new_p1=np.array(new_p)
        r=generator.rand()
        cumsum_val=np.cumsum(prob_val)
        new_center=np.argmax(cumsum_val>r)
        centers.append(new_center)

    # DO NOT CHANGE CODE BELOW THIS LINE
    return centers

 





# Vanilla initialization method for KMeans
def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)



class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple in the following order:
                  - final centroids, a n_cluster X D numpy array, 
                  - a length (N,) numpy array where cell i is the ith sample's assigned cluster's index (start from 0), 
                  - number of times you update the assignment, an Int (at most self.max_iter)
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        self.generator.seed(42)
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)
        ###################################################################
        # TODO: Update means and membership until convergence 
        #   (i.e., average K-mean objective changes less than self.e)
        #   or until you have made self.max_iter updates.
        ###################################################################
        k=self.n_cluster
        centroids=np.zeros([k,D])
        for i in range(self.n_cluster):
            centroids[i]=x[self.centers[i]]
        temp=np.zeros(N)
        #distortion=sum(X-centroid)^2
        val=[np.sum(np.power((x[temp==i]-centroids[i]),2)) for i in range(k)]
        distortion=np.sum(val)/N
        iter_val=0
        while iter_val<self.max_iter:
            val=np.power((x-np.expand_dims(centroids,axis=1)),2)
            temp=np.argmin(np.sum(val,axis=2),axis=0)
            dis_val=np.sum([np.sum((x[temp==i]-centroids[i])**2) for i in range(k)])/N
            ab_diff=np.absolute(distortion-dis_val) 
            if(ab_diff<self.e):
                break
            distortion=dis_val
            update_centroid=np.array([np.mean(x[temp==i],axis=0) for i in range(k)])
            nan_val=np.isnan(update_centroid)
            index=np.where(nan_val)
            update_centroid[index]=centroids[index]
            centroids=update_centroid
            iter_val=iter_val+1

        self.max_iter=iter_val

        return(centroids,temp,self.max_iter)


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Store following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (numpy array of length n_cluster)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        ################################################################
        # TODO:
        # - assign means to centroids (use KMeans class you implemented, 
        #      and "fit" with the given "centroid_func" function)
        # - assign labels to centroid_labels
        ################################################################
        k=self.n_cluster
        k_means=KMeans(k,self.max_iter,self.e,self.generator)
        centroids,membership,iter=k_means.fit(x,centroid_func)
        centroid_labels=np.zeros([k])
        val=[y[membership==i].tolist() for i in range(k)]
        labels=[[] for i in range(k)]
        for i in range(N):
            labels[membership[i]].append(y[i])
        centroid_labels=np.array([max(set(val[i]),key=val[i].count) for i in range(len(val))])

        
        # DO NOT CHANGE CODE BELOW THIS LINE
        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        ##########################################################################
        # TODO:
        # - for each example in x, predict its label using 1-NN on the stored 
        #    dataset (self.centroids, self.centroid_labels)
        ##########################################################################
        k=self.n_cluster
        norm=[np.power(np.linalg.norm(x-self.centroids[i],axis=1,ord=2),2) for i in range(k)]
        labels=np.argmin(np.array(norm),axis=0)
        return(np.array(self.centroid_labels[labels]))
        





def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors (aka centroids)

        Return a new image by replacing each RGB value in image with the nearest code vector
          (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'
    ##############################################################################
    # TODO
    # - replace each pixel (a 3-dimensional point) by its nearest code vector
    ##############################################################################
    
    a,b,c=image.shape
    img_shape=image.shape
    reshape_img=np.reshape(image,(a*b,c))
    norm=np.linalg.norm(np.expand_dims(code_vectors, axis=1)-reshape_img,axis=2)
    val=np.argmin(norm,axis=0)
    img_t=code_vectors[val].reshape(img_shape)
    return(img_t)
