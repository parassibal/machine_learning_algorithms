
import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY CODES ABOVE 
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels)==len(predicted_labels)
    false_p=0
    false_n=0
    true_p=0
    for i in range(len(real_labels)):
        if(real_labels[i]==0 and predicted_labels[i]==1):
            false_p=false_p+1
        elif(real_labels[i]==1 and predicted_labels[i]==0):
            false_n=false_n+1
        elif(real_labels[i]==1 and predicted_labels[i]==1):
            true_p=true_p+1
    f1_s=float(true_p)/(true_p+0.5*(false_n+false_p))
    return(f1_s)
    raise NotImplementedError


class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        dist_point=0.0
        assert(len(point1)==len(point2))
        for i in range(len(point1)):
            dist_point+=pow(abs(point1[i]-point2[i]),3)
        dist_point=pow(dist_point,1.0/3)

        return(dist_point)
        raise NotImplementedError

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        dist_point=0.0
        assert(len(point1)==len(point2))
        for i in range(len(point1)):
            dist_point+=pow(abs(point1[i]-point2[i]),2)
        dist_point=pow(dist_point,1.0/2)

        return(dist_point)
        raise NotImplementedError

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        if(len(point1)!=len(point2)):
            raise NotImplementedError
        val1=val2=val3=0
        val4=val5=val6=0
        for i in range(len(point1)):
            val1=val1+(point1[i]*point2[i])
            val2=val2+pow(point1[i],2)
            val3=val3+pow(point2[i],2)
        val2=pow(val2,1.0/2)
        val3=pow(val3,1.0/2)
        dist=1-val1/(val2*val3)
        return(dist)

class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you need to try different distance functions you implemented in part 1.1 and different values of k (among 1, 3, 5, ... , 29), and find the best model with the highest f1-score on the given validation set.
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] training labels to train your KNN model
        :param x_val:  List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), and model (an instance of KNN) and assign them to self.best_k,
        self.best_distance_function, and self.best_model respectively.
        NOTE: self.best_scaler will be None.

        NOTE: When there is a tie, choose the model based on the following priorities:
        First check the distance function:  euclidean > Minkowski > cosine_dist 
		(this will also be the insertion order in "distance_funcs", to make things easier).
        For the same distance function, further break tie by prioritizing a smaller k.
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_model = None
        l=len(x_train)
        l1=len(y_train)
        f1_s=0
        for d in distance_funcs:
            for i in range(1,min(31,l+1),2):
                model=KNN(i,distance_funcs[d])
                model.train(x_train,y_train)
                val=model.predict(x_val)
                cal_f1=f1_score(y_val,val)
                if(cal_f1>f1_s):
                    self.best_k=i
                    self.best_distance_function=d
                    self.best_model=model
                    f1_s=cal_f1

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is the same as "tuning_without_scaling", except that you also need to try two different scalers implemented in Part 1.3. More specifically, before passing the training and validation data to KNN model, apply the scalers in scaling_classes to both of them. 
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param scaling_classes: dictionary of scalers (key is the scaler name, value is the scaler class) you need to try to normalize your data
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), scaler (its name), and model (an instance of KNN), and assign them to self.best_k, self.best_distance_function, best_scaler, and self.best_model respectively.
        
        NOTE: When there is a tie, choose the model based on the following priorities:
        First check scaler, prioritizing "min_max_scale" over "normalize" (which will also be the insertion order of scaling_classes). Then follow the same rule as in "tuning_without_scaling".
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None
        l=len(x_train)
        l1=len(y_train)
        f1_s=0
        for i in range(l1):
            continue
        for scaler_type,scaling_class in scaling_classes.items():
            scaler=scaling_class()
            scaler_x_train=scaler(features=x_train)
            scaler_x_val=scaler(features=x_val)
            for d in distance_funcs:
                for i in range(1,min(31,l+1),2):
                    model=KNN(i,distance_funcs[d])
                    model.train(scaler_x_train,y_train)
                    val=model.predict(scaler_x_val)
                    cal_f1=f1_score(y_val,val)
                    if(cal_f1>f1_s):
                        self.best_k=i
                        self.best_distance_function=d
                        self.best_model=model
                        self.best_scaler=scaler_type
                        f1_s=cal_f1


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        norm_features=[]
        for i in features:
            temp=pow(sum([x*x for x in i]),1/2)
            if(temp==0):
                val=i
            else:
                val=[x/temp for x in i]
            norm_features.append(val)
        return(norm_features)


class MinMaxScaler:
    def __init__(self):
        pass

    # TODO: min-max normalize data
    def __call__(self, features):
        """
		For each feature, normalize it linearly so that its value is between 0 and 1 across all samples.
        For example, if the input features are [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]].
		This is because: take the first feature for example, which has values 2, -1, and 0 across the three samples.
		The minimum value of this feature is thus min=-1, while the maximum value is max=2.
		So the new feature value for each sample can be computed by: new_value = (old_value - min)/(max-min),
		leading to 1, 0, and 0.333333.
		If max happens to be same as min, set all new values to be zero for this feature.
		(For further reference, see https://en.wikipedia.org/wiki/Feature_scaling.)

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        norm_feature=[]
        max_val=np.max(features,axis=0)
        min_val=np.min(features,axis=0)
        for i in features:
            norm=[]
            for j in range(len(i)):
                if(max_val[j]==min_val[j]):
                    norm.append(0.0)
                else:
                    f_norm=float(i[j]-min_val[j])/(max_val[j]-min_val[j])
                    norm.append(f_norm)
            norm_feature.append(norm)
        return(norm_feature)
