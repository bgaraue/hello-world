import numpy as np

def is1Darray(point):
    if not isinstance(point, np.ndarray) or point.ndim != 1:
        raise TypeError("Point to be compared is not a 1D array")
        
def haveSameLength(first_point, second_point):
    if len(first_point) != len(second_point):
        raise ValueError("The two points to be compared\
                         have different lengths") 
                         
def wrapperDistanceFunction(distanceFunction):
    def distanceFunctionWithChecks(*args):
        Knn_object = args[0]
        point = args[1]
        
        is1Darray(point)
            
        for x_point in Knn_object.X:
            is1Darray(x_point)
            haveSameLength(x_point, point)

        return distanceFunction(*args)
    
    return distanceFunctionWithChecks
    
class Knn: 
    def __init__(self,
                 n_neighbors,
                 metric='euclidean', 
                 p=2, 
                 weights='uniform'):
        
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        try:
            self.distanceFunction = getattr(self, "_{}_distance".format(metric))
        except AttributeError:
            raise ValueError("Non valid metric")
            
    def fit(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y dimensions are not consistent")
        self.X = X
        self.y = y

    @wrapperDistanceFunction
    def _manhattan_distance(self, point):
        return np.sum(abs(self.X - point), axis=1)
    
    @wrapperDistanceFunction
    def _euclidean_distance(self, point):
        return np.sqrt(np.sum((self.X - point)**2, axis=1))

    @wrapperDistanceFunction
    def _minkowski_distance(self, point):
        return np.power(np.sum(np.power(abs(self.X - point), self.p), axis=1), (1./self.p))
    
    def _uniform_weights(self, distances):
        return np.array([(1., distance) for distance in distances])

    def _distance_weights(self, distances):
        return np.array([(1., distance)
                         if distance == 0 
                         else (1./distance, distance) 
                         for distance in distances])

    
    def findNeighborClasses(self, point):
        distances = list(self.distanceFunction(point))
        classes = list(self.y)
        neighbor_classes = []
        neighbor_distances = []

        for k in range(self.n_neighbors): 
            position = np.argmin(distances)
            neighbor_classes.append(classes.pop(position)) 
            neighbor_distances.append(distances.pop(position)) 

        return neighbor_classes, neighbor_distances 
    
    def _predict_point(self, point):
        '''First we find the classes of the k-nn.
        Next we use the distance weights to choose our point's class'''
        
        neighbor_classes, neighbor_distances = self.findNeighborClasses(point)

        weights_function = getattr(self, "_{}_weights".format(self.weights))     
        weighted_distances = [elem[0] for elem in weights_function(neighbor_distances)]
        
        weights_by_class = zip(neighbor_classes, weighted_distances)
         
        dict_weights_by_class = dict.fromkeys(set(neighbor_classes), 0)
        
        for pair in weights_by_class:
            this_class = pair[0]
            weight = pair[1]
            dict_weights_by_class[this_class] += weight

        predicted_class = max(dict_weights_by_class,
                              key = dict_weights_by_class.get)

        return predicted_class
    
    def predict(self, x): 
        new_classes = []
        for new_point in x:
            predicted_class = self._predict_point(new_point)
            new_classes.append(predicted_class)
        return np.array(new_classes)


    