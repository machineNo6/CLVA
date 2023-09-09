import gc
from pickle import TRUE
from scipy.spatial import distance_matrix
from keras.models import Model
import keras.backend as K
import numpy as np
import keras

def get_unlabeled_idx(X_train, labeled_idx):
    """
    Given the training set and the indices of the labeled examples, return the indices of the unlabeled examples.
    """
    return np.arange(X_train.shape[0])[np.logical_not(np.in1d(np.arange(X_train.shape[0]), labeled_idx))]


class QueryMethod:
    """
    A general class for query strategies, with a general method for querying examples to be labeled.
    """

    def __init__(self, represent, input_shape=(28,28), num_labels=10, gpu=1):
        # self.model = keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(input_shape[0], input_shape[1], 3), pooling='avg')
        self.represent = represent
        self.input_shape = input_shape
        self.num_labels = num_labels
        self.gpu = gpu

    def query(self, X_train, Y_train, labeled_idx, amount):
        """
        get the indices of labeled examples after the given amount have been queried by the query strategy.
        :param X_train: the training set
        :param Y_train: the training labels
        :param labeled_idx: the indices of the labeled examples
        :param amount: the amount of examples to query
        :return: the new labeled indices (including the ones queried)
        """
        return NotImplemented

    def update_model(self, new_model):
        del self.model
        gc.collect()
        self.model = new_model

class CoreSetMIPSampling(QueryMethod):
    """
    An implementation of the core set query strategy with the MIP formulation using gurobi as our optimization solver.
    """

    def __init__(self, represent, input_shape, num_labels, gpu):
        super().__init__(represent, input_shape, num_labels, gpu)
        self.subsample = False

    def greedy_k_center(self, labeled, unlabeled, amount):

        greedy_indices = []
        # print("1111111111111111111111111111---------------------------------")
        # print(labeled.shape)
        # print("1111111111111111111111111111---------------------------------")
        # print( unlabeled.shape)
        # print("1111111111111111111111111111---------------------------------")
        # print(amount)
        # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
        min_dist = np.min(distance_matrix(labeled[0, :].reshape((1, labeled.shape[1])), unlabeled), axis=0)
        min_dist = min_dist.reshape((1, min_dist.shape[0]))
        # print(min_dist)
        # print(min_dist.shape)
        for j in range(1, labeled.shape[0], 100):
            if j + 100 < labeled.shape[0]:
                dist = distance_matrix(labeled[j:j+100, :], unlabeled)
            else:
                dist = distance_matrix(labeled[j:, :], unlabeled)
            min_dist = np.vstack((min_dist, np.min(dist, axis=0).reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
        # print(min_dist)
        # print(min_dist.shape)
        # iteratively insert the farthest index and recalculate the minimum distances:
        farthest = np.argmax(min_dist)
        greedy_indices.append(farthest)
        # print(greedy_indices)
        for i in range(amount-1):
            dist = distance_matrix(unlabeled[greedy_indices[-1], :].reshape((1,unlabeled.shape[1])), unlabeled)
            min_dist = np.vstack((min_dist, dist.reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            farthest = np.argmax(min_dist)
            greedy_indices.append(farthest)
            # print(greedy_indices)

        # print(np.array(greedy_indices, dtype=int), np.max(min_dist))
        return np.array(greedy_indices, dtype=int), np.max(min_dist)

    def get_distance_matrix(self, X, Y):

        x_input = K.placeholder((X.shape))
        y_input = K.placeholder(Y.shape)
        dot = K.dot(x_input, K.transpose(y_input))
        x_norm = K.reshape(K.sum(K.pow(x_input, 2), axis=1), (-1, 1))
        y_norm = K.reshape(K.sum(K.pow(y_input, 2), axis=1), (1, -1))
        dist_mat = x_norm + y_norm - 2.0*dot
        sqrt_dist_mat = K.sqrt(K.clip(dist_mat, min_value=0, max_value=10000))
        dist_func = K.function([x_input, y_input], [sqrt_dist_mat])

        return dist_func([X, Y])[0]

    def get_neighborhood_graph(self, representation, delta):

        graph = {}
        # print(representation.shape)
        for i in range(0, representation.shape[0], 1000):

            if i+1000 > representation.shape[0]:
                distances = self.get_distance_matrix(representation[i:], representation)
                amount = representation.shape[0] - i
            else:
                distances = self.get_distance_matrix(representation[i:i+1000], representation)
                amount = 1000

            distances = np.reshape(distances, (amount, -1))
            for j in range(i, i+amount):
                graph[j] = [(idx, distances[j-i, idx]) for idx in np.reshape(np.where(distances[j-i, :] <= delta),(-1))]

        # print("Finished Building Graph!")
        return graph

    def get_graph_max(self, representation, delta):

        # print("Getting Graph Maximum...")

        maximum = 0
        for i in range(0, representation.shape[0], 1000):
            # print("At Point " + str(i))

            if i+1000 > representation.shape[0]:
                distances = self.get_distance_matrix(representation[i:], representation)
            else:
                distances = self.get_distance_matrix(representation[i:i+1000], representation)

            distances = np.reshape(distances, (-1))
            distances[distances > delta] = 0
            maximum = max(maximum, np.max(distances))

        return maximum

    def get_graph_min(self, representation, delta):

        # print("Getting Graph Minimum...")

        minimum = 10000
        for i in range(0, representation.shape[0], 1000):
            # print("At Point " + str(i))

            if i+1000 > representation.shape[0]:
                distances = self.get_distance_matrix(representation[i:], representation)
            else:
                distances = self.get_distance_matrix(representation[i:i+1000], representation)

            distances = np.reshape(distances, (-1))
            distances[distances < delta] = 10000
            minimum = min(minimum, np.min(distances))

        return minimum

    def mip_model(self, representation, labeled_idx, budget, delta, outlier_count, greedy_indices=None):

        import gurobipy as gurobi

        model = gurobi.Model("Core Set Selection")
        model.setParam('OutputFlag', 0)
        # set up the variables:
        points = {}
        outliers = {}
        for i in range(representation.shape[0]):
            if i in labeled_idx:
                points[i] = model.addVar(ub=1.0, lb=1.0, vtype="B", name="points_{}".format(i))
            else:
                points[i] = model.addVar(vtype="B", name="points_{}".format(i))
        for i in range(representation.shape[0]):
            outliers[i] = model.addVar(vtype="B", name="outliers_{}".format(i))
            outliers[i].start = 0

        # initialize the solution to be the greedy solution:
        if greedy_indices is not None:
            for i in greedy_indices:
                points[i].start = 1.0

        # set the outlier budget:
        model.addConstr(sum(outliers[i] for i in outliers) <= outlier_count, "budget")

        # build the graph and set the constraints:
        model.addConstr(sum(points[i] for i in range(representation.shape[0])) == budget, "budget")
        neighbors = {}
        graph = {}
        #print("Updating Neighborhoods In MIP Model...")
        for i in range(0, representation.shape[0], 1000):
            # print("At Point " + str(i))

            if i+1000 > representation.shape[0]:
                distances = self.get_distance_matrix(representation[i:], representation)
                amount = representation.shape[0] - i
            else:
                distances = self.get_distance_matrix(representation[i:i+1000], representation)
                amount = 1000

            distances = np.reshape(distances, (amount, -1))
            for j in range(i, i+amount):
                graph[j] = [(idx, distances[j-i, idx]) for idx in np.reshape(np.where(distances[j-i, :] <= delta),(-1))]
                neighbors[j] = [points[idx] for idx in np.reshape(np.where(distances[j-i, :] <= delta),(-1))]
                neighbors[j].append(outliers[j])
                model.addConstr(sum(neighbors[j]) >= 1, "coverage+outliers")

        model.__data = points, outliers
        model.Params.MIPFocus = 1
        model.params.TIME_LIMIT = 180

        return model, graph

    def mip_model_subsample(self, data, subsample_num, budget, dist, delta, outlier_count, greedy_indices=None):

        import gurobipy as gurobi

        model = gurobi.Model("Core Set Selection")

        # calculate neighberhoods:
        data_1, data_2 = np.where(dist <= delta)

        # set up the variables:
        points = {}
        outliers = {}
        for i in range(data.shape[0]):
            if i >= subsample_num:
                points[i] = model.addVar(ub=1.0, lb=1.0, vtype="B", name="points_{}".format(i))
            else:
                points[i] = model.addVar(vtype="B", name="points_{}".format(i))
        for i in range(data.shape[0]):
            outliers[i] = model.addVar(vtype="B", name="outliers_{}".format(i))
            outliers[i].start = 0

        # initialize the solution to be the greedy solution:
        if greedy_indices is not None:
            for i in greedy_indices:
                points[i].start = 1.0

        # set up the constraints:
        model.addConstr(sum(points[i] for i in range(data.shape[0])) == budget, "budget")
        neighbors = {}
        for i in range(data.shape[0]):
            neighbors[i] = []
            neighbors[i].append(outliers[i])
        for i in range(len(data_1)):
            neighbors[data_1[i]].append(points[data_2[i]])
        for i in range(data.shape[0]):
            model.addConstr(sum(neighbors[i]) >= 1, "coverage+outliers")
        model.addConstr(sum(outliers[i] for i in outliers) <= outlier_count, "budget")
        model.setObjective(sum(outliers[i] for i in outliers), gurobi.GRB.MINIMIZE)

        model.__data = points, outliers
        model.Params.MIPFocus = 1

        return model

    def query_regular(self, X_train, Y_train, labeled_idx, amount):

        import gurobipy as gurobi

        unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)

        # use the learned representation for the k-greedy-center algorithm:
        # representation_model = Model(inputs=self.model.input, outputs=self.model.layers[-1].output)
        representation = self.represent
        # print(representation.shape)
        # print("Calculating Greedy K-Center Solution...")
        # print(amount)
        new_indices, max_delta = self.greedy_k_center(representation[labeled_idx], representation[unlabeled_idx], amount)
        # print(new_indices)
        new_indices = unlabeled_idx[new_indices]
        outlier_count = int(X_train.shape[0] / 10000)
        # outlier_count = 250
        submipnodes = 20000

        # iteratively solve the MIP optimization problem:
        eps = 0.00001
        upper_bound = max_delta
        lower_bound = max_delta / 2.0
        # if max_delta == 0:
        #     return False
        # else:
        #     return True
        # print("Building MIP Model...")
        model, graph = self.mip_model(representation, labeled_idx, len(labeled_idx) + amount, upper_bound, outlier_count, greedy_indices=new_indices)
        model.Params.SubMIPNodes = submipnodes
        points, outliers = model.__data
        model.optimize()
        indices = [i for i in graph if points[i].X == 1]
        current_delta = upper_bound
        while upper_bound - lower_bound > eps:

            # print("upper bound is {ub}, lower bound is {lb}".format(ub=upper_bound, lb=lower_bound))
            if model.getAttr(gurobi.GRB.Attr.Status) in [gurobi.GRB.INFEASIBLE, gurobi.GRB.TIME_LIMIT]:
                # print("Optimization Failed - Infeasible!")

                lower_bound = max(current_delta, self.get_graph_min(representation, current_delta))
                current_delta = (upper_bound + lower_bound) / 2.0

                del model
                gc.collect()
                model, graph = self.mip_model(representation, labeled_idx, len(labeled_idx) + amount, current_delta, outlier_count, greedy_indices=indices)
                points, outliers = model.__data
                model.Params.SubMIPNodes = submipnodes

            else:
                # print("Optimization Succeeded!")
                upper_bound = min(current_delta, self.get_graph_max(representation, current_delta))
                current_delta = (upper_bound + lower_bound) / 2.0
                indices = [i for i in graph if points[i].X == 1]

                del model
                gc.collect()
                model, graph = self.mip_model(representation, labeled_idx, len(labeled_idx) + amount, current_delta, outlier_count, greedy_indices=indices)
                points, outliers = model.__data
                model.Params.SubMIPNodes = submipnodes

            if upper_bound - lower_bound > eps:
                model.optimize()

        return np.array(indices)

    def query_subsample(self, X_train, Y_train, labeled_idx, amount):

        import gurobipy as gurobi

        unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)

        submipnodes = 20000
        subsample_num = 30000
        subsample_idx = np.random.choice(unlabeled_idx, subsample_num, replace=False)
        subsample = np.vstack((X_train[labeled_idx], X_train[subsample_idx]))
        new_labeled_idx = np.arange(len(labeled_idx))
        new_indices = self.query_regular(subsample, Y_train, new_labeled_idx, amount)
        return np.array(subsample_idx[new_indices - len(labeled_idx)])


    def query(self, X_train, Y_train, labeled_idx, amount):

        if self.subsample:
            return self.query_subsample(X_train, Y_train, labeled_idx, amount)
        else:
            return self.query_regular(X_train, Y_train, labeled_idx, amount)