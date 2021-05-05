import altsklearn.myutils as myutils
from statistics import mean, mode
from random import seed, random, shuffle
from copy import deepcopy
from os import popen
from pprint import pp

class MySimpleLinearRegressor:
    """Represents a simple linear regressor.

    Attributes:
        slope(float): m in the equation y = mx + b
        intercept(float): b in the equation y = mx + b

    Notes:
        Loosely based on sklearn's LinearRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    
    def __init__(self, slope=None, intercept=None):
        """Initializer for MySimpleLinearRegressor.

        Args:
            slope(float): m in the equation y = mx + b (None if to be computed with fit())
            intercept(float): b in the equation y = mx + b (None if to be computed with fit())
        """
        self.slope = slope 
        self.intercept = intercept

    def fit(self, x_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            x_train(list of list of numeric vals): The list of training samples
                The shape of X_train is (n_train_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]
            y_train(list of numeric vals): The target y values (parallel to X_train) 
                The shape of y_train is n_train_samples
        """
        xmean = mean(x_train)
        ymean = mean(y_train)

        slope_num = 0.0
        slope_den = 0.0
        corr_den_y = 0.0 # The third summation that shows up in calculating r. Saves an iteration
        for i in range(len(x_train)):
            slope_num += (x_train[i] - xmean) * (y_train[i] - ymean)
            slope_den += (x_train[i] - xmean) ** 2
            corr_den_y += (y_train[i] - ymean) ** 2
        
        self.slope = slope = slope_num / slope_den
        self.intercept = ymean - slope*xmean

    def predict(self, X_test):
        """Makes predictions for test samples in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]

        Returns:
            y_predicted(list of numeric vals): The predicted target y values (parallel to X_test)
        """
        preds = []

        for val in X_test:
            preds.append(self.slope * val + self.intercept)

        return preds

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        categorical_indices(list of int): list of categorical attribute indices 

    Notes:
        Loosely based on sklearn's KNeighborsClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3, categorical_indices=None):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        if(categorical_indices == None): categorical_indices = []

        self.n_neighbors = n_neighbors
        self.X_train = None 
        self.y_train = None
        self.categorical_indices=categorical_indices

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train 
        self.y_train = y_train 

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances 
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        neighbors = []
        for test in X_test:
            dis_index = []
            for i, val in enumerate(self.X_train):
                dis_index.append((myutils.distance(test, val, self.categorical_indices), i))
            
            dis_index.sort(key=lambda x: x[0])
            dis, indices = map(list, zip(*dis_index))

            distances.append(dis)
            neighbors.append(indices)

        distances = [ x[:self.n_neighbors] for x in distances ]
        neighbors = [ x[:self.n_neighbors] for x in neighbors ]
        return distances, neighbors

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """

        y_predicted = []

        _distances, indices = self.kneighbors(X_test)

        # seed rng with 0 for consistency
        seed(0)
        for i in range(len(X_test)):
            yneighbor_vals = [self.y_train[val] for val in indices[i]]
            
            # statistics.mode choses the earliest value in a tie, so we shuffle
            # to ensure a random outcome. In testing, this should be accounted for
            shuffle(yneighbor_vals)
            
            y_predicted.append(mode(yneighbor_vals))

        return y_predicted

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        priors(dict of obj): The prior probabilities computed for each
            label in the training set.
        posteriors(list of dict of dict of obj): The posterior probabilities computed for each
            attribute value/label pair in the training set. Accessed like posteriors[attrib_index][attrib_value][label]

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.priors = None 
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        self.X_train = X_train
        self.y_train = y_train

        freq = myutils.get_frequency(self.y_train)
        self.priors = deepcopy(freq)
        self.posteriors = []

        for key in self.priors:
            self.priors[key] /= len(self.y_train)

        # get the domain an range and generate the data structure to be used for counting
        x_domain = []
        y_range = list(freq.keys())
        x_columns = list(map(list, zip(*self.X_train)))
        for col in x_columns:
            x_domain.append(set(col))
        
        for col in x_columns:
            count_dict = {}
            for val in col:
                count_dict[val] = {}
                for label in y_range:
                    count_dict[val][label] = 0.0

            self.posteriors.append(count_dict)

        for row, label in zip(self.X_train, self.y_train):
            for i, cell in enumerate(row):
                self.posteriors[i][cell][label] += 1        

        for attribute in self.posteriors:
            for value in attribute:
                for label in attribute[value]:
                    attribute[value][label] /= freq[label]

        

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for test_row in X_test:
            results = {}
            for label in self.priors:
                prob = self.priors[label]
                for i, value in enumerate(test_row):
                    if(value not in self.posteriors[i]):
                        prob *= 0
                    else:
                        prob *= self.posteriors[i][value][label]
                results[label] = prob
            
            y_predicted.append(sorted(results.items(), reverse=True, key=lambda x: x[1])[0][0])
        
        return y_predicted

class MyZeroRClassifier:
    """Represents a zero rule classifier
    
    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        most_common(obj): The label that occurs most often. Equivalent to mode(y_train)
    """
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.most_common = None

    def fit(self, X_train, y_train):
        """Fits the Zero-Rule classifier to the dataset given

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        self.X_train = X_train
        self.y_train = y_train
        self.most_common = mode(y_train)

    def predict(self, X_test):
        """Predicts the labels of a given set of classes

        Args:
            X_test(list of list of obj): The list of test instances (samples).
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted class labels.
                The shape of y_test is n_test_samples
        """

        return [self.most_common] * len(X_test)

class MyRandomClassifier:
    """Represents a random classifier (returns a weighted random label)

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        priors(dict of obj): The prior probabilities stored so that it can be used
            alongside an rng easily. So instead of {'label1': 0.5, 'label2': 0.25, 'label3': 0.25}
            it is {'label1': 0.5, 'label2': 0.75, 'label3': 1.0}

    """

    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.priors = None

    
    def fit(self, X_train, y_train):
        """Stores the datasets and the frequencies of the labels.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        self.X_train = X_train
        self.y_train = y_train
        freq = myutils.get_frequency(y_train)

        prev = 0.0
        for label in freq:
            freq[label] /= len(y_train)
            freq[label] += prev
            prev = freq[label]

        self.priors = freq
    
    def predict(self, X_test, set_seed=None):
        """Randomly predicts the labels of a set of samples.
        The randomness is weighted by the prior probability of
        each label

        Args:
            X_test:(list of list of obj): The list of test instances (samples).
                The shape of X_test is (n_test_samples, n_features)
            set_seed(int or NoneType): if not none, seed the random number
                generator with the given value
        
        Returns:
            y_predicted(list of obj): The predicted class labels.
                The shape of y_test is n_test_samples
        """
        if(set_seed is not None):
            seed(set_seed)
        y_predicted = []

        for _test in X_test:
            val = random()

            # note: should always return since the last label should be 1.0 and
            # the range of random() is [0,1.0)
            for label in self.priors:
                if(val < self.priors[label]):
                    y_predicted.append(label)
                    break
        
        return y_predicted

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.tree = None

        self.X_domain = None
        self.y_range = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train

        cols = list(map(list, zip(*X_train)))
        instances = list(map(list, zip(*cols, y_train)))

        domain = []
        for i in range(len(instances[0])):
            domain.append([])
        
        for instance in instances:
            for i, value in enumerate(instance):
                if (value not in domain[i]):
                    domain[i].append(value)
        
        for attribute in domain:
            attribute.sort()

        
        # create tree
        self.tree = myutils.tdidt(instances, list(range(len(domain) - 1)), domain)
        
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []


        for test in X_test:
            curr_node = self.tree
            while(curr_node[0] != "Leaf"):
                att_index = int(curr_node[1][3:])
                for val in curr_node[2:]:
                    if(val[1] == test[att_index]):
                        curr_node = val[2]
                        break
            
            y_predicted.append(curr_node[1])

        return y_predicted

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """

        myutils.print_traversal(self.tree, [], attribute_names, class_name)

    # BONUS METHOD
    def visualize_tree(self, dot_fname, outfile_fname, attribute_names=None, fmt='pdf'):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).
            fmt(str): The format of the output file. Default: pdf. Other supported filetypes include but are not limitted to:
                - png
                - svg
                - jpg
                - gif


        Notes: 
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """

        lines = myutils.graphviz_traversal(self.tree, attribute_names)
        
        with open(dot_fname, 'w') as fp:
            fp.write('\n'.join(lines))
        
        popen("dot -T{fmt} -o {outfile} {infile}".format(fmt=fmt, outfile=outfile_fname, infile=dot_fname))