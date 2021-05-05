import math
from uuid import uuid1
from collections import defaultdict

def normalize(dataset, tests=None):
    """Normalizes a given dataset using min-max normalization
    
    Args:
        dataset(list of lists): List of attribute values for each yval
        tests(list of lists): (Optional) List of test values to be normalized against the dataset

    Returns:
        list of lists: Same attributes as before, but normalized
    """
    # ensure that all values are of the same length
    try:
        exp_length = len(dataset[0])
        for val in dataset:
            assert len(val) == exp_length
        if(tests):
            for val in tests:
                assert len(val) == exp_length
    except TypeError:
        print("Error: Value in dataset is not list")
    except AssertionError:
        print("Error: Values in dataset are not all of the same length")

    attribs = list(zip(*dataset))
    new_attribs = []

    for attr in attribs:
        attr_norm = []
        for val in attr:
            attr_norm.append((val - min(attr)) / ((max(attr) - min(attr)) * 1.0))
        new_attribs.append(attr_norm)

    dataset_norm = list(map(list, zip(*new_attribs)))

    if(tests):
        attribs_test = list(zip(*tests))
        new_attribs_test = []
        for i, attr in enumerate(attribs_test):
            attr_norm = []
            for val in attr:
                attr_norm.append((val - min(attribs[i])) / ((max(attribs[i]) - min(attribs[i])) * 1.0))
            new_attribs_test.append(attr_norm)

        tests_norm = list(map(list, zip(*new_attribs_test)))

    if(tests):
        return dataset_norm, tests_norm
    else:
        return dataset_norm

def distance(point1, point2, categorical_list):
    """Finds the euclidian distance between two points

    Args:
        point1(list of float): The first point being measured
        point2(list of float): The second point being measured
    
    Returns:
        float: The euclidian distance between the two points

    Notes: Raises an AssertionError if len(point1) != len(point2)
    """
    assert len(point1) == len(point2)
    indiv_dis = []

    for i in range(len(point1)):
        if(i in categorical_list):
            indiv_dis.append(point1[i] == point2[i])
        else:
            indiv_dis.append((point1[i] - point2[i])**2)

    return math.sqrt(sum(indiv_dis))

def separate_by_value(yvals):
    """Gets the indices of all occurances of each value on the list

    Args:
        yvals(list of obj): The values being counted

    Returns:
        freq(dict): The total times each value occurs in the list, 
            indexed by those values
    """
    bins = {}
    for i, val in enumerate(yvals):
        if(val not in bins):
            bins[val] = []
        bins[val].append(i)
    
    return bins

def get_mpg_class(mpg):
    """Gets the DoE rating of associated with a specific milage

    Args:
        mpg(float): the mpg being rated

    Returns:
        rating(int): a value from 1-10
    """

    if(mpg >= 45):
        return 10
    elif(mpg >= 37 and mpg < 45):
        return 9
    elif(mpg >= 31 and mpg < 37):
        return 8
    elif(mpg >= 27 and mpg < 31):
        return 7
    elif(mpg >= 24 and mpg < 27):
        return 6
    elif(mpg >= 20 and mpg < 24):
        return 5
    elif(mpg >= 17 and mpg < 20):
        return 4
    elif(mpg >= 15 and mpg < 17):
        return 3
    elif(mpg >= 14 and mpg < 15):
        return 2
    else:
        return 1

def get_weight_class(weight):
    """ Gets the NHTSA weight ranking for a car of a given weight

    Args:
        weight(float): the weight of the car

    Returns:
        ranking(int): the NHTSA ranking of the car
    """

    if(weight >= 3500):
        return 5
    elif(weight >= 3000 and weight < 3500):
        return 4
    elif(weight >= 2500 and weight < 3000):
        return 3
    elif(weight >= 2000 and weight < 2500):
        return 2
    else:
        return 1

def get_frequency(arr):
    """Gets the frequency of each value in a given array and returns as
    a dictionary indexed by each value

    Args:
        arr(list of obj): List of values to get the frequency of

    Returns:
        freq(dict): Dictionary containing the total occurances of each 
        value in the list
    """
    freq = {}
    for i in arr:
        if(i not in freq):
            freq[i] = 0
        freq[i] += 1
    
    return freq

def flatten_posteriors(posteriors):
    """Flattens a set of posteriors in the format list of dict of dict of obj

    Args:
        posteriors(list of dict of dict of obj): the posteriors to be flattened

    Returns:
        flat_posts(list of obj): a sorted list of values in the posteriors
    """

    flat_posts = []

    for attribute in posteriors:
        for value in attribute:
            for label in attribute[value]:
                flat_posts.append(attribute[value][label])
    
    flat_posts.sort()

    return flat_posts

## Decision Tree ##
def generate_subtable(X_table, y_values, rules):
    """ Generates a sub-table of the given table that has all samples
        meeting a given condition. 

    Args:
        X_table(list of list of obj): The list of values to check 
            The shape of X_table is (n_train_samples, n_features)
        y_values(list of obj): The target y values (parallel to X_train)
            The shape of y_values is n_train_samples
        rules(list of tuple): Rules in the form of (index, value)
    
    Returns:
        X_new(list of list of obj): The modified table
        y_new(list of obj): The corresponding labels for the table
    
    """
    X_new, y_new = [], []
    for row, label in zip(X_table, y_values):
        row_passed = True
        for index, value in rules:
            if(row[index] != value):
                row_passed = False
                break
        if(row_passed):
            X_new.append(row)
            y_new.append(label)

    return X_new, y_new

def partition_instances(curr_instances, attribute, attrib_domains):
    partitions = {}
    for value in attrib_domains[attribute]:
        partitions[value] = []
    
    for instance in curr_instances:
        partitions[instance[attribute]].append(instance)
    
    return partitions

def entropy(*args):
    """ Calculates the entropy for a given dataset or slice. Should work with
        datasets of any number of labels

    Args:
        *args(floats): should be a set of floats less than 1 that represent the 
            distribution of each label in the set/slice. Naturally, this
            means that the sum of args should be 1, but this is not validated.
    
    Returns:
        entropy(float): the calculated entropy
        leaf(int): if there is a value with an occurance rate of 1.0, returns the index
            used to find leaf nodes in tree generation. Otherwise, returns -1
    """


    values = []
    leaf = -1

    for i, val in enumerate(args):
        if(val != 0):
            values.append(val * math.log(val, len(args)))
            if(val == 1):
                leaf = i
    
    return -sum(values), leaf

def entropy_new(instances, att_index, domain):
    """ Computes E_new for a given attribute on a given dataset.

    Args:
        X_table(list of list of obj): List of currently available instances
        att_index(int): the index of the attribute being tested
        domain(list of obj): the domain of every instance in the tree

    Returns:
        entropy(float): the agregate entropy of the given attribute
        leaf_candidates(list of tuple(obj, obj)): values of attribute for which
            a label always occurs 
    """
    e_new = 0.0
    leaf_candidates = []

    # initializes a dictionary that will contain the counts of every values
    counts = {}
    for val in domain[att_index]:
        counts[val] = [0 for x in range(len(domain[-1]))]
    
    # counts all the instances
    for instance in instances:
        counts[instance[att_index]][domain[-1].index(instance[-1])] += 1
    
    for value in domain[att_index]:
        total_occs = sum(counts[value])
        if(total_occs == 0):
            continue
        occ_rate = total_occs / len(instances)
        rates = []
        for count in counts[value]:
            rates.append(count / total_occs)
        
        e_slice, leaf = entropy(*rates)
        e_new += occ_rate * e_slice

        if(leaf != -1):
            leaf_candidates.append((value, domain[-1][leaf], total_occs, len(instances)))
    
    return e_new, leaf_candidates

def select_attribute(instances, available_attributes, domain):
    """ Computes the entropy of all available attributes and
        returns the attribute with the minimum entropy and any
        case 1 leaf nodes

        Args:
            instances(list of list): the remaining instances
            available_attributes(list of inst): indices of the remaining
                attributes
            domain(list of list): List of all values for each attribute

        Returns:
            next_attrib(int): index of the next attribute to partition on
            leaves(list of tuple): list of case 1 leaf nodes found in the
                returned attribute
    """


    entropies = {}
    for att in available_attributes:
        entropies[att] = entropy_new(instances, att, domain)
    
    next_attrib, (_ent, leaves) = min(list(entropies.items()), key=lambda x: x[1][0])
    
    return next_attrib, leaves

def all_same(instances, index):
    """ Simple boolean function that returns true if
        all instances are have the same value at the given instance

        Args:
            instances(list of obj): list of remaining labels
            index: the index of the attribute

        Returns:
            all_same(bool): whether or not there is only 1 unique 
                value in the given partiiton
    """
    if(len(instances) == 0):
        return True
    first = instances[0][index]
    for i in range(1, len(instances)):
        if(instances[i][index != first]):
            return False
    
    return True

def partition_stats(instances, index):
    """ breaks up instances into partitions based on the value of the given
        attribute

        Args:
            instances(list of list): The instances to partition
            index(int): the index of the attribute to partition on

        Returns:
            partitions(dict of list of list): Partitions indexed by attribute value.
    """

    counts = defaultdict(int)
    total = len(instances)

    for inst in instances:
        counts[inst[index]] += 1
    
    stats = []
    for val in counts:
        stats.append([val, counts[val], total])
    
    return stats

def tdidt(instances, available_attributes, domain):
    """ Generates a decision tree using TDIDT

        Args:
            instances(list of list of obj): List of all remaining instances with the label
                at the last index
            available_attributes(list of int): the currently available attributes to split on
            domain(list of obj): List of domains for each attribute including the class label 
        
        Returns:
            tree(list of [str, str, tree] or [str, str, int, int]): The node generated from that
                level of tdidt. This can be either an attribute node or a leaf node.
    """

    # Select attribute, initialize tree
    split_attrib, leaves = select_attribute(instances, available_attributes, domain)
    tree = ["Attribute", "att" + str(split_attrib)]
    rem_attribs = [ x for x in available_attributes if x != split_attrib]

    # partition on attribute
    for value, partition in partition_instances(instances, split_attrib, domain).items():
        val_segment = ["Value", value]

        
        # CASE 1: all values are the same
        if(len(leaves) > 0 and value == leaves[0][0]):
            val_segment.append(["Leaf", *leaves[0][1:]])
            leaves.pop(0)

        # CASE 2: there are instances, but no more attributes
        elif(len(partition) > 0 and len(rem_attribs) == 0):
            most_common = max(partition_stats(partition, -1), key=lambda x: x[1])
            val_segment.append(["Leaf", *most_common])
        
        # CASE 3: no instances, replace attribute with leaf node
        elif(len(partition) == 0):
            most_common = max(partition_stats(instances, -1), key=lambda x: x[1])
            return ["Leaf", *most_common]
        
        # Default case. Recurse on partition
        else:
            val_segment.append(tdidt(partition, rem_attribs, domain))

        tree.append(val_segment)
    return tree    

def get_rule_string(stack, value, attribute_names, class_name):
    """ Converts a tree stack into a rule string

    Args:
        stack(list of tuple<str, str>): list of tuple in the format 
            (attribute, value) that correspond to the rule conditions
        value(str): the value of the leaf node at the end
        attribute_names(list of str or NoneType): attribute names passed
            from MyDecisionTreeClassifier.print_decision_rules(), if None,
            just uses default values
        class_name(str): the name of the class label
    
    Returns:
        rule_string(str): string describing a single decision rule

    """

    # lambda so I don't have to clutter my code with boring condition
    # checks or make a new list
    if(attribute_names == None):
        get_name = lambda att: att
    else:
        get_name = lambda att: attribute_names[int(att[3:])]

    rule = "IF "
    for node in stack:
        # print(node, flush=True)
        rule += get_name(node[0]) + " == " + str(node[1]) + " "
        if(node == stack[-1]):
            rule += "THEN "
        else:
            rule += "AND "
    
    rule += class_name + " = " + str(value)

    return rule

def print_traversal(curr_item, stack, attribute_names, class_name):
    """ Method for traversing a tree for the purpose of printing it

        Args:
            curr_item(obj): the current node of the tree
            stack(list of tuple): The current stack of attribute rules
            attribute_names(list of str or None): see print_decision_rules
            class_name(str): see print_decision_rules
        
        Returns: None
    """
    if(curr_item[0] == "Attribute"):
        for value in curr_item[2:]:
            stack_add = [(curr_item[1], value[1])]
            print_traversal(value[2], stack + stack_add, attribute_names, class_name)
    elif(curr_item[0] == "Leaf"):
        print(get_rule_string(stack, curr_item[1], attribute_names, class_name))

def graphviz_traversal(curr_item, attribute_names, lines=None):
    """ Traverses a tree while writing graph information to a given function
        pointer
        
        Args:
            curr_item: the current node of the tree being viewed
            attribute_names: if provided, names for each attribute
            lines(list of str or None):
                If given, the set of lines to write to. If it is none, initializes
                an empty array that will be passed to each recursion and returns
                it at the end.

        Returns:
            lines(list of str): Only for top level call
            -- OR --
            prev_id: the previous id added 
    """

    if(lines is None):
        is_top = True
        lines = [ 
            "graph g {",
            "    rankdir=TB;"
        ]
    else:
        is_top = False

    prev_id = ""

    if(attribute_names == None):
        get_name = lambda att: att
    else:
        get_name = lambda att: attribute_names[int(att[3:])]

    if(curr_item[0] == "Attribute"):
        prev_id = curr_item[1] + "_" + str(uuid1()).replace('-','')
        lines.append("    {id} [style=filled fillcolor=cornsilk shape=box label={name}];".format(
            id=prev_id, name=get_name(curr_item[1]).replace(" ", "_")))
        for val in curr_item[2:]:
            child_id = graphviz_traversal(val[2], attribute_names, lines=lines)
            lines.append("    {n1} -- {n2} [label={value}];".format(n1=prev_id, n2=child_id, value=val[1]))
        
    elif(curr_item[0] == "Leaf"):
        prev_id = "leaf_" + str(uuid1()).replace('-','')
        lines.append("    {id} [style=filled fillcolor=lightskyblue1 label={value}];".format(
            id=prev_id, value=curr_item[1]))
        

    if(is_top):
        lines.append("}")
        return lines
    else:
        return prev_id
    


