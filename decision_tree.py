import csv
import random
import math
import operator
import numpy
from collections import Counter

from numpy.lib.function_base import iterable


def read_data(csv_path):
    """Read in the training data from a csv file.

    The examples are returned as a list of Python dictionaries, with column names as keys.
    """
    examples = []
    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for example in csv_reader:
            for k, v in example.items():
                if v == '':
                    example[k] = None
                else:
                    try:
                        example[k] = float(v)
                    except ValueError:
                        example[k] = v
            examples.append(example)
    return examples


def train_test_split(examples, test_perc):
    """Randomly data set (a list of examples) into a training and test set."""
    test_size = round(test_perc*len(examples))
    shuffled = random.sample(examples, len(examples))
    return shuffled[test_size:], shuffled[:test_size]


class TreeNodeInterface():
    """Simple "interface" to ensure both types of tree nodes must have a classify() method."""

    def classify(self, example):
        pass


class DecisionNode(TreeNodeInterface):
    """Class representing an internal node of a decision tree."""

    def __init__(self, test_attr_name, test_attr_threshold, child_lt, child_ge, child_miss):
        """Constructor for the decision node.  Assumes attribute values are continuous.

        Args:
            test_attr_name: column name of the attribute being used to split data
            test_attr_threshold: value used for splitting
            child_lt: DecisionNode or LeafNode representing examples with test_attr_name
                values that are less than test_attr_threshold
            child_ge: DecisionNode or LeafNode representing examples with test_attr_name
                values that are greater than or equal to test_attr_threshold
            child_miss: DecisionNode or LeafNode representing examples that are missing a
                value for test_attr_name                 
        """
        self.test_attr_name = test_attr_name
        self.test_attr_threshold = test_attr_threshold
        self.child_ge = child_ge
        self.child_lt = child_lt
        self.child_miss = child_miss

    def classify(self, example):
        """Classify an example based on its test attribute value.

        Args:
            example: a dictionary { attr name -> value } representing a data instance

        Returns: a class label and probability as tuple
        """
        test_val = example[self.test_attr_name]
        if test_val is None:
            return self.child_miss.classify(example)
        elif test_val < self.test_attr_threshold:
            return self.child_lt.classify(example)
        else:
            return self.child_ge.classify(example)

    def __str__(self):
        return "test: {} < {:.4f}".format(self.test_attr_name, self.test_attr_threshold)


class LeafNode(TreeNodeInterface):
    """Class representing a leaf node of a decision tree.  Holds the predicted class."""

    def __init__(self, pred_class, pred_class_count, total_count):
        """Constructor for the leaf node.

        Args:
            pred_class: class label for the majority class that this leaf represents
            pred_class_count: number of training instances represented by this leaf node
            total_count: the total number of training instances used to build the whole tree
        """
        self.pred_class = pred_class
        self.pred_class_count = pred_class_count
        self.total_count = total_count
        # probability of having the class label
        self.prob = pred_class_count / total_count

    def classify(self, example):
        """Classify an example.

        Args:
            example: a dictionary { attr name -> value } representing a data instance

        Returns: a class label and probability as tuple as stored in this leaf node.  This will be
            the same for all examples!
        """
        return self.pred_class, self.prob

    def __str__(self):
        return "leaf {} {}/{}={:.2f}".format(self.pred_class, self.pred_class_count,
                                             self.total_count, self.prob)


class DecisionTree:
    """Class representing a decision tree model."""

    def __init__(self, examples, id_name, class_name, min_leaf_count=1):
        """Constructor for the decision tree model.  Calls learn_tree().

        Args:
            examples: training data to use for tree learning, as a list of dictionaries
            id_name: the name of an identifier attribute (ignored by learn_tree() function)
            class_name: the name of the class label attribute (assumed categorical)
            min_leaf_count: the minimum number of training examples represented at a leaf node
        """
        self.id_name = id_name
        self.class_name = class_name
        self.min_leaf_count = min_leaf_count

        # build the tree!
        self.root = self.learn_tree(examples)

    def entropy(self, examples):
        """Calculate importance of attribute based on entropy and information gain

        Args:
            attribute: Single attribute of a given list of examples
            examples: training data to use for tree learning, as a list of dictionaries.  The
                attribute stored in self.id_name is ignored, and self.class_name is consided
                the class label.

        Returns: information gain value for a single attribute
        """

        # Initializing variables and probabilities to be used
        p_target_vals = list()

        #total_index = numpy.concatenate([(example.items()) for example in examples])
        total_index = 0
        for example in examples:
            for attribute in examples:
                total_index +=1

        #Find unique target and frequency 
        unique_target_vals, target_val_freq = numpy.unique([x for x in example for example in examples], return_counts=True)
        for n in range(0, len(unique_target_vals)):
            p_target_vals.append(target_val_freq[n] / sum(target_val_freq))

        # Ges entropy of data set
        entropy = 0
        for p in p_target_vals:
            entropy += -(p)*math.log(p, 2)
        return entropy

    def plurality_value(self, examples):
        """Selects the most common output value among a set of examples

        Args:
            examples: training data to use for tree learning, as a list of dictionaries.  The
                attribute stored in self.id_name is ignored, and self.class_name is consided
                the class label.

        Returns: A leaf node
        """
        # Dictionary counting class label in data set
        output_counter = Counter([example[self.class_name] for example in examples])
        return LeafNode(max(output_counter, key=output_counter.get), output_counter[max(output_counter, key=output_counter.get)], len(examples))
    
    def minmax_threshold(self, examples, attribute):
        #get minimum and maximum values as threshold
        max_thresh, min_thresh = -math.inf, math.inf
        for example in range(len(examples)):
            temp = examples[example][attribute]
            if temp == None:
                continue
            elif temp > max_thresh:
                max_thresh = temp
            elif temp < min_thresh:
                min_thresh = temp
        return (max_thresh, min_thresh)
            
    def decision_tree_learning(self, examples, parent_examples):
        """Build the decision tree based on entropy and information gain.

        Args:
            examples: training data to use for tree learning, as a list of dictionaries. The
                attribute stored in self.id_name is ignored, and self.class_name is consided
                the class label.

        Returns: a DecisionNode or LeafNode representing the tree
        """
        if len(examples) <= self.min_leaf_count*2:
            return self.plurality_value(examples)
        # Base case: All examples have the same label
        elif (all(x[self.class_name] == examples[0][self.class_name] for x in examples)):
            return self.plurality_value(parent_examples)
        # Base case: There are no attributes left to split the examples
        elif(len([example for example in examples]) == 0):
            return self.plurality_value(examples)
        else:
            max_info = -math.inf
            attr, best_threshold, best_lt_child, best_ge_child= '', 0, 0, 0
            # Split examples based on attribute test and minumum threshold
            for attribute in examples[0].keys():
                if attribute == self.id_name or attribute == self.class_name:
                    continue
                else:
                    #Find floor and ceil of threshold
                    thresh = self.minmax_threshold(examples, attribute)
                    max_thresh, min_thresh = thresh[0], thresh[1]

                    #Incrementor
                    increment = (max_thresh - min_thresh)/self.min_leaf_count
                    threshold = min_thresh + increment

                    #Split data into two seperate groups based on threshold
                    while threshold <= max_thresh:
                        ge_children, lt_children = [], []
                        for i in range(len(examples)):
                            ex = examples[i]
                            if ex[attribute] == None:
                                continue
                            elif ex[attribute] < threshold:
                                lt_children.append(ex)
                            else:
                                ge_children.append(ex)

                        #Edge case if size is out of leaf count range
                        if len(lt_children) < self.min_leaf_count or len(ge_children) < self.min_leaf_count:
                            threshold += increment
                            continue

                        # Test all possible attribute splits and picks the attribute with highest information gain
                        less_entropy, greater_entropy = self.entropy(lt_children), self.entropy(ge_children)
                        lt, ge = len(lt_children)/len(examples) * less_entropy, len(ge_children)/len(examples) * greater_entropy

                        #info=entropy-remainder
                        information_gain = self.entropy(examples) - (lt + ge)

                        # Sets best to be updated values
                        if information_gain > max_info:
                            max_info = information_gain
                            attr = attribute
                            best_threshold = threshold
                            #best split
                            best_lt_child = lt_children
                            best_ge_child = ge_children
                            
                        #increment threshold
                        threshold += increment
            #Initialize split node
            lt_node, ge_node = self.decision_tree_learning(best_lt_child, examples), self.decision_tree_learning(best_ge_child, examples)
            
            #Sets child_miss node to the bigger set
            if(len(best_lt_child) > len(best_ge_child)):
                child_miss = lt_node
            else:
                child_miss = ge_node

            return DecisionNode(attr, best_threshold, lt_node, ge_node, child_miss)

    def learn_tree(self, examples):
        return self.decision_tree_learning(examples, examples)

    def classify(self, example):
        """Perform inference on a single example.

        Args:
            example: the instance being classified

        Returns: a tuple containing a class label and a probability
        """
        return self.root.classify(example)

    def __str__(self):
        """String representation of tree, calls _ascii_tree()."""
        ln_bef, ln, ln_aft = self._ascii_tree(self.root)
        return "\n".join(ln_bef + [ln] + ln_aft)

    def _ascii_tree(self, node):
        """Super high-tech tree-printing ascii-art madness."""
        indent = 7  # adjust this to decrease or increase width of output
        if type(node) == LeafNode:
            return [""], "leaf {} {}/{}={:.2f}".format(node.pred_class, node.pred_class_count, node.total_count, node.prob), [""]
        else:
            child_ln_bef, child_ln, child_ln_aft = self._ascii_tree(
                node.child_ge)
            lines_before = [" "*indent*2 + " " + " " *
                            indent + line for line in child_ln_bef]
            lines_before.append(
                " "*indent*2 + u'\u250c' + " >={}----".format(node.test_attr_threshold) + child_ln)
            lines_before.extend(
                [" "*indent*2 + "|" + " "*indent + line for line in child_ln_aft])

            line_mid = node.test_attr_name

            child_ln_bef, child_ln, child_ln_aft = self._ascii_tree(
                node.child_lt)
            lines_after = [" "*indent*2 + "|" + " " *
                           indent + line for line in child_ln_bef]
            lines_after.append(" "*indent*2 + u'\u2514' +
                               "- <{}----".format(node.test_attr_threshold) + child_ln)
            lines_after.extend(
                [" "*indent*2 + " " + " "*indent + line for line in child_ln_aft])

            return lines_before, line_mid, lines_after


def confusion4x4(labels, vals):
    """Create an normalized predicted vs. actual confusion matrix for four classes."""
    n = sum([v for v in vals.values()])
    abbr = ["".join(w[0] for w in lab.split()) for lab in labels]
    s = ""
    s += " actual ___________________________________  \n"
    for ab, labp in zip(abbr, labels):
        row = [vals.get((labp, laba), 0)/n for laba in labels]
        s += "       |        |        |        |        | \n"
        s += "  {:^4s} | {:5.2f}  | {:5.2f}  | {:5.2f}  | {:5.2f}  | \n".format(
            ab, *row)
        s += "       |________|________|________|________| \n"
    s += "          {:^4s}     {:^4s}     {:^4s}     {:^4s} \n".format(*abbr)
    s += "                     predicted \n"
    return s


#############################################

if __name__ == '__main__':

    path_to_csv = 'town_data.csv'
    class_attr_name = '2020_label'
    id_attr_name = 'town'
    min_examples = 50  # minimum number of examples for a leaf node

    # read in the data
    examples = read_data(path_to_csv)
    train_examples, test_examples = train_test_split(examples, 0.25)

    # learn a tree from the training set
    tree = DecisionTree(train_examples, id_attr_name,
                        class_attr_name, min_examples)

    # test the tree on the test set and see how we did
    correct = 0
    almost = 0  # within one level of correct answer
    ordering = ['red', 'light blue', 'medium blue',
                'wicked blue']  # used to count "almost" right
    test_act_pred = {}
    for example in test_examples:
        actual = example[class_attr_name]
        pred, prob = tree.classify(example)
        print("{:30} pred {:15} ({:.2f}), actual {:15} {}".format(example[id_attr_name] + ':',
                                                                "'" + pred + "'", prob,
                                                                "'" + actual + "'",
                                                                '*' if pred == actual else ''))
        if pred == actual:
            correct += 1
        if abs(ordering.index(pred) - ordering.index(actual)) < 2:
            almost += 1
        test_act_pred[(actual, pred)] = test_act_pred.get(
            (actual, pred), 0) + 1

    print("\naccuracy: {:.2f}".format(correct/len(test_examples)))
    print("almost:   {:.2f}\n".format(almost/len(test_examples)))
    print(confusion4x4(['red', 'light blue', 'medium blue', 'wicked blue'], test_act_pred))
    print(tree)  # visualize the tree in sweet, 8-bit text
