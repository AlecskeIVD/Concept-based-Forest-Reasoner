import torch
from treeUtils import GradTree, ProbabilisticTree, fastProbabilisticTree

if __name__ == "__main__":
    test_tree = GradTree()

    concepts = torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
    representation = torch.tensor([[[0.3, 0.4, 0.3]]])
    leafValues = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.float32)
    nOutputs = 2
    depth = 1
    #output = test_tree.evaluate(concepts, representation, leafValues, nOutputs, depth)
    #print("Output of the GradTree evaluation:", output)

    #test_tree = ProbabilisticTree()
    #print("Output of the ProbabilisticTree evaluation:", test_tree.evaluate(concepts, representation, leafValues, nOutputs, depth))

    test_tree = fastProbabilisticTree(depth=depth, batchSize=2, memorySize=1)
    print("Output of the fastProbabilisticTree evaluation:", test_tree.evaluate(concepts, representation, leafValues, nOutputs, depth))
    test_tree.printTree(representation, leafValues, concepts.shape[1], depth)
    for tree in range(1):
        print(f"Tree {tree}:")
        treeVars = representation[tree, :]
        splitVars = treeVars
        leafValuesU = leafValues[tree,:].view(2**depth, leafValues.shape[2])
        test_tree.printTree(splitVars, leafValues, concepts.shape[1], depth)
        print("---------")


