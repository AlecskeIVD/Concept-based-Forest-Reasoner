"""
Abstract class to represent decision trees
"""
from abc import abstractmethod
import torch


class DecisionTree():
    @abstractmethod
    def evaluate(self, concepts, representation, leafValues, nOutputs, depth):
        pass


# Try gradtree with linear

# Probabilistic splits ipv gradtree
class GradTree(DecisionTree):
    def evaluate(self, concepts, representation, leafValues, nOutputs, depth, splitScale = 8):
        """
        concepts: tensor of shape (batch size, nConcepts) with concept activations
        representation: vector with splits (one value for each concept) (nConcepts*(2**depth - 1))
        leafValues: tensor of shape (2**depth, nOutputs) with output values for each leaf
        nOutputs: number of output classes
        depth: depth of the decision tree
        """
        # Calculate output of decision tree based on GradTree paper
        concepts = concepts.detach() > 0.5 # Hardcut for now
        nConcepts = len(concepts[0, :])

        device = concepts.device  # <- add this

        y = torch.zeros((len(concepts[:,0]), nOutputs), device=device) # Initialize output tensor
        
        # Iterate over all possible leaves
        for leaf in range(2**depth):
            probabilityLeaf = 1.0
            for level in range(1, depth+1):
                #allSplitValuesOfThisLevel = representation[(2**(level-1)-1)*nConcepts : (2**level -1)*nConcepts]
                parentNode = 2**(level-1) + leaf // (2**(depth - (level-1))) -1
                leftOrRight = (leaf // (2**(depth - level))) % 2
                s = torch.sigmoid(splitScale * (0.5-torch.sum(representation[parentNode * nConcepts : (parentNode + 1) * nConcepts] * concepts, dim=1))) # Subtract inner product of split values parent node and concepts of 0.5
                probabilityLeaf = probabilityLeaf * (leftOrRight*(1-s)+(1-leftOrRight)*s)
            y = y + probabilityLeaf.unsqueeze(1) * leafValues[leaf, :].unsqueeze(0)
        return y


"""
For every split, we use hard splits, but the variable on which we split is chosen probabilistically
=> output is weighted average over all possible trees with hard splits, weights given by probabilities of choosing variables at each split
"""
class ProbabilisticTree(DecisionTree):
    def evaluate(self, concepts, representation, leafValues, nOutputs, depth):
        """
        concepts: tensor of shape (batch size, nConcepts) with concept values
        representation: vector with splits (one value for each concept) (nConcepts*(2**depth - 1))
        leafValues: tensor of shape (2**depth, nOutputs) with output values for each leaf
        nOutputs: number of output classes
        depth: depth of the decision tree
        """
        concepts = concepts.detach() > 0.5 # Hardcut for now
        nConcepts = len(concepts[0,:])
        batchSize = len(concepts[:,0])

        device = concepts.device

        result = torch.zeros((batchSize, nOutputs), device=device) # Initialize output tensor

        def recursiveEvaluate(concepts, currentIndex):
            if currentIndex >= 2**depth - 1:
                leafIndex = currentIndex - (2**depth - 1)
                return leafValues[leafIndex, :].unsqueeze(0).to(device)
            else:
                splitVars = representation[currentIndex * nConcepts : (currentIndex + 1) * nConcepts]
                # Initialize output vector for one entry of batch
                y = torch.zeros((1, nOutputs), device=device)
                leftOutput = recursiveEvaluate(concepts, 2 * currentIndex + 1)
                rightOutput = recursiveEvaluate(concepts, 2 * currentIndex + 2)

                # probRight = sum over all concepts of (concept value * split prob for that concept)
                probRight = torch.sum(concepts * splitVars, dim=1)
                probLeft = 1 - probRight
                y = y + probLeft.unsqueeze(1) * leftOutput + probRight.unsqueeze(1) * rightOutput
                return y
                #for concept, conProb in enumerate(splitVars):
                #    if concepts[0, concept] == 0:
                #        leftOutput = recursiveEvaluate(concepts, 2 * currentIndex + 1)
                #        y = y + conProb * leftOutput
                #    else:
                #        rightOutput = recursiveEvaluate(concepts, 2 * currentIndex + 2)
                #        y = y + conProb * rightOutput
                #return y
            # Hergebruik evalueren van subtrees 

        
        for entry in range(batchSize):
            entryConcepts = concepts[entry, :].unsqueeze(0)
            entryResult = recursiveEvaluate(entryConcepts, 0)
            result[entry, :] = entryResult
        return result


class fastProbabilisticTree(DecisionTree):
    def __init__(self, depth, batchSize, memorySize):
        super().__init__()
        L = 2**depth
        self.pathUnexpanded = torch.tensor([[ (i >> d) & 1 for d in reversed(range(depth)) ] for i in range(L)], dtype=torch.bool) # (L, depth)
        self.path = self.pathUnexpanded.unsqueeze(0).unsqueeze(0).expand(batchSize, memorySize, L, depth)  # (B,M,L,depth)
    
        levels = torch.arange(depth)         # [0,1,2]
        leaf_range = torch.arange(L)         # [0..7]
        self.visitedNodes = ((2**levels - 1).unsqueeze(0) + torch.div(leaf_range.unsqueeze(1), 2**(depth - levels), rounding_mode='floor')).long()  # shape (L, depth)
        self.index = self.visitedNodes.unsqueeze(0).unsqueeze(0).expand(batchSize, memorySize, L, depth)  # (B,M,L,depth)

        # keep visitedNodes and index on device in case batch size changes

        self.batchSize = batchSize
        self.memorySize = memorySize


    
    def evaluate(self, concepts: torch.Tensor, representation: torch.Tensor, leafValues: torch.Tensor, nOutputs, depth):
        """
        concepts: tensor of shape (batch size, nConcepts) with concept values
        representation: Tensor with splits (one value for each concept) (memorySize, (2**depth - 1), nConcepts)
        leafValues: tensor of shape (memorySize, 2**depth, nOutputs) with output values for each leaf
        nOutputs: number of output classes
        depth: depth of the decision tree
        """
        concepts = (concepts.detach() > 0.5).float() # Hardcut for now
        nConcepts = len(concepts[0,:])
        batchSize = len(concepts[:,0])
        memorySize = representation.shape[0]
        nLeaves = 2**depth
        nInternalNodes = nLeaves - 1

        device = concepts.device
        if batchSize == self.batchSize:
            self.index = self.index.to(device)
            self.path = self.path.to(device)
        else:
            self.batchSize = batchSize
            self.index = self.visitedNodes.unsqueeze(0).unsqueeze(0).expand(batchSize, memorySize, 2**depth, depth).to(device)  # (B,M,L,depth)
            self.path = self.pathUnexpanded.unsqueeze(0).unsqueeze(0).expand(batchSize, memorySize, 2**depth, depth).to(device)  # (B,M,L,depth)
            


        # repeat concepts such that it has dimension (batch size, memory size, (2**depth-1), nConcepts)
        concepts_bmnc = concepts.unsqueeze(1).unsqueeze(1).expand(-1, memorySize, nInternalNodes, -1) # (B, M,(2**depth-1), nConcepts)
        #print("concepts_bmnc:", concepts_bmnc.shape)
        # repeat representation such that it has dimension (B, memory size, (2**depth-1), nConcepts)
        representation_bmnc = representation.unsqueeze(0).expand(batchSize, -1, -1, -1).to(device) # (B, M, (2**depth-1), nConcepts)
        #print("representation_bmnc:", representation_bmnc.shape)
        goRightProbs = torch.sum(concepts_bmnc * representation_bmnc, dim=-1) # (B, M, (2**depth-1))
        #print("goRightProbs:", goRightProbs.shape)
        #goLeftProbs = 1 - goRightProbs # (B, M, (2**depth-1))

        # make sure gather works along last dim (N)
        goRightProbs_bmli = goRightProbs.unsqueeze(2).expand(batchSize, memorySize, nLeaves, nInternalNodes)
        goLeftProbs_bmli = 1-goRightProbs_bmli

        nodeProbsRight = torch.gather(goRightProbs_bmli, 3, self.index) # (B, M, L, depth)
        nodeProbsLeft  = torch.gather(goLeftProbs_bmli, 3, self.index) # (B, M, L, depth)
        # npr[n, m, l, d] = prob of going right at node on path from root to leaf l on depth d for tree m for entry n
        #print("nodeProbsRight:", nodeProbsRight.shape)
        #print("nodeProbsLeft:", nodeProbsLeft.shape)

        # Now we combine the relevant probabilities based on self.paths

        combinedNodeProbs_bmld = torch.where(self.path, nodeProbsRight, nodeProbsLeft) # (B, M, L, depth)
        #print("combinedNodeProbs_bc:", combinedNodeProbs_bmld.shape)

        leafProbs_bml = torch.prod(combinedNodeProbs_bmld, dim=-1) # (B, M, L)
        #print("leafProbs_bml:", leafProbs_bml.shape)

        # Now combine with leaf values
        # leafValues_bmlo = leafValues.unsqueeze(0).expand(batchSize, -1, -1, -1) # (B, M, L, nOutputs)
        # print("leafValues_bmlo:", leafValues_bmlo.shape)

        # Now do weighted sum over leaves
        output_bmo = torch.einsum('bml,mlo->bmo', leafProbs_bml, leafValues) # (B, M, nOutputs)
        #print("output_bmo:", output_bmo.shape)
        #output_bo = torch.sum(output_bmo, dim=1) # (B, nOutputs)
        #print("output_bo:", output_bo.shape)
        return output_bmo, leafProbs_bml
    

    def printTree(self, splitVars, leafValues, nConcepts, depth):
        """
        splitVars: tensor with splits (one value for each concept) ((2**depth - 1), nConcepts)
        leafValues: tensor with outputs (one vector for each leaf) (2**depth, nOutputs)
        """
        output = ""
        translation = {}
        lOrR = {0: "L", 1: "R"}
        maxDigit = nConcepts // 2
        for i in range(nConcepts):
            translation[i] = str(lOrR[i//maxDigit]) + str(i%maxDigit)
        splitVars = splitVars.detach().cpu()
        leafValues = leafValues.detach().cpu()
        nLeaves = 2**depth
        for depthIndex in range(depth):
            levelNodes = 2**depthIndex
            for node in range(levelNodes):
                nodeIndex = 2**depthIndex - 1 + node
                splitVar = splitVars[nodeIndex, :]
                chosenConcept = torch.argmax(splitVar).item()
                print(translation[chosenConcept], end=' | ')
                output += translation[chosenConcept] + " | "
            print()
            output += "\n"
        for leaf in range(nLeaves):
            print(torch.argmax(leafValues[leaf, :]).item(), end=" | ")
            output += str(torch.argmax(leafValues[leaf, :]).item()) + " | "
        print()
        output += "\n"
        return output
    

    def collectLeafProbs(self, concepts: torch.Tensor, representation: torch.Tensor, depth):
        concepts = (concepts.detach() > 0.5).float() # Hardcut for now
        nConcepts = len(concepts[0,:])
        batchSize = len(concepts[:,0])
        memorySize = representation.shape[0]
        nLeaves = 2**depth
        nInternalNodes = nLeaves - 1

        device = concepts.device
        if batchSize == self.batchSize:
            self.index = self.index.to(device)
            self.path = self.path.to(device)
        else:
            self.batchSize = batchSize
            self.index = self.visitedNodes.unsqueeze(0).unsqueeze(0).expand(batchSize, memorySize, 2**depth, depth).to(device)  # (B,M,L,depth)
            self.path = self.pathUnexpanded.unsqueeze(0).unsqueeze(0).expand(batchSize, memorySize, 2**depth, depth).to(device)  # (B,M,L,depth)
            


        # repeat concepts such that it has dimension (batch size, memory size, (2**depth-1), nConcepts)
        concepts_bmnc = concepts.unsqueeze(1).unsqueeze(1).expand(-1, memorySize, nInternalNodes, -1) # (B, M,(2**depth-1), nConcepts)
        #print("concepts_bmnc:", concepts_bmnc.shape)
        # repeat representation such that it has dimension (B, memory size, (2**depth-1), nConcepts)
        representation_bmnc = representation.unsqueeze(0).expand(batchSize, -1, -1, -1).to(device) # (B, M, (2**depth-1), nConcepts)
        #print("representation_bmnc:", representation_bmnc.shape)
        goRightProbs = torch.sum(concepts_bmnc * representation_bmnc, dim=-1) # (B, M, (2**depth-1))
        #print("goRightProbs:", goRightProbs.shape)
        #goLeftProbs = 1 - goRightProbs # (B, M, (2**depth-1))

        # make sure gather works along last dim (N)
        goRightProbs_bmli = goRightProbs.unsqueeze(2).expand(batchSize, memorySize, nLeaves, nInternalNodes)
        goLeftProbs_bmli = 1-goRightProbs_bmli

        nodeProbsRight = torch.gather(goRightProbs_bmli, 3, self.index) # (B, M, L, depth)
        nodeProbsLeft  = torch.gather(goLeftProbs_bmli, 3, self.index) # (B, M, L, depth)
        # npr[n, m, l, d] = prob of going right at node on path from root to leaf l on depth d for tree m for entry n
        #print("nodeProbsRight:", nodeProbsRight.shape)
        #print("nodeProbsLeft:", nodeProbsLeft.shape)

        # Now we combine the relevant probabilities based on self.paths

        combinedNodeProbs_bmld = torch.where(self.path, nodeProbsRight, nodeProbsLeft) # (B, M, L, depth)
        #print("combinedNodeProbs_bc:", combinedNodeProbs_bmld.shape)

        leafProbs_bml = torch.prod(combinedNodeProbs_bmld, dim=-1) # (B, M, L)
        return leafProbs_bml


        



# E = (S * C).sum(dim=-1) (S: softmax over split variables (batch, Depth, 2^(D-1), nConcepts), C: binary matrix (B, D, 2^(D-1), nConcepts))

# create one row for each leaf with relevant E entries (or complement)
# torch.where gebruiken, en anders 1 om maal te kunnen gebruiken

class regressionFPT(DecisionTree):
    def __init__(self, depth, batchSize, memorySize):
        super().__init__()
        L = 2**depth
        self.pathUnexpanded = torch.tensor([[ (i >> d) & 1 for d in reversed(range(depth)) ] for i in range(L)], dtype=torch.bool) # (L, depth)
        self.path = self.pathUnexpanded.unsqueeze(0).unsqueeze(0).expand(batchSize, memorySize, L, depth)  # (B,M,L,depth)
    
        levels = torch.arange(depth)         # [0,1,2]
        leaf_range = torch.arange(L)         # [0..7]
        self.visitedNodes = ((2**levels - 1).unsqueeze(0) + torch.div(leaf_range.unsqueeze(1), 2**(depth - levels), rounding_mode='floor')).long()  # shape (L, depth)
        self.index = self.visitedNodes.unsqueeze(0).unsqueeze(0).expand(batchSize, memorySize, L, depth)  # (B,M,L,depth)

        # keep visitedNodes and index on device in case batch size changes

        self.batchSize = batchSize
        self.memorySize = memorySize


    
    def evaluate(self, concepts: torch.Tensor, representation: torch.Tensor, leafValues: torch.Tensor, depth):
        """
        concepts: tensor of shape (batch size, nConcepts) with concept values
        representation: Tensor with splits (one value for each concept) (memorySize, (2**depth - 1), nConcepts)
        leafValues: tensor of shape (memorySize, 2**depth, nOutputs) with output values for each leaf
        nOutputs: number of output classes
        depth: depth of the decision tree
        """
        concepts = (concepts.detach() > 0.5).float() # Hardcut for now
        nConcepts = len(concepts[0,:])
        batchSize = len(concepts[:,0])
        memorySize = representation.shape[0]
        nLeaves = 2**depth
        nInternalNodes = nLeaves - 1

        device = concepts.device
        if batchSize == self.batchSize:
            self.index = self.index.to(device)
            self.path = self.path.to(device)
        else:
            self.batchSize = batchSize
            self.index = self.visitedNodes.unsqueeze(0).unsqueeze(0).expand(batchSize, memorySize, 2**depth, depth).to(device)  # (B,M,L,depth)
            self.path = self.pathUnexpanded.unsqueeze(0).unsqueeze(0).expand(batchSize, memorySize, 2**depth, depth).to(device)  # (B,M,L,depth)
            


        # repeat concepts such that it has dimension (batch size, memory size, (2**depth-1), nConcepts)
        concepts_bmnc = concepts.unsqueeze(1).unsqueeze(1).expand(-1, memorySize, nInternalNodes, -1) # (B, M,(2**depth-1), nConcepts)
        #print("concepts_bmnc:", concepts_bmnc.shape)
        # repeat representation such that it has dimension (B, memory size, (2**depth-1), nConcepts)
        representation_bmnc = representation.unsqueeze(0).expand(batchSize, -1, -1, -1).to(device) # (B, M, (2**depth-1), nConcepts)
        #print("representation_bmnc:", representation_bmnc.shape)
        goRightProbs = torch.sum(concepts_bmnc * representation_bmnc, dim=-1) # (B, M, (2**depth-1))
        #print("goRightProbs:", goRightProbs.shape)
        #goLeftProbs = 1 - goRightProbs # (B, M, (2**depth-1))

        # make sure gather works along last dim (N)
        goRightProbs_bmli = goRightProbs.unsqueeze(2).expand(batchSize, memorySize, nLeaves, nInternalNodes)
        goLeftProbs_bmli = 1-goRightProbs_bmli

        nodeProbsRight = torch.gather(goRightProbs_bmli, 3, self.index) # (B, M, L, depth)
        nodeProbsLeft  = torch.gather(goLeftProbs_bmli, 3, self.index) # (B, M, L, depth)
        # npr[n, m, l, d] = prob of going right at node on path from root to leaf l on depth d for tree m for entry n
        #print("nodeProbsRight:", nodeProbsRight.shape)
        #print("nodeProbsLeft:", nodeProbsLeft.shape)

        # Now we combine the relevant probabilities based on self.paths

        combinedNodeProbs_bmld = torch.where(self.path, nodeProbsRight, nodeProbsLeft) # (B, M, L, depth)
        #print("combinedNodeProbs_bc:", combinedNodeProbs_bmld.shape)

        leafProbs_bml = torch.prod(combinedNodeProbs_bmld, dim=-1) # (B, M, L)
        #print("leafProbs_bml:", leafProbs_bml.shape)

        # Now combine with leaf values
        # leafValues_bmlo = leafValues.unsqueeze(0).expand(batchSize, -1, -1, -1) # (B, M, L, nOutputs)
        # print("leafValues_bmlo:", leafValues_bmlo.shape)

        # Now do weighted sum over leaves
        output_bm = torch.einsum('bml,ml->bm', leafProbs_bml, leafValues) # (B, M)
        #print("output_bmo:", output_bmo.shape)
        #output_bo = torch.sum(output_bmo, dim=1) # (B, nOutputs)
        #print("output_bo:", output_bo.shape)
        return output_bm, leafProbs_bml
    

    def printTree(self, splitVars, leafValues, nConcepts, depth):
        """
        splitVars: tensor with splits (one value for each concept) ((2**depth - 1), nConcepts)
        leafValues: tensor with outputs (one vector for each leaf) (2**depth, nOutputs)
        """
        translation = {}
        lOrR = {0: "L", 1: "R"}
        maxDigit = nConcepts // 2
        for i in range(nConcepts):
            translation[i] = str(lOrR[i//maxDigit]) + str(i%maxDigit)
        splitVars = splitVars.detach().cpu()
        leafValues = leafValues.detach().cpu()
        nLeaves = 2**depth
        for depthIndex in range(depth):
            levelNodes = 2**depthIndex
            for node in range(levelNodes):
                nodeIndex = 2**depthIndex - 1 + node
                splitVar = splitVars[nodeIndex, :]
                chosenConcept = torch.argmax(splitVar).item()
                print(translation[chosenConcept], end=' | ')
            print()
        for leaf in range(nLeaves):
            print(leafValues[leaf, :].item(), end=" | ")
        print()