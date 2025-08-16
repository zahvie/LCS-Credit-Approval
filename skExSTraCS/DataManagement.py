import numpy as np
import math

class DataManagement:
    def __init__(self,dataFeatures,dataPhenotypes,model):
        self.savedRawTrainingData = [dataFeatures,dataPhenotypes]
        self.numAttributes = dataFeatures.shape[1]  # The number of attributes in the input file.
        self.attributeInfoType = [0] * self.numAttributes  # stores false (d) or true (c) depending on its type, which points to parallel reference in one of the below 2 arrays
        self.attributeInfoContinuous = [[np.inf,-np.inf] for _ in range(self.numAttributes)] #stores continuous ranges and NaN otherwise
        self.attributeInfoDiscrete = [0] * self.numAttributes  # stores arrays of discrete values or NaN otherwise.
        for i in range(0, self.numAttributes):
            self.attributeInfoDiscrete[i] = AttributeInfoDiscreteElement()

        # About Phenotypes
        self.discretePhenotype = True  # Is the Class/Phenotype Discrete? (False = Continuous)
        self.phenotypeList = []  # Stores all possible discrete phenotype states/classes or maximum and minimum values for a continuous phenotype
        self.phenotypeRange = None  # Stores the difference between the maximum and minimum values for a continuous phenotype
        self.isDefault = True  # Is discrete attribute limit an int or string
        self.classCount = {}
        self.majorityClass = None
        try:
            int(model.discrete_attribute_limit)
        except:
            self.isDefault = False

        #Initialize some variables
        self.continuousCount = 0
        self.classPredictionWeights = {}
        self.averageStateCount = 0

        # About Dataset
        self.numTrainInstances = dataFeatures.shape[0]  # The number of instances in the training data
        self.discriminateClasses(dataPhenotypes)

        self.discriminateAttributes(dataFeatures, model)
        self.characterizeAttributes(dataFeatures, model)

        #Rule Specificity Limit
        # Rule Specificity Limit Calculation
        if model.rule_specificity_limit == None:
            i = 1
            uniqueCombinations = math.pow(self.averageStateCount,i)
            while uniqueCombinations < self.numTrainInstances:
                i += 1
                uniqueCombinations = math.pow(self.averageStateCount, i)
        
            if (model.use_feature_ranked_RSL and model.feature_importance_rank is not None):
#                 ✅ New Feature-Ranked RSL Calculation
#                print("⚡ Using Feature-Ranked Rule Specificity Limit (RSL)")
                top_features = model.feature_importance_rank  # Select top features
                print("✅ New Feature-Ranked RSL Calculation top_features - ",top_features)
                print("model.feature_importance_rank -", model.feature_importance_rank)
                model.rule_specificity_limit = len(model.feature_importance_rank)  # Adjust limit
#                model.rule_specificity_limit = min(i, len(model.feature_importance_rank))
                print("len(model.feature_importance_rank)", len(model.feature_importance_rank))
                print("model.feature_importance_rank-->", model.feature_importance_rank)
                print("model.rule_specificity_limit", model.rule_specificity_limit)
            else:
                # ✅ Original RSL Calculation
                print("🛑 Using Standard Rule Specificity Limit Calculation")
                model.rule_specificity_limit = min(i, self.numAttributes)  # Original method
                print(f"i is {i} and self.numAttributes is {self.numAttributes} ")
                print(f"model.rule_specificity_limit is set to {model.rule_specificity_limit }")


        self.trainFormatted = self.formatData(dataFeatures, dataPhenotypes, model)  # The only np array

    def discriminateClasses(self,phenotypes):
        currentPhenotypeIndex = 0
        while (currentPhenotypeIndex < self.numTrainInstances):
            target = phenotypes[currentPhenotypeIndex]
            if target in self.phenotypeList:
                self.classCount[target]+=1
                self.classPredictionWeights[target] += 1
            else:
                self.phenotypeList.append(target)
                self.classCount[target] = 1
                self.classPredictionWeights[target] = 1
            currentPhenotypeIndex+=1
        self.majorityClass = max(self.classCount)
        total = 0
        for eachClass in list(self.classCount.keys()):
            total += self.classCount[eachClass]
        for eachClass in list(self.classCount.keys()):
            self.classPredictionWeights[eachClass] = 1 - (self.classPredictionWeights[eachClass]/total)

    def discriminateAttributes(self,features,model): #determines whether each attribute (column) in the dataset is discrete or continuous
        for att in range(self.numAttributes): #att iterates over each column (attribute) in the dataset. The loop ensures that each column is processed one at a time.
            attIsDiscrete = True
            if self.isDefault:
                currentInstanceIndex = 0 #currentInstanceIndex = 0 initializes row traversal.
                stateDict = {}
                while attIsDiscrete and len(list(stateDict.keys())) <= model.discrete_attribute_limit and currentInstanceIndex < self.numTrainInstances:
                    target = features[currentInstanceIndex,att] #Extracts values from the same column. Keeps going down the column row by row.
                    if target in list(stateDict.keys()):
                        stateDict[target] += 1
                    elif np.isnan(target):
                        pass
                    else:
                        stateDict[target] = 1
                    currentInstanceIndex+=1

                if len(list(stateDict.keys())) > model.discrete_attribute_limit:
                    attIsDiscrete = False
            elif model.discrete_attribute_limit == "c":
                if att in model.specified_attributes:
                    attIsDiscrete = False
                else:
                    attIsDiscrete = True
            elif model.discrete_attribute_limit == "d":
                if att in model.specified_attributes:
                    attIsDiscrete = True
                else:
                    attIsDiscrete = False

            if attIsDiscrete:
                self.attributeInfoType[att] = False # indicating discrete
            else:
                self.attributeInfoType[att] = True # indicating continuous
                self.continuousCount += 1

    def characterizeAttributes(self,features,model): # Stores min/max for continuous attributes & unique values for discrete ones
        for currentFeatureIndexInAttributeInfo in range(self.numAttributes): # Iterate through all attributes (columns).
            for currentInstanceIndex in range(self.numTrainInstances): # Iterate through all instances (rows) within the current column.
                target = features[currentInstanceIndex,currentFeatureIndexInAttributeInfo] # Retrieves the value at the current row (currentInstanceIndex) and column (currentFeatureIndexInAttributeInfo).
                if not self.attributeInfoType[currentFeatureIndexInAttributeInfo]:#if attribute is discrete (False = discrete value.)
                    if target in self.attributeInfoDiscrete[currentFeatureIndexInAttributeInfo].distinctValues or np.isnan(target): # If the value is already in the distinctValues list, or it's NaN, --> do nothing. np.isnan(target) ensures missing values (NaNs) are ignored.
                        pass
                    else:
                        self.attributeInfoDiscrete[currentFeatureIndexInAttributeInfo].distinctValues.append(target) # If the discrete value is new, it is added to distinctValues.
                        self.averageStateCount += 1 # averageStateCount is incremented, which helps track the number of unique states in the dataset.
                else:                               # if attribute is continuous - True = continuous value
                    if np.isnan(target):            # If the value is NaN (missing), it is skipped.
                        pass
                    elif float(target) > self.attributeInfoContinuous[currentFeatureIndexInAttributeInfo][1]:
                        self.attributeInfoContinuous[currentFeatureIndexInAttributeInfo][1] = float(target)
                    elif float(target) < self.attributeInfoContinuous[currentFeatureIndexInAttributeInfo][0]:
                        self.attributeInfoContinuous[currentFeatureIndexInAttributeInfo][0] = float(target)
                    else:
                        pass
            if self.attributeInfoType[currentFeatureIndexInAttributeInfo]: #if attribute is continuous
                self.averageStateCount += 2
        self.averageStateCount = self.averageStateCount/self.numAttributes

    def formatData(self,features,phenotypes,model):
        formatted = np.insert(features,self.numAttributes,phenotypes,1) #Combines features and phenotypes into one array

        self.shuffleOrder = np.random.choice(self.numTrainInstances,self.numTrainInstances,replace=False) #e.g. first element in this list is where the first element of the original list will go
        shuffled = []
        for i in range(self.numTrainInstances):
            shuffled.append(None)
        for instanceIndex in range(self.numTrainInstances):
            shuffled[self.shuffleOrder[instanceIndex]] = formatted[instanceIndex]
        formatted = np.array(shuffled)

        shuffledFeatures = formatted[:,:-1].tolist()
        shuffledLabels = formatted[:,self.numAttributes].tolist()
        for i in range(len(shuffledFeatures)):
            for j in range(len(shuffledFeatures[i])):
                if np.isnan(shuffledFeatures[i][j]):
                    shuffledFeatures[i][j] = None
            if np.isnan(shuffledLabels[i]):
                shuffledLabels[i] = None
        return [shuffledFeatures,shuffledLabels]


class AttributeInfoDiscreteElement():
    def __init__(self):
        self.distinctValues = []
