import random
import copy
import numpy as np
import math
import csv
import os

class Classifier:
    def __init__(self,model):
        self.specifiedAttList = []
        self.condition = []
        self.phenotype = None

        self.fitness = model.init_fitness
        self.accuracy = 0
        self.numerosity = 1
        self.aveMatchSetSize = None
        self.deletionProb = None

        self.timeStampGA = None
        self.initTimeStamp = None
        self.epochComplete = False

        self.matchCount = 0
        self.correctCount = 0
        self.matchCover = 0
        self.correctCover = 0

    def initializeByCopy(self,toCopy,iterationCount):
        self.specifiedAttList = copy.deepcopy(toCopy.specifiedAttList)
        self.condition = copy.deepcopy(toCopy.condition)
        self.phenotype = copy.deepcopy(toCopy.phenotype)
        self.timeStampGA = iterationCount
        self.initTimeStamp = iterationCount
        self.aveMatchSetSize = copy.deepcopy(toCopy.aveMatchSetSize)
        self.fitness = toCopy.fitness
        self.accuracy = toCopy.accuracy

    def initializeByCovering(self,model,setSize,state,phenotype):
        self.timeStampGA = model.iterationCount
        self.initTimeStamp = model.iterationCount
        self.aveMatchSetSize = setSize
        self.phenotype = phenotype

#        print("model.rule_specificity_limit / # of Attributes selected - ", model.rule_specificity_limit)
        toSpecify = random.randint(1, model.rule_specificity_limit) # Determines how many attributes will be part of the rule.

        if model.use_midpoint_distance_filter:
#            print("⚡ Using Midpoint Distance Filtering for Classifier Creation")

#            toSpecify = model.rule_specificity_limit  # Ensure we select the full RSL number of attributes
#            print("toSpecify / how many attributes will be part of the rule - ", toSpecify)
            # Compute midpoints for each attribute in the dataset AFTER classifier creation
            self.midpoints = {}

            # Create the classifier by selecting features (the logic is unchanged)
            potentialSpec = random.sample(range(model.env.formatData.numAttributes),toSpecify)
#            print("Selected Attributes",potentialSpec)

            for attRef in potentialSpec:
                if state[attRef] is not None:
                    # Build the condition with a random buffer and calculate midpoints
                    self.specifiedAttList.append(attRef)
                    self.condition.append(self.buildMatch(model, attRef, state))  

        elif model.use_feature_ranked_RSL and model.feature_importance_rank is not None:
            self.midpoints = {}
            randomized_percent = 50
            # ✅ New Feature-Ranked RSL Enabled
#            print("⚡ Using Feature-Ranked Rule Specificity Limit (RSL)")
            toSpecify = model.rule_specificity_limit  # Ensure we select the full RSL number of attributes
#            print("Ensure we select the full RSL number of attributes-->toSpecify", toSpecify)

            if model.doExpertKnowledge:
                i = 0
                while len(self.specifiedAttList) < toSpecify and i < len(model.feature_importance_rank):
                    target = model.feature_importance_rank[i]  # Pick features based on ranking
                    if state[target] is not None:
                        self.specifiedAttList.append(target)
                        self.condition.append(self.buildMatch(model, target, state))
                    i += 1
            else:
                # Copy the feature importance rank to a new variable
                shuffled_feature_rank = model.feature_importance_rank.copy()                

                # Shuffle the copied list
                random.shuffle(shuffled_feature_rank)
#                print("shuffled_feature_rank",shuffled_feature_rank)
#                print("New toSpecify",toSpecify)

                potentialSpec = shuffled_feature_rank[:toSpecify]  # Use top-ranked features
#                potentialSpec = model.feature_importance_rank[:toSpecify]  # Use top-ranked features
#                print("Selected fields to make the rule (Feature Ranked) --> potentialSpec ", potentialSpec)
#                print("# of potentialSpec", len(potentialSpec))
                ###########################################
                RSL_50_percent = math.ceil(len(model.feature_importance_rank) * (randomized_percent / 100))  # 50% of RSL
#                print("50% of feature importance rank is -->", RSL_50_percent )
                randomized_rsl_features = random.sample(potentialSpec, RSL_50_percent)  # Randomly select from the top-ranked features
#                print("randomized top 50% rsl features", randomized_rsl_features)

                # Get the remaining (unselected) features
#                print("self.all_feature_list", model.all_feature_list)
#                print("list(self.all_feature_list)", list(model.all_feature_list))
#                print("randomized_rsl_features",randomized_rsl_features)
#                print("list(randomized_rsl_features)",list(randomized_rsl_features))
                remaining_features = list(model.all_feature_list - set(randomized_rsl_features))
#                print("remaining_features --Final",remaining_features)
#                randomized_remaining_features = random.sample(remaining_features, 2)  # Randomly select 2 features from the entire set

                # Calculate the number of features to select as remaining features
                remaining_feature_count = len(model.feature_importance_rank) - RSL_50_percent
#                print("remaining_feature_count", remaining_feature_count)
                
                # Ensure we don't sample more features than available
                remaining_feature_count = max(0, remaining_feature_count)  # Prevent negative values
#                print("Prevent negative -->remaining_feature_count",remaining_feature_count)
                
                # Randomly select the remaining features dynamically
                randomized_remaining_features = random.sample(remaining_features, remaining_feature_count)
#                print("Randomly select remaining features",randomized_remaining_features)
                
#                print("randomized_remaining_features", randomized_remaining_features)

                # Combine the randomized features
                final_selected_features = randomized_rsl_features + randomized_remaining_features
#                print("final_selected_features",final_selected_features)
                ###########################################
                
#                for attRef in potentialSpec:
                for attRef in final_selected_features: # modified to get from Combined the randomized features
                    if state[attRef] is not None:
                        self.specifiedAttList.append(attRef)
                        self.condition.append(self.buildMatch(model, attRef, state))
        else:
            # ✅ Original Method: Standard Attribute Selection
            if model.doExpertKnowledge:
                i = 0
                while len(self.specifiedAttList) < toSpecify and i < model.env.formatData.numAttributes - 1:
                    target = model.EK.EKRank[i] # Use expert knowledge ranking
                    if state[target] != None:
                        self.specifiedAttList.append(target)
                        self.condition.append(self.buildMatch(model,target,state))
                    i += 1
            else:
#                print("model.rule_specificity_limit", model.rule_specificity_limit)
                #print("toSpecify-toSpecify BEFORE", toSpecify)
                #print("TYPE toSpecify", type(toSpecify))
                potentialSpec = random.sample(range(model.env.formatData.numAttributes),toSpecify)
#                print("Selected fields to make the rule (Random-->potentialSpec) -",potentialSpec)
    
                for attRef in potentialSpec:
                    if state[attRef] != None:
                        self.specifiedAttList.append(attRef)
                        self.condition.append(self.buildMatch(model,attRef,state))
#        print(f"✅ Final selected attributes: {self.condition}")
#        print("model.midpoints", model.midpoints)

    def buildMatch(self,model,attRef,state):
        attributeInfoType = model.env.formatData.attributeInfoType[attRef]
        if not (attributeInfoType):  # Discrete -->False
            attributeInfoValue = model.env.formatData.attributeInfoDiscrete[attRef]
        else:                        # Continuous -->True
            attributeInfoValue = model.env.formatData.attributeInfoContinuous[attRef]

        if model.use_midpoint_distance_filter or model.use_feature_ranked_RSL:

            if attributeInfoType: #Continuous Attribute
                attRange = attributeInfoValue[1] - attributeInfoValue[0] # calculates a range (low & high values)
#                rangeRadius = random.randint(25, 75) * 0.01 * attRange / 2.0  # Continuous initialization domain radius.
#                random.seed(42)  # Fixing the seed
                # Add a random buffer to the range (upper and lower bounds)
#                buffer_low = random.uniform(-0.1, 0.1) * attRange  # Random buffer for the lower bound
#                buffer_high = random.uniform(0.1, 0.2) * attRange  # Random buffer for the upper bound

                buffer_low = random.random() * 0.35  # Random buffer for the lower bound
                buffer_high = random.random() * 0.35  # Random buffer for the upper bound

#                print("attributeInfoValue[1]", attributeInfoValue[1])
#                print("attributeInfoValue[0]", attributeInfoValue[0])
#                print("attRange", attRange)
#                print("random.uniform for buffer_low", random.uniform)
#                print("buffer_low",buffer_low)
#                print("buffer_high",buffer_high)

                # Compute lower and upper bounds with added random buffer
#                Low = state[attRef] - rangeRadius - buffer_low
#                High = state[attRef] + rangeRadius + buffer_high
                Low = state[attRef] - buffer_low
                High = state[attRef] + buffer_high
#                print("state[attRef]",state[attRef])
#                print("rangeRadius",rangeRadius)
#                print("buffer_low",buffer_low)
#                print("buffer_high",buffer_high)
#                print("Low",Low)
#                print("High",High)

                # Compute midpoint as the average of the lower and upper bounds
                midpoint = (Low + High) / 2
#                print("midpoint",midpoint)

                # 🔹 Check if midpoint is None and log it
                if midpoint is None:
#                    print(f"Logging None midpoint: state={state},attRef={attRef},Low{Low},High{High}")
                    self.log_none_midpoints(attRef, state)
#                    print("attRef is", attRef)
#                    print("Low",Low)
#                    print("High",High)
                else:
                    # Store midpoint for later use (for distance calculations)
                    model.midpoints[attRef] = midpoint
     
                condList = [Low, High]
            else:  # Discrete Attribute -->False
                condList = state[attRef]  # Directly assigns a value

#                print("state[attRef]",state[attRef])
                midpoint = state[attRef] / 2
#                print("Discrete", state[attRef])
#                print("Discrete - midpoint",midpoint)

                # 🔹 Check if midpoint is None and log it
                if midpoint is None:
#                    print(f"Logging None midpoint: state{state},attRef={attRef}, state={state}")
                    self.log_none_midpoints(attRef, state)
                else:
                    # Store midpoint for later use (for distance calculations)
                    model.midpoints[attRef] = midpoint
#                    print("condList", condList)
            return condList   
        else:
            if attributeInfoType: #Continuous Attribute -->True
                attRange = attributeInfoValue[1] - attributeInfoValue[0] # calculates a range (low & high values)
                rangeRadius = random.randint(25, 75) * 0.01 * attRange / 2.0  # Continuous initialization domain radius.
                Low = state[attRef] - rangeRadius
                High = state[attRef] + rangeRadius
                condList = [Low, High]
            else:
                condList = state[attRef] # directly assigns a value
            return condList

    def updateEpochStatus(self,model): # checks whether a classifier (rule) has completed an epoch (i.e., has seen the entire training dataset at least once)
        if not self.epochComplete and (model.iterationCount - self.initTimeStamp - 1) >= model.env.formatData.numTrainInstances:
            self.epochComplete = True
#            print("self.epochComplete -", self.epochComplete)
#            print("model.iterationCount -", model.iterationCount)
#            print("self.initTimeStamp -", self.initTimeStamp)

    def match(self, model, state):
        for i in range(len(self.condition)): # Loop through each condition (the loop runs 6 times because self.condition has 6 elements - [0.0, [-0.5700000000000038, 40.730000000000004], [-136.39999999999998, 536.4], [5.75, 12.25], [-7.245, 7.245], 1.0]
            specifiedIndex = self.specifiedAttList[i]
            attributeInfoType = model.env.formatData.attributeInfoType[specifiedIndex] # Checks if the attribute is continuous (True) or discrete (False).
            # Continuous --> True
            if attributeInfoType:                     # Continuous Attribute
                instanceValue = state[specifiedIndex]
                if instanceValue == None:
                    return False
                elif self.condition[i][0] < instanceValue < self.condition[i][1]:
#                    print("self.condition[i][0]", self.condition[i][0])
#                    print("instanceValue", instanceValue)
#                    print("self.condition[i][1]", self.condition[i][1])
                    pass
                else:
                    return False

            # Discrete --> False
            else:
                stateRep = state[specifiedIndex]
                if stateRep == self.condition[i]:
                    pass
                elif stateRep == None:
                    return False
                else:
                    return False
        return True

    def equals(self,cl):
        if cl.phenotype == self.phenotype and len(cl.specifiedAttList) == len(self.specifiedAttList):
            clRefs = sorted(cl.specifiedAttList)
            selfRefs = sorted(self.specifiedAttList)
            if clRefs == selfRefs:
                for i in range(len(cl.specifiedAttList)):
                    tempIndex = self.specifiedAttList.index(cl.specifiedAttList[i])
                    if not (cl.condition[i] == self.condition[tempIndex]):
                        return False
                return True
        return False

    def updateExperience(self):
        self.matchCount += 1
        if self.epochComplete:  # Once epoch Completed, number of matches for a unique rule will not change, so do repeat calculation
            pass
        else:
            self.matchCover += 1

    def updateMatchSetSize(self, model,matchSetSize):
        if self.matchCount < 1.0 / model.beta:
            self.aveMatchSetSize = (self.aveMatchSetSize * (self.matchCount-1)+ matchSetSize) / float(self.matchCount)
        else:
            self.aveMatchSetSize = self.aveMatchSetSize + model.beta * (matchSetSize - self.aveMatchSetSize)

    def updateCorrect(self):
        self.correctCount += 1
        if self.epochComplete: #Once epoch Completed, number of correct for a unique rule will not change, so do repeat calculation
            pass
        else:
            self.correctCover += 1

    def updateAccuracy(self):
        self.accuracy = self.correctCount / float(self.matchCount)

    def updateFitness(self,model):
        self.fitness = pow(self.accuracy, model.nu)

    def updateNumerosity(self, num):
        """ Alters the numberosity of the classifier.  Notice that num can be negative! """
        self.numerosity += num

    def isSubsumer(self, model):
        if self.matchCount > model.theta_sub and self.accuracy > model.acc_sub:
            return True
        return False

    def subsumes(self,model,cl):
        return cl.phenotype == self.phenotype and self.isSubsumer(model) and self.isMoreGeneral(model,cl)

    def isMoreGeneral(self,model, cl):
        if len(self.specifiedAttList) >= len(cl.specifiedAttList):
            return False
        for i in range(len(self.specifiedAttList)):
            attributeInfoType = model.env.formatData.attributeInfoType[self.specifiedAttList[i]]
            if self.specifiedAttList[i] not in cl.specifiedAttList:
                return False

            # Continuous
            if attributeInfoType:
                otherRef = cl.specifiedAttList.index(self.specifiedAttList[i])
                if self.condition[i][0] < cl.condition[otherRef][0]:
                    return False
                if self.condition[i][1] > cl.condition[otherRef][1]:
                    return False
        return True

    def updateTimeStamp(self, ts):
        """ Sets the time stamp of the classifier. """
        self.timeStampGA = ts

    def uniformCrossover(self,model,cl):
        p_self_specifiedAttList = copy.deepcopy(self.specifiedAttList)
        p_cl_specifiedAttList = copy.deepcopy(cl.specifiedAttList)

        useAT = model.do_attribute_feedback and random.random() < model.AT.percent

        comboAttList = []
        for i in p_self_specifiedAttList:
            comboAttList.append(i)
        for i in p_cl_specifiedAttList:
            if i not in comboAttList:
                comboAttList.append(i)
            elif not model.env.formatData.attributeInfoType[i]:  # Attribute specified in both parents, and the attribute is discrete (then no reason to cross over)
                comboAttList.remove(i)
        comboAttList.sort()

        changed = False
        for attRef in comboAttList:
            attributeInfoType = model.env.formatData.attributeInfoType[attRef]
            if useAT:
                probability = model.AT.getTrackProb()[attRef]
            else:
                probability = 0.5

            ref = 0
            if attRef in p_self_specifiedAttList:
                ref += 1
            if attRef in p_cl_specifiedAttList:
                ref += 1

            if ref == 0:
                pass
            elif ref == 1:
                if attRef in p_self_specifiedAttList and random.random() > probability:
                    i = self.specifiedAttList.index(attRef)
                    cl.condition.append(self.condition.pop(i))

                    cl.specifiedAttList.append(attRef)
                    self.specifiedAttList.remove(attRef)
                    changed = True

                if attRef in p_cl_specifiedAttList and random.random() < probability:
                    i = cl.specifiedAttList.index(attRef)
                    self.condition.append(cl.condition.pop(i))

                    self.specifiedAttList.append(attRef)
                    cl.specifiedAttList.remove(attRef)
                    changed = True
            else:
                # Continuous Attribute
                if attributeInfoType:
                    i_cl1 = self.specifiedAttList.index(attRef)
                    i_cl2 = cl.specifiedAttList.index(attRef)
                    tempKey = random.randint(0, 3)
                    if tempKey == 0:
                        temp = self.condition[i_cl1][0]
                        self.condition[i_cl1][0] = cl.condition[i_cl2][0]
                        cl.condition[i_cl2][0] = temp
                    elif tempKey == 1:
                        temp = self.condition[i_cl1][1]
                        self.condition[i_cl1][1] = cl.condition[i_cl2][1]
                        cl.condition[i_cl2][1] = temp
                    else:
                        allList = self.condition[i_cl1] + cl.condition[i_cl2]
                        newMin = min(allList)
                        newMax = max(allList)
                        if tempKey == 2:
                            self.condition[i_cl1] = [newMin, newMax]
                            cl.condition.pop(i_cl2)

                            cl.specifiedAttList.remove(attRef)
                        else:
                            cl.condition[i_cl2] = [newMin, newMax]
                            self.condition.pop(i_cl1)

                            self.specifiedAttList.remove(attRef)

                # Discrete Attribute
                else:
                    pass

        #Specification Limit Check
        if len(self.specifiedAttList) > model.rule_specificity_limit:
            self.specLimitFix(model,self)

        if len(cl.specifiedAttList) > model.rule_specificity_limit:
            self.specLimitFix(model,cl)

        tempList1 = copy.deepcopy(p_self_specifiedAttList)
        tempList2 = copy.deepcopy(cl.specifiedAttList)
        tempList1.sort()
        tempList2.sort()
        if changed and (tempList1 == tempList2):
            changed = False
        return changed

    def specLimitFix(self, model, cl):
        """ Lowers classifier specificity to specificity limit. """
        if model.do_attribute_feedback:
            # Identify 'toRemove' attributes with lowest AT scores
            while len(cl.specifiedAttList) > model.rule_specificity_limit:
                minVal = model.AT.getTrackProb()[cl.specifiedAttList[0]]
                minAtt = cl.specifiedAttList[0]
                for j in cl.specifiedAttList:
                    if model.AT.getTrackProb()[j] < minVal:
                        minVal = model.AT.getTrackProb()[j]
                        minAtt = j
                i = cl.specifiedAttList.index(minAtt)  # reference to the position of the attribute in the rule representation
                cl.specifiedAttList.remove(minAtt)
                cl.condition.pop(i)  # buildMatch handles both discrete and continuous attributes

        else:
            # Randomly pick 'toRemove'attributes to be generalized
            toRemove = len(cl.specifiedAttList) - model.rule_specificity_limit
            genTarget = random.sample(cl.specifiedAttList, toRemove)
            for j in genTarget:
                i = cl.specifiedAttList.index(j)  # reference to the position of the attribute in the rule representation
                cl.specifiedAttList.remove(j)
                cl.condition.pop(i)  # buildMatch handles both discrete and continuous attributes

    def setAccuracy(self, acc):
        """ Sets the accuracy of the classifier """
        self.accuracy = acc

    def setFitness(self, fit):
        """  Sets the fitness of the classifier. """
        self.fitness = fit

    def mutation(self, model, state):
        """ 
        Mutates the condition of the classifier. Also handles phenotype mutation.
        This is a niche mutation, ensuring the classifier still matches the current instance.
        """
    
        pressureProb = 0.5  # Probability that if EK is activated, it will be applied.
        useAT = model.do_attribute_feedback and random.random() < model.AT.percent
        changed = False
    
        steps = 0
        keepGoing = True
        while keepGoing:
            if random.random() < model.mu:
                steps += 1
            else:
                keepGoing = False
    
        # Define Spec Limits
        if (len(self.specifiedAttList) - steps) <= 1:
            lowLim = 1
        else:
            lowLim = len(self.specifiedAttList) - steps
    
        if (len(self.specifiedAttList) + steps) >= model.rule_specificity_limit:
            highLim = model.rule_specificity_limit
        else:
            highLim = len(self.specifiedAttList) + steps
    
        if len(self.specifiedAttList) == 0:
            highLim = 1
    
        # Get new rule specificity
        newRuleSpec = random.randint(lowLim, highLim)
    
        # ✅ NEW: Use Ranked Features if `use_feature_ranked_RSL` is ON
        if model.use_feature_ranked_RSL and model.feature_importance_rank is not None:
            #print("⚡ Using Feature-Ranked Rule Specificity Limit (RSL) for Mutation")
    
            ranked_features = model.feature_importance_rank  # Ranked list of features
    
            # MAINTAIN SPECIFICITY
            if newRuleSpec == len(self.specifiedAttList) and random.random() < (1 - model.mu):
                # Remove a feature based on ranking
                if not model.doExpertKnowledge or random.random() > pressureProb:
                    genTarget = random.sample(self.specifiedAttList, 1)  # Select randomly
                else:
                    genTarget = self.selectGeneralizeRW(model, 1)  # Use EK-based selection
    
                if genTarget[0] in ranked_features:  # Ensure removal is from ranked list
                    attributeInfoType = model.env.formatData.attributeInfoType[genTarget[0]]
                    if not attributeInfoType or random.random() > 0.5:
                        if not useAT or random.random() > model.AT.getTrackProb()[genTarget[0]]:
                            i = self.specifiedAttList.index(genTarget[0])
                            self.specifiedAttList.remove(genTarget[0])
                            self.condition.pop(i)
                            changed = True
                    else:
                        self.mutateContinuousAttributes(model, useAT, genTarget[0])
    
                # Add a ranked feature
                if len(self.specifiedAttList) < len(state):
                    available_features = [f for f in ranked_features if f not in self.specifiedAttList]
                    if available_features:
                        specTarget = available_features[:1]  # Select top-ranked available feature
                        if state[specTarget[0]] is not None:
                            self.specifiedAttList.append(specTarget[0])
                            self.condition.append(self.buildMatch(model, specTarget[0], state))
                            changed = True
    
                if len(self.specifiedAttList) > model.rule_specificity_limit:
                    self.specLimitFix(model, self)
    
        else:  # 🔄 FALLBACK TO ORIGINAL LOGIC
            # MAINTAIN SPECIFICITY
            if newRuleSpec == len(self.specifiedAttList) and random.random() < (1 - model.mu):
                # Remove random condition element
                if not model.doExpertKnowledge or random.random() > pressureProb:
                    genTarget = random.sample(self.specifiedAttList, 1)
                else:
                    genTarget = self.selectGeneralizeRW(model, 1)
    
                attributeInfoType = model.env.formatData.attributeInfoType[genTarget[0]]
                if not attributeInfoType or random.random() > 0.5:
                    if not useAT or random.random() > model.AT.getTrackProb()[genTarget[0]]:
                        i = self.specifiedAttList.index(genTarget[0])  # Find position
                        self.specifiedAttList.remove(genTarget[0])
                        self.condition.pop(i)
                        changed = True
                else:
                    self.mutateContinuousAttributes(model, useAT, genTarget[0])
    
                # Add random condition element
                if len(self.specifiedAttList) < len(state):
                    if not model.doExpertKnowledge or random.random() > pressureProb:
                        pickList = list(range(model.env.formatData.numAttributes))
                        for i in self.specifiedAttList:
                            pickList.remove(i)
                        specTarget = random.sample(pickList, 1)
                    else:
                        specTarget = self.selectSpecifyRW(model, 1)
    
                    if state[specTarget[0]] is not None and (not useAT or random.random() < model.AT.getTrackProb()[specTarget[0]]):
                        self.specifiedAttList.append(specTarget[0])
                        self.condition.append(self.buildMatch(model, specTarget[0], state))
                        changed = True
    
                if len(self.specifiedAttList) > model.rule_specificity_limit:
                    self.specLimitFix(model, self)
    
            # Increase Specificity
            elif newRuleSpec > len(self.specifiedAttList):
                change = newRuleSpec - len(self.specifiedAttList)
                if not model.doExpertKnowledge or random.random() > pressureProb:
                    pickList = list(range(model.env.formatData.numAttributes))
                    for i in self.specifiedAttList:
                        pickList.remove(i)
                    specTarget = random.sample(pickList, change)
                else:
                    specTarget = self.selectSpecifyRW(model, change)
    
                for j in specTarget:
                    if state[j] is not None and (not useAT or random.random() < model.AT.getTrackProb()[j]):
                        self.specifiedAttList.append(j)
                        self.condition.append(self.buildMatch(model, j, state))
                        changed = True
    
            # Decrease Specificity
            elif newRuleSpec < len(self.specifiedAttList):
                change = len(self.specifiedAttList) - newRuleSpec
                if not model.doExpertKnowledge or random.random() > pressureProb:
                    genTarget = random.sample(self.specifiedAttList, change)
                else:
                    genTarget = self.selectGeneralizeRW(model, change)
    
                for j in genTarget:
                    attributeInfoType = model.env.formatData.attributeInfoType[j]
                    if not attributeInfoType or random.random() > 0.5:
                        if not useAT or random.random() > model.AT.getTrackProb()[j]:
                            i = self.specifiedAttList.index(j)
                            self.specifiedAttList.remove(j)
                            self.condition.pop(i)
                            changed = True
                    else:
                        self.mutateContinuousAttributes(model, useAT, j)
    
        return changed
        
        
    def selectGeneralizeRW(self,model,count):
        probList = []
        for attribute in self.specifiedAttList:
            probList.append(1/model.EK.scores[attribute])
        if sum(probList) == 0:
            probList = (np.array(probList) + 1).tolist()

        probList = np.array(probList)/sum(probList) #normalize
        return np.random.choice(self.specifiedAttList,count,replace=False,p=probList).tolist()

    # def selectGeneralizeRW(self,model,count):
    #     EKScoreSum = 0
    #     selectList = []
    #     currentCount = 0
    #     specAttList = copy.deepcopy(self.specifiedAttList)
    #     for i in self.specifiedAttList:
    #         # When generalizing, EK is inversely proportional to selection probability
    #         EKScoreSum += 1 / float(model.EK.scores[i] + 1)
    #
    #     while currentCount < count:
    #         choicePoint = random.random() * EKScoreSum
    #         i = 0
    #         sumScore = 1 / float(model.EK.scores[specAttList[i]] + 1)
    #         while choicePoint > sumScore:
    #             i = i + 1
    #             sumScore += 1 / float(model.EK.scores[specAttList[i]] + 1)
    #         selectList.append(specAttList[i])
    #         EKScoreSum -= 1 / float(model.EK.scores[specAttList[i]] + 1)
    #         specAttList.pop(i)
    #         currentCount += 1
    #     return selectList

    def selectSpecifyRW(self,model,count):
        pickList = list(range(model.env.formatData.numAttributes))
        for i in self.specifiedAttList:  # Make list with all non-specified attributes
            pickList.remove(i)

        probList = []
        for attribute in pickList:
            probList.append(model.EK.scores[attribute])
        if sum(probList) == 0:
            probList = (np.array(probList) + 1).tolist()
        probList = np.array(probList) / sum(probList)  # normalize
        return np.random.choice(pickList, count, replace=False, p=probList).tolist()

    # def selectSpecifyRW(self, model,count):
    #     """ EK applied to the selection of an attribute to specify for mutation. """
    #     pickList = list(range(model.env.formatData.numAttributes))
    #     for i in self.specifiedAttList:  # Make list with all non-specified attributes
    #         pickList.remove(i)
    #
    #     EKScoreSum = 0
    #     selectList = []
    #     currentCount = 0
    #
    #     for i in pickList:
    #         # When generalizing, EK is inversely proportional to selection probability
    #         EKScoreSum += model.EK.scores[i]
    #
    #     while currentCount < count:
    #         choicePoint = random.random() * EKScoreSum
    #         i = 0
    #         sumScore = model.EK.scores[pickList[i]]
    #         while choicePoint > sumScore:
    #             i = i + 1
    #             sumScore += model.EK.scores[pickList[i]]
    #         selectList.append(pickList[i])
    #         EKScoreSum -= model.EK.scores[pickList[i]]
    #         pickList.pop(i)
    #         currentCount += 1
    #     return selectList

    def mutateContinuousAttributes(self, model,useAT, j):
        # -------------------------------------------------------
        # MUTATE CONTINUOUS ATTRIBUTES
        # -------------------------------------------------------
        if useAT:
            if random.random() < model.AT.getTrackProb()[j]:  # High AT probability leads to higher chance of mutation (Dives ExSTraCS to explore new continuous ranges for important attributes)
                # Mutate continuous range - based on Bacardit 2009 - Select one bound with uniform probability and add or subtract a randomly generated offset to bound, of size between 0 and 50% of att domain.
                attRange = float(model.env.formatData.attributeInfoContinuous[j][1]) - float(model.env.formatData.attributeInfoContinuous[j][0])
                i = self.specifiedAttList.index(j)  # reference to the position of the attribute in the rule representation
                mutateRange = random.random() * 0.5 * attRange
                if random.random() > 0.5:  # Mutate minimum
                    if random.random() > 0.5:  # Add
                        self.condition[i][0] += mutateRange
                    else:  # Subtract
                        self.condition[i][0] -= mutateRange
                else:  # Mutate maximum
                    if random.random() > 0.5:  # Add
                        self.condition[i][1] += mutateRange
                    else:  # Subtract
                        self.condition[i][1] -= mutateRange
                # Repair range - such that min specified first, and max second.
                self.condition[i].sort()
                changed = True
        elif random.random() > 0.5:
            # Mutate continuous range - based on Bacardit 2009 - Select one bound with uniform probability and add or subtract a randomly generated offset to bound, of size between 0 and 50% of att domain.
            attRange = float(model.env.formatData.attributeInfoContinuous[j][1]) - float(model.env.formatData.attributeInfoContinuous[j][0])
            i = self.specifiedAttList.index(j)  # reference to the position of the attribute in the rule representation
            mutateRange = random.random() * 0.5 * attRange
            if random.random() > 0.5:  # Mutate minimum
                if random.random() > 0.5:  # Add
                    self.condition[i][0] += mutateRange
                else:  # Subtract
                    self.condition[i][0] -= mutateRange
            else:  # Mutate maximum
                if random.random() > 0.5:  # Add
                    self.condition[i][1] += mutateRange
                else:  # Subtract
                    self.condition[i][1] -= mutateRange
            # Repair range - such that min specified first, and max second.
            self.condition[i].sort()
            changed = True
        else:
            pass


    def rangeCheck(self,model):
        """ Checks and prevents the scenario where a continuous attributes specified in a rule has a range that fully encloses the training set range for that attribute."""
        for attRef in self.specifiedAttList:
            if model.env.formatData.attributeInfoType[attRef]: #Attribute is Continuous
                trueMin = model.env.formatData.attributeInfoContinuous[attRef][0]
                trueMax = model.env.formatData.attributeInfoContinuous[attRef][1]
                i = self.specifiedAttList.index(attRef)
                valBuffer = (trueMax-trueMin)*0.1
                if self.condition[i][0] <= trueMin and self.condition[i][1] >= trueMax: # Rule range encloses entire training range
                    self.specifiedAttList.remove(attRef)
                    self.condition.pop(i)
                    return
                elif self.condition[i][0]+valBuffer < trueMin:
                    self.condition[i][0] = trueMin - valBuffer
                elif self.condition[i][1]- valBuffer > trueMax:
                    self.condition[i][1] = trueMin + valBuffer
                else:
                    pass

    def getDelProp(self, model, meanFitness):
        """  Returns the vote for deletion of the classifier. """
        if self.fitness / self.numerosity >= model.delta * meanFitness or self.matchCount < model.theta_del:
            deletionVote = self.aveMatchSetSize * self.numerosity
        elif self.fitness == 0.0:
            deletionVote = self.aveMatchSetSize * self.numerosity * meanFitness / (model.init_fitness / self.numerosity)
        else:
            deletionVote = self.aveMatchSetSize * self.numerosity * meanFitness / (self.fitness / self.numerosity)
        return deletionVote

   

    def calculateEuclideanDistance(self, model, state):
        """ Calculate Euclidean distance between the current classifier's condition and a given state. """
        distance = 0
        for i, attRef in enumerate(self.specifiedAttList):

            # Retrieve the midpoint for the current attribute from the pre-calculated midpoints
            midpoint = model.midpoints.get(attRef, None)
#            print(f"midpoint of {attRef} is {midpoint}")            

            if midpoint is not None:
                # Calculate squared difference (Euclidean distance for each attribute)
                distance += (state[attRef] - midpoint) ** 2
            else:
                # If midpoint is not available, handle the case (perhaps skip the attribute or handle differently)
                print(f"Warning: Midpoint for attribute {attRef} not found. State{state}")
#                print("state is", state)
    
        # Return the square root of the sum of squared differences
#        print("SQRT of distance-->",distance ** 0.5)
        return distance ** 0.5
