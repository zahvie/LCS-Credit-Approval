import csv
import numpy as np
import os
from cvxpy.atoms import length

class IterationRecord():
    '''
    IterationRecord Tracks 1 dictionary:
    1) Tracking Dict: Cursory Iteration Evaluation. Frequency determined by trackingFrequency param in ExSTraCS. For each iteration evaluated, it saves:
        KEY-iteration number
        0-accuracy (approximate from correct array in ExSTraCS)
        1-average population generality
        2-macropopulation size
        3-micropopulation size
        4-match set size
        5-correct set size
        6-average iteration age of correct set classifiers
        7-number of classifiers subsumed (in iteration)
        8-number of crossover operations performed (in iteration)
        9-number of mutation operations performed (in iteration)
        10-number of covering operations performed (in iteration)
        11-number of deleted macroclassifiers performed (in iteration)
        12-number of rules removed via compaction
        13-total global time at end of iteration
        14-total matching time at end of iteration
        15-total covering time at end of iteration
        16-total crossover time at end of iteration
        17-total mutation time at end of iteration
        18-total AT time at end of iteration
        19-total EK time at end of iteration
        20-total init time at end of iteration
        21-total add time at end of iteration
        22-total RC time at end of iteration
        23-total deletion time at end of iteration
        24-total subsumption time at end of iteration
        25-total selection time at end of iteration
        26-total evaluation time at end of iteration
        27-self.distance_threshold
        28-self.classifier_distance
    '''

    def __init__(self):
        self.trackingDict = {}

    def addToTracking(self,iterationNumber,accuracy,avgPopGenerality,macroSize,microSize,mSize,cSize,iterAvg,
                      subsumptionCount,crossoverCount,mutationCount,coveringCount,deletionCount,RCCount,
                      globalTime,matchingTime,coveringTime,crossoverTime,mutationTime,ATTime,EKTime,initTime,addTime,
                      RCTime,deletionTime,subsumptionTime,selectionTime,evaluationTime,distance_threshold):

        self.trackingDict[iterationNumber] = [accuracy,avgPopGenerality,macroSize,microSize,mSize,cSize,iterAvg,
                                   subsumptionCount,crossoverCount,mutationCount,coveringCount,deletionCount,RCCount,
                                   globalTime,matchingTime,coveringTime,crossoverTime,mutationTime,ATTime,EKTime,initTime,
                                   addTime,RCTime,deletionTime,subsumptionTime,selectionTime,evaluationTime,distance_threshold]


    def exportTrackingToCSV(self,filename=r'D:\Python\Thesis\ExSTraCS\test\Logs\iterationData.csv'):
        #Exports each entry in Tracking Array as a column
        with open(filename,mode='w', newline='') as file: # Add newline=''
            writer = csv.writer(file,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)

            writer.writerow(["Iteration","Accuracy (approx)", "Average Population Generality","Macropopulation Size",
                             "Micropopulation Size", "Match Set Size", "Correct Set Size", "Average Iteration Age of Correct Set Classifiers",
                             "# Classifiers Subsumed in Iteration","# Crossover Operations Performed in Iteration","# Mutation Operations Performed in Iteration",
                             "# Covering Operations Performed in Iteration","# Deletion Operations Performed in Iteration","# Rules Removed via Rule Compaction",
                             "Total Global Time","Total Matching Time","Total Covering Time","Total Crossover Time",
                             "Total Mutation Time","Total Attribute Tracking Time","Total Expert Knowledge Time","Total Model Initialization Time",
                             "Total Classifier Add Time","Total Rule Compaction Time",
                             "Total Deletion Time","Total Subsumption Time","Total Selection Time","Total Evaluation Time","Distance Threshold","Classifer Distance"])


            for k, v in sorted(self.trackingDict.items()):
                writer.writerow([k] + v)

#            for k,v in sorted(self.trackingDict.items()):
#                writer.writerow([k,v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10],v[11],v[12],v[13],v[14],v[15],v[16],v[17],v[18],v[19],v[20],v[21],v[22],v[23],v[24],v[25],v[26]])
            print(f"Tracking data saved to {filename}")
        file.close()

    def exportPop(self,model,popSet,headerNames=np.array([]),className='Class',filename='D:\Python\Thesis\ExSTraCS\test\Logs\populationData.csv'):
        numAttributes = model.env.formatData.numAttributes

        headerNames = headerNames.tolist() #Convert to Python List

        #Default headerNames if none provided
        if len(headerNames) == 0:
            for i in range(numAttributes):
                headerNames.append("N"+str(i))

        if len(headerNames) != numAttributes:
            raise Exception("# of Header Names provided does not match the number of attributes in dataset instances - exportPop.")

        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            writer.writerow(headerNames+[className]+["Fitness","Accuracy","Numerosity","Avg Match Set Size","TimeStamp GA","Iteration Initialized","Specificity","Deletion Probability","Correct Count","Match Count","Epoch Complete"])
            classifiers = popSet
            for classifier in classifiers:
                a = []
                for attributeIndex in range(numAttributes):
                    if attributeIndex in classifier.specifiedAttList:
                        specifiedLocation = classifier.specifiedAttList.index(attributeIndex)
                        if not isinstance(classifier.condition[specifiedLocation],list): #if discrete
                            a.append(classifier.condition[specifiedLocation])
                        else: #if continuous
                            conditionCont = classifier.condition[specifiedLocation] #cont array [min,max]
                            s = str(conditionCont[0])+","+str(conditionCont[1])
                            a.append(s)
                    else:
                        a.append("#")

                if isinstance(classifier.phenotype,list):
                    s = str(classifier.phenotype[0])+","+str(classifier.phenotype[1])
                    a.append(s)
                else:
                    a.append(classifier.phenotype)
                a.append(classifier.fitness)
                a.append(classifier.accuracy)
                a.append(classifier.numerosity)
                a.append(classifier.aveMatchSetSize)
                a.append(classifier.timeStampGA)
                a.append(classifier.initTimeStamp)
                a.append(len(classifier.specifiedAttList)/numAttributes)
                a.append(classifier.deletionProb)
                a.append(classifier.correctCount)
                a.append(classifier.matchCount)
                a.append(classifier.epochComplete)
                writer.writerow(a)
        file.close()

    def exportPopPatches(self,model,popSet,headerNames=np.array([]),className='Class',filename='D:\Python\Thesis\ExSTraCS\test\Logs\populationData.csv'):
        
        #self,popSet=self.population.popSet,filename=pop_file_name
        numAttributes = model.env.formatData.numAttributes

        headerNames = headerNames.tolist() #Convert to Python List
        print("numAttributes - ", numAttributes)
        print("headerNames - ",headerNames)

        #Default headerNames if none provided
        if len(headerNames) == 0:
            for i in range(model.total_patches):
                for j in range(model.patch_len):
                    patch_no = "P"+str(i)
                    lower_cond = "N"+str(j*2)
                    upper_cond = "N"+str(j*2+1)
                    headerNames.append(patch_no+lower_cond)
                    headerNames.append(patch_no+upper_cond)

        if len(headerNames) != 2*numAttributes:
            print("# of Header Names provided does not match the number of attributes in dataset instances - exportPopPatches.")
            #raise Exception("# of Header Names provided does not match the number of attributes in dataset instances.")

        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            writer.writerow(headerNames+[className]+["Fitness","Accuracy","Numerosity","Avg Match Set Size","TimeStamp GA","Iteration Initialized","Specificity","Deletion Probability","Correct Count","Match Count","Epoch Complete"])
            classifiers = popSet
            for classifier in classifiers:
                a = []
                for ptch_no in range(model.total_patches):
                    if ptch_no in classifier.specifiedAttList:
                        for p_idx in range(model.patch_len):
                            s=str(classifier.condition[ptch_no][p_idx][0])
                            a.append(s)
                            s=str(classifier.condition[ptch_no][p_idx][1])
                            a.append(s)
                    else:
                        for p_idx in range(model.patch_len):
                            a.append("#")
                            a.append("#") 
                
                a.append(classifier.phenotype)
                a.append(classifier.fitness)
                a.append(classifier.accuracy)
                a.append(classifier.numerosity)
                a.append(classifier.aveMatchSetSize)
                a.append(classifier.timeStampGA)
                a.append(classifier.initTimeStamp)
                a.append(len(classifier.specifiedAttList)/numAttributes)
                a.append(classifier.deletionProb)
                a.append(classifier.correctCount)
                a.append(classifier.matchCount)
                a.append(classifier.epochComplete)
                writer.writerow(a)
        file.close()
        
    def exportPopDCAL(self,model,popSet,headerNames=np.array([]),className='Class',filename='D:\Python\Thesis\ExSTraCS\test\Logs\populationData.csv'):

        numAttributes = model.env.formatData.numAttributes

        headerNames = headerNames.tolist() #Convert to Python List

        #Default headerNames if none provided
        if len(headerNames) == 0:
            for i in range(numAttributes):
                headerNames.append("N"+str(i))

        if len(headerNames) != numAttributes:
            raise Exception("# of Header Names provided does not match the number of attributes in dataset instances - exportPopDCAL.")

        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            writer.writerow(["Specified Values","Specified Attribute Names"]+[className]+["Fitness","Accuracy","Numerosity","Avg Match Set Size","TimeStamp GA","Iteration Initialized","Specificity","Deletion Probability","Correct Count","Match Count","Epoch Complete"])

            classifiers = popSet
            for classifier in classifiers:
                a = []

                #Add attribute information
                headerString = ""
                valueString = ""
                for attributeIndex in range(numAttributes):
                    if attributeIndex in classifier.specifiedAttList:
                        specifiedLocation = classifier.specifiedAttList.index(attributeIndex)
                        headerString+=str(headerNames[attributeIndex])+", "
                        if not isinstance(classifier.condition[specifiedLocation],list): #if discrete
                            valueString+= str(classifier.condition[specifiedLocation])+", "
                        else: #if continuous
                            conditionCont = classifier.condition[specifiedLocation] #cont array [min,max]
                            s = "["+str(conditionCont[0])+","+str(conditionCont[1])+"]"
                            valueString+= s+", "

                a.append(valueString[:-2])
                a.append(headerString[:-2])

                #Add phenotype information
                if isinstance(classifier.phenotype, list):
                    s = str(classifier.phenotype[0]) + "," + str(classifier.phenotype[1])
                    a.append(s)
                else:
                    a.append(classifier.phenotype)

                #Add statistics
                a.append(classifier.fitness)
                a.append(classifier.accuracy)
                a.append(classifier.numerosity)
                a.append(classifier.aveMatchSetSize)
                a.append(classifier.timeStampGA)
                a.append(classifier.initTimeStamp)
                a.append(len(classifier.specifiedAttList) / numAttributes)
                a.append(classifier.deletionProb)
                a.append(classifier.correctCount)
                a.append(classifier.matchCount)
                a.append(classifier.epochComplete)
                writer.writerow(a)
        file.close()

    def trackPopulationData(self, population, iterationCount, movingAvgCount):
        """
        Tracks the population-related data for the iteration and adds it to the tracking dictionary.
        """
        if iterationCount % movingAvgCount == 0:
            popNumerosity = sum([pop.numerosity for pop in population.popSet])  # Assuming population has popSet
            macropopulation_size = len(population.popSet)  # Assuming population contains classifiers
            micropopulation_size = len(population.popSet)  # Refine as per the actual micropopulation size logic
            total_classifiers = len(population.popSet)
            total_match_set_size = sum([pop.numerosity for pop in population.popSet])
            total_correct_set_size = sum([pop.correctCount for pop in population.popSet])
            total_subsumed_classifiers = sum([1 for pop in population.popSet if pop.isSubsumer(model)])  # Example method for subsumed classifiers
            total_crossover_operations = sum([pop.matchCount for pop in population.popSet])  # Example method for crossover operations
            total_mutation_operations = sum([pop.correctCount for pop in population.popSet])  # Example method for mutation operations
            total_covering_operations = sum([pop.matchCover for pop in population.popSet])  # Example method for covering operations
            total_deletion_operations = sum([pop.deletionProb for pop in population.popSet])  # Example method for deletion operations

            # Store the tracking data in trackingDict
            self.trackingDict[iterationCount] = [
                popNumerosity, macropopulation_size, micropopulation_size, total_classifiers, total_match_set_size,
                total_correct_set_size, total_subsumed_classifiers, total_crossover_operations, total_mutation_operations,
                total_covering_operations, total_deletion_operations
            ]
            print(f"Population data for iteration {iterationCount} saved.")