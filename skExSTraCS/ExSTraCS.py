from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score
import numpy as np
import pandas as pd
from skExSTraCS.Timer import Timer
from skExSTraCS.OfflineEnvironment import OfflineEnvironment
from skExSTraCS.ExpertKnowledge import ExpertKnowledge
from skExSTraCS.AttributeTracking import AttributeTracking
from skExSTraCS.ClassifierSet import ClassifierSet
from skExSTraCS.Prediction import Prediction
from skExSTraCS.RuleCompaction import RuleCompaction
from skExSTraCS.IterationRecord import IterationRecord
from sklearn.feature_selection import mutual_info_classif  # to compute feature importance based on mutual information (MI)
from scipy.spatial.distance import euclidean
import copy
import time
import pickle
import random
import logging
from datetime import datetime
import os
from sympy.logic.boolalg import false

# from Cython.Includes.cpython import type

#log_dir = r'D:\Python\Thesis\ExSTraCS\test\Logs'
print('New Development Source')

class ExSTraCS(BaseEstimator, ClassifierMixin):

    def __init__(self, learning_iterations=100000, N=1000, nu=1, chi=0.8, mu=0.04, theta_GA=25, theta_del=20, theta_sub=20,
                 acc_sub=0.99, beta=0.2, delta=0.1, init_fitness=0.01, fitness_reduction=0.1, theta_sel=0.5, rule_specificity_limit=None,
                 do_correct_set_subsumption=False, do_GA_subsumption=True, selection_method='tournament', do_attribute_tracking=False,
                 do_attribute_feedback=False, attribute_tracking_method='add', attribute_tracking_beta=0.1, expert_knowledge=None,
                 rule_compaction='QRF', reboot_filename=None, discrete_attribute_limit=10, specified_attributes=np.array([]),
                 track_accuracy_while_fit=True, random_state=None, total_patches=25, patch_len=-1, init_best_pop=False, log_dir="",
                 use_feature_ranked_RSL=False, log_trainingfile_name="log_trainingfile_name.csv", log_popfile_name="",
                 feature_selection_percentage=0.75, all_feature_list=None, use_midpoint_distance_filter=False, midpoints={},
                 distance_threshold=0, classifier_distance=0):
        '''
        :param learning_iterations:          Must be nonnegative integer. The number of training cycles to run.
        :param N:                           Must be nonnegative integer. Maximum micro classifier population size (sum of classifier numerosities).
        :param nu:                          Must be a float. Power parameter (v) used to determine the importance of high accuracy when calculating fitness.
        :param chi:                         Must be float from 0 - 1. The probability of applying crossover in the GA (X).
        :param mu:                     Must be float from 0 - 1. The probability (u) of mutating an allele within an offspring
        :param theta_GA:                    Must be nonnegative float. The GA threshold. The GA is applied in a set when the average time (# of iterations) since the last GA in the correct set is greater than theta_GA.
        :param theta_del:                   Must be a nonnegative integer. The deletion experience threshold; The calculation of the deletion probability changes once this threshold is passed.
        :param theta_sub:                   Must be a nonnegative integer. The subsumption experience threshold
        :param acc_sub:                     Must be float from 0 - 1. Subsumption accuracy requirement
        :param beta:                        Must be float. Learning parameter; Used in calculating average correct set size
        :param delta:                       Must be float. Deletion parameter; Used in determining deletion vote calculation.
        :param init_fitness:                Must be float. The initial fitness for a new classifier. (typically very small, approaching but not equal to zero)
        :param fitness_reduction:            Must be float. Initial fitness reduction in GA offspring rules.
        :param theta_sel:                   Must be float from 0 - 1. The fraction of the correct set to be included in tournament selection.
        :param rule_specificity_limit:        Must be nonnegative integer or None. If not None, overrides the automatically computed RSL.
        :param do_correct_set_subsumption:     Must be boolean. Determines if subsumption is done in [C] after [C] updates.
        :param do_GA_subsumption:             Must be boolean. Determines if subsumption is done between offspring and parents after GA
        :param selection_method:             Must be either "tournament" or "roulette". Determines GA selection method. Recommended: tournament
        :param do_attribute_tracking:         Must be boolean. Whether to have AT
        :param do_attribute_feedback:         Must be boolean. Whether to have AF
        :param attribute_tracking_method:   Must be 'add' or 'wh' (widrow hoff). Update AT method
        :param attribute_tracking_beta:     Must be float
        :param expert_knowledge:             Must be np.ndarray or list or None. If None, don't use EK. If not, attribute filter scores are the array (in order)
        :param rule_compaction:              Must be None or String. QRF or PDRC or QRC or CRA2 or Fu2 or Fu1. If None, no rule compaction is done
        :param reboot_filename:              Must be String or None. Filename of pickled model to be rebooted
        :param discrete_attribute_limit:      Must be nonnegative integer OR "c" OR "d". Multipurpose param. If it is a nonnegative integer, discrete_attribute_limit determines the threshold that determines
                                            if an attribute will be treated as a continuous or discrete attribute. For example, if discrete_attribute_limit == 10, if an attribute has more than 10 unique
                                            values in the dataset, the attribute will be continuous. If the attribute has 10 or less unique values, it will be discrete. Alternatively,
                                            discrete_attribute_limit can take the value of "c" or "d". See next param for this.
        :param specified_attributes:         Must be an ndarray type of nonnegative integer attributeIndices (zero indexed).
                                            If "c", attributes specified by index in this param will be continuous and the rest will be discrete. If "d", attributes specified by index in this
                                            param will be discrete and the rest will be continuous.
                                            If this value is given, and discrete_attribute_limit is not "c" or "d", discrete_attribute_limit overrides this specification
        :param track_accuracy_while_fit     Must be boolean. Determines if accuracy is tracked during model training
        :param random_state:                Must be an integer or None. Set a constant random seed value to some integer
        :param use_feature_ranked_RSL:      Must be boolean. Determines if use_feature_ranked_RSL is used in the calculation or not
        :param feature_selection_percentage:Must be a float. Use to calculate a feature_selection_percentage
        '''

        print("Log Dir Name --> ", log_dir)

        # learning_iterations
        if not self.checkIsInt(learning_iterations):
            raise Exception("learning_iterations param must be nonnegative integer")

        if learning_iterations < 0:
            raise Exception("learning_iterations param must be nonnegative integer")

        #N
        if not self.checkIsInt(N):
            raise Exception("N param must be nonnegative integer")

        if N < 0:
            raise Exception("N param must be nonnegative integer")

        #nu
        if not self.checkIsFloat(nu):
            raise Exception("nu param must be float")

        #chi
        if not self.checkIsFloat(chi):
            raise Exception("chi param must be float from 0 - 1")

        if chi < 0 or chi > 1:
            raise Exception("chi param must be float from 0 - 1")

        #mu
        if not self.checkIsFloat(mu):
            raise Exception("mu param must be float from 0 - 1")

        if mu < 0 or mu > 1:
            raise Exception("mu param must be float from 0 - 1")

        #theta_GA
        if not self.checkIsFloat(theta_GA):
            raise Exception("theta_GA param must be nonnegative float")

        if theta_GA < 0:
            raise Exception("theta_GA param must be nonnegative float")

        #theta_del
        if not self.checkIsInt(theta_del):
            raise Exception("theta_del param must be nonnegative integer")

        if theta_del < 0:
            raise Exception("theta_del param must be nonnegative integer")

        #theta_sub
        if not self.checkIsInt(theta_sub):
            raise Exception("theta_sub param must be nonnegative integer")

        if theta_sub < 0:
            raise Exception("theta_sub param must be nonnegative integer")

        #acc_sub
        if not self.checkIsFloat(acc_sub):
            raise Exception("acc_sub param must be float from 0 - 1")

        if acc_sub < 0 or acc_sub > 1:
            raise Exception("acc_sub param must be float from 0 - 1")

        #beta
        if not self.checkIsFloat(beta):
            raise Exception("beta param must be float")

        #delta
        if not self.checkIsFloat(delta):
            raise Exception("delta param must be float")

        #init_fitness
        if not self.checkIsFloat(init_fitness):
            raise Exception("init_fitness param must be float")

        #fitness_reduction
        if not self.checkIsFloat(fitness_reduction):
            raise Exception("fitness_reduction param must be float")

        #theta_sel
        if not self.checkIsFloat(theta_sel):
            raise Exception("theta_sel param must be float from 0 - 1")

        if theta_sel < 0 or theta_sel > 1:
            raise Exception("theta_sel param must be float from 0 - 1")

        #rule_specificity_limit
        if rule_specificity_limit != None:
            if not self.checkIsInt(rule_specificity_limit):
                raise Exception("rule_specificity_limit param must be nonnegative integer or None")

            if rule_specificity_limit < 0:
                raise Exception("rule_specificity_limit param must be nonnegative integer or None")

        #do_correct_set_subsumption
        if not(isinstance(do_correct_set_subsumption,bool)):
            raise Exception("do_correct_set_subsumption param must be boolean")

        #do_GA_subsumption
        if not (isinstance(do_GA_subsumption, bool)):
            raise Exception("do_GA_subsumption param must be boolean")

        #selection_method
        if selection_method != "tournament" and selection_method != "roulette":
            raise Exception("selection_method param must be 'tournament' or 'roulette'")

        #do_attribute_tracking
        if not(isinstance(do_attribute_tracking,bool)):
            raise Exception("do_attribute_tracking param must be boolean")

        #do_attribute_feedback
        if not(isinstance(do_attribute_feedback,bool)):
            raise Exception("do_attribute_feedback param must be boolean")

        #attribute_tracking_method
        if attribute_tracking_method != 'add' and attribute_tracking_method != 'wh':
            raise Exception("attribute_tracking_method param must be 'add' or 'wh'")

        # attribute_tracking_beta
        if not self.checkIsFloat(attribute_tracking_beta):
            raise Exception("attribute_tracking_beta param must be float")

        #expert_knowledge
        if not (isinstance(expert_knowledge, np.ndarray)) and not (isinstance(expert_knowledge, list)) and expert_knowledge != None:
            raise Exception("expert_knowledge param must be None or list/ndarray")
        if isinstance(expert_knowledge,np.ndarray):
            expert_knowledge = expert_knowledge.tolist()

        #rule_compaction
        if rule_compaction != None and rule_compaction != 'QRF' and rule_compaction != 'PDRC' and rule_compaction != 'QRC' and rule_compaction != 'CRA2' and rule_compaction != 'Fu2' and rule_compaction != 'Fu1':
            raise Exception("rule_compaction param must be None or 'QRF' or 'PDRC' or 'QRC' or 'CRA2' or 'Fu2' or 'Fu1'")

        #reboot_filename
        if reboot_filename != None and not isinstance(reboot_filename, str):
            raise Exception("reboot_filename param must be None or String from pickle")

        #discrete_attribute_limit
        if discrete_attribute_limit != "c" and discrete_attribute_limit != "d":
            try:
                dpl = int(discrete_attribute_limit)
                if not self.checkIsInt(discrete_attribute_limit):
                    raise Exception("discrete_attribute_limit param must be nonnegative integer or 'c' or 'd'")
                if dpl < 0:
                    raise Exception("discrete_attribute_limit param must be nonnegative integer or 'c' or 'd'")
            except:
                raise Exception("discrete_attribute_limit param must be nonnegative integer or 'c' or 'd'")

        #specified_attributes
        if not (isinstance(specified_attributes,np.ndarray)):
            raise Exception("specified_attributes param must be ndarray")

        for spAttr in specified_attributes:
            if not self.checkIsInt(spAttr):
                raise Exception("All specified_attributes elements param must be nonnegative integers")
            if int(spAttr) < 0:
                raise Exception("All specified_attributes elements param must be nonnegative integers")

        #track_accuracy_while_fit
        if not(isinstance(track_accuracy_while_fit,bool)):
            raise Exception("track_accuracy_while_fit param must be boolean")

        #random_state
        if random_state != None:
            try:
                if not self.checkIsInt(random_state):
                    raise Exception("random_state param must be integer")
            except:
                raise Exception("random_state param must be integer or None")

        self.learning_iterations = learning_iterations
        self.N = N
        self.nu = nu
        self.chi = chi
        self.mu = mu
        self.theta_GA = theta_GA
        self.theta_del = theta_del
        self.theta_sub = theta_sub
        self.acc_sub = acc_sub
        self.beta = beta
        self.delta = delta
        self.init_fitness = init_fitness
        self.fitness_reduction = fitness_reduction
        self.theta_sel = theta_sel
        self.rule_specificity_limit = rule_specificity_limit
        self.do_correct_set_subsumption = do_correct_set_subsumption
        self.do_GA_subsumption = do_GA_subsumption
        self.selection_method = selection_method
        self.do_attribute_tracking = do_attribute_tracking
        self.attribute_tracking_beta = attribute_tracking_beta
        if self.do_attribute_tracking == False:
            self.do_attribute_feedback = False
        else:
            self.do_attribute_feedback = do_attribute_feedback
        if not (isinstance(expert_knowledge, np.ndarray)) and not (isinstance(expert_knowledge, list)):
            self.doExpertKnowledge = False
        else:
            self.doExpertKnowledge = True
        self.attribute_tracking_method = attribute_tracking_method
        self.expert_knowledge = expert_knowledge
        self.rule_compaction = rule_compaction
        self.reboot_filename = reboot_filename
        self.discrete_attribute_limit = discrete_attribute_limit
        self.specified_attributes = specified_attributes
        self.track_accuracy_while_fit = track_accuracy_while_fit
        self.random_state = random_state
        self.use_feature_ranked_RSL = use_feature_ranked_RSL  # # This controls which method to use
        self.feature_importance_rank = None  # Feature importance ranking (to be assigned in fit())

        self.hasTrained = False
        self.feature_selection_percentage = 0.75
        self.all_feature_list = None
        self.use_midpoint_distance_filter = use_midpoint_distance_filter
        self.midpoints = {}
        self.distance_threshold = 0
        self.trackingObj = TempTrackingObj()
        self.record = IterationRecord()

        #Reboot Population
        if self.reboot_filename != None:
            self.rebootPopulation()
            self.hasTrained = True
        else:
            self.iterationCount = 0
            self.population = ClassifierSet()

        self.patch_len = patch_len
        self.total_patches = total_patches
        self.init_best_pop = init_best_pop

        self.log_popfile_path = os.path.join(log_dir, log_popfile_name)
        log_trainingfile_path = os.path.join(log_dir, log_trainingfile_name)

        self.log_trainingfile = open(log_trainingfile_path, 'w', newline='')
        print("Printing Header -", "IterationNo Accuracy PopulationNumerosity Population Theshold classifier_distance\n")
        self.log_trainingfile.write("IterationNo,Accuracy,PopulationNumerosity,Population,matchSetSize,correctSetSize,macroPopSize,microPopSize,avgIterAge,subsumptionCount,crossOverCount,mutationCount,coveringCount,deletionCount,threshold,classifer_distance\n")

    def checkIsInt(self, num):
        try:
            n = float(num)
            if num - int(num) == 0:
                return True
            else:
                return False
        except:
            return False

    def checkIsFloat(self, num):
        try:
            n = float(num)
            return True
        except:
            return False

    ##*************** Fit ****************
    def fit(self, X, y):
        """Scikit-learn required: Supervised training of exstracs
             Parameters
            X: array-like {n_samples, n_features} Training instances. ALL INSTANCE ATTRIBUTES MUST BE NUMERIC or NAN
            y: array-like {n_samples} Training labels. ALL INSTANCE PHENOTYPES MUST BE NUMERIC NOT NAN OR OTHER TYPE
            Returns self
        """

        # Check if X and Y are numeric
        try:
            for instance in X:
                for value in instance:
                    if not (np.isnan(value)):
                        float(value)
            for value in y:
                float(value)
        except:
            raise Exception("X and y must be fully numeric")

        # Handle repeated fit calls
        if self.learning_iterations == self.iterationCount and self.reboot_filename != None:
            raise Exception("You cannot call fit(X,y) a second time with a rebooted population.")

        if self.random_state != None:
            random.seed(int(self.random_state))
            np.random.seed(int(self.random_state))

        if self.reboot_filename == None and self.hasTrained:
            self.iterationCount = 0
            self.population = ClassifierSet()

        # Reboot Timer, if necessary
        if self.reboot_filename == None:
            self.timer = Timer()
        else:
            self.rebootTimer()

        ################## Compute feature importance #######################

        # Compute feature importance before training
#        important_features = self.get_feature_importance(X, y)
#        feature_importance_rank, feature_importance_df = self.get_feature_importance(X, y, percentage=0.30)
#        print("important_features from get_feature_importance(X, y) - ", feature_importance_rank)
#        print("type(important_features) - ",type(important_features))
        if self.use_feature_ranked_RSL:
            self.feature_importance_rank, _ = self.get_feature_importance(X, y)  # Step 1: Compute feature importance before training
#        print("self.feature_importance_rank Saved in - ", self.feature_importance_rank )
#        print("important_features count- IMPORTANT FEATURES",len(self.feature_importance_rank))

        # Create a sorted ranking of features (highest MI first)
        # feature_ranks = np.argsort(-important_features)  # Negative for descending order
        
        print("important_features - ", self.feature_importance_rank)
        print("all_features - ", self.all_feature_list)
        # Select the top 30% of features as the dynamic RSL
        # self.rule_specificity_limit = max(1, int(len(important_features) * 0.3))
        
        # print("important_features OUTPUT", important_features)
        # print(f"Dynamic Rule Specificity Limit (RSL) set to: {self.rule_specificity_limit}")

        ################### End of Computing feature importance #####################

        self.timer.startTimeInit()

        # Set up offline environment w/ dataset
        self.env = OfflineEnvironment(X,y,self)

        # Set up tracking metrics
        self.trackingAccuracy = []
        self.movingAvgCount = 100
        aveGenerality = 0
        aveGeneralityFreq = min(self.env.formatData.numTrainInstances, 1000)

        if self.doExpertKnowledge:
            if len(self.expert_knowledge) != self.env.formatData.numAttributes:
                raise Exception("length of expertKnowledge param must match the # of data instance attributes")

        if self.doExpertKnowledge:
            self.timer.startTimeEK()
            self.EK = ExpertKnowledge(self)
            self.timer.stopTimeEK()
        if self.do_attribute_tracking and (self.reboot_filename == None or (self.reboot_filename != None and self.AT == None)):
            self.timer.startTimeAT()
            self.AT = AttributeTracking(self)
            self.timer.stopTimeAT()
        elif not self.do_attribute_tracking and self.reboot_filename == None:
            self.AT = None
        self.timer.stopTimeInit()
        
		##############################################################################
        # Initialize clfrs population at the start
        pop_count = 0
        if self.init_best_pop == True:
            for img_idx, img_best_parts in imgs_best_samples.items():
                state_phenotype = self.env.getTrainInstanceAtIdx(img_idx)
                self.population.initClfr4BestPop(self, state_phenotype, img_best_parts)
                # Increment pop count
                pop_count += 1
                
        '''
        #def implementation
        while pop_count < self.init_pop:
            state_phenotype = self.env.getTrainInstanceRandom()
            self.population.initClfr4Pop(self,state_phenotype)
            # Increment pop count
            pop_count += 1
        '''
        
        pop_len_temp = len(self.population.popSet)
        pop_intrvl = self.learning_iterations / 4
		##############################################################################
        
        while self.iterationCount < self.learning_iterations:
            state_phenotype = self.env.getTrainInstance()
            self.runIteration(state_phenotype)

            self.timer.startTimeEvaluation()
            if self.iterationCount % aveGeneralityFreq == aveGeneralityFreq - 1:
                aveGenerality = self.population.getAveGenerality(self)

            if len(self.trackingAccuracy) != 0:
                accuracy = sum(self.trackingAccuracy)/len(self.trackingAccuracy)
            else:
                accuracy = 0

            className = ""
            headerName = ""
            pop_data_file_name = r'D:\Python\Thesis\ExSTraCS\test\Logs\populationDataDetail.csv'

            self.record.exportPop(model=self, popSet=self.population.popSet, headerNames=np.array([]), className=className, filename=pop_data_file_name)

#            print("******self.distance_threshold-In FIT-->",self.distance_threshold)
#            print("******model.distance_threshold-In FIT-->",model.distance_threshold)
            self.trackingObj.distance_threshold = self.distance_threshold
            ###############################################################################################
            if self.iterationCount % self.movingAvgCount == 0:
                popNmrsty = 0
                for i in range(len(self.population.popSet)):
                    popNmrsty += self.population.popSet[i].numerosity
#                print("self.trackingObj.distance_threshold -->",self.trackingObj.distance_threshold)
#                print("model.distance_threshold", model.distance_threshold)
                self.log_trainingfile.write("{:d},{:.4f},{:d},{:d},{:d},{:d},{:d},{:d},{:.4f},{:d},{:d},{:d},{:d},{:d},{:.4f}\n".format(self.iterationCount, accuracy, popNmrsty, len(self.population.popSet), self.trackingObj.matchSetSize, self.trackingObj.correctSetSize, self.trackingObj.macroPopSize, self.trackingObj.microPopSize, self.trackingObj.avgIterAge, self.trackingObj.subsumptionCount,
                                                                                                                                      self.trackingObj.crossOverCount, self.trackingObj.mutationCount, self.trackingObj.coveringCount, self.trackingObj.deletionCount, self.trackingObj.distance_threshold))
                print(self.iterationCount, accuracy, popNmrsty, len(self.population.popSet), self.trackingObj.distance_threshold)
                # self.log_trainingfile.flush()            

                pop_data_file_name = r'D:\Python\Thesis\ExSTraCS\test\Logs\populationData.csv'
                className = ""
                headerName = ""
#                self.record.exportPop(self, model, self.population.popSet, headerName, className, pop_data_file_name)
                self.record.exportPop(model=self, popSet=self.population.popSet, headerNames=np.array([]), className=className, filename=pop_data_file_name)
#                self.record.exportPop(self, model, popSet, headerNames=np.array([]), className='Class', filename='path.csv')

                '''
                self.iterationCount, 
                accuracy, 
                aveGenerality, 
                self.trackingObj.macroPopSize,
                  self.trackingObj.microPopSize, 
                  self.trackingObj.matchSetSize,
                  self.trackingObj.correctSetSize,
                  self.trackingObj.avgIterAge, self.trackingObj.subsumptionCount,self.trackingObj.crossOverCount,
                  self.trackingObj.mutationCount, self.trackingObj.coveringCount,self.trackingObj.deletionCount,
                  self.trackingObj.RCCount, self.timer.globalTime, self.timer.globalMatching,self.timer.globalCovering,
                  self.timer.globalCrossover, self.timer.globalMutation, self.timer.globalAT,self.timer.globalEK,
                  self.timer.globalInit, self.timer.globalAdd, self.timer.globalRuleCmp,self.timer.globalDeletion,
                  self.timer.globalSubsumption, self.timer.globalSelection, self.timer.globalEvaluation
                '''

				###############################################################
                
                #####*****************************************************#####
#                self.record.trackPopulationData(self.population, self.iterationCount, self.movingAvgCount)
                '''
                population_tracker = PopulationTracker(population, movingAvgCount=100)

                # During the training or evolution process
                population_tracker.trackPopulationData()
                
                # When you want to export the data (e.g., after the training is done or periodically)
                population_tracker.exportTrackingToCSV(filename='population_tracking_data.csv')
                '''
                #####*****************************************************#####
                
                is_log_pop = False
                pop_file_name = ""
                if self.iterationCount % pop_intrvl == 0:
                    is_log_pop = True
                    pop_file_name = self.log_popfile_path + "_pop_" + str(int(self.iterationCount / 10000)) + ".csv"
                    print("pop_file_name -", pop_file_name)

#                if is_log_pop:
#                    self.record.exportPopPatches(self,popSet=self.population.popSet,filename=pop_file_name)

            ###############################################################################################
            self.timer.updateGlobalTimer()
            self.addToTracking(accuracy,aveGenerality)
            self.timer.stopTimeEvaluation()

            # Increment Instance & Iteration
            self.iterationCount += 1
            self.env.newInstance()

        self.preRCPop = copy.deepcopy(self.population.popSet)
        #Rule Compaction
        if self.rule_compaction != None:
            self.trackingObj.resetAll()
            self.timer.startTimeRuleCmp()
            RuleCompaction(self)
            self.timer.stopTimeRuleCmp()
            self.timer.startTimeEvaluation()
            if len(self.trackingAccuracy) != 0:
                accuracy = sum(self.trackingAccuracy) / len(self.trackingAccuracy)
            else:
                accuracy = 0
            aveGenerality = self.population.getAveGenerality(self)
            self.trackingObj.macroPopSize = len(self.population.popSet)
            self.population.microPopSize = 0
            for rule in self.population.popSet:
                self.trackingObj.microPopSize += rule.numerosity
                self.population.microPopSize += rule.numerosity
            self.population.clearSets()
            self.trackingObj.avgIterAge = self.population.getInitStampAverage()
            self.timer.stopTimeEvaluation()
            self.timer.updateGlobalTimer()
            self.addToTracking(accuracy, aveGenerality)

        self.saveFinalMetrics()
        self.hasTrained = True
        
        self.record.exportTrackingToCSV(filename=r'D:\Python\Thesis\ExSTraCS\test\Logs\iterationData.csv')

        # Export the trained classifier population
#        self.record.exportPop(self, self.popSet, headerNames=self.train_headers, className='Class', filename='D:\\Python\\Thesis\\ExSTraCS\\test\\Logs\\populationData.csv')
#        self.record.exportPop(self, self.popSet, headerNames=self.train_headers, className='Class', filename=r'D:\Python\Thesis\ExSTraCS\test\Logs\populationData.csv')

        # Export population with detailed condition-action mappings
#        self.record.exportPopDCAL(self, self.popSet, headerNames=self.train_headers, className='Class', filename=r'D:\Python\Thesis\ExSTraCS\test\Logs\populationData_DCAL.csv')
         
        return self

    def addToTracking(self,accuracy,aveGenerality):
        self.record.addToTracking(self.iterationCount, accuracy, aveGenerality, self.trackingObj.macroPopSize,
                                  self.trackingObj.microPopSize, self.trackingObj.matchSetSize, self.trackingObj.correctSetSize,
                                  self.trackingObj.avgIterAge, self.trackingObj.subsumptionCount, self.trackingObj.crossOverCount,
                                  self.trackingObj.mutationCount, self.trackingObj.coveringCount, self.trackingObj.deletionCount,
                                  self.trackingObj.RCCount, self.timer.globalTime, self.timer.globalMatching, self.timer.globalCovering,
                                  self.timer.globalCrossover, self.timer.globalMutation, self.timer.globalAT, self.timer.globalEK,
                                  self.timer.globalInit, self.timer.globalAdd, self.timer.globalRuleCmp, self.timer.globalDeletion,
                                  self.timer.globalSubsumption, self.timer.globalSelection, self.timer.globalEvaluation, self.distance_threshold)

    def runIteration(self,state_phenotype):
        # Reset tracking object counters
        self.trackingObj.resetAll()

        #Make [M]
        self.population.makeMatchSet(self,state_phenotype)

        #Track Training Accuracy
        if self.track_accuracy_while_fit:
            self.timer.startTimeEvaluation()
            prediction = Prediction(self,self.population)
            phenotypePrediction = prediction.getDecision()

            if phenotypePrediction == state_phenotype[1]:
                if len(self.trackingAccuracy) == self.movingAvgCount:
                    del self.trackingAccuracy[0]
                self.trackingAccuracy.append(1)
            else:
                if len(self.trackingAccuracy) == self.movingAvgCount:
                    del self.trackingAccuracy[0]
                self.trackingAccuracy.append(0)

            self.timer.stopTimeEvaluation()

        #Make [C]
        self.population.makeCorrectSet(state_phenotype[1])

        #Update Parameters
        self.population.updateSets(self)

        #[C] Subsumption
        if self.do_correct_set_subsumption:
            self.timer.startTimeSubsumption()
            self.population.do_correct_set_subsumption(self)
            self.timer.stopTimeSubsumption()

        #AT
        if self.do_attribute_tracking:
            self.timer.startTimeAT()
            self.AT.updateAttTrack(self,self.population)
            if self.do_attribute_feedback:
                self.AT.updatePercent(self)
                self.AT.genTrackProb(self)
            self.timer.stopTimeAT()

        #Run GA
        self.population.runGA(self,state_phenotype[0],state_phenotype[1])

        #Deletion
        self.population.deletion(self)

        self.trackingObj.macroPopSize = len(self.population.popSet)
        self.trackingObj.microPopSize = self.population.microPopSize
        self.trackingObj.matchSetSize = len(self.population.matchSet)
        self.trackingObj.correctSetSize = len(self.population.correctSet)
        self.trackingObj.avgIterAge = self.population.getInitStampAverage()

        #Clear Sets
        self.population.clearSets()

    ##*************** Population Reboot ****************
    def saveFinalMetrics(self):
        self.finalMetrics = [self.learning_iterations,self.timer.globalTime, self.timer.globalMatching,self.timer.globalCovering,
                             self.timer.globalCrossover, self.timer.globalMutation, self.timer.globalAT,self.timer.globalEK,
                             self.timer.globalInit, self.timer.globalAdd, self.timer.globalRuleCmp,self.timer.globalDeletion,
                             self.timer.globalSubsumption, self.timer.globalSelection, self.timer.globalEvaluation,copy.deepcopy(self.AT),
                             copy.deepcopy(self.env),copy.deepcopy(self.population.popSet),self.preRCPop]

    def pickle_model(self,filename=None,saveRCPop=False):
        if self.hasTrained and self.learning_iterations == self.iterationCount: # Check hasFit, and there is new stuff to pickle
            if filename == None:
                filename = 'pickled'+str(int(time.time()))
            outfile = open(filename, 'wb')
            finalMetricsCopy = copy.deepcopy(self.finalMetrics)
            if saveRCPop==False:
                finalMetricsCopy.pop(len(self.finalMetrics) - 2)
            else:
                finalMetricsCopy.pop(len(self.finalMetrics) - 1)
            pickle.dump(finalMetricsCopy, outfile)
            outfile.close()
        elif self.hasTrained and self.learning_iterations != self.iterationCount:
            raise Exception("Pickle not allowed, as there is nothing new to pickle.")
        else:
            raise Exception("There is no final model to pickle, as the ExSTraCS model has not been trained")

    def rebootPopulation(self):
        file = open(self.reboot_filename,'rb')
        rawData = pickle.load(file)
        file.close()

        popSet = rawData[len(rawData)-1]
        microPopSize = 0
        for rule in popSet:
            microPopSize += rule.numerosity
        set = ClassifierSet()
        set.popSet = popSet
        set.microPopSize = microPopSize
        self.population = set
        self.learning_iterations += rawData[0]
        self.iterationCount = rawData[0]
        self.AT = rawData[15]
        self.env = rawData[16]

    def rebootTimer(self):
        file = open(self.reboot_filename, 'rb')
        rawData = pickle.load(file)
        file.close()

        self.timer = Timer()
        self.timer.addedTime = rawData[1]
        self.timer.globalMatching = rawData[2]
        self.timer.globalCovering = rawData[3]
        self.timer.globalCrossover = rawData[4]
        self.timer.globalMutation = rawData[5]
        self.timer.globalAT = rawData[6]
        self.timer.globalEK = rawData[7]
        self.timer.globalInit = rawData[8]
        self.timer.globalAdd = rawData[9]
        self.timer.globalRuleCmp = rawData[10]
        self.timer.globalDeletion = rawData[11]
        self.timer.globalSubsumption = rawData[12]
        self.timer.globalSelection = rawData[13]
        self.timer.globalEvaluation = rawData[14]

    ##*************** Predict and Score ****************
    def predict(self, X, is_test=False):
        """Scikit-learn required: Test Accuracy of ExSTraCS
            Parameters
            X: array-like {n_samples, n_features} Test instances to classify. ALL INSTANCE ATTRIBUTES MUST BE NUMERIC

            Returns
            y: array-like {n_samples} Classifications.
        """
        try:
            for instance in X:
                for value in instance:
                    if not (np.isnan(value)):
                        float(value)
        except:
            raise Exception("X must be fully numeric")

        instances = X.shape[0]
        predList = []
        probList = []  # List to store probabilities for each instance

        for inst in range(instances):  # loops through each instance (or row) in X

            state = X[inst]  # Retrieves the inst-th row of the dataset X & stores in "state"
            self.population.makeEvalMatchSet(self, state)  # Creates a match set for the current instance
            prediction = Prediction(self, self.population)  # A new Prediction object is created, passing the model and its population as parameters to generate the prediction for the current instance
            phenotypeSelection = prediction.getDecision()  # Retrieves the predicted class (phenotype) for the current instance
            predList.append(phenotypeSelection)  # Adds the predicted class to the list of predictions.
#            print(phenotypeSelection)

            if is_test:
                # Get probabilities for the current instance (only for testing data)
                probabilities = prediction.getProbabilities()
                probList.append(probabilities)  # Store probabilities                
#                print(f"Instance {inst + 1} Probabilities: {probabilities}") # Print the probabilities for the current instance

            self.population.clearSets()  # Clears out references in the match and correct sets to prepare for the next instance
#            print("np.array(predList -)", np.array(predList))

        if is_test:
            return np.array(predList), np.array(probList)  # Return both predictions and probabilities into a NumPy array and returns it
        else:
            return np.array(predList)  # Return only predictions for training data

    def predict_proba(self, X):
        """Scikit-learn required: Test Accuracy of ExSTraCS
            Parameters
            X: array-like {n_samples, n_features} Test instances to classify. ALL INSTANCE ATTRIBUTES MUST BE NUMERIC

            Returns
            y: array-like {n_samples} Classifications.
        """
        try:
            for instance in X:
                for value in instance:
                    if not (np.isnan(value)):
                        float(value)
        except:
            raise Exception("X must be fully numeric")

        instances = X.shape[0]
        predList = []

        for inst in range(instances):
            state = X[inst]
            self.population.makeEvalMatchSet(self,state)
            prediction = Prediction(self, self.population)
            probs = prediction.getProbabilities()
            predList.append(probs)
            self.population.clearSets()
        return np.array(predList)

    def score(self, X, y, is_test=False):

        if is_test:  # For testing data, predict() returns both predictions and probabilities
            predList, probabilities = self.predict(X, is_test)
        else:  # For training data, predict() returns only predictions
            predList = self.predict(X, is_test)

        if is_test:
#            print("probabilities array-", probabilities)
    
            # Print class labels and probabilities
#            self.print_probabilities_with_labels(probabilities)            

            # Export combined data to CSV
            self.export_probabilities_with_labels(X, probabilities, y, filename=r"D:\Python\Thesis\ExSTraCS\test\Logs\test_data_with_probabilities_LCS.csv")

#        print("predList - ", predList)    
        accuracy_test = balanced_accuracy_score(y, predList)
#        self.log_trainingfile.write("Testtttt Accuracy: {:.4f}\n".format(accuracy_test))
#        self.log_trainingfile.close()

        return accuracy_test
        # return balanced_accuracy_score(y,predList)

#################################

    def get_feature_importance(self, X, y):
        """
        Computes feature importance using Mutual Information (Entropy-Based).
        
        Parameters:
        - X: Feature matrix (numpy array or pandas DataFrame)
        - y: Target labels (numpy array or pandas Series)
        
        Returns:
        - selected_features: List of top 5 feature names
        - feature_importance_df: DataFrame with all feature names and MI scores
        """
        print("Calculating Feature Importance using Mutual Information...")

        # If X is a NumPy array, convert it to a DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)  # Convert NumPy array to DataFrame (columns will be indexed 0,1,2,...)
        
        # Compute MI scores
        mi_scores = mutual_info_classif(X, y)

        # Create a DataFram e with feature names and importance scores
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,  # Now X is a DataFrame, so it has columns
            'Importance': mi_scores
        }).sort_values(by='Importance', ascending=False)
        print("feature_importance_df -->", feature_importance_df)

        print("self.feature_selection_percentage", self.feature_selection_percentage)
        percentage = self.feature_selection_percentage  # Use the model’s stored feature selection percentage
        print("after assigning to the percentage -->", percentage)
        # Calculate the number of top features to select based on the given percentage
        num_features_to_select = max(1, int(len(X.columns) * percentage))  # Ensure at least one feature is selected

#        print("X.columns - ", X.columns)
#        print("percentage -", percentage)
        print("num_features_to_select -", num_features_to_select)
        
        # Select the top features
        selected_features = feature_importance_df.nlargest(num_features_to_select, 'Importance')['Feature'].tolist()
        print(f"Top {num_features_to_select} Selected Features: {selected_features}")
#        print("selected_features", selected_features)
        
        # Get all feature names
        self.all_feature_list = set(feature_importance_df['Feature'])
        
        return selected_features, feature_importance_df  # Returns a sorted DataFrame with feature names

#################################

    def export_probabilities_with_labels(self, X, probabilities, y, filename="test_results.csv"):
        # Get class labels from phenotypeList
        class_labels = self.env.formatData.phenotypeList
    
        # Create a DataFrame with input data, probabilities, and original classes
        combined_data = []
        for i in range(len(X)):
            row = list(X[i])  # Input features
            row.extend(probabilities[i])  # Probabilities
            row.append(y[i])  # Original class
            combined_data.append(row)
    
        # Define column headers
        feature_headers = [f"Feature_{j}" for j in range(len(X[0]))]
        probability_headers = [f"Prob_{label}" for label in class_labels]
        headers = feature_headers + probability_headers + ["Original_Class"]
    
        # Convert to DataFrame
        df = pd.DataFrame(combined_data, columns=headers)
    
        # Export to CSV
        df.to_csv(filename, index=False)
        print(f"Combined data exported to '{filename}'")

#################################

    ##*************** More Evaluation Methods ****************

    def get_final_training_accuracy(self,RC=False):
        if self.hasTrained or RC:
            originalTrainingData = self.env.formatData.savedRawTrainingData
            return self.score(originalTrainingData[0], originalTrainingData[1])
        else:
            raise Exception("There is no final training accuracy to return, as the ExSTraCS model has not been trained")

    def get_final_instance_coverage(self):
        if self.hasTrained:
            numCovered = 0
            originalTrainingData = self.env.formatData.savedRawTrainingData
            for state in originalTrainingData[0]:
                self.population.makeEvalMatchSet(self,state)
                predictionArray = Prediction(self, self.population)
                if predictionArray.hasMatch:
                    numCovered += 1
                self.population.clearSets()
            return numCovered/len(originalTrainingData[0])
        else:
            raise Exception("There is no final instance coverage to return, as the ExSTraCS model has not been trained")

    def get_final_attribute_specificity_list(self):
        if self.hasTrained:
            return self.population.getAttributeSpecificityList(self)
        else:
            raise Exception("There is no final attribute specificity list to return, as the ExSTraCS model has not been trained")

    def get_final_attribute_accuracy_list(self):
        if self.hasTrained:
            return self.population.getAttributeAccuracyList(self)
        else:
            raise Exception("There is no final attribute accuracy list to return, as the ExSTraCS model has not been trained")

    def get_final_attribute_tracking_sums(self):
        if self.hasTrained and self.AT != None:
            return self.AT.getSumGlobalAttTrack(self)
        elif not self.do_attribute_tracking:
            raise Exception("There are no final attribute tracking sums to return, as AT was False")
        else:
            raise Exception("There are no final attribute tracking sums to return, as the ExSTraCS model has not been trained")

    def get_final_attribute_coocurrences(self,headers,maxNumAttributesToTrack=50):
        #Return a 2D list of [[attr1Name, attr2Name, specificity, accuracy weighted specificity]...]
        if self.hasTrained:
            if not isinstance(headers,np.ndarray):
                raise Exception("headers param must be ndarray")
            if len(headers.tolist()) != self.env.formatData.numAttributes:
                raise Exception("length of headers param must match the # of data instance attributes")

            if self.env.formatData.numAttributes <= maxNumAttributesToTrack:
                attList = []
                for i in range(0,self.env.formatData.numAttributes):
                    attList.append(i)
            else:
                attList = sorted(self.get_final_attribute_specificity_list(),reverse=True)[:maxNumAttributesToTrack]

            comboList = []
            for i in range(0,len(attList)-1):
                for j in range(i+1,len(attList)):
                    comboList.append([headers[attList[i]],headers[attList[j]],0,0])

            for cl in self.population.popSet:
                counter = 0
                for i in range(0, len(attList) - 1):
                    for j in range(i + 1, len(attList)):
                        if attList[i] in cl.specifiedAttList and attList[j] in cl.specifiedAttList:
                            comboList[counter][2] += cl.numerosity
                            comboList[counter][3] += cl.numerosity * cl.accuracy
                        counter+=1

            return sorted(comboList,key=lambda test:test[3],reverse=True)

        else:
            raise Exception(
                "There are no final attribute cooccurences to return, as the ExSTraCS model has not been trained")

    def get_attribute_tracking_scores(self,instance_labels=np.array([])):
        if self.hasTrained:
            retList = []
            if instance_labels.size != self.env.formatData.numTrainInstances:
                raise Exception('# of Instance Labels must match # of training instances')

            for i in range(self.env.formatData.numTrainInstances):
                retList.append([instance_labels[i], self.AT.attAccuracySums[self.env.formatData.shuffleOrder[i]]])
            return retList
        else:
            raise Exception("There is no AT scores to return, as the ExSTraCS model has not been trained")

    # #Export Methods##
    def export_iteration_tracking_data(self, filename='D:\Python\Thesis\ExSTraCS\test\Logs\iterationData.csv'):
        if self.hasTrained:
            self.record.exportTrackingToCSV(filename)
        else:
            raise Exception("There is no tracking data to export, as the ExSTraCS model has not been trained")

    def export_final_rule_population(self,headerNames=np.array([]),className="phenotype",filename='populationData.csv',DCAL=True,RCPopulation=False):
        if self.hasTrained:
            if RCPopulation:
                popSet = self.population.popSet
            else:
                popSet = self.preRCPop

            if DCAL:
                self.record.exportPopDCAL(self,popSet,headerNames,className,filename)
            else:
                self.record.exportPop(self,popSet,headerNames, className, filename)
        else:
            raise Exception("There is no rule population to export, as the ExSTraCS model has not been trained")

    ##Rule Compaction Method ##
    def post_training_rule_compaction(self,method='QRF'):
        if self.hasTrained:
            oldRC = copy.deepcopy(self.rule_compaction)
            self.rule_compaction = method
            RuleCompaction(self)
            self.rule_compaction = oldRC
            self.saveFinalMetrics()
        else:
            raise Exception("There is no rule population to compact, as the ExSTraCS model has not been trained")

class TempTrackingObj():
    #Tracks stats of every iteration (except accuracy, avg generality, and times)
    def __init__(self):
        self.macroPopSize = 0
        self.microPopSize = 0
        self.matchSetSize = 0
        self.correctSetSize = 0
        self.avgIterAge = 0
        self.subsumptionCount = 0
        self.crossOverCount = 0
        self.mutationCount = 0
        self.coveringCount = 0
        self.deletionCount = 0
        self.RCCount = 0
        self.distance_threshold = 0
        self.classifier_distance = 0

    def resetAll(self):
        self.macroPopSize = 0
        self.microPopSize = 0
        self.matchSetSize = 0
        self.correctSetSize = 0
        self.avgIterAge = 0
        self.subsumptionCount = 0
        self.crossOverCount = 0
        self.mutationCount = 0
        self.coveringCount = 0
        self.deletionCount = 0
        self.RCCount = 0
