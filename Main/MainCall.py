'''
Created on Dec 16, 2024

@author: MSII
'''
from skExSTraCS import ExSTraCS,StringEnumerator
import os
import csv
from datetime import datetime
import time
from pathlib import Path

try:
    import google.colab
    in_colab = True
except ImportError:
    in_colab = False

print("Running in Google Colab?", in_colab)


# Default: local file path (Windows)

#Loan Approval Data #########
train_file = r'D:\Python\Thesis\ExSTraCS\test\DataSets\Real\Data\Loan Approval Data\train_Loan_Approval_Data.csv'
test_file = r'D:\Python\Thesis\ExSTraCS\test\DataSets\Real\Data\Loan Approval Data\test_Loan_Approval_Data.csv'

# Loan Approval_Prediction Data
#train_file = r'D:\Python\Thesis\ExSTraCS\test\DataSets\Real\Data\Loan_Approval_Prediction\train_Loan_Approval_Prediction.csv'
#test_file = r'D:\Python\Thesis\ExSTraCS\test\DataSets\Real\Data\Loan_Approval_Prediction\test_Loan_Approval_Prediction.csv'



# Alternative: Colab path

#Loan Approval Data #########
train_colab_path = "/content/repo/data/LAD/train_Loan_Approval_Data.csv"
test_colab_path = "/content/repo/data/LAD/test_Loan_Approval_Data.csv"
    

# Loan Approval_Prediction Data
#train_colab_path = "/content/repo/data/LAP/train_Loan_Approval_Prediction.csv"
#test_colab_path = "/content/repo/data/LAP/test_Loan_Approval_Prediction.csv"
    
# Decide which train and test file to use
if in_colab:
    if os.path.exists(train_colab_path):
        print("✅ Running in Colab & train file exists:", train_colab_path)
        train_file_path = train_colab_path
        print("Google Colab Path",os.path.exists(train_file_path))
        print("Google Colab train file is used -->", in_colab)
    else:
        train_file_path = train_colab_path
        print("⚠ Running in Colab but train file not found:", train_file_path)

    if os.path.exists(test_colab_path):
        print("✅ Running in Colab & test file exists:", test_colab_path)
        test_file_path = test_colab_path
        print("Google Colab Path",os.path.exists(test_file_path))
        print("Google Colab test file is used -->", in_colab)
    else:
        test_file_path = test_colab_path
        print("⚠ Running in Colab but test file not found:", test_file_path)
else:
    if os.path.exists(train_file):
        print("⚠ Running using Physical train file:", train_file)
        train_file_path = train_file
        print("Physical file Path",os.path.exists(train_file_path))
        print("Physical train file is used -->", in_colab)
    else:
        train_file_path = train_file
        print("💻 Not running in Colab but physical train file not found:", train_file_path)

    if os.path.exists(test_file):
        print("⚠ Running using Physical test file:", test_file)
        test_file_path = test_file
        print("Physical file Path",os.path.exists(test_file_path))
        print("Physical test file is used -->", in_colab)
    else:
        test_file_path = test_file
        print("💻 Not running in Colab but physical test file not found:", test_file_path)


# Default: local log path (Windows)
log_dir = r'D:\Python\Thesis\ExSTraCS\test\Logs'

# Alternative: Colab log path
colab_log_dir = '/content/repo/logs'


# Decide which train and test file to use
if in_colab:
    if os.path.exists(colab_log_dir):
        print("✅ Running in Colab & log directory exists:", colab_log_dir)
        log_dir_path = colab_log_dir
        print("Google Colab log directory is used", log_dir_path)
    else:
        log_dir_path = colab_log_dir
        print("⚠ Running in Colab but log directory not found:", colab_log_dir)
else:
    log_dir_path = log_dir
    print("⚠ Running using Physical log directory:", log_dir)
print(">>> DEBUG: Using this file for training:", train_file_path)

print('train file path is -->',train_file_path)


train_converter = StringEnumerator(train_file_path,'Class')
train_headers, train_classLabel, train_dataFeatures, train_dataPhenotypes = train_converter.get_params()
#print(train_dataFeatures, train_dataPhenotypes)
#print("Debugging: Features Loaded - train_dataFeatures", train_dataFeatures[:5])
#print("Training Data Shape:", train_dataFeatures.shape)

#print("Debugging: Features Loaded - train_dataPhenotypes", train_dataPhenotypes)
#print(train_dataPhenotypes)

######################################


# Prepare the results file with headers
results_file = os.path.join(log_dir_path, 'testing_results.csv')
print('results_file path is -->',results_file )

# Extract the data file name (without extension) from the training file path
data_file_name = os.path.splitext(os.path.basename(train_file_path))[0]
print("Extract the data file name (without extension) -",data_file_name)

# Remove "train_" prefix if it exists
if data_file_name.startswith("train_"):
    data_file_name = data_file_name.replace("train_", "", 1)
print("Extract the data file name (without extension) and 'train' -",data_file_name)



# Check if results file exists and if not, write headers
if not os.path.exists(results_file):
    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Iteration', 'Population Size', 'Fitness', 'Crossover', 'Mutating', 'Training Accuracy',
                          'Testing Accuracy', 'Specificity', 'Data File Name'])

# Dynamically create a new log file for each iteration change

current_time_for_file = datetime.now().strftime('%Y-%m-%d_%H-%M')  # Include seconds for uniqueness
log_trainingfile_name = f"log_training_{current_time_for_file}.txt"  # Name the log file with learning_iterations and timestamp
log_file_path = os.path.join(log_dir_path, log_trainingfile_name)


 
last_iterations = 100
max_iterations = 10000
N=7000
rule_specificity_limit = 8
use_feature_ranked_RSL= True
use_midpoint_distance_filter = True
log_file = open("log_trainingfile.csv", 'w', newline='')  # Open the log file

# Initialize the model with the new learning_iterations
#model = ExSTraCS(learning_iterations=max_iterations, N=10000, nu=10)
#model = ExSTraCS(learning_iterations=max_iterations, N=15000, nu=10, log_dir=log_dir, log_trainingfile_name=log_trainingfile_name)
#model = ExSTraCS(learning_iterations=max_iterations, N=N, nu=10)
#model = ExSTraCS(learning_iterations=max_iterations, N=N, nu=10, rule_specificity_limit = rule_specificity_limit,
#                 use_feature_ranked_RSL=True)
#model = ExSTraCS(learning_iterations=max_iterations, N=N, nu=10,use_feature_ranked_RSL=use_feature_ranked_RSL)
#model = ExSTraCS(learning_iterations=max_iterations, N=N, nu=10, rule_specificity_limit = rule_specificity_limit, use_feature_ranked_RSL=use_feature_ranked_RSL)

#model = ExSTraCS(learning_iterations=max_iterations, N=N, nu=10)
#model = ExSTraCS(learning_iterations=max_iterations, N=N, nu=10, rule_specificity_limit = rule_specificity_limit)
model = ExSTraCS(learning_iterations=max_iterations, N=N, nu=10, log_dir=log_dir, 
                 use_feature_ranked_RSL=use_feature_ranked_RSL)
#model = ExSTraCS(learning_iterations=max_iterations, N=N, nu=10, 
#                 use_midpoint_distance_filter=use_midpoint_distance_filter)

print("Model training in progress ...")

# Train the model
model.fit(model,train_dataFeatures, train_dataPhenotypes)
print("Model training Ends")

#Export the match set after the fit method completes
#model.export_match_set_from_population(matchSet=model.population.matchSet, filename=r'D:\Python\Thesis\ExSTraCS\test\Logs\matchSetData.csv')

accuracy = model.score(train_dataFeatures, train_dataPhenotypes, is_test=False)
training_accuracy = accuracy

print("Model Training Accuracy ", training_accuracy)
model.log_trainingfile.write("Training Accuracy: {:.4f}\n".format(training_accuracy))

# Test data conversion
test_converter = StringEnumerator(test_file_path, 'Class')
test_headers, test_classLabel, test_dataFeatures, test_dataPhenotypes = test_converter.get_params()

print("Testing Data Shape:", test_dataFeatures.shape)

print("Model Testing in progress ...")
accuracy = model.score(test_dataFeatures, test_dataPhenotypes, is_test=True)
testing_accuracy = accuracy
print("Model Testing Ends")
print("Model Testing Accuracy ", accuracy)
model.log_trainingfile.write("Testing Accuracy: {:.4f}\n".format(testing_accuracy))
print("Prog End")

# Update last_iterations for the next run
last_iterations = max_iterations  # Set the last_iterations to the newly calculated value

# Retry logic
max_retries = 5  # Maximum number of retries
retry_interval = 1  # Interval between retries in seconds
retry_count = 0

while retry_count < max_retries:
    try:
        # Attempt to open the file and append the results
        with open(results_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([max_iterations, model.N, model.nu, model.chi, model.mu, training_accuracy, 
                             testing_accuracy, rule_specificity_limit, data_file_name,
                             (training_accuracy - testing_accuracy), (training_accuracy - testing_accuracy) * 100])
        break  # If successful, exit the loop
    except PermissionError:
        retry_count += 1
        print(f"Attempt {retry_count} failed. Retrying in {retry_interval} seconds...")
        time.sleep(retry_interval)
else:
    print(f"Failed to write to the file after {max_retries} attempts.")

# Once the loop ends, the script will stop if learning_iterations >= 1,000,001
print("Maximum learning_iterations reached. Stopping execution.")

