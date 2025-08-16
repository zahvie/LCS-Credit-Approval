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
train_file = r'D:\Python\Thesis\ExSTraCS\test\DataSets\Real\Data\Loan Approval Data\train_Loan_Approval_Data.csv'
test_file = r'D:\Python\Thesis\ExSTraCS\test\DataSets\Real\Data\Loan Approval Data\test_Loan_Approval_Data.csv'

# Alternative: Colab path
train_colab_path = "/content/repo/data/LAD/train_Loan_Approval_Data.csv"
test_colab_path = "/content/repo/data/LAD/test_Loan_Approval_Data.csv"
    

# Decide which one to use
if in_colab:
    if os.path.exists(train_colab_path):
        print("✅ Running in Colab & train file exists:", train_colab_path)
        train_file_path = train_colab_path
        print("OS Path",os.path.exists(train_file_path))
        print("Google Colab file is used", in_colab)
    else:
        print("⚠ Running in Colab but train file not found:", train_file_path)
        train_file_path = train_file
        print("OS Path",train_file_path)

    if os.path.exists(test_colab_path):
        print("✅ Running in Colab & test file exists:", test_colab_path)
        test_file_path = test_colab_path
        print("OS Path",os.path.exists(test_file_path))
        print("Google Colab file is used", in_colab)
    else:
        print("⚠ Running in Colab but test file not found:", train_file_path)
        test_file_path = test_file
        print("OS Path",test_file_path)
else:
    print("⚠ Running in Colab but file not found:", train_colab_path)
    print("⚠ Running in Colab but file not found:", test_colab_path)
    train_file_path = train_file
    test_file_path = test_file
    print("OS Path",train_file_path)
    print("OS Path",test_file_path)
    print("💻 Not running in Colab")




'''
log_dir = r'D:\Python\Thesis\ExSTraCS\test\Logs'
#log_trainingfile_name = f"log_training_{current_time}.txt"  # Name the log file with the current time
#log_file_path = os.path.join(log_dir, log_trainingfile_name)

#logging.basicConfig(filename=log_file_path, level=logging.INFO, 
#                    format='%(asctime)s - %(message)s')

#Default Credit Card Clients
#train_file = r'D:\Python\Thesis\ExSTraCS\test\DataSets\Real\train_CC_data.csv'
#test_file = r'D:\Python\Thesis\ExSTraCS\test\DataSets\Real\test_CC_data.csv'

#Clean Data
#train_file = r'D:\Python\Thesis\ExSTraCS\test\DataSets\Real\Data\Clean\train_clean_dataset.csv'
#test_file = r'D:\Python\Thesis\ExSTraCS\test\DataSets\Real\Data\Clean\test_clean_dataset.csv'

#Credit Risk #########
#train_file = r'D:\Python\Thesis\ExSTraCS\test\DataSets\Real\Data\Credit_Risk\train_credit_risk.csv'
#test_file = r'D:\Python\Thesis\ExSTraCS\test\DataSets\Real\Data\Credit_Risk\test_credit_risk.csv'

#train_file = r'D:\Python\Thesis\ExSTraCS\test\DataSets\Real\train_application_data.csv'
#test_file = r'D:\Python\Thesis\ExSTraCS\test\DataSets\Real\test_application_data.csv'

#train_file = r'D:\Python\Thesis\ExSTraCS\test\DataSets\Real\Data\Credit_Risk\train_credit_risk.csv'
#test_file = r'D:\Python\Thesis\ExSTraCS\test\DataSets\Real\Data\Credit_Risk\test_credit_risk.csv'

# Loan Approval_Prediction #########
#train_file = r'D:\Personal\Studies\MSc\Thesis\Code Backup\Laptop\32-2025-05-13_After_Mixed_Method\skExSTraCS'
#test_file = r'D:\Python\Thesis\ExSTraCS\test\DataSets\Real\Data\Loan_Approval_Prediction\test_Loan_Approval_Prediction.csv'

#############################
#Loan Approval Data #########
train_file = r'D:\Python\Thesis\ExSTraCS\test\DataSets\Real\Data\Loan Approval Data\train_Loan_Approval_Data.csv'
test_file = r'D:\Python\Thesis\ExSTraCS\test\DataSets\Real\Data\Loan Approval Data\test_Loan_Approval_Data.csv'

# Application Data
#train_file = r'D:\Python\Thesis\ExSTraCS\test\DataSets\Real\Data\Application_Data\train_Application_Data.csv'
#test_file = r'D:\Python\Thesis\ExSTraCS\test\DataSets\Real\Data\Application_Data\test_Application_Dataa.csv'

#train_file = r'D:\Python\Thesis\ExSTraCS\test\DataSets\Real\Data\Application_Data\train_application_data_sample3.csv'
#test_file = r'D:\Python\Thesis\ExSTraCS\test\DataSets\Real\Data\Application_Data\test_application_data_sample3.csv'

#train_file = r'D:\Python\Thesis\ExSTraCS\latent_train_features_with_class_cnn.csv'
#test_file = r'D:\Python\Thesis\ExSTraCS\latent_test_features_with_class_cnn.csv'

#train_file = r'D:\Python\Thesis\ExSTraCS\latent_train_features_with_class_MLP.csv'
#test_file = r'D:\Python\Thesis\ExSTraCS\latent_test_features_with_class_MLP.csv'

#train_file = r'D:\Python\Thesis\ExSTraCS\test\DataSets\Real\train_loan_approval_data_cleaned.csv'
#test_file = r'D:\Python\Thesis\ExSTraCS\test\DataSets\Real\test_loan_approval_data_cleaned.csv'

#train_file = r'D:\Python\Thesis\ExSTraCS\feature_selected_train_RFE.csv'
#test_file = r'D:\Python\Thesis\ExSTraCS\feature_selected_test_RFE.csv'

#train_file = r'D:\Python\Thesis\ExSTraCS\latent_train_features_with_class_XGBoost.csv'
#test_file = r'D:\Python\Thesis\ExSTraCS\latent_test_features_with_class_XGBoost.csv'

#train_file = r'D:\Python\Thesis\ExSTraCS\test\DataSets\Real\train_loan_approval_data.csv'
#test_file = r'D:\Python\Thesis\ExSTraCS\test\DataSets\Real\test_loan_approval_data.csv'

#train_file = r'D:\Python\Thesis\ExSTraCS\latent_train_features_with_class_XGBoost.csv'
#test_file = r'D:\Python\Thesis\ExSTraCS\latent_test_features_with_class_XGBoost.csv'

#train_file = r'D:\Python\Thesis\ExSTraCS\test\DataSets\Real\train_Clean_data_FE_Concat.csv'
#test_file = r'D:\Python\Thesis\ExSTraCS\test\DataSets\Real\test_Clean_data_FE_Concat.csv'

#train_file = r'D:\Python\Thesis\ExSTraCS\test\DataSets\Real\train_Clean_Sample2.csv'
#test_file = r'D:\Python\Thesis\ExSTraCS\test\DataSets\Real\test_Clean_Sample2.csv'

#PLC_Data
#train_file = r'D:\Python\Thesis\ExSTraCS\test\DataSets\Real\Data\PLC_Data\train_plc_data.csv'
#test_file = r'D:\Python\Thesis\ExSTraCS\test\DataSets\Real\Data\PLC_Data\test_plc_data.csv'

#print("Debugging: Loading train file", train_file)

train_converter = StringEnumerator(train_file,'Class')
train_headers, train_classLabel, train_dataFeatures, train_dataPhenotypes = train_converter.get_params()
#print(train_dataFeatures, train_dataPhenotypes)
#print("Debugging: Features Loaded - train_dataFeatures", train_dataFeatures[:5])
#print("Training Data Shape:", train_dataFeatures.shape)

#print("Debugging: Features Loaded - train_dataPhenotypes", train_dataPhenotypes)
#print(train_dataPhenotypes)

######################################


# Prepare the results file with headers
results_file = os.path.join(log_dir, 'testing_results.csv')

# Extract the data file name (without extension) from the training file path
data_file_name = os.path.splitext(os.path.basename(train_file))[0]
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
log_file_path = os.path.join(log_dir, log_trainingfile_name)


 
last_iterations = 100
max_iterations = 1000
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
model = ExSTraCS(learning_iterations=max_iterations, N=N, nu=10, use_feature_ranked_RSL=use_feature_ranked_RSL)
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
test_converter = StringEnumerator(test_file, 'Class')
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

'''