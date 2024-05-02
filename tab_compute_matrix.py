import pandas as pd
import sys
import os
def simple_clf(true_label,predicted_label,total):
    # Count occurrences where values in Column1 and Column2 are the same
    same_values_count = (true_label == predicted_label).sum()
    accuracy = same_values_count / total
    return accuracy

def input_processor(true_label,predicted_label, total):
    same_values_count = (true_label == predicted_label).sum()
    accuracy = same_values_count / total
    different_value = (true_label != predicted_label).sum()
    missclassification = different_value/total
    return accuracy,missclassification


folder_path = './results/tabular/NIDS/'  # Replace this with the path to your folder

# Get a list of all files in the folder
files = os.listdir(folder_path)

# Filter out only the CSV files
csv_files = [file for file in files if file.endswith('.csv')]
output_file = 'output.txt'
if os.path.exists(output_file):
    os.remove(output_file)
# Iterate through each CSV file
for csv_file in csv_files:
    file_name = csv_file
    print(file_name)
    file_path = os.path.join(folder_path, csv_file)


    # file_name = 'Result_with_ResNet_AlexNet.csv'
    df = pd.read_csv(file_path)
    length = len(df)
    probability_criteria = 0.99996
    probability_criteria_2 = 0.8

    # Split the string based on the underscore character
    split_values = file_name.split('_')

    # Extract 'ResNet' and 'AlexNet'
    binary_clf = split_values[0]
    multiclass_clf = split_values[1].split('.')[0]  # Removing the '.csv' extension



    # Open a file in write mode to redirect the output
    with open(output_file, 'a') as f:
        # Redirect stdout to the file
        sys.stdout = f

        print("Binary CLF: ",binary_clf)
        print("Multiclass CLF: ",multiclass_clf)

        predict_clf_label = df['Predicted CLF']
        # print("===========================")
        print("Simple Class Classifier")
        # print("===========================")

        true_clf_label = df['True CLF Labels']
        simple_accuracy = simple_clf(true_clf_label,predict_clf_label,length)

        simple_miss = 1 - simple_accuracy
        print(f'Accuracy: {simple_accuracy},Misclassification: {simple_miss}')

        print("******************************************************************************")

        # print("===========================")
        print("Input Processor")
        # print("===========================")
        filtered_ip = df[df['Predicted Label'] == 1]
        true_clf_label_filtered = filtered_ip['True CLF Labels']
        predict_clf_label_filtered = filtered_ip['Predicted CLF']
        ip_accuracy,ip_missclassification = input_processor(true_clf_label_filtered,predict_clf_label_filtered,length)
        # print("Accuracy: ",ip_accuracy)
        # print("Misclassification:", ip_missclassification)

        df_phi = df[df['Predicted Label']== 0]
        ip_phi = df_phi['Predicted Label'].count()/length
        # print("Phi: ",ip_phi)

        same_values_count = (df_phi['True CLF Labels'] == df_phi['Predicted CLF']).sum()
        ip_phi_c = same_values_count/len(df_phi)
        # print("Phi_c:",ip_phi_c)

        diff_values_count = (df_phi['True CLF Labels'] != df_phi['Predicted CLF']).sum()
        ip_phi_m = diff_values_count/len(df_phi)
        # print("Phi_m:",ip_phi_m)

        print(f'Accuracy: {ip_accuracy}, Misclassification: {ip_missclassification}, Phi: {ip_phi}, Phi_c: {ip_phi_c}, Phi_m: {ip_phi_m}')
        print("******************************************************************************")

        # print("===========================")
        print(f"Output Processor with {probability_criteria}")
        # print("===========================")
        filtered_op = df[df['Probability'] >= probability_criteria]
        true_clf_label_filtered = filtered_op['True CLF Labels']
        predict_clf_label_filtered = filtered_op['Predicted CLF']
        op_accuracy,op_missclassification = input_processor(true_clf_label_filtered,predict_clf_label_filtered,length)
        # print("Accuracy: ",op_accuracy)
        # print("Misclassification:", op_missclassification)

        df_phi = df[df['Probability']< probability_criteria]
        op_phi = df_phi['Predicted Label'].count()/length
        # print("Phi: ",op_phi)

        same_values_count = (df_phi['True CLF Labels'] == df_phi['Predicted CLF']).sum()
        op_phi_c = same_values_count/len(df_phi)
        # print("Phi_c:", op_phi_c)

        diff_values_count = (df_phi['True CLF Labels'] != df_phi['Predicted CLF']).sum()
        op_phi_m = diff_values_count/len(df_phi)
        # print("Phi_m:", op_phi_m)

        print(f'Accuracy: {op_accuracy}, Misclassification: {op_missclassification}, Phi: {op_phi}, Phi_c: {op_phi_c}, Phi_m: {op_phi_m}')


        print("******************************************************************************")

        # print("===========================")
        print(f"Safety Wrapper with {probability_criteria}")
        # print("===========================")

        filtered_sw = df[(df['Predicted Label'] == 1) & (df['Probability'] >= probability_criteria)]
        true_clf_label_filtered = filtered_sw['True CLF Labels']
        predict_clf_label_filtered = filtered_sw['Predicted CLF']
        sw_accuracy,sw_missclassification = input_processor(true_clf_label_filtered,predict_clf_label_filtered,length)
        # print("Accuracy: ",sw_accuracy)
        # print("Misclassification:", sw_missclassification)

        df_phi = df[(df['Predicted Label'] == 0) | (df['Probability'] < probability_criteria)]
        sw_phi = df_phi['Predicted Label'].count()/length
        # print("Phi: ",sw_phi)

        same_values_count = (df_phi['True CLF Labels'] == df_phi['Predicted CLF']).sum()
        sw_phi_c = same_values_count/len(df_phi)
        # print("Phi_c:",sw_phi_c)

        diff_values_count = (df_phi['True CLF Labels'] != df_phi['Predicted CLF']).sum()
        sw_phi_m = diff_values_count/len(df_phi)
        # print("Phi_m:",sw_phi_m)

        print(f'Accuracy: {sw_accuracy}, Misclassification: {sw_missclassification}, Phi: {sw_phi}, Phi_c: {sw_phi_c}, Phi_m: {sw_phi_m}')


        print("######################################################################")

        print("===========================")
        print(f"Output Processor {probability_criteria_2}")
        print("===========================")
        filtered_op2 = df[df['Probability'] >= probability_criteria_2]
        true_clf_label_filtered = filtered_op2['True CLF Labels']
        predict_clf_label_filtered = filtered_op2['Predicted CLF']
        op_accuracy, op_missclassification = input_processor(true_clf_label_filtered, predict_clf_label_filtered, length)
        # print("Accuracy: ", op_accuracy)
        # print("Misclassification:", op_missclassification)

        df_phi = df[df['Probability'] < probability_criteria_2]
        op_phi = df_phi['Predicted Label'].count() / length
        # print("Phi: ", op_phi)

        same_values_count = (df_phi['True CLF Labels'] == df_phi['Predicted CLF']).sum()
        op_phi_c = same_values_count / len(df_phi)
        # print("Phi_c:", op_phi_c)

        diff_values_count = (df_phi['True CLF Labels'] != df_phi['Predicted CLF']).sum()
        op_phi_m = diff_values_count / len(df_phi)
        # print("Phi_m:", op_phi_m)

        print(f'Accuracy: {op_accuracy}, Misclassification: {op_missclassification}, Phi: {op_phi}, Phi_c: {op_phi_c}, Phi_m: {op_phi_m}')

        print("******************************************************************************")

        # print("===========================")
        print(f'Safety Wrapper with {probability_criteria_2}')
        # print("===========================")

        filtered_sw2 = df[(df['Predicted Label'] == 1) & (df['Probability'] >= probability_criteria_2)]
        true_clf_label_filtered = filtered_sw2['True CLF Labels']
        predict_clf_label_filtered = filtered_sw2['Predicted CLF']
        sw_accuracy, sw_missclassification = input_processor(true_clf_label_filtered, predict_clf_label_filtered, length)
        # print("Accuracy: ", sw_accuracy)
        # print("Misclassification:", sw_missclassification)

        df_phi = df[(df['Predicted Label'] == 0) | (df['Probability'] < probability_criteria_2)]
        sw_phi = df_phi['Predicted Label'].count() / length
        # print("Phi: ", sw_phi)

        same_values_count = (df_phi['True CLF Labels'] == df_phi['Predicted CLF']).sum()
        sw_phi_c = same_values_count / len(df_phi)
        # print("Phi_c:", sw_phi_c)

        diff_values_count = (df_phi['True CLF Labels'] != df_phi['Predicted CLF']).sum()
        sw_phi_m = diff_values_count / len(df_phi)
        # print("Phi_m:", sw_phi_m)

        print(f'Accuracy: {sw_accuracy}, Misclassification: {sw_missclassification}, Phi: {sw_phi}, Phi_c: {sw_phi_c}, Phi_m: {sw_phi_m}')


        print("######################################################################")

    # Reset stdout to the console
    sys.stdout = sys.__stdout__