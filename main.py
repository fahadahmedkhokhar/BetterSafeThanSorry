import argparse
import pickle
import numpy as np
# Press the green button in the gutter to run the script.
import torch
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import yaml
import pandas as pd
from GenericDataset import GenericDataset
from GenericDatasetc import GenericDatasetLoader
import torchvision.transforms as transforms
from models.model import ResNet,AlexNet,Voting,Stacking,TabularClassifier,InceptionV3
import joblib
from sklearn import  svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from torchsummary import summary
from sklearn.metrics import confusion_matrix
def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Classification")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")

    args = parser.parse_args()

    # Load arguments from YAML file
    config = load_config(args.config)

    classifier_names = config['MODEL']['CLASSIFIER_NAME']
    epochs = config['TRAINING']['EPOCHS']
    num_classes = config['MODEL']['NUM_CLASSES']
    batch_size = config['TRAINING']['BATCH_SIZE']
    learning_rate = config['TRAINING']['LEARNING_RATE']
    dataset_name = config['DATASET']['DATASET_NAME']
    save_dir = config['TRAINING']['SAVE_DIR']
    dataset_dir = config['DATASET']['DATASET_DIR']
    target = config['DATASET']['TARGET']
    base_model_list = config['MODEL']['BASE_MODEL']


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # dataset = GenericDataset(dataset_name=dataset_name, csv_file=dataset_dir,target = target)
    #dataset = GenericDataset(dataset_name=dataset_name, batch_size=batch_size, train=True, dataset_location=dataset_dir)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Create GenericDatasetLoader object for CIFAR
    #cifar_loader = GenericDatasetLoader(dataset_name=dataset_name, root_dir=dataset_dir, transform=transform)


    custom_data = GenericDatasetLoader(dataset_name=dataset_name, root_dir = dataset_dir, transform= transform, batch_size=1)
    custom_data_orig = GenericDatasetLoader(dataset_name=dataset_name, root_dir = './dataset/image/stl10/original', transform= transform,batch_size=8)

    train_loader_orig = custom_data_orig.create_dataloader(split='train')
    # Create DataLoader for test set

    test_loader_orig = custom_data_orig.create_dataloader(split='test')
    test_loader_anomolous = custom_data.create_dataloader(split='test')

    # Get the number of unique classes
    num_classes_anomolous = len(test_loader_anomolous.dataset.dataset.classes)
    num_classes_original = len(test_loader_orig.dataset.dataset.classes)

    print("Number of classes Anomolous:", num_classes_anomolous)
    print("Number of classes Original:", num_classes_original)

    # model_orig = InceptionV3(num_classes=num_classes_original, channel=3, num_epochs=epochs, save_dir=save_dir)
    # model_orig.create_model()
    # # model_orig.load_model('dataset/image/stl10/trained_model/InceptionV3/InceptionV3_42_model_weights.pth')
    # # predictions = model_orig.predict_proba(train_loader_orig)
    # # print(predictions)
    # # #
    # model_orig.fit(device,train_loader_orig)



    X_test, y_test = custom_data.dataloader_to_dataframe(test_loader_anomolous)

    # test_loader_anom_svm = custom_data.dataloader_to_dataframe(test_loader_anomolous)

    # print(type(test_loader_anom_svm['input_data']))






    # Class IDx along with there Labels
    class_idx = test_loader_orig.dataset.dataset.class_to_idx



    channel = 3
    #dataset.get_channel()


    # clf1 = clf = svm.SVC(kernel='linear')
    # model = TabularClassifier(clf1)
    # model.fit(test_loader_svm['input_data'], test_loader_svm['target'])

    # # Load the saved model from file
    loaded_model = joblib.load('dataset/image/stl10/abnormal/trained_model/svm_model.pkl')

    # Make predictions using the loaded model
    y_pred = loaded_model.predict(X_test)
    df = pd.read_csv('SVM_abnormal.csv')
    df['Predicted Label'] = y_pred
    df.to_csv('SVM_abnormal.csv',  index=False)


    # Create an instance of the Anomolous ResNet class
    # model_anom = ResNet(num_classes=num_classes_anomolous,channel=channel, num_epochs=epochs, save_dir = save_dir)
    # model_anom.create_model()
    # model_anom.load_model('dataset/image/stl10/abnormal/trained_model/ResNet_29_model_weights.pth')
    # # model.fit(device,train_loader)
    # resnet_predictions_anom = model_anom.predict_proba(test_loader_anomolous)
    # labels = model_anom.predict(test_loader_anomolous)
    # model_anom.save_csv('abnormal.csv')
    # df = pd.read_csv('abnormal.csv')




    def find_class(text):
        for class_name, value in class_idx.items():
            if class_name in text:
                return value
        return None


    # Add new column to DataFrame
    df['True CLF Labels'] = df['File Path'].apply(find_class)


    custom_data = GenericDatasetLoader( dataset_name= 'CUSTOM',data_frame=df, transform=transform,batch_size=1)

    # Create data loader
    data_loader = custom_data.create_dataloader(split='test')


    # Create an instance of the Main CLF class
    model_orig = InceptionV3(num_classes=num_classes_original,channel=channel, num_epochs=epochs, save_dir = save_dir)
    model_orig.create_model()
    model_orig.load_model('dataset/image/stl10/trained_model/InceptionV3/InceptionV3_42_model_weights.pth')
    # model.fit(device,train_loader)
    predictions_orig = model_orig.predict_proba(data_loader)
    labels = model_orig.predict(data_loader)
    print(predictions_orig)

    # df = pd.read_csv('filtered_predictions_with_mainCLF.csv')

    df['Predicted CLF'] = predictions_orig
    df['True CLF Labels'] = labels
    df['Probabilities'] = model_orig.get_probability()

    df.to_csv('Result_with_SVM_InceptionV3.csv', index = False)





    # clf1 = clf = svm.SVC(kernel='linear')
    # model = TabularClassifier(clf1)
    # model.fit(X_train, y_train)
    # joblib.dump(model, save_dir+'/svm_model.pkl')
    # print(model.predict(X_test))
    #
    # clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    # model2 = TabularClassifier(clf1)
    #
    # clf3 = GaussianNB()
    # model3 = TabularClassifier(clf1)
    # eclf1 = VotingClassifier(estimators=[
    #             ('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
    #
    # model = TabularClassifier(eclf1)
    # model.fit(X_train,y_train)
    # print(model.predict(dataset.X_test))

    # estimators = [
    #          ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    #     ('svr', make_pipeline(StandardScaler(), LinearSVC(dual="auto", random_state=42)))
    #     ]
    #
    # clf = StackingClassifier(
    #          estimators=estimators, final_estimator=LogisticRegression()
    #     )
    #
    # # svm = SVC(probability=True)
    # model = TabularClassifier(clf)
    # model.fit(X_train, y_train)


    # proba = model.predict_proba(dataset.X_test)
    # print(proba)

    #base_model_list = ["AlexNet", "ResNet"]
#     base_models_created = [resnet_model, alexnet_model]
#     meta_model =  svm.SVC(probability=True, kernel='rbf')
    # model = Stacking(base_models_created,meta_model)
    #model = Voting(base_models_created)

    # model.fit(device,dataset,is_stacked=False)
    #model.fit(device,dataset)

    # print(model.predict_proba(dataset))
    #print(models.model.called_classes)



    # for batch_data, batch_file_names in test_loader:
    #     # Perform prediction on each batch_data
    #     predictions = model.predict_proba(batch_data)
    #     # Process predictions and associate them with file_names as needed
    #     for prediction, file_name in zip(predictions, batch_file_names):
    #         # Associate each prediction with its corresponding file_name
    #         print("Prediction for {}: {}".format(file_name, prediction))





    # Create an instance of the AlexNet class
    # model = AlexNet(num_classes=10,channel=channel, num_epochs= airplane)
    # model.create_model()
    # model.fit(device,dataset)
    # alexnet_predictions = model.predict_proba(dataset)
    # labels = model.predict(dataset)
    #
    # confusion_matrix = model.ConfusionMatrix(alexnet_predictions,labels)
    # print(confusion_matrix)
    #
    # model_list = Voting([resnet_predictions,alexnet_predictions])
    # # Apply majority voting on the predictions
    # ensemble_result = model_list.majority_voting()
    #
    # print("Majority Vote Result:", ensemble_result)


# Create an instance of the AlexNet class
    #model = Voting(classifierstovote=['Alex']AlexNet(num_classes=10,channel=channel, num_epochs= airplane)
    # model.create_model()
    # model.fit(device,dataset)
    # predictions = model.predict_proba(dataset)
    # labels = model.predict(dataset)


