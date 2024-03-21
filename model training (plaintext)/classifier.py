import argparse
import torch
import pickle
from model import CNN_Classifier, DataLoader, Dataset
import sys
import os

def load_data(filename):
    print("Loading data：", filename)
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data

def get_dataset(pathname):
    pathname = pathname + "/" if pathname[-1] != "/" else pathname
    train_df = load_data(pathname + "train.pkl")
    eval_df = load_data(pathname + "valid.pkl")
    test_df = load_data(pathname + "test.pkl")
    return train_df, eval_df, test_df

def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='The dir path of dataset', type=str, required=True)
    args = parser.parse_args()
    return args

def main():
    args = parse_options()
    hidden_size = 256
    data_path = args.input
    train_df, eval_df, test_df = get_dataset(data_path)
    classifier = CNN_Classifier(result_save_path = data_path.replace("dataset", "models"), epochs = 100, hidden_size = hidden_size)
    classifier.preparation(
        X_train=train_df['data'],
        y_train=train_df['label'],
        X_valid=eval_df['data'],
        y_valid=eval_df['label'],
    )
    classifier.train()
    
    for metric in ['F1']:
        model_path = os.path.join(data_path.replace("dataset", "models"), f'best_model_{metric}.pt')
        misclassified_filenames = []
        if os.path.exists(model_path):
            print(f"\nLoading the best model for {metric}...")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            classifier.model.load_state_dict(torch.load(model_path, map_location=device))
            classifier.model.eval()

            test_set = Dataset(test_df['data'], test_df['label'], classifier.max_len, classifier.hidden_size)
            test_loader = DataLoader(test_set, batch_size=classifier.batch_size, shuffle=False)

            print(f"\nEvaluating the best model for {metric} on the test set...")
            # test_loss, test_score, classification_details = classifier.eval_with_classification_details(test_loader, test_df)
            # test_loss, test_score, misclassified_indices = classifier.eval_with_misclassified_indices(test_loader)
            test_loss, test_score = classifier.eval()
            print(f"Test loss: {test_loss}")
            print(f"Test score for {metric}: {test_score}\n")
            
            # misclassified_filenames = [test_df['filename'][i] for i in misclassified_indices]
        
            # # Save misclassified filenames to a text file
            # with open(os.path.join(data_path.replace("pkl", "models"), "misclassified_filenames.txt"), "w") as f:
            #     for filename in misclassified_filenames:
            #         f.write(f"{filename}\n")

            # print(f"\nMisclassified filenames are saved to misclassified_filenames.txt.")
            
            # for category in ['TP', 'TN', 'FP', 'FN']:
            #     filenames = [test_df['filename'][i] for i in classification_details[category][:50]]
                
            #     with open(os.path.join(data_path.replace("pkl", "models"), f"{category}_filenames.txt"), "w") as f:
            #         for filename in filenames:
            #             f.write(f"{filename}\n")
                
            #     print(f"{category} filenames are saved.\n")

        else:
            print(f"\nNo best model found for {metric}.")


if __name__ == "__main__":
    main()