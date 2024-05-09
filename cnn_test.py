import os
import torch
import glob
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, hamming_loss
import numpy as np

directory_path_test = './mfcc_post_processing_test'
random.seed(666)
file_pattern = "*.csv"
csv_files_test = glob.glob(os.path.join(directory_path_test, file_pattern))
dataframes_test = []

for file in csv_files_test:
    df = pd.read_csv(file)
    dataframes_test.append(df)
pd_test = pd.concat(dataframes_test, ignore_index=True)

# Process labels for multi-label classification
def process_labels(value):
    value = str(value)
    return [int(v) for v in value.split(';')]

test_labels = pd_test['Instruments'].apply(process_labels)
all_labels = list(range(0,12)) 
mlb = MultiLabelBinarizer(classes = all_labels)
labels_encoded = mlb.fit_transform(test_labels)

#print(labels_encoded[1])

features = pd_test.drop('Instruments', axis=1)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

test_features = torch.tensor(features_scaled).float()
test_labels = torch.tensor(labels_encoded).float()

test_data = TensorDataset(test_features, test_labels)
batch_size = 32

test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last = True)


class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(32, 64, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(64, hidden_size, kernel_size=3, stride=1)
        self.pool = nn.MaxPool1d(2, stride=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(7, 12)  
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = torch.flatten(x,1)
        x = self.fc1(x)
        return torch.sigmoid(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(input_size = 13, hidden_size=32, num_layers=2, num_classes=3).to(device)
model_path = "./cnn_v4_state.pth"  # Provide the path to your .pth file
##THE MODEL CNN_V2 IS HIGH RECALL 
##MODEL CNN_V1 HAS LOW HAMMING LOSS

model.load_state_dict(torch.load(model_path))
model.eval()

true_labels = []
model_predictions = []
ran = False
with torch.no_grad():
    for inputs, labels in test_loader:
        # Move data to the device
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Make predictions using the model
        outputs = model(inputs)
        # Convert outputs to probabilities and get the predicted class labels
        predicted_labels = torch.sigmoid(outputs)  # For BCEWithLogitsLoss, use sigmoid activation
        predicted_labels = (predicted_labels > 0.5).float()  # Convert probabilities to binary labels
        
        true_labels.extend(labels.cpu().numpy())
        model_predictions.extend(predicted_labels.cpu().numpy())
model_predictions = np.concatenate(model_predictions).tolist()
true_labels = np.concatenate(true_labels).tolist()


# Calculate performance metrics
hamming = hamming_loss(true_labels, model_predictions)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, model_predictions, average='macro')

        
# Print performance metrics
print(f'Hamming Loss: {hamming:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')