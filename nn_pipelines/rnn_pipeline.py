import glob
import os
import torch
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence

directory_path_train = './mfcc_post_processing'
file_pattern = "*.csv"
csv_files_train = glob.glob(os.path.join(directory_path_train, file_pattern))

if not csv_files_train:
    raise Exception("No CSV files found. Check the directory path and file pattern.")

sample_percentage = 1
num_files_to_sample = max(1, int(len(csv_files_train) * (sample_percentage / 100.0)))
csv_files_train = random.sample(csv_files_train, num_files_to_sample)

sequences = []
labels_list = []

for file in csv_files_train:
    df = pd.read_csv(file)
    if df.empty:
        continue
    features = df.drop('Instruments', axis=1)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    sequences.append(torch.tensor(scaled_features).float())
    
    # Process labels correctly assuming they are separated by semicolons
    labels = df['Instruments'].iloc[0].split(';')  # Assuming labels are in one cell separated by semicolons
    labels_list.append([int(label) for label in labels])

# Ensure all sequences have the same number of features
if not sequences:
    raise ValueError("No data was loaded into sequences. Please check the input files and preprocessing steps.")

input_size = sequences[0].shape[1]  # Number of features per timestep

# Process labels for multi-label classification
mlb = MultiLabelBinarizer()
labels_encoded = mlb.fit_transform(labels_list)
labels_encoded = [torch.tensor(label).float() for label in labels_encoded]

# Split the data
X_train, X_val, y_train, y_val = train_test_split(sequences, labels_encoded, test_size=0.2, random_state=42)

# Pad sequences and create dataloaders
def collate_fn(batch):
    sequences, labels = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return sequences_padded, labels

train_dataset = list(zip(X_train, y_train))
val_dataset = list(zip(X_val, y_val))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# RNN Model
class RNNInstrumentClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNInstrumentClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNNInstrumentClassifier(input_size=input_size, hidden_size=100, num_layers=2, num_classes=len(mlb.classes_))
model.to(device)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Training function
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.unsqueeze(1)  # Add a sequence dimension
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)

# Validation function
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.unsqueeze(1)  # Add a sequence dimension
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)


# Training loop
for epoch in range(10):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = validate_epoch(model, val_loader, criterion, device)
    print(f'Epoch {epoch+1}/30 - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# Save the trained model and scaler
torch.save(model.state_dict(), './rnn_model_state.pth')
torch.save(model, './rnn_model.pth')

print('Finished Training')
