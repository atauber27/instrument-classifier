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

# Assuming you have a list of CSV file paths
directory_path_train = './mfcc_post_processing'
random.seed(666)
file_pattern = "*.csv"
csv_files_train = glob.glob(os.path.join(directory_path_train, file_pattern))
sample_percentage = 100
num_files_to_sample = int(len(csv_files_train) * (sample_percentage / 100.0))
csv_files_train = random.sample(csv_files_train, num_files_to_sample)
dataframes_train = []

for file in csv_files_train:
    df = pd.read_csv(file)
    dataframes_train.append(df)
pd_train = pd.concat(dataframes_train, ignore_index=True)

# Process labels for multi-label classification
def process_labels(value):
    value = str(value)
    return [int(v) for v in value.split(';')]
    
labels = pd_train['Instruments'].apply(process_labels)
all_labels = list(range(0,12)) 
mlb = MultiLabelBinarizer(classes = all_labels)
labels_encoded = mlb.fit_transform(labels)


# Prepare features
features = pd_train.drop('Instruments', axis=1)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

X_train, X_val, y_train, y_val = train_test_split(features_scaled, labels_encoded, test_size=.2, random_state=42)

# Convert to tensors
train_features = torch.tensor(X_train).float()
train_labels = torch.tensor(y_train).float()
val_features = torch.tensor(X_val).float()
val_labels = torch.tensor(y_val).float()

train_data = TensorDataset(train_features, train_labels)
for idx, (inputs, labels) in enumerate(train_data):
    print(f"Sample {idx}:")
    print(f"  Inputs: {inputs}")
    print(f"  Labels: {labels}")
    # If you want to print a limited number of samples (e.g., the first 5), you can use an if statement
    if idx >= 5:  # Limit to printing the first 5 samples for brevity
        break

val_data = TensorDataset(val_features, val_labels)
batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last = True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last = True)

# Pad or truncate sequences to a fixed length
# You may need to adjust this based on your data
# For example, you can use torch.nn.utils.rnn.pad_sequence

# Step 2: Define the CNN Model
class CNN(nn.Module): ##THIS NEEDS TO BE HEAVILY EDITTED IDK WHAT IM DOING >< 
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(CNN, self).__init__()
        # Define your CNN architecture here
        self.conv1 = nn.Conv1d(32, 64, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(64, hidden_size, kernel_size=3, stride=1)
        self.pool = nn.MaxPool1d(2, stride=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(7, 12)  
    def forward(self, x):
        # Define the forward pass of your CNN
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

# Step 3: Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = train_features.shape[1]
num_classes = train_labels.shape[1]
print(input_size, num_classes)
model = CNN(input_size=train_features.shape[1], hidden_size=32, num_layers=5, num_classes=train_labels.shape[1]).to(device)
model_path = "./cnn_v4_state.pth"  # Provide the path to your .pth file
model.load_state_dict(torch.load(model_path))
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
num_epochs = 40

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)


for epoch in range(num_epochs):
    print("epoch number: " + str(epoch))
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = validate_epoch(model, val_loader, criterion, device)
    print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

torch.save(model.state_dict(), './cnn_v4_state.pth')
torch.save(model, './cnn_v4_pth')