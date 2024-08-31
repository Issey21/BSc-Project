import torch as th
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time  
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# MLP Model 
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        layers = []
        in_features = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size)) #For linear transformation
            layers.append(nn.ReLU())  # Activation function
            in_features = hidden_size
        layers.append(nn.Linear(in_features, output_size)) #Applying linear on the output
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# Trainig the model
def train_model(train_loader, val_loader, input_size, hidden_sizes, output_size, num_epochs=10):
    print(f'Input size: {input_size}')
    
    # Start timing the training process
    script_start = time.time()

    model = MLP(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) #optimised learning rate

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model.to(device)
    
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        training_loss = 0.0
        
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            outputs = model(batch_data)
            loss = criterion(outputs.squeeze(), batch_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            training_loss += loss.item() * batch_data.size(0)

        # Validation loss
        model.eval()
        val_loss = 0.0
        
        with th.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                outputs = model(batch_data)
                loss = criterion(outputs.squeeze(), batch_labels)
                val_loss += loss.item() * batch_data.size(0)
        
        training_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        
        train_losses.append(training_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {training_loss:.4f}, Validation Loss: {val_loss:.4f}')
        

    # Plot learning curves
    plt.figure(figsize=(12, 6))
    plt.plot(range(num_epochs), train_losses, label='Training Loss')
    plt.plot(range(num_epochs), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.show()

    # Calculate and print the total training time
    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')

    return model



def evaluate_model(model, test_loader):
    model.eval()
    all_predictions = []
    all_targets = []

    with th.no_grad():
        for batch_data, batch_labels in test_loader:
            outputs = model(batch_data)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(batch_labels.cpu().numpy())

    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    # Calculate metrics
    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)

    # Print metrics
    print(f'MSE: {mse:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'R-squared: {r2:.4f}')
    
    # Plot predictions vs. actual values
    plt.figure(figsize=(12, 6))
    plt.scatter(all_targets, all_predictions, alpha=0.5)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs. Actual Values')
    plt.plot([min(all_targets), max(all_targets)], [min(all_targets), max(all_targets)], 'r--', lw=2)
    plt.show()
