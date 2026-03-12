import torch
import time
from torch.cuda.amp import autocast, GradScaler

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")

# Create large random dataset
print("Creating dataset...")
X = torch.randn(100000, 512, device=device)  # 100k samples, 512 features
y = torch.randint(0, 10, (100000,), device=device)  # 10 classes

# Simple neural network
model = torch.nn.Sequential(
    torch.nn.Linear(512, 1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 10)
).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

scaler = GradScaler()

# Training
print("Starting training...\n")
batch_size = 512
epochs = 30

start_time = time.time()

for epoch in range(epochs):
    epoch_loss = 0
    for i in range(0, X.size(0), batch_size):
        # Get batch
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        
        # Forward pass
        #outputs = model(X_batch)
        #loss = criterion(outputs, y_batch)
        

        # Enable Tensor Cores via AMP
        with autocast():
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)


        # Backward pass
        #optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()


        # Scaled backprop
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / (X.size(0) // batch_size)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

elapsed_time = time.time() - start_time
print(f"\nTraining completed in {elapsed_time:.2f} seconds")
print(f"Average time per epoch: {elapsed_time/epochs:.2f} seconds")
