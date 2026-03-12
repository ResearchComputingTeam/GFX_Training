import torch
import time
from mpi4py import MPI                          # ADDED

# ADDED: MPI diagnostics
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(f"MPI initialized — rank {rank}/{size}, library: {MPI.Get_library_version()}")

# Check if GPU is available
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")  #ADDED
print(f"Rank {rank} → {device} ({torch.cuda.get_device_name(rank)})")

if rank == 0:                                   # CHANGED: only rank 0 prints
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")

# Create large random dataset
if rank == 0:
    print("Creating dataset...")
X = torch.randn(100000, 512, device=device)
y = torch.randint(0, 10, (100000,), device=device)

# ADDED: each MPI rank works on its own slice of the data
chunk = 100000 // size
X = X[rank * chunk:(rank + 1) * chunk]
y = y[rank * chunk:(rank + 1) * chunk]

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

# Training
if rank == 0:
    print("Starting training...\n")
batch_size = 512
epochs = 30
start_time = time.time()

for epoch in range(epochs):
    epoch_loss = 0
    for i in range(0, X.size(0), batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # ADDED: aggregate loss across all ranks
    avg_loss = epoch_loss / (X.size(0) // batch_size)
    global_loss = comm.reduce(avg_loss, op=MPI.SUM, root=0)
    if rank == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {global_loss/size:.4f}")

elapsed_time = time.time() - start_time
if rank == 0:
    print(f"\nTraining completed in {elapsed_time:.2f} seconds")
    print(f"Average time per epoch: {elapsed_time/epochs:.2f} seconds")
