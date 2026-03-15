# raad2-gfx GPU Training Cheatsheet

Quick reference of commands used during the **raad2-gfx GPU training**.

---

## Table of Contents

**Part 1: Essential Commands**
1. [Connect to the Cluster](#1-connect-to-the-cluster)
2. [Request GPU Session](#2-request-gpu-session)
3. [Check GPU Availability](#3-check-gpu-availability)
4. [Software Environment](#4-software-environment)
5. [Running Python Scripts](#5-running-python-scripts)
6. [PBS Job Management](#6-pbs-job-management)
7. [Monitor Jobs](#7-monitor-jobs)

**Part 2: Advanced Commands**
8. [PBS Advanced Examples](#8-pbs-advanced-examples)
9. [GPU Monitoring & Debugging](#9-gpu-monitoring--debugging)
10. [File Transfer](#10-file-transfer)
11. [Environment Management](#11-environment-management)
12. [Performance Optimization](#12-performance-optimization)
13. [Troubleshooting](#13-troubleshooting)
14. [Jupyter Lab Setup](#14-jupyter-lab-setup)
15. [Linux Commands Reference](#15-linux-commands-reference)
16. [Resources & Support](#16-resources--support)

---

# Part 1: Essential Commands

## 1. Connect to the Cluster

Connect using SSH:
```bash
ssh username@raad2-gfx.hbku.edu.qa
```

> **Note:** If VPN is required, connect to the **QBRI VPN** first.

---

## 2. Request GPU Session

**Interactive GPU session (recommended for testing):**
```bash
sinteractive
```

**Manual request with specific resources:**
```bash
qsub -I -l select=1:ncpus=8:ngpus=1 -l walltime=02:00:00
```

Exit interactive session:
```bash
exit
```

---

## 3. Check GPU Availability

Check GPUs on node:
```bash
nvidia-smi
```

Continuous monitoring:
```bash
watch -n 1 nvidia-smi
```

---
## 4. Software Environment
### 4.1 Environment Modules
```bash
module avail       # List available modules
module list        # List loaded modules
module load name   # Load module
module unload name # Unload module
module purge       # Unload all modules
```

### 4.2 Conda Environments

**Load conda:**
```bash
module load anaconda
```

**Create environment with Python 3.11:**
```bash
conda create -n torch-gpu python=3.11
```

**Activate environment:**
```bash
conda activate torch-gpu
```

**Install PyTorch with GPU support:**
```bash
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```

**Deactivate environment:**
```bash
conda deactivate
```

**List environments:**
```bash
conda env list
```

---

## 5. Running Python Scripts

**Run training script:**
```bash
python train.py
```

**Quick GPU test:**
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

---

## 6. PBS Job Management

**Create job script:**
```bash
vi train_gpu.pbs
```

**Basic PBS script template:**
```bash
#!/bin/bash
#PBS -N pytorch_training
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=04:00:00
#PBS -j oe

cd $PBS_O_WORKDIR

module load anaconda
conda activate torch-gpu

python train.py
```

**Submit job:**
```bash
qsub train_gpu.pbs
```

**Delete job:**
```bash
qdel JOB_ID
```

---

## 7. Monitor Jobs

**List your jobs:**
```bash
qstat
```

**Detailed job info:**
```bash
qstat -f JOB_ID
```

**View job output in real-time:**
```bash
tail -f pytorch_training.oJOBID
```

**Check queue status:**
```bash
qstat -Q
```

---

# Part 2: Advanced Commands

## 8. PBS Advanced Examples

### Multi-GPU Training
```bash
#!/bin/bash
#PBS -N multi_gpu_job
#PBS -l select=1:ncpus=16:ngpus=2
#PBS -l walltime=08:00:00
#PBS -j oe

cd $PBS_O_WORKDIR
module load anaconda
conda activate torch-gpu

python -m torch.distributed.launch --nproc_per_node=2 train.py
```

### Array Jobs (Multiple Experiments)
```bash
#!/bin/bash
#PBS -N experiment_array
#PBS -J 1-5
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=02:00:00
#PBS -j oe

cd $PBS_O_WORKDIR
module load anaconda
conda activate torch-gpu

python train.py --config config_${PBS_ARRAY_INDEX}.yaml
```

### Job with Email Notifications
```bash
#!/bin/bash
#PBS -N training_notify
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=04:00:00
#PBS -m abe
#PBS -M your.email@hbku.edu.qa
#PBS -j oe

cd $PBS_O_WORKDIR
module load anaconda
conda activate torch-gpu

python train.py
```

> **Email flags:** `-m abe` = abort, begin, end

---

## 9. GPU Monitoring & Debugging

### Advanced GPU Monitoring

**Monitor specific GPU metrics:**
```bash
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total --format=csv -l 10
```

**Log GPU stats to file:**
```bash
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total --format=csv -l 10 > gpu_log.csv
```

**Show GPU processes:**
```bash
nvidia-smi pmon -c 1
```

### Python GPU Debugging

**Detailed CUDA information:**
```bash
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('cuDNN version:', torch.backends.cudnn.version()); print('Device count:', torch.cuda.device_count())"
```

**Check GPU memory usage:**
```python
import torch
print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
print(f"GPU Memory Reserved: {torch.cuda.memory_reserved(0)/1e9:.2f} GB")
print(f"GPU Memory Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0))/1e9:.2f} GB")
```

**Clear GPU cache:**
```python
import torch
torch.cuda.empty_cache()
```

**Monitor GPU in real-time from another terminal:**
```bash
# SSH to the compute node running your job
ssh gfx2  # Replace with your node name
watch -n 1 nvidia-smi
```

---

## 10. File Transfer

### Basic Transfer

**Download file from cluster:**
```bash
scp username@raad2-gfx.hbku.edu.qa:/path/to/file .
```

**Upload file to cluster:**
```bash
scp file.py username@raad2-gfx.hbku.edu.qa:/path/to/destination/
```

**Copy directories recursively:**
```bash
scp -r folder username@raad2-gfx.hbku.edu.qa:/path/to/destination/
```

### Advanced Transfer (Large Files)

**Use rsync for resumable transfers:**
```bash
rsync -avz --progress dataset/ username@raad2-gfx.hbku.edu.qa:/path/to/data/
```

**Compress before transfer:**
```bash
# On local machine
tar -czf dataset.tar.gz dataset/

# Transfer
scp dataset.tar.gz username@raad2-gfx.hbku.edu.qa:/path/

# On cluster - extract
tar -xzf dataset.tar.gz
```

**Transfer with bandwidth limit (nice to others):**
```bash
rsync -avz --progress --bwlimit=10000 dataset/ username@raad2-gfx.hbku.edu.qa:/path/
```

---

## 11. Environment Management

### Export and Share Environments

**Export complete environment:**
```bash
conda env export > environment.yml
```

**Create environment from file:**
```bash
conda env create -f environment.yml
```

**Export minimal package list:**
```bash
conda list --export > requirements.txt
```

**Clone existing environment:**
```bash
conda create --name new-env --clone existing-env
```

### Clean Up Environments

**Remove unused packages:**
```bash
conda clean --all
```

**Remove environment:**
```bash
conda env remove -n old-env
```

**List package sizes:**
```bash
conda list --show-channel-urls
```

---

## 12. Performance Optimization

### Mixed Precision Training (Faster on V100)
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Optimize Data Loading
```python
from torch.utils.data import DataLoader

# Pin memory for faster GPU transfer
train_loader = DataLoader(
    dataset, 
    batch_size=32, 
    pin_memory=True, 
    num_workers=4
)
```

### Set Threading
```bash
# In PBS script or before running Python
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

### Benchmark Mode (Faster for Fixed Input Sizes)
```python
import torch
torch.backends.cudnn.benchmark = True
```

---

## 13. Troubleshooting

### Problem: `torch.cuda.is_available()` returns False
```bash
# Check PyTorch installation
conda list | grep torch
# Should show pytorch with cuda in build string

# Reinstall if needed
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia --force-reinstall
```

### Problem: `ImportError: undefined symbol: iJIT_NotifyEvent`
```bash
# Fix MKL version
conda install mkl=2023.1.0
```

### Problem: Out of GPU Memory
```python
# In your script - clear cache
import torch
torch.cuda.empty_cache()

# Or reduce batch size
batch_size = 16  # Instead of 32
```
```bash
# Check what's using GPU memory
nvidia-smi
```

### Problem: Job Stuck in Queue
```bash
# Check queue status
qstat -Q

# Check detailed job info
qstat -f JOB_ID

# Check node availability
pbsnodes -a
```

### Problem: Module Not Found After Installing
```bash
# Make sure environment is activated
conda activate torch-gpu

# Verify package is installed
conda list | grep package-name

# Try reinstalling
conda install package-name --force-reinstall
```

### Problem: Conda Environment Not Found in PBS Job
```bash
# In your PBS script, make sure to source conda
source /path/to/anaconda/etc/profile.d/conda.sh
conda activate torch-gpu
```

---

## 14. Jupyter Lab Setup

### Launch Jupyter on GPU Node

**Start Jupyter:**
```bash
# Request GPU node first
sinteractive

# Load modules
module load anaconda
conda activate torch-gpu

# Start Jupyter
jupyter-lab --no-browser --port=8888
```

### SSH Tunnel from Local Machine

**On your local machine:**
```bash
ssh -L 8888:localhost:8888 username@raad2-gfx.hbku.edu.qa
```

Then open the URL shown in Jupyter output in your browser.

### SSH Tunnel to Specific Compute Node

**If Jupyter is running on compute node (e.g., gfx2):**
```bash
# On your local machine
ssh -L 8888:gfx2:8888 username@raad2-gfx.hbku.edu.qa
```

---

## 15. Linux Commands Reference

### File Operations
```bash
pwd                # Show current directory
ls                 # List files
ls -lh             # Detailed list with human-readable sizes
ls -lht            # Sort by modification time
cd directory_name  # Change directory
mkdir myproject    # Create directory
rm file.txt        # Remove file
rm -r folder/      # Remove directory recursively
cp file1 file2     # Copy file
mv old new         # Rename/move file
```

### File Viewing
```bash
cat file.txt       # Display entire file
head file.txt      # First 10 lines
tail file.txt      # Last 10 lines
tail -n 50 file.txt # Last 50 lines
tail -f output.log  # Follow file in real-time
less file.txt      # Page through file
```

### File Editing
```bash
nano script.py     # Simple editor
vi script.py       # Advanced editor
```

### System Monitoring
```bash
top                # CPU/memory usage
htop               # Better top (if available)
df -h              # Disk usage
du -sh *           # Size of directories
free -h            # Memory usage
```

### Process Management
```bash
ps aux             # List all processes
ps aux | grep python  # Find Python processes
kill PID           # Kill process by ID
killall python     # Kill all Python processes
```

---

## 16. Resources & Support

### Documentation

- **PyTorch Documentation:** https://pytorch.org/docs
- **PyTorch Tutorials:** https://pytorch.org/tutorials
- **NVIDIA CUDA:** https://developer.nvidia.com/cuda-zone
- **NVIDIA GPU Cloud:** https://ngc.nvidia.com
- **PBS Professional:** https://www.altair.com/pbs-professional/
- **Conda Documentation:** https://docs.conda.io

### Best Practices & Tips

- **Never run training on login nodes** - Always use `sinteractive` or submit jobs
- **Save checkpoints regularly** - Don't lose hours of training
- **Monitor resources** - Check GPU utilization to ensure efficient usage
- **Don't share credentials** - Keep passwords and API keys private
- **Log your experiments** - Keep track of hyperparameters and results
- **Clean up old files** - Remove unused environments and datasets
- **Test interactively first** - Debug on `sinteractive` before batch jobs
- **Request appropriate walltime** - Don't request more than needed

### RCCG Support

For questions or technical issues, contact:

📧 **rccg@hbku.edu.qa**

---

## Quick Command Summary

| Task | Command |
|------|---------|
| Connect to cluster | `ssh username@raad2-gfx.hbku.edu.qa` |
| Get GPU node | `sinteractive` |
| Check GPU | `nvidia-smi` |
| Load conda | `module load anaconda` |
| Activate env | `conda activate torch-gpu` |
| Run script | `python train.py` |
| Submit job | `qsub train.pbs` |
| Check jobs | `qstat` |
| Delete job | `qdel JOB_ID` |
| View output | `tail -f jobname.oJOBID` |

---

**Last Updated:** March 2026
