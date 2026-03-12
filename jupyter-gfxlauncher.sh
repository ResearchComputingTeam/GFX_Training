#!/bin/bash
#
# Jupyter Launcher version 1.0
# Author: itamboli@hbku.edu.qa
# Group: Research Computing Core Group @ HBKU
# Usage: ./jupyter-gfxlauncher.sh <username> [port]
#
#############################################


USER_NAME=$1
#PORT=${2:-8787}

# Define colors
GREEN='\e[32m'
YELLOW='\e[33m'
RED='\e[31m'
RESET='\e[0m'

# Generate a free random port between 50000-60000
find_free_port() {
    while :; do
        port=$((50000 + RANDOM % 10000))
        if ! nc -z localhost $port 2>/dev/null; then
            echo $port
            return
        fi
    done
}

if [ -z "$USER_NAME" ]; then
  printf "\nPlease provide your raad2-gfx username.\n   Usage: $0 <username> \n Example: $0 itambol89\n\n"
  read -p 'Username: ' USER_NAME
  #  echo "Usage: $0 <username> [port]"
  # exit 1
fi

if [ -z "$USER_NAME" ]; then
  echo "Usage: $0 <username> [port]"
  exit 1
fi

PORT=$(find_free_port)
echo "[INFO] Using port $PORT"

echo -e ""
echo -e ">>> Logging into HPC as ${GREEN}$USER_NAME${RESET}"
echo -e ""
echo -e ">>> Will request PBS interactive session, Jupyter on port ${GREEN}$PORT${RESET}"

ssh -t ${USER_NAME}@raad2-gfx.biolab.local "
  echo '>>> When inside compute node, Jupyter will auto-start.'

  # Create helper script inside login node
  mkdir -p ~/.rccg
  cat > ~/.rccg/start_Jupyter_inner.sh <<'INNER'
#!/bin/bash
PORT=$PORT
NODE=\$(hostname)
# Define colors
GREEN='\e[32m'
RESET='\e[0m'
YELLOW='\e[33m'
RED='\e[31m'

echo -e ''
echo -e '>>> Running on compute node:' \${GREEN}\$NODE\${RESET}

# Load environment for modules
if [ -f /etc/profile ]; then
    source /etc/profile
fi


source /cm/shared/apps/anaconda/2024.10/etc/profile.d/conda.sh

#conda activate myJupyter
conda activate /ddn/sw/cm/shared/apps/conda/envs/pytorch_demo

LOGFILE=\$HOME/jupyter_\$\$.log

jupyter lab --no-browser --ip=0.0.0.0 --port=\$PORT > \$LOGFILE 2>&1 &

sleep 5

RAW_URL=\$(grep -m 1 -o 'http://[^ ]*token=[^ ]*' \$LOGFILE)
TOKEN_URL=\$(echo \$RAW_URL | sed 's@http://[^:]*:[0-9]*/@http://localhost:$PORT/@')

#TOKEN_URL=\$(grep -m 1 -o 'http://[^ ]*token=[^ ]*' \$LOGFILE)
echo '=================================================================='
echo -e ''
echo -e '>>> Open another terminal and run following command:'
echo ''
echo -e '    '\${YELLOW}'ssh -L '\$PORT':'\$NODE':'\$PORT' '\$USER'@raad2-gfx.biolab.local'\${RESET}
echo ''
echo -e '>>> Open this URL in your browser:'
echo ''
echo -e '    '\${YELLOW} \$TOKEN_URL \${RESET}
#echo -e '    '\${YELLOW}'http://localhost:'\$PORT\${RESET}
echo ''
echo '=================================================================='

bash
INNER

  chmod +x ~/.rccg/start_Jupyter_inner.sh

  # Launch interactive PBS job and run helper script inside
  qsub -I -N RStudioSession -l select=1:ncpus=4:ngpus=1:mem=16gb -l walltime=01:00:00 -- ~/.rccg/start_Jupyter_inner.sh
"
