#!/bin/bash
if ! which python &> /dev/null; then
  echo "Python interpreter not found. Please install Python 3."
  exit 1
fi
if which python3 &> /dev/null; then
  PYTHON="python3"
else
  PYTHON="python"
fi
if ! which wget &> /dev/null; then
  echo "wget command not found. Installing wget..."
  brew install wget
fi

echo "Downloading latest AEIONic Diffusion Deluxe and running in a Python Virtual Environment"
if [ ! -d "./venv" ] ; then
  $PYTHON -m venv ./venv || { echo "Failed to create the virtual environment"; exit 1; }
fi

# Download script using curl if wget is not available
if ! which wget &> /dev/null; then
  curl -o ./venv/AEIONic-Diffusion-Deluxe.py https://raw.githubusercontent.com/Skquark/AEIONic/main/AEIONic-Diffusion-Deluxe/AEIONic-Diffusion-Deluxe.py || { echo "Failed to download AEIONic Diffusion Deluxe script"; exit 1; }
else
  wget -O ./venv/AEIONic-Diffusion-Deluxe.py https://raw.githubusercontent.com/Skquark/AEIONic/main/AEIONic-Diffusion-Deluxe/AEIONic-Diffusion-Deluxe.py || { echo "Failed to download AEIONic Diffusion Deluxe script"; exit 1; }
fi

cd ./venv
if ! source ./bin/activate; then
  source ./venv/bin/activate
fi
if [ $? -ne 0 ]; then
  echo "Failed to activate the virtual environment"; exit 1;
fi

./bin/pip install --upgrade --quiet pip
./bin/pip install --upgrade --quiet flet
$PYTHON ./AEIONic-Diffusion-Deluxe.py --saved_settings_json ./aeionic-settings.json || { echo "Failed to run AEIONic Diffusion Deluxe script"; exit 1; }

deactivate
exit
