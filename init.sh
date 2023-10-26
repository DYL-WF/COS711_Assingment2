#======== INITIALIZATION =========#
# Initialize env
python3 -m venv env

# Activate python env
source ./env/bin/activate

# Install Dependencies
pip install -r requirements.txt

#======= DOWNLAD IMAGES ======#
#!this will take a while
python3 download_data.py