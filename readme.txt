# run the bash `init.sh` script if you're on linux, else follow steps bellow.
./init.sh

#======== INITIALIZATION =========#
# Initialize env
python -m venv env

# Activate python env
source ./env/bin/activate

# Install Dependencies
pip install -r requirements.txt


#======= DOWNLAD IMAGES ======#
#!this will take a while
python download_data