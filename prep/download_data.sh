# Download HumanML3D dataset for Generation only
git clone https://github.com/EricGuo5513/HumanML3D.git
unzip ./HumanML3D/HumanML3D/texts.zip -d ./HumanML3D/HumanML3D/
cp -r HumanML3D/HumanML3D external/motion-diffusion-model/dataset/HumanML3D
rm -fdr HumanML3D

# Download NTU RGB+D data
mkdir -p data/NTU60
wget -P data/NTU60 https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_3danno.pkl

# Done!
echo "Stored datasets on data/ directory"