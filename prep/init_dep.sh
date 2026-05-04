cd external/motion-diffusion-model

# Body models
mkdir -p body_models
unzip smpl.zip -d body_models/
rm smpl.zip

# Glove dep.
unzip glove.zip
rm glove.zip

#rm -rf kit
unzip t2m.zip
unzip kit.zip
rm t2m.zip
rm kit.zip

cd ../../
echo "Done installing MDM dependancies!"