dirname=${PWD##*/} 
if [ "$dirname" != "src" ]; then
    echo "please run from the script directory"
    exit
fi

## link up datasets

# LEAStereo

ln -s "/media/makspll/General Storage/Datasets/datasets/kitti2015" ../reproductions/LEAStereo/dataset
# ln -s /mnt/data/StereoDataset/dataset/kitti2012 ./dataset
# ln -s /mnt/data/StereoDataset/dataset/SceneFlow ./dataset
# ln -s /mnt/data/StereoDataset/dataset/MiddEval3 ./dataset



