python -W ignore ./fddb_test.py --trained_model weights/WIDERFace_DSFD_RES152.pth --split_dir /home/ubuntu/dataset/FDDB-folds/ \
--det_dir ../evaluation/rime/g2 --gamma 2  --data_dir /home/ubuntu/dataset/fddb_images/ \
--preprocess rime --pretrain_under weights/sice_under_old.pth --pretrain_mix weights/sice_mix_old.pth --pretrain_over weights/sice_over_old.pth

python -W ignore ./fddb_test.py --trained_model weights/WIDERFace_DSFD_RES152.pth --split_dir /home/ubuntu/dataset/FDDB-folds/ \
--det_dir ../evaluation/rime/g5 --gamma 5  --data_dir /home/ubuntu/dataset/fddb_images/ \
--preprocess rime --pretrain_under weights/sice_under_old.pth --pretrain_mix weights/sice_mix_old.pth --pretrain_over weights/sice_over_old.pth

python -W ignore ./fddb_test.py --trained_model weights/WIDERFace_DSFD_RES152.pth --split_dir /home/ubuntu/dataset/FDDB-folds/ \
--det_dir ../evaluation/rime/g10 --gamma 10  --data_dir /home/ubuntu/dataset/fddb_images/ \
--preprocess rime --pretrain_under weights/sice_under_old.pth --pretrain_mix weights/sice_mix_old.pth --pretrain_over weights/sice_over_old.pth

python -W ignore ./fddb_test.py --trained_model weights/WIDERFace_DSFD_RES152.pth --split_dir /home/ubuntu/dataset/FDDB-folds/ \
--det_dir ../evaluation/rime/g20 --gamma 20  --data_dir /home/ubuntu/dataset/fddb_images/ \
--preprocess rime --pretrain_under weights/sice_under_old.pth --pretrain_mix weights/sice_mix_old.pth --pretrain_over weights/sice_over_old.pth

python -W ignore ./fddb_test.py --trained_model weights/WIDERFace_DSFD_RES152.pth --split_dir /home/ubuntu/dataset/FDDB-folds/ \
--det_dir ../evaluation/rime/g30 --gamma 30  --data_dir /home/ubuntu/dataset/fddb_images/ \
--preprocess rime --pretrain_under weights/sice_under_old.pth --pretrain_mix weights/sice_mix_old.pth --pretrain_over weights/sice_over_old.pth

python -W ignore ./fddb_test.py --trained_model weights/WIDERFace_DSFD_RES152.pth --split_dir /home/ubuntu/dataset/FDDB-folds/ \
--det_dir ../evaluation/rime/g50 --gamma 50  --data_dir /home/ubuntu/dataset/fddb_images/ \
--preprocess rime --pretrain_under weights/sice_under_old.pth --pretrain_mix weights/sice_mix_old.pth --pretrain_over weights/sice_over_old.pth