import os
os.chdir('./dataset/valid_rgb/val_data/')
dirs = os.listdir(os.getcwd())
for dir in dirs:
    os.rename(dir,dir.zfill(4))

