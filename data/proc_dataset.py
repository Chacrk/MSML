import zipfile
import os

# unzip dataset
ds_file_zip = zipfile.ZipFile('mini-imagenet.zip')
ds_file_zip.extractall()
ds_file_zip.close()

# make filenames list
images_list = []
for root, dirs, files in os.walk('miniimagenet', topdown=False):
    for name in files:
        if '.jpg' in name:
            images_list.append(os.path.join(root, name))
print('len of image list={}'.format(len(images_list)))

if os.path.exists('miniimagenet/images') is False:
    os.mkdir('miniimagenet/images')

# move files to /images folder
for index, s in enumerate(images_list):
    if (index + 1 ) % 50 == 0:
        print('\r>> [{}/{}], cp {}'.format(index+1, len(images_list), s), end='')
    os.system('cp {} {}'.format(s, 'miniimagenet/images/'))
