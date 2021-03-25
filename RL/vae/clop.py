from PIL import Image
import glob,os

#files = glob.glob(os.path.join('/content/dataset_root', DATASET_DIR, '*.jpg'))
files = glob.glob(os.path.join('./RL/vae/dataset_img/original_data/dataset/train_data2/*.jpg'))


print(len(files))

output_dir='./RL/vae/dataset_img/crop_data/crop_64_dataset'

for f in files:
  #print(files)
  #print(f)
  #print(f.split('/')[-2])
  #print(f.split('\\')[-1])
  #exit()
  try:
    image = Image.open(f)
    
  except OSError:
    print('Delete' + f)
    #!rm -rf f

  image_name=f.split('\\')[-1]
  image = image.resize((160,120))
  image.crop((0, 56, 160, 120)).save(f'{output_dir}/{image_name}', quality=95)
  #image = image.resize((160,80))
  print(image.size)
  