# python Data_preprocessing.py --modality DWI --fold 1 --mode train
# python Data_preprocessing.py --modality DWI --fold 1 --mode test
# python Data_preprocessing.py --modality DCE --fold 1 --mode train
# python Data_preprocessing.py --modality DCE --fold 1 --mode test
# python Data_preprocessing.py --modality T2 --fold 1 --mode train
# python Data_preprocessing.py --modality T2 --fold 1 --mode test

import os
import pandas as pd
import numpy as np

from scipy import ndimage
import SimpleITK as sitk
import tensorflow as tf

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--modality', type=str, default='DWI', help='DWI / T2 / DCE')
parser.add_argument('--fold', type=int, default=1, help='fold 1 ~ 5')
parser.add_argument('--mode', type=str, default='train', help='train / test')


def preprocessing_2d_tumor_patch(img_pathes,roi_pathes,cnt):
    # Data Load
    img, roi = sitk.ReadImage(img_pathes), sitk.ReadImage(roi_pathes) 
    img_np, roi_np = sitk.GetArrayFromImage(img),sitk.GetArrayFromImage(roi)
    img_np, roi_np = img_np.transpose(1,2,0), roi_np.transpose(1,2,0)
    
    # Get the tumor slice
    _,_,nz = np.nonzero(roi_np)
    nnz = np.sort(np.unique(nz))
    mid_z = nnz[len(nnz)//2]
    print(f'ROI checking..\n nnz:{nnz},\n mid_z:{mid_z}')
    
    # 2.5D tumor slices
    window = 32
    if mid_z>=2 and mid_z<=img_np.shape[2]-2:
        img_slice = img_np[:,:,mid_z-2:mid_z+3:2] # 2.5d 224x224x3
        roi_slice = roi_np[:,:,mid_z-2:mid_z+3:2] # 2.5d 224x224x3
    
    elif mid_z>=1 and mid_z<=img_np.shape[2]-1: # If z slice +-2 is not possible, +-1
        img_slice = img_np[:,:,mid_z-1:mid_z+2] 
        roi_slice = roi_np[:,:,mid_z-1:mid_z+2] 
        
    elif mid_z==0: # If z slice is the first slice
        img_slice = img_np[:,:,mid_z:mid_z+3] 
        roi_slice = roi_np[:,:,mid_z:mid_z+3] 
    else:
        print(f'ROI checking..\n nnz:{nnz},\n mid_z:{mid_z}')
        print('CANNOT middle tumor slice...')
        return None
        
    if img_slice.shape[0]!=224:
        resize_img = ndimage.zoom(img_slice,(224/img_slice.shape[0],224/img_slice.shape[1],1)) 
        resize_roi = ndimage.zoom(roi_slice,(224/roi_slice.shape[0],224/img_slice.shape[1],1)) 
    else:
        resize_img = img_slice
        resize_roi = roi_slice

    # Get the patch location
    nx,ny = np.nonzero(resize_roi[:,:,1])
    nnx,nny = map(lambda x:np.sort(np.unique(x)),[nx,ny])
    mid_x,mid_y= map(lambda x: x[len(x)//2], [nnx,nny])

    # Crop to the patch
    resize_img = resize_img[mid_x-window:mid_x+window,mid_y-window:mid_y+window,:] # 2.5d 64x64x3
    resize_roi = resize_roi[mid_x-window:mid_x+window,mid_y-window:mid_y+window,:]

    # Noramlization
    resize_img = (resize_img-np.min(resize_img)) / (np.max(resize_img) -np.min(resize_img))
    resize_roi = resize_roi>0.5    
    norm_img=resize_img

    if np.min(norm_img)!=0.0 or np.max(norm_img)!=1.0:
        print('Wrong in normalization')
        print('pat # folder : ',img_pathes.split('/')[-2])
        print(f'1) Before shape and spacing:{img.GetSize()},{img.GetSpacing()}')
        print(f'2) Img np shape:{img_np.shape}')            
        print(f'3) Resize shape:{resize_img.shape}')
        print('4) Before min, max, mean',np.min(resize_img),np.max(resize_img),np.mean(resize_img))
        print('5) After min, max, mean',np.min(norm_img),np.max(norm_img),np.mean(norm_img))
        return None
    else:
        return resize_img
    

def create_tfrecord_tumor_one_patch(dataset,save_path):
    print("Start converting...")
    clinic=pd.read_excel('clinical_variables.xlsx',engine='openpyxl')
    writer = tf.io.TFRecordWriter(path=save_path)
    cnt,pat_cnt=0,0
    for data_img, data_roi in zip(dataset["img_path"],dataset["roi_path"]):
        pat_num=data_img.split('/')[-2] # img's folder name
        print('\n\n START!',pat_num, 'th patient processing and save...\n ###### slices num:',cnt, 'pat num :',pat_cnt) # patient number
        try:
            norm_img = preprocessing_2d_tumor_patch(data_img, data_roi, pat_num)

            pat_clinic = clinic[clinic['data_num']==int(pat_num)]
            pat_gt = int(pat_clinic['BCR'])
            pat_gt_dur = int(pat_clinic['date'])
            pat_gt_id = int(pat_clinic['ID'])
            print(f'{pat_num} patient has BCR : {pat_gt}, dur : {pat_gt_dur}, id : {pat_gt_id}')

            if norm_img is None  :
                print(pat_num,' this patient has something NONE!!')
                continue
            feature = {}
            feature['img']=tf.train.Feature(float_list=tf.train.FloatList(value = norm_img.flatten()))
            feature['gt']=tf.train.Feature(int64_list=tf.train.Int64List(value=[pat_gt]))
            feature['duration']=tf.train.Feature(int64_list=tf.train.Int64List(value=[pat_gt_dur]))
            feature['ID']=tf.train.Feature(int64_list=tf.train.Int64List(value=[pat_gt_id]))

            features = tf.train.Features(feature=feature)
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())
            cnt+=1
        except:
             print('something files wrong... ',pat_num,'this pat failed to save !!')
        pat_cnt+=1
    writer.close() 
    print("Done...")


def main():
    args = parser.parse_args()

    path = 'prostate_nii/'
    cv = pd.read_excel('5-fold_cv.xlsx', engine='openpyxl', header=None)
    clinic = pd.read_excel('clinical_variables.xlsx', engine='openpyxl')
    clinic = clinic.iloc[:,1:]
    
    if args.mode == 'train':
        clinic = clinic[cv[args.fold] == True]
    else:
        clinic = clinic[cv[args.fold] == False]

    dataset = dict()
    dataset['img_path'], dataset['roi_path'] = [], []
    bcr_0, bcr_1 = 0, 0

    data = clinic
    for i in range(data.shape[0]):
        one_data = data.iloc[i]
        img_num = str(int(one_data['data_num']))
        bcr = clinic[clinic['data_num'] == int(img_num)]['BCR'].iloc[0]
        if bcr == 0:
            bcr_0 += 1
        elif bcr == 1:
            bcr_1 += 1
        
        one_id = str(one_data['ID'])
        print(f'from clinic : {i}, to img num : {img_num}, this is ID : {one_id}')
        
        dataset['img_path'].append(os.path.join(path, img_num, args.modality+'.nii.gz'))
        dataset['roi_path'].append(os.path.join(path, img_num, args.modality+'_roi.nii.gz'))
    length=[len(dataset[str(key)]) for key in dataset.keys()]
    print(f'dwi, roi keys len checking : {length}')
    print('bcr_0, bcr_1',bcr_0,bcr_1)

    create_tfrecord_tumor_one_patch(dataset, str(args.fold)+"_"+args.mode+"_"+args.modality+'.tfrecords')

if __name__ == "__main__":
    main()