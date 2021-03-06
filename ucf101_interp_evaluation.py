import numpy as np
from pathlib import Path, WindowsPath
from PIL import Image

# from skimage.measure import compare_ssim, compare_mse, compare_psnr, compare_nrmse
import tensorflow as tf
import pandas as pd

eval_dir_str = r'./ucf101_interp_ours'
eval_dir = Path(eval_dir_str)


def imread(path, gray=False):
    path_str = path.as_posix()
    with open(path_str, 'rb') as f:
        img = Image.open(f)
        if gray:
            img = img.convert('L')
        img_np= np.array(img)
    return img_np

out_dict = {}

""" graph """
# gt_t = tf.placeholder(dtype=tf.uint8, shape=gt_np.shape)
# gen_t = tf.placeholder(dtype=tf.uint8, shape=gen_np.shape)
gt_t = tf.placeholder(dtype=tf.uint8)
gen_t = tf.placeholder(dtype=tf.uint8)
ssim_val_t = tf.image.ssim(gt_t, gen_t, max_val=255)
psnr_val_t = tf.image.psnr(gt_t, gen_t, max_val=255)
ms_ssim_val_t = tf.image.ssim_multiscale(gt_t, gen_t, max_val=255)
"""   """

for avi_dir in eval_dir.glob('*/'):
    if avi_dir.is_dir():
        print(avi_dir)
        gt_file = avi_dir / 'frame_01_gt.png'
        gt_np = imread(gt_file)
        avi_id = avi_dir.name
        for gen_file in avi_dir.glob('**/frame_01_*.png'):
            if gen_file.name != 'frame_01_gt.png':
                gen_np = imread(gen_file)
                if True:


                    with tf.Session() as sess:
                        ssim_val, psnr_val, ms_ssim_val = sess.run([ssim_val_t, psnr_val_t, ms_ssim_val_t], feed_dict={gt_t: gt_np, gen_t: gen_np})
                    model_id = gen_file.name[len('frame_01_'):len('frame_01_2019-04-22T082112')]
                    num_steps = gen_file.name[len('frame_01_2019-04-22T082112_model.ckpt-'):len('frame_01_2019-04-22T082112_model.ckpt-28069')]
                    num_steps = num_steps.split('.png')[0]
                    out_dict[(avi_id, model_id, num_steps)] = [ssim_val, psnr_val, ms_ssim_val]
        # break

# print(out_dict)

df = pd.DataFrame.from_dict(out_dict, orient='index', columns=['ssim_val', 'psnr_val', 'ms_ssim_val'])
df.index = pd.MultiIndex.from_tuples(df.index)
df.index.names = ['triplet_id', 'model_id', 'num_steps']
# print(df)
df.reset_index(inplace=True)
df.index.name = 'index'
df.to_csv(eval_dir / 'ucf101_interp_evaluation.csv')
print('finished.')