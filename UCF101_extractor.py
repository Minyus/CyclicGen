import numpy as np
from pathlib import Path
import cv2
import traceback

top_dir = r'./UCF-101'
f_interval = 5  # 25

top_dir_path = Path(top_dir)

w_top_dir = top_dir + '_frames'
w_top_dir_path = Path(w_top_dir)

label_dir_paths = list(top_dir_path.glob('*'))
# label_dir = label_dirs[0]
for label_index, label_dir_path in enumerate(label_dir_paths):
    print('Label: {} | Directory: {}'.format(label_index, label_dir_path))
    w_label_dir_path = w_top_dir_path / label_dir_path.stem

    for video_file_path in list(label_dir_path.glob('*')):
        # print(video_file_path)
        try:
            cap = cv2.VideoCapture(video_file_path.as_posix())
            # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # fps = cap.get(cv2.CAP_PROP_FPS)
            # print("width:{}, height:{}, count:{}, fps:{}".format(width,height,count,fps))

            w_video_dir_path = w_label_dir_path / video_file_path.stem

            # print('Video Directory Path to Write:', w_video_dir_path)
            num_triplets = (count // f_interval) // 3
            # print('Number of Triplets:', num_triplets)
            pos_last = num_triplets * 3 * f_interval

            for i in np.arange(0, pos_last, f_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                frame_valid, frame = cap.read()
                if not frame_valid:
                    print('Invalid frame from: {}'.format(video_file_path))
                    break

                # triplet_id = 3 * f_interval * ((i // f_interval) // 3)
                # triplet_index = (i // f_interval) % 3
                # w_image_file_path = w_video_dir_path / 'f_{:04d}/frame_{}.png'.format(triplet_id, triplet_index+1)
                w_image_file_path = w_video_dir_path / 'frame_{:04d}.png'.format(i)

                w_image_file_path.parent.mkdir(parents=True, exist_ok=True)
                # print('file: ', w_image_file_path)
                cv2.imwrite(w_image_file_path.as_posix(), frame)
        except KeyboardInterrupt:
            break
        except:
            print('### An exception occurred during processing:', video_file_path)
            print(traceback.format_exc())
            print('### Continue running.')
print('### End of code.')