from pathlib import Path
import random

random.seed(0)


def get_random_child(path):
    return random.choice(list(path.glob('*')))


def get_random_child_r(path):
    child_path = get_random_child(path)
    if child_path.is_dir():
        return get_random_child_r(child_path)
    else:
        return child_path


f_interval = 5

top_dir = r'./UCF-101'
top_dir_path = Path(top_dir)

w_top_dir = top_dir + '_frames'
w_top_dir_path = Path(w_top_dir)

f1_list_path = Path(r'data_list/ucf101_train_files_frame1.txt')
f2_list_path = Path(r'data_list/ucf101_train_files_frame2.txt')
f3_list_path = Path(r'data_list/ucf101_train_files_frame3.txt')
t_list_path = [f1_list_path, f2_list_path, f3_list_path]
f1_list_path.parent.mkdir(exist_ok=True)
for f_list_path in t_list_path:
    with f_list_path.open('w') as file:
        pass

num_iter = 276917

i = 0
# for i in range(num_iter):
while i < num_iter:
    img_path = get_random_child_r(w_top_dir_path)
    try:
        f_index = int(img_path.stem[-4:])
    except:
        img_path.unlink()
        print('img_path:', img_path)
        continue
    f1_index = (f_index // (3 * f_interval)) * 3 * f_interval
    f2_index = f1_index + f_interval
    f3_index = f2_index + f_interval
    t_indices = [f1_index, f2_index, f3_index]
    for f_list_path, f_index in zip(t_list_path, t_indices):
        with f_list_path.open('a') as file:
            file.write((img_path.parent / 'frame_{:04d}.png\r\n'.format(f1_index)).as_posix())
    if i % 10000 == 0:
        print('Finished iteration: ', i)
    i += 1
