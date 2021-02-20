# coding=utf-8
"""从数据集生成需要的文件
"""
import os
import pickle
import shutil
import random
from collections import defaultdict
from pathlib import Path
from math import ceil


def process2model_color(root: Path):
  """处理所有标注有model和color数据

  从model_attr.txt, color_attr.txt中提取不重复的vid映射为trainid，
  将映射关系存到vid2TrainID.pkl, trainID2Vid.pkl，将原始的ids列表
  存储到MC_IDs.pkl。
  如果存在img2vid.txt，将id和图片列表的映射存到ID2imgs.pkl。

  Arguments:
    root: a Path of VehicleID dataset dir.
  """

  # 求model_attr.txt和color_attr.txt中ID交集
  model_attr_txt = root / 'attribute/model_attr.txt'
  color_attr_txt = root / 'attribute/color_attr.txt'
  img2vid_txt = root / 'attribute/img2vid.txt'

  assert model_attr_txt.is_file() and \
      color_attr_txt.is_file()

  model_ids = set()
  with model_attr_txt.open('r', encoding='utf-8') as f_h_model:
    for line in f_h_model.readlines():
      model_ids.add(int(line.strip().split()[0]))

  color_ids = set()
  with color_attr_txt.open('r', encoding='utf-8') as f_h_color:
    for line in f_h_color.readlines():
      color_ids.add(int(line.strip().split()[0]))

  # get intersection
  model_color_ids = list(model_ids & color_ids)
  model_color_ids.sort()
  # print(model_color_ids)
  print('=> toal %d vehicle IDs with model and color labeled.' %
        len(model_color_ids))

  # class ID mapping: vid_2_trainid <=> trainid_2_vid
  vid_2_trainid, trainid_2_vid = defaultdict(int), defaultdict(int)
  for train_id, vid in enumerate(model_color_ids):
    vid_2_trainid[vid] = train_id
    trainid_2_vid[train_id] = vid

  # 序列化用于model,color的Vehicle mapping，序列化model_color_ids
  vid_2_trainid_path = root / 'attribute/vid2TrainID.pkl'
  trainid_2_vid_path = root / 'attribute/trainID2Vid.pkl'
  model_color_ids_path = root / 'attribute/MC_IDs.pkl'
  with vid_2_trainid_path.open('wb') as f_h_1:
    pickle.dump(vid_2_trainid, f_h_1)
    print('=> %s dumped.' % vid_2_trainid_path)
  with trainid_2_vid_path.open('wb') as f_h_2:
    pickle.dump(trainid_2_vid, f_h_2)
    print('=> %s dumped.' % trainid_2_vid_path)
  with model_color_ids_path.open('wb') as f_h_2:
    pickle.dump(model_color_ids, f_h_2)
    print('=> %s dumped.' % model_color_ids_path)

  # statistics of ID2img_list
  if img2vid_txt.is_file():
    all_vid = set()
    vid_2_imgs = defaultdict(list)  # vid到图像名列表的映射

    with img2vid_txt.open('r', encoding='utf-8') as f_h:
      sample_cnt = 0
      for line in f_h.readlines():
        img, vid = line.strip().split()

        all_vid.add(vid)
        vid_2_imgs[int(vid)].append(img)
        sample_cnt += 1

    print('=> total %d vehicles have total %d IDs' %
          (sample_cnt, len(all_vid)))

    # 序列化vid_2_imgs到attribute子目录
    vid_2_imgs_path = root / 'attribute/ID2imgs.pkl'
    # ID2imgs = sorted(ID2imgs.items(),
    #                  key=lambda x: int(x[0]),
    #                  reverse=False)
    print(len(vid_2_imgs))
    with vid_2_imgs_path.open('wb') as f_h_1:
      pickle.dump(vid_2_imgs, f_h_1)
      print('=> %s dumped.' % vid_2_imgs_path)


# model, color: multi-label classification
def split_train_and_test(root, test_rate=0.1):
  """
  根据ID2imgs和MC_IDS划分到新目录
  @TODO: 还需要对生成的数据集进行可视化验证
  """

  model_color_ids_path = root / 'attribute/MC_IDs.pkl'
  id2imgs_f_path = root / 'attribute/ID2imgs.pkl'
  model_attr_txt = root / 'attribute/model_attr.txt'
  color_attr_txt = root / 'attribute/color_attr.txt'

  assert model_color_ids_path.is_file() \
      and id2imgs_f_path.is_file() \
      and model_attr_txt.is_file() \
      and color_attr_txt.is_file()

  # 读取veh2model和veh2color
  vid2mid, vid2cid, _ = defaultdict(int), defaultdict(int), defaultdict(int)
  with model_attr_txt.open('r', encoding='utf-8') as fh_1:
    for line in fh_1.readlines():  # vid to model id
      line = line.strip().split()
      vid, modelid = line[:2]
      vid2mid[int(vid)] = int(modelid)
  with color_attr_txt.open('r', encoding='utf-8') as fh_2:
    for line in fh_2.readlines():  # vid to color id
      line = line.strip().split()
      vid, colorid = line[:2]
      vid2cid[int(vid)] = int(colorid)

  with open(model_color_ids_path, 'rb') as f_h_1, \
          open(id2imgs_f_path, 'rb') as f_h_2:
    model_color_ids = pickle.load(f_h_1)
    id2imgs = pickle.load(f_h_2)

    train_txt = root / 'attribute/train_all.txt'
    test_txt = root / 'attribute/test_all.txt'

    # 按照Vehicle ID的顺序生成训练和测试数据
    train_cnt, test_cnt = 0, 0
    with train_txt.open('w', encoding='utf-8') as train_txt_f_h, \
            test_txt.open('w', encoding='utf-8') as test_txt_f_h:
      for _, vid in enumerate(model_color_ids):
        if vid in model_color_ids:
          imgs_list = id2imgs[vid]
          random.shuffle(imgs_list)
          for i, img in enumerate(imgs_list):
            # get image and label: img + model_id + color_id
            model_id, color_id = vid2mid[vid], vid2cid[vid]
            img_label = img + ' ' + str(model_id) \
                + ' ' + str(color_id) + ' ' + str(vid) + '\n'

            # split to train.txt and test.txt
            if i < int(ceil((1 - test_rate) * len(imgs_list))):
              train_txt_f_h.write(img_label)
              train_cnt += 1
            else:
              test_txt_f_h.write(img_label)
              test_cnt += 1
          # print('=> Vehicle ID %d samples generated.' % vid)

      print('=> %d img files splitted to train set' % train_cnt)
      print('=> %d img files splitted to test set' % test_cnt)
      print('=> total %d img files in root dataset.' % (train_cnt + test_cnt))


def process_vehicleID(root, TH=15):
  """
    统计VehicleID ID数
    """
  # 遍历所有图片
  img2vid_f_path = root + '/attribute/img2vid.txt'
  if os.path.isfile(img2vid_f_path):
    IDs = set()
    ID2imgs = defaultdict(list)

    with open(img2vid_f_path, 'r', encoding='utf-8') as f_h:
      sample_cnt = 0
      for line in f_h.readlines():
        line = line.strip().split()

        img, _id = line

        IDs.add(_id)
        ID2imgs[_id].append(img)
        sample_cnt += 1

  # print(ID2Num)
  print('=> total %d vehicles have total %d IDs' % (sample_cnt, len(IDs)))

  # 序列化满足条件的Vehicle IDS
  ID2imgs_path = root + '/attribute/ID2imgs.pkl'
  ID2imgs = sorted(ID2imgs.items(), key=lambda x: int(x[0]), reverse=False)

  # print(ID2imgs)
  ID2imgs_sort = defaultdict(list)
  for item in ID2imgs:
    if len(item[1]) >= TH:
      ID2imgs_sort[item[0]] = item[1]
  print('=> total %d Ids meet requirements' % len(ID2imgs_sort.keys()))

  # print(ID2imgs_sort)
  print('=> Last 10 vehicle ids: ', list(ID2imgs_sort.keys())[-10:])

  with open(ID2imgs_path, 'wb') as f_h:
    pickle.dump(ID2imgs_sort, f_h)
    print('=> %s dumped.' % ID2imgs_path)

  # 验证筛选结果...
  fetch_from_vechicle(ID2imgs_path, root, img)


def fetch_from_vechicle(ID2imgs_path, root, img):
  # 验证筛选结果...
  print('=> testing...')
  with open(ID2imgs_path, 'rb') as f_h:
    ID2imgs = pickle.load(f_h)

    img_root = root + '/image'
    dst_root = 'f:/VehicleID_Part'

    # 按子目录存放
    for i, (k, v) in enumerate(ID2imgs.items()):
      _, imgs = k, v
      imgs = [img_root + '/' + img + '.jpg' for img in imgs]
      # print(imgs)

      dst_sub_dir = dst_root + '/' + str(i)
      if not os.path.isdir(dst_sub_dir):
        os.makedirs(dst_sub_dir)

      # copy to test_result
      for img in imgs:
        shutil.copy(img, dst_sub_dir)
      print('=> %s processed.' % dst_sub_dir)


def get_ext_files(root, ext, f_list):
  """
    递归搜索指定文件
    :param root:
    :param ext:
    :param f_list:
    :return:
    """
  for x in os.listdir(root):
    x_path = root + '/' + x
    if os.path.isfile(x_path) and x_path.endswith(ext):
      f_list.append(x_path)
    elif os.path.isdir(x_path):
      get_ext_files(x_path, ext, f_list)


def split(data_root, RATIO=0.1):
  """
    将按照子目录存放的数据划分为训练数据集和测试数据集
    """
  if not os.path.isdir(data_root):
    print('=> invalid data root.')
    return

  train_txt = data_root + '/train.txt'
  test_txt = data_root + '/test.txt'

  # 写train.txt, test.txt
  train_cnt, test_cnt = 0, 0
  with open(train_txt, 'w', encoding='utf-8') as f_train, \
          open(test_txt, 'w', encoding='utf-8') as f_test:
    # 从根目录遍历每一个子目录
    sub_dirs = [sub for sub in os.listdir(data_root) if sub.isdigit()]
    sub_dirs.sort(key=lambda x: int(x))
    for sub in sub_dirs:
      sub_path = data_root + '/' + sub
      if os.path.isdir(sub_path):
        for img in os.listdir(sub_path):
          # 写txt文件
          img_path = sub_path + '/' + img
          relative_apth = '/'.join(img_path.split('/')[2:])
          if random.random() > RATIO:
            f_train.write(relative_apth + '\n')
            train_cnt += 1
          else:
            f_test.write(relative_apth + '\n')
            test_cnt += 1
    print('=> %d img files splitted to train set' % train_cnt)
    print('=> %d img files splitted to test set' % test_cnt)
    print('=> total %d img files in root dataset.' % (train_cnt + test_cnt))


def form_cls_name(root):
  """
    加载类别和类别序号的映射,
    序列化到attribute目录
    """
  model_names_txt = root + '/attribute/model_names.txt'
  color_names_txt = root + '/attribute/color_names.txt'
  if not (os.path.isfile(model_names_txt) and os.path.isfile(color_names_txt)):
    print('=> [Err]: invalid class names file.')
    return

  modelID2name, colorID2name = defaultdict(str), defaultdict(str)
  with open(model_names_txt, 'r', encoding='utf-8') as fh_1, \
          open(color_names_txt, 'r', encoding='utf-8') as fh_2:
    for line in fh_1.readlines():
      line = line.strip().split()
      modelID2name[int(line[1])] = line[0]

    for line in fh_2.readlines():
      line = line.strip().split()
      colorID2name[int(line[1])] = line[0]
  print(modelID2name)
  print(colorID2name)

  # 序列化到硬盘
  modelID2name_path = root + '/attribute/modelID2name.pkl'
  colorID2name_path = root + '/attribute/colorID2name.pkl'
  with open(modelID2name_path, 'wb') as fh_1, \
          open(colorID2name_path, 'wb') as fh_2:
    pickle.dump(modelID2name, fh_1)
    pickle.dump(colorID2name, fh_2)

  print('=> %s dumped.' % modelID2name_path)
  print('=> %s dumped.' % colorID2name_path)


# ----------------- 从10086之外取test-pairs数据用来测试
def gen_test_pairs(root):
  """
  """
  return root


if __name__ == '__main__':
  # process_vehicleID(root='e:/VehicleID_V1.0')
  # split(data_root='f:/VehicleID_Part')
  root_dir = Path('./dataset/Glodon_Veh_V1.0')
  process2model_color(root_dir)
  split_train_and_test(root_dir, test_rate=0.2)

  # form_cls_name(root='e:/VehicleID_V1.0')

  print('=> Done.')
