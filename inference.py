# coding=utf-8

import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

import torch.nn as nn

import random
import torchvision

import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io
# from data import VehicleID_MC, VehicleID_All, id2name
import glob
from tqdm import tqdm
# import matplotlib as mpl
from pylab import mpl
from matplotlib.font_manager import *
from collections import defaultdict

from InitRepNet import InitRepNet

# 解决负号'-'显示为方块的问题
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams[u'font.sans-serif'] = ['simhei']

from network import RepNet
from vehicleID_dataset import VehicleID_All, VehicleID_MC


class FocalLoss(nn.Module):
  """
    Focal loss: focus more on hard samples
    """
  def __init__(self, gamma=0, eps=1e-7):
    """
        :param gamma:
        :param eps:
        """
    super(FocalLoss, self).__init__()
    self.gamma = gamma
    self.eps = eps
    self.ce = torch.nn.CrossEntropyLoss()

  def forward(self, data, target):
    """
        :param data:
        :param target:
        :return:
        """
    log_p = self.ce(data, target)
    p = torch.exp(-log_p)
    loss = (1.0 - p)**self.gamma * log_p
    return loss.mean()


# --------------------------------------- methods
def get_predict_mc(output, model_attr=250):
  """
    softmax归一化,然后统计每一个标签最大值索引
    :param output:
    :return:
    """
  # 计算预测值
  output = output.cpu()  # 从GPU拷贝出来
  pred_model = output[:, :model_attr]
  pred_color = output[:, model_attr:]

  model_idx = pred_model.max(1, keepdim=True)[1]
  color_idx = pred_color.max(1, keepdim=True)[1]

  # 连接pred
  pred = torch.cat((model_idx, color_idx), dim=1)
  return pred


def count_correct(pred, label):
  """
    :param output:
    :param label:
    :return:
    """
  assert pred.size(0) == label.size(0)
  correct_num = 0
  for one, two in zip(pred, label):
    if torch.equal(one, two):
      correct_num += 1
  return correct_num


def count_attrib_correct(pred, label, idx):
  """
    :param pred:
    :param label:
    :param idx:
    :return:
    """
  assert pred.size(0) == label.size(0)
  correct_num = 0
  for one, two in zip(pred, label):
    if one[idx] == two[idx]:
      correct_num += 1
  return correct_num


# @TODO: 可视化分类结果...
def ivt_tensor_img(data, title=None):
  """
  Imshow for Tensor.
  """
  data = data.numpy().transpose((1, 2, 0))

  # 转变数组格式 RGB图像格式：rows * cols * channels
  # 灰度图则不需要转换，只有(rows, cols)而不是（rows, cols, 1）
  # (3, 228, 906)   #  (228, 906, 3)
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])

  # 去标准化，对应transforms
  data = std * data + mean

  # 修正 clip 限制inp的值，小于0则=0，大于1则=1
  output = np.clip(data, 0, 1)

  # plt.imshow(data)
  # if title is not None:
  #     plt.title(title)
  # plt.pause(0.001)  # pause a bit so that plots are updated

  return output


def viz_results(resume,
                data_root,
                out_ids=10086,
                model_attr=250,
                color_attr=27,
                save_dir=''):
  """
  :param resume:
  :param data_root:
  :return:
  """
  out_attribs = model_attr + color_attr
  test_set = VehicleID_All(root=data_root, transforms=None, mode='test')
  test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=1)

  net = RepNet(out_ids=out_ids, out_attribs=out_attribs).to(device)
  print('=> Mix difference network:\n', net)

  # 从断点启动
  if resume is not None:
    if os.path.isfile(resume):
      # 加载模型
      net.load_state_dict(torch.load(resume, map_location=torch.device('cpu')))
      print('=> net resume from {}'.format(resume))
    else:
      print('=> [Err]: invalid resume path @ %s' % resume)

  # 测试模式
  net.eval()

  # 加载类别id映射和类别名称
  modelID2name_path = data_root + '/attribute/modelID2name.pkl'
  colorID2name_path = data_root + '/attribute/colorID2name.pkl'
  trainID2Vid_path = data_root + '/attribute/trainID2Vid.pkl'
  if not (os.path.isfile(modelID2name_path) and \
          os.path.isfile(colorID2name_path) and \
          os.path.isfile((trainID2Vid_path))):
    print('=> [Err]: invalid file.')
    return

  with open(modelID2name_path, 'rb') as fh_1, \
          open(colorID2name_path, 'rb') as fh_2, \
          open(trainID2Vid_path, 'rb') as fh_3:
    modelID2name = pickle.load(fh_1)
    colorID2name = pickle.load(fh_2)
    trainID2Vid = pickle.load(fh_3)

  # load pred_vid image
  vid_2_images_path = data_root + '/attribute/ID2imgs.pkl'
  with open(vid_2_images_path, 'rb') as vid_2_images_f_h:
    vid_2_images = pickle.load(vid_2_images_f_h)

  # 测试
  print('=> testing...')
  for i, (data, label) in enumerate(test_loader):
    # 放入GPU.
    data, label = data.to(device), label.to(device).long()

    # 前向运算: 预测车型、车身颜色
    output_attrib = net.forward(X=data, branch=1, label=None)
    pred_mc = get_predict_mc(output_attrib, model_attr).cpu()[0]
    pred_m_id, pred_c_id = pred_mc[0].item(), pred_mc[1].item()
    pred_m_name = modelID2name[pred_m_id]
    pred_c_name = colorID2name[pred_c_id]

    # 前向运算: 预测Vehicle ID
    output_id = net.forward(X=data, branch=3, label=label[:, 2])
    # print('output_id', output_id.cpu())
    # print('output_id shape', output_id.size())
    _, pred_tid = torch.max(output_id, 1)
    pred_tid = pred_tid.cpu()[0].item()
    pred_vid = trainID2Vid[pred_tid]

    # 获取实际result
    img_path = test_loader.dataset.imgs_path[i]
    img_name = os.path.split(img_path)[-1][:-4]

    result = label.cpu()[0]
    res_m_id, res_c_id, res_vid = result[0].item(), result[1].item(), \
                                  trainID2Vid[result[2].item()]
    res_m_name = modelID2name[res_m_id]
    res_c_name = colorID2name[res_c_id]

    # 图像标题
    title = 'pred: ' + pred_m_name + ' ' + pred_c_name \
            + ', vehicle ID ' + str(pred_vid) \
            + '\n' + \
            'resu: ' + res_m_name + ' ' + res_c_name \
            + ', vehicle ID ' + str(res_vid)
    print('=> result: ', title)

    # 绘图
    img = ivt_tensor_img(data.cpu()[0])
    plt.figure(figsize=(12, 6))
    plt.suptitle(title)
    # plt.title(title)
    ax1 = plt.subplot(121)
    ax1.imshow(img)

    pred_vid_image_path = os.path.join(data_root, 'image',
                                       vid_2_images[pred_vid][0] + '.jpg')
    pred_vid_image = io.imread(pred_vid_image_path)
    ax2 = plt.subplot(122)
    ax2.imshow(pred_vid_image)

    if save_dir:
      save_path = os.path.join(save_dir,
                               img_name + f'-{pred_vid}-{res_vid}.png')
      plt.savefig(save_path)
    plt.show()


def gen_test_pairs(test_txt, dst_dir, num=10000):
  """
    生成测试pair数据: 一半positive，一半negative
    :param test_txt:
    :return:
    """
  if not os.path.isfile(test_txt):
    print('[Err]: invalid file.')
    return
  print('=> genarating %d samples...' % num)

  with open(test_txt, 'r') as f_h:
    valid_list = f_h.readlines()
    print('=> %s loaded.' % test_txt)

    # 映射: img_name => cls_id
    valid_dict = {
        x.strip().split()[0]: int(x.strip().split()[3])
        for x in valid_list
    }

    # 映射: cls_id => img_list
    inv_dict = defaultdict(list)
    for k, v in valid_dict.items():
      inv_dict[v].append(k)

    # 统计样本数不少于2的id
    big_ids = [k for k, v in inv_dict.items() if len(v) > 1]

  # 添加测试样本
  pair_set = set()
  while len(pair_set) < num:
    if random.random() <= 0.7:  # positive
      # 随机从big_ids中选择一个
      pick_id = random.sample(big_ids, 1)[0]  # 不放回抽取

      anchor = random.sample(inv_dict[pick_id], 1)[0]
      positive = random.choice(inv_dict[pick_id])
      while positive == anchor:
        positive = random.choice(inv_dict[pick_id])

      pair_set.add(anchor + '\t' + positive + '\t1')
    else:  # negative
      pick_id_1 = random.sample(big_ids, 1)[0]  # 不放回抽取
      pick_id_2 = random.sample(big_ids, 1)[0]  # 不放回抽取
      while pick_id_2 == pick_id_1:
        pick_id_2 = random.sample(big_ids, 1)[0]
      assert pick_id_2 != pick_id_1
      anchor = random.choice(inv_dict[pick_id_1])
      negative = random.choice(inv_dict[pick_id_2])

      pair_set.add(anchor + '\t' + negative + '\t0')
  print(list(pair_set)[:5])
  print(len(pair_set))

  # 序列化pair_set到dst_dir
  pair_set_f_path = dst_dir + '/' + 'pair_set_vehicle.txt'
  with open(pair_set_f_path, 'w') as f_h:
    for x in pair_set:
      f_h.write(x + '\n')
  print('=> %s generated.' % pair_set_f_path)


# 获取每张测试图片对应的特征向量
def gen_feature_map(resume,
                    imgs_path,
                    batch_size=16,
                    out_ids=10086,
                    out_attribs=257):
  """
    根据图相对生成每张图象的特征向量, 映射: img_name => img_feature vector
    :param resume:
    :param imgs_path:
    :return:
    """
  net = RepNet(out_ids=out_ids, out_attribs=out_attribs).to(device)
  print('=> Mix difference network:\n', net)

  # 从断点启动
  if resume is not None:
    if os.path.isfile(resume):
      # 加载模型
      net.load_state_dict(torch.load(resume, map_location=torch.device('cpu')))
      print('=> net resume from {}'.format(resume))
    else:
      print('=> [Err]: invalid resume path @ %s' % resume)

  # 图像数据变换
  transforms = torchvision.transforms.Compose([
      torchvision.transforms.Resize(224),
      torchvision.transforms.CenterCrop(224),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
  ])

  # load model, image and forward
  data, features = None, None
  for i, img_path in tqdm(enumerate(imgs_path)):
    # load image
    img = Image.open(img_path)

    # tuen to RGB
    if img.mode == 'L' or img.mode == 'I':  # 8bit或32bit灰度图
      img = img.convert('RGB')

    # image data transformations
    img = transforms(img)
    img = img.view(1, 3, 224, 224)

    if data is None:
      data = img
    else:
      data = torch.cat((data, img), dim=0)

    if data.shape[0] % batch_size == 0 or i == len(imgs_path) - 1:

      # collect a batch of image data
      data = data.to(device)

      output = net.forward(X=data, branch=5, label=None)

      batch_features = output.data.cpu().numpy()
      if features is None:
        features = batch_features
      else:
        features = np.vstack((features, batch_features))

      # clear a batch of images
      data = None

  # generate feature map
  feature_map = {}
  for i, img_path in enumerate(imgs_path):
    feature_map[img_path] = features[i]

  print('=> feature map size: %d' % (len(feature_map)))
  return feature_map


def cosin_metric(x1, x2):
  cosin = np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
  return cosin


def euclidean_metric(x1, x2):
  euclidean_dis = np.linalg.norm(x1 - x2)
  euclidean_sim = 1.0 / (1.0 + euclidean_dis)
  return euclidean_sim


def cal_accuracy(y_score, y_true):
  """
    :param y_score:
    :param y_true:
    :return:
    """
  y_score = np.asarray(y_score)
  y_true = np.asarray(y_true)
  best_acc = 0
  best_th = 0
  for i in range(len(y_score)):
    th = y_score[i]
    y_test = (y_score >= th)
    acc = np.mean((y_test == y_true).astype(int))
    print('=> th: %.3f, acc: %.3f' % (th, acc))

    if acc > best_acc:
      best_acc = acc
      best_th = th

  return (best_acc, best_th)


# 统计阈值和准确率: Vehicle ID数据集
def get_th_acc_VID(resume,
                   pair_set_txt,
                   img_dir,
                   batch_size=16,
                   out_ids=10086,
                   out_attribs=257):
  """
    :param resume:
    :param pair_set_txt:
    :param img_dir:
    :param batch_size:
    :return:
    """
  if not os.path.isfile(pair_set_txt):
    print('=> [Err]: invalid file.')
    return

  pairs, imgs_path = [], []
  with open(pair_set_txt, 'r', encoding='utf-8') as fh:
    for line in fh.readlines():
      pair = line.strip().split()

      imgs_path.append(img_dir + '/' + pair[0] + '.jpg')
      imgs_path.append(img_dir + '/' + pair[1] + '.jpg')

      pairs.append(pair)

  print('=> total %d pairs.' % (len(pairs)))
  print('=> total %d image samples.' % (len(imgs_path)))
  imgs_path.sort()

  # generate feature dict
  feature_map = gen_feature_map(resume=resume,
                                imgs_path=imgs_path,
                                batch_size=batch_size,
                                out_ids=out_ids,
                                out_attribs=out_attribs)

  sims, labels = [], []
  for pair in pairs:
    img_path_1 = img_dir + '/' + pair[0] + '.jpg'
    img_path_2 = img_dir + '/' + pair[1] + '.jpg'
    sim = cosin_metric(feature_map[img_path_1], feature_map[img_path_2])
    label = int(pair[2])
    sims.append(sim)
    labels.append(label)

  # 统计最佳阈值及其对应的准确率
  acc, th = cal_accuracy(sims, labels)
  print('=> best threshold: %.3f, accuracy: %.3f%%' % (th, acc * 100.0))
  return acc, th


# 统计阈值和准确率: Car Match数据集
def test_car_match_data(resume, pair_set_txt, img_root, batch_size=16):
  """
    :param resume:
    :param pair_set_txt:
    :param batch_size:
    :return:
    """
  if not os.path.isfile(pair_set_txt):
    print('=> [Err]: invalid file.')
    return

  pairs, imgs_path = [], []
  with open(pair_set_txt, 'r', encoding='utf-8') as fh:
    for line in fh.readlines():
      line = line.strip().split()

      imgs_path.append(img_root + '/' + line[0] + '.jpg')
      imgs_path.append(img_root + '/' + line[1] + '.jpg')

      pairs.append(line)

  print('=> total %d pairs.' % (len(pairs)))
  print('=> total %d image samples.' % (len(imgs_path)))
  imgs_path.sort()

  # 计算特征向量字典
  feature_map = gen_feature_map(resume=resume,
                                imgs_path=imgs_path,
                                batch_size=batch_size)

  # 计算所有pair的sim
  sims, labels = [], []
  for pair in pairs:
    img_path_1 = img_root + '/' + pair[0]
    img_path_2 = img_root + '/' + pair[1]
    sim = cosin_metric(feature_map[img_path_1], feature_map[img_path_2])
    label = int(pair[2])
    sims.append(sim)
    labels.append(label)

  # 统计最佳阈值及其对应的准确率
  acc, th = cal_accuracy(sims, labels)
  print('=> best threshold: %.3f, accuracy: %.3f%%' % (th, acc * 100.0))
  return acc, th


def test_accuracy(net, data_loader):
  """
    测试VehicleID分类在测试集上的准确率
    :param net:
    :param data_loader:
    :return:
    """
  net.eval()  # 测试模式,前向计算

  num_correct = 0
  num_total = 0

  # 每个属性的准确率
  num_model = 0
  num_color = 0
  total_time = 0.0

  print('=> testing...')
  for data, label in data_loader:
    # 放入GPU.
    data, label = data.to(device), label.to(device).long()

    # 前向运算, 预测Vehicle ID
    output = net.forward(X=data, branch=3, label=label[:, 2])

    # 统计总数
    num_total += label.size(0)

    # 统计全部属性都预测正确正确数
    _, pred = torch.max(output.data, 1)
    batch_correct = (pred == label[:, 2]).sum().item()
    num_correct += batch_correct

  # test-set总的统计
  accuracy = 100.0 * float(num_correct) / float(num_total)
  print('=> test accuracy: {:.3f}%'.format(accuracy))

  return accuracy


def test_mc_accuracy(net, data_loader):
  """
    :param net:
    :param data_loader:
    :return:
    """
  net.eval()  # 测试模式,前向计算

  num_correct = 0
  num_total = 0

  # 每个属性的准确率
  num_model = 0
  num_color = 0
  total_time = 0.0

  print('=> testing...')
  for data, label in data_loader:
    # 放入GPU.
    data, label = data.to(device), label.to(device)

    # 将label转化为cpu, long
    label = label.cpu().long()

    # 前向运算, 预测
    output = net.forward(X=data, branch=1)  # 默认在device(GPU)中推理运算
    pred = get_predict_mc(output)  # 返回的pred存在于host端

    # 统计总数
    num_total += label.size(0)

    # 统计全部属性都预测正确正确数
    num_correct += count_correct(pred, label)

    # 统计各属性正确率
    num_model += count_attrib_correct(pred, label, 0)
    num_color += count_attrib_correct(pred, label, 1)

  # 总统计
  accuracy = 100.0 * float(num_correct) / float(num_total)
  model_acc = 100.0 * float(num_model) / float(num_total)
  color_acc = 100.0 * float(num_color) / float(num_total)

  print('=> test accuracy: {:.3f}%, RAModel accuracy: {:.3f}%, '
        'color accuracy: {:.3f}%'.format(accuracy, model_acc, color_acc))
  return accuracy


def eval_mc(resume,
            root='./dataset/VehicleID_V1.0',
            out_ids=10086,
            out_attribs=257):
  """
    训练RepNet: RAModel and color multi-label classification
    :param freeze_feature:
    :return:
    """
  net = RepNet(out_ids=out_ids, out_attribs=out_attribs).to(device)
  print('=> Mix difference network:\n', net)

  if os.path.isfile(resume):
    net.load_state_dict(torch.load(resume,
                                   map_location=torch.device('cpu')))  # 加载模型
    print('=> net resume from {}'.format(resume))
  else:
    print('=> [Err]: invalid resume path @ %s' % resume)

  # 数据集
  test_set = VehicleID_MC(root=root, transforms=None, mode='test')
  test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                            batch_size=32,
                                            shuffle=False,
                                            num_workers=2)

  # 计算测试集准确度
  test_acc = test_mc_accuracy(net=net, data_loader=test_loader)


def eval(resume, root='./dataset/VehicleID_V1.0'):
  """
    :param resume:
    :return:
    """
  # net = RepNet(out_ids=10086,
  #              out_attribs=257).to(device)

  vgg16_pretrain = torchvision.models.vgg16(pretrained=True)
  net = InitRepNet(vgg_orig=vgg16_pretrain, out_ids=10086,
                   out_attribs=257).to(device)

  print('=> Mix difference network:\n', net)

  if os.path.isfile(resume):
    net.load_state_dict(torch.load(resume,
                                   map_location=torch.device('cpu')))  # 加载模型
    print('=> net resume from {}'.format(resume))
  else:
    print('=> [Err]: invalid resume path @ %s' % resume)

  # 数据集
  test_set = VehicleID_All(root=root, transforms=None, mode='test')

  test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                            batch_size=16,
                                            shuffle=False,
                                            num_workers=4)

  # calculate test-set accuracy
  test_acc = test_accuracy(net=net, data_loader=test_loader)

  print('test_acc: \t\t%4.2f%%' % (test_acc))


def featrue_map_test(resume,
                     image_root,
                     metric_func=cosin_metric,
                     out_ids=10086,
                     out_attribs=257):
  imgs_path = sorted(glob.glob(os.path.join(image_root, '*.jpg')))

  feature_map = gen_feature_map(resume,
                                imgs_path,
                                batch_size=1,
                                out_ids=out_ids,
                                out_attribs=out_attribs)

  images_num = len(imgs_path)
  all_result_str = []
  for i in range(images_num):
    print('image_path: ', imgs_path[i])
    sims_list = []
    for j in range(images_num):
      f1 = feature_map[imgs_path[i]]
      f2 = feature_map[imgs_path[j]]
      sim = metric_func(f1, f2)
      sims_list.append(sim)
    sims_list = list(map(str, sims_list))
    one_line = ', '.join([os.path.basename(imgs_path[i])] + sims_list)
    all_result_str.append(one_line)
  with open('cosine_sim_matrix.csv', 'w') as f:
    f.write('\n'.join(all_result_str))


def main():
  # resume = './models/pretrain_epoch_14.pth'
  pair_set_txt = './test_pair_set.txt'
  # data_root = './dataset/VehicleID_V1.0'
  resume = './checkpoints/epoch_31.pth'
  data_root = './dataset/Glodon_Veh_V1.0'
  save_dir = './checkpoints/outputs'
  img_root = os.path.join(data_root, 'image')
  out_ids = 105
  model_attribs = 25
  color_attribs = 7
  # test_car_match_data(resume, pair_set_txt, img_root, batch_size=1)
  # get_th_acc_VID(resume, pair_set_txt, img_root, batch_size=1)

  # 结果可视化
  viz_results(resume,
              data_root,
              out_ids=out_ids,
              model_attr=model_attribs,
              color_attr=color_attribs,
              save_dir=save_dir)

  # featrue_map_test(resume,
  #                  img_root,
  #                  metric_func=euclidean_metric,
  #                  out_ids=out_ids,
  #                  out_attribs=out_attribs)
  print('=> Done.')


if __name__ == '__main__':
  main()
