"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.S3DISDataLoader import ScannetDatasetWholeScene
from data_utils.indoor3d_util import g_label2color
from data_utils.data_util import voxelize
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes_ = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter']

classes = ['ceiling', 'floor', 'wall', 'door', 'window', 'table', 'chair', 'cabinet', 'shelf',
           'bin', 'box', 'board', 'screen', 'clutter']

class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=20000, help='point number [default: 4096]')
    parser.add_argument('--voxel_size', type=float, default=0.04, help='voxel size')
    parser.add_argument('--log_dir', type=str, default='pointnet2_att_trans', help='experiment root')
    parser.add_argument('--visual', action='store_true', default=True, help='visualize result [default: False]')
    parser.add_argument('--test_area', type=int, default=5, help='area for testing, option: 1-6 [default: 5]')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting [default: 5]')
    return parser.parse_args()


def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


def input_normalize(coord, feat):
    coord_min = np.min(coord, 0)
    coord -= coord_min
    feat = feat / 255.
    return coord, feat


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/sem_seg/' + args.log_dir
    visual_dir = experiment_dir + '/visual/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    NUM_CLASSES = 14
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point

    root = 'data/craslab3d/'

    TEST_DATASET_WHOLE_SCENE = ScannetDatasetWholeScene(root, split='test', test_area=args.test_area, block_points=NUM_POINT)
    log_string("The number of test data is: %d" % len(TEST_DATASET_WHOLE_SCENE))

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    # ground_truth = []
    # predictions = []
    # unique = np.arange(NUM_CLASSES)

    with torch.no_grad():
        scene_id = TEST_DATASET_WHOLE_SCENE.file_list
        scene_id = [x[:-4] for x in scene_id]
        num_batches = len(TEST_DATASET_WHOLE_SCENE)

        total_seen_class = [0 for _ in range(len(classes))]
        total_correct_class = [0 for _ in range(len(classes))]
        total_iou_deno_class = [0 for _ in range(len(classes))]

        log_string('---- EVALUATION WHOLE SCENE----')

        for batch_idx in range(num_batches):
            print("Inference [%d/%d] %s ..." % (batch_idx + 1, num_batches, scene_id[batch_idx]))
            total_seen_class_tmp = [0 for _ in range(len(classes))]
            total_correct_class_tmp = [0 for _ in range(len(classes))]
            total_iou_deno_class_tmp = [0 for _ in range(len(classes))]
            if args.visual:
                fout = open(os.path.join(visual_dir, scene_id[batch_idx] + '_pred.obj'), 'w')
                fout_gt = open(os.path.join(visual_dir, scene_id[batch_idx] + '_gt.obj'), 'w')

            whole_scene_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[batch_idx]
            whole_scene_label = TEST_DATASET_WHOLE_SCENE.semantic_labels_list[batch_idx]
                   
            
            coord, feat, label, idx_data = data_load(whole_scene_data, whole_scene_label)
            pred = torch.zeros((label.size, NUM_CLASSES)).cuda()

            idx_size = len(idx_data)
            idx_list, coord_list, feat_list, offset_list  = [], [], [], []
            
            for i in range(idx_size):
                idx_part = idx_data[i]
                coord_part, feat_part = coord[idx_part], feat[idx_part]
 
                if args.num_point and coord_part.shape[0] > args.num_point:
                    coord_p, idx_uni, cnt = np.random.rand(coord_part.shape[0]) * 1e-3, np.array([]), 0
                    while idx_uni.size != idx_part.shape[0]:
                        init_idx = np.argmin(coord_p)
                        dist = np.sum(np.power(coord_part - coord_part[init_idx], 2), 1)
                        idx_crop = np.argsort(dist)[:args.num_point]
                        coord_sub, feat_sub, idx_sub = coord_part[idx_crop], feat_part[idx_crop], idx_part[idx_crop]
                        dist = dist[idx_crop]
                        delta = np.square(1 - dist / np.max(dist))
                        coord_p[idx_crop] += delta
                        coord_sub, feat_sub = input_normalize(coord_sub, feat_sub)
                        idx_list.append(idx_sub), coord_list.append(coord_sub), feat_list.append(feat_sub), offset_list.append(idx_sub.size)
                        idx_uni = np.unique(np.concatenate((idx_uni, idx_sub)))
                else:
                    coord_part, feat_part = input_normalize(coord_part, feat_part)
                    idx_list.append(idx_part), coord_list.append(coord_part), feat_list.append(feat_part), offset_list.append(idx_part.size)
            batch_num = int(np.ceil(len(idx_list) / args.batch_size))
            for i in range(batch_num):
                s_i, e_i = i * args.batch_size, min((i + 1) * args.batch_size, len(idx_list))
                idx_part, coord_part, feat_part, offset_part = idx_list[s_i:e_i], coord_list[s_i:e_i], feat_list[s_i:e_i], offset_list[s_i:e_i]
                idx_part = np.concatenate(idx_part)
                coord_part = torch.FloatTensor(np.concatenate(coord_part)).cuda(non_blocking=True)
                feat_part = torch.FloatTensor(np.concatenate(feat_part)).cuda(non_blocking=True)
                offset_part = torch.IntTensor(np.cumsum(offset_part)).cuda(non_blocking=True)
                with torch.no_grad():
                    pred_part = classifier([coord_part, feat_part, offset_part])[0]  # (n, k)
                torch.cuda.empty_cache()
                pred[idx_part, :] += pred_part

            pred = pred.max(1)[1].data.cpu().numpy()

            pred_label = pred
            
            # pred_label = np.ones(pred.shape) * -1

            # for i, class_name in enumerate(classes_):
            #     if class_name in classes:
            #         index = classes.index(class_name)
            #         pred_label[pred == i] = index
            #     else:
            #         index = classes.index('clutter')
            #         pred_label[pred == i] = index

            for l in range(len(classes)):
                total_seen_class_tmp[l] += np.sum((whole_scene_label == l))
                total_correct_class_tmp[l] += np.sum((pred_label == l) & (whole_scene_label == l))
                total_iou_deno_class_tmp[l] += np.sum(((pred_label == l) | (whole_scene_label == l)))
                total_seen_class[l] += total_seen_class_tmp[l]
                total_correct_class[l] += total_correct_class_tmp[l]
                total_iou_deno_class[l] += total_iou_deno_class_tmp[l]

            iou_map = np.array(total_correct_class_tmp) / (np.array(total_iou_deno_class_tmp, dtype=np.float32) + 1e-6)
            print(iou_map)
            arr = np.array(total_seen_class_tmp)
            tmp_iou = np.mean(iou_map[arr != 0])
            log_string('Mean IoU of %s: %.4f' % (scene_id[batch_idx], tmp_iou))
            print('----------------------------')

            filename = os.path.join(visual_dir, scene_id[batch_idx] + '.txt')
            with open(filename, 'w') as pl_save:
                for i in pred_label:
                    pl_save.write(str(int(i)) + '\n')
                pl_save.close()
            
            # data_trans = np.concatenate([whole_scene_data, pred_label[:, np.newaxis], whole_scene_label[:, np.newaxis]], axis=-1)
            # np.save(os.path.join(BASE_DIR, 'data/craslab3d_add',scene_id[batch_idx] + '.npy'), data_trans)
            
            for i in range(whole_scene_label.shape[0]):
                color = g_label2color[pred_label[i]]
                color_gt = g_label2color[whole_scene_label[i]]
                if args.visual:
                    fout.write('v %f %f %f %d %d %d\n' % (
                        whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color[0], color[1],
                        color[2]))
                    fout_gt.write(
                        'v %f %f %f %d %d %d\n' % (
                            whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color_gt[0],
                            color_gt[1], color_gt[2]))
            if args.visual:
                fout.close()
                fout_gt.close()


        # predictions = np.concatenate(predictions)
        # ground_truth = np.concatenate(ground_truth)
        # cm = confusion_matrix(ground_truth, predictions)
        # cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # sns.set(font_scale=1.5)
        # sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
        # plt.xlabel('Predicted Labels', fontsize=16)
        # plt.ylabel('True Labels', fontsize=16)
        # plt.title('Confusion Matrix', fontsize=16)

        IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float32) + 1e-6)
        iou_per_class_str = '------- IoU --------\n'
        for l in range(len(classes)):
            iou_per_class_str += 'class %s, IoU: %.3f \n' % (
                seg_label_to_cat[l] + ' ' * (14+1 - len(seg_label_to_cat[l])),
                total_correct_class[l] / float(total_iou_deno_class[l]))
        log_string(iou_per_class_str)
        log_string('eval point avg class IoU: %f' % np.mean(IoU))
        log_string('eval whole scene point avg class acc: %f' % (
            np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float32) + 1e-6))))
        log_string('eval whole scene point accuracy: %f' % (
                np.sum(total_correct_class) / float(np.sum(total_seen_class) + 1e-6)))

        print("Done!")


def data_load(points, label):

    coord, feat = points[:, :3], points[:, 3:6]

    idx_data = []

    coord_min = np.min(coord, 0)
    coord -= coord_min
    idx_sort, count = voxelize(coord, args.voxel_size, mode=1)
    for i in range(count.max()):
        idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
        idx_part = idx_sort[idx_select]
        idx_data.append(idx_part)

    return coord, feat, label, idx_data

if __name__ == '__main__':
    args = parse_args()
    main(args)
