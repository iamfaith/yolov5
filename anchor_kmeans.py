import numpy as np
from sklearn.cluster import KMeans
import argparse

# def bbox_iou(box, clusters):
#     """计算 box 与多个 anchor 的 IoU"""
#     x = np.minimum(clusters[:, 0], box[0])
#     y = np.minimum(clusters[:, 1], box[1])
#     intersection = x * y
#     box_area = box[0] * box[1]
#     cluster_area = clusters[:, 0] * clusters[:, 1]
#     iou = intersection / (box_area + cluster_area - intersection + 1e-16)
#     return iou



def bbox_iou(box, anchors):
    inter = np.minimum(box[0], anchors[:, 0]) * np.minimum(box[1], anchors[:, 1])
    union = box[0] * box[1] + anchors[:, 0] * anchors[:, 1] - inter
    return inter / (union + 1e-16)

def avg_iou(boxes, clusters):
    return np.mean([np.max(bbox_iou(box, clusters)) for box in boxes])

def kmeans(boxes, k, seed=1):
    np.random.seed(seed)
    indices = np.random.choice(len(boxes), k, replace=False)
    clusters = boxes[indices]
    last_clusters = np.full(len(boxes), -1)

    while True:
        # 每个 box 与所有 cluster 的 IoU（shape: n x k）
        distances = np.array([1 - bbox_iou(box, clusters) for box in boxes])  # shape: (n, k)
        nearest = np.argmin(distances, axis=1)  # shape: (n,)

        if np.array_equal(last_clusters, nearest):
            break

        for i in range(k):
            if np.any(nearest == i):  # 避免空聚类
                clusters[i] = np.median(boxes[nearest == i], axis=0)
            else:
                # 若该聚类为空，随机重新初始化
                clusters[i] = boxes[np.random.choice(len(boxes))]

        last_clusters = nearest

    return clusters

def load_labels(path):
    boxes = []
    with open(path, 'r') as f:
        for label_file in f.readlines():
            label_file = label_file.strip()
            with open(label_file, 'r') as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, x, y, w, h = map(float, parts)
                        boxes.append([w, h])
    return np.array(boxes)




def compute_bpr(boxes, anchors, threshold=0.5):
    matched = 0
    for box in boxes:
        ious = bbox_iou(box, anchors)
        if (ious > threshold).any():
            matched += 1
    return matched / len(boxes)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--labels', type=str, required=True, help='Path to YOLO txt labels', default='/home/faith/coco2017labels-person/coco/labels/val.txt')
    parser.add_argument('--k', type=int, default=9, help='Number of anchors')
    parser.add_argument('--img-size', type=int, default=640, help='Input image size')
    # defaults = {
    #     "labels": '/home/faith/coco2017labels-person/coco/labels/val.txt'
    # }
    # parser.set_defaults(**defaults)
    labels = '/home/faith/coco2017labels-person/coco/labels/val.txt'
    args = parser.parse_args()

    boxes = load_labels(labels) * args.img_size  # 归一化框还原
    anchors = kmeans(boxes, args.k)
    anchors = anchors[np.argsort(anchors[:, 0] * anchors[:, 1])]  # 按面积排序

    print(f"Anchors (w,h):\n{anchors.astype(int)}")
    print(f"Best Possible Recall: {avg_iou(boxes, anchors):.4f}")

    bpr = compute_bpr(boxes, anchors)
    print(f"BPR = {bpr:.4f}")
    
    # 示例：使用 COCO 默认 anchor（单位：像素）
    anchors = np.array([
        [10,13], [16,30], [33,23],
        [30,61], [62,45], [59,119],
        [116,90], [156,198], [373,326]
    ])


    bpr = compute_bpr(boxes, anchors)
    print(f"BPR = {bpr:.4f}")
    print(f"Best Possible Recall: {avg_iou(boxes, anchors):.4f}")
