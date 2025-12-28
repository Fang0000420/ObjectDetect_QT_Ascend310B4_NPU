import cv2
import numpy as np
import torch
import os
from ais_bench.infer.interface import InferSession

# --------------------------
# 直接引用你的工具类
# --------------------------
from det_utils import letterbox, scale_coords, nms

class YOLODetectorWrapper:
    def __init__(self, model_path, label_path, conf_thres=0.4, iou_thres=0.45, input_shape=640):
        # 配置参数
        self.cfg = {
            'conf_thres': conf_thres,
            'iou_thres': iou_thres,
            'input_shape': (input_shape, input_shape)
        }
        
        # 1. 加载模型
        self.model = InferSession(0, model_path)
        
        # 2. 加载标签
        self.labels_dict = self._get_labels(label_path)

    def _get_labels(self, path):
        labels = {}
        if os.path.exists(path):
            with open(path, 'r') as f:
                for i, line in enumerate(f):
                    labels[i] = line.strip()
        return labels

    def preprocess_image(self, image):
        """
        针对 YOLOv5 的预处理，使用 det_utils.letterbox
        """
        # 1. Letterbox (缩放 + Padding)
        # 注意：det_utils.letterbox 返回: img, (ratio_x, ratio_y), (pad_w, pad_h)
        img, ratio, pad = letterbox(image, new_shape=self.cfg['input_shape'], auto=False, scaleup=True)

        # 2. BGR 转 RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 3. HWC 转 CHW (3, 640, 640)
        img = img.transpose((2, 0, 1))

        # 4. 转换为连续内存并转 float32
        # 注意：这里不除以 255.0，假设 .om 模型转换时已配置 AIPP 或模型首层处理归一化
        img = np.ascontiguousarray(img, dtype=np.float32)

        return img[None], ratio, pad

    def detect_frame(self, frame):
        # 1. 预处理
        img, ratio, pad = self.preprocess_image(frame)

        # 2. NPU 推理
        # output通常是 list，取第一个输出
        output = self.model.infer([img])[0] 

        # 3. 格式转换 (Numpy -> Tensor)
        # YOLOv5s 输出通常已经是 [Batch, Anchors, 85]，直接转 Tensor 即可
        pred_tensor = torch.from_numpy(output)

        # 4. NMS (使用 det_utils 中的 nms)
        # 注意：det_utils.nms 内部调用 non_max_suppression，处理 xywh->xyxy
        box_out = nms(pred_tensor, conf_thres=self.cfg['conf_thres'], iou_thres=self.cfg['iou_thres'])
        
        # 取第一张图片的结果
        results_tensor = box_out[0]

        results = []
        # 如果有检测结果
        if results_tensor.shape[0] > 0:
            # 5. 坐标还原 (Scale Coords)
            # 将 Tensor 转回 Numpy 以便后续处理
            pred_all = results_tensor.cpu().numpy()
            
            # det_utils.scale_coords 需要的 ratio_pad 参数正是 letterbox 返回的 (ratio, pad)
            scale_coords(self.cfg['input_shape'], pred_all[:, :4], frame.shape, ratio_pad=(ratio, pad))

            # 6. 封装结果
            for *xyxy, conf, cls in pred_all:
                class_index = int(cls)
                class_name = self.labels_dict.get(class_index, f"class_{class_index}")
                
                bbox = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                
                results.append({
                    "class": class_name,
                    "confidence": float(conf),
                    "bbox": bbox
                })

        # 7. 绘制结果
        draw_frame = self.draw_detection_boxes(frame, results)
        return results, draw_frame

    def draw_detection_boxes(self, frame, results):
        img = frame.copy()
        for res in results:
            x1, y1, x2, y2 = res['bbox']
            label = f"{res['class']} {res['confidence']:.2f}"
            
            # 画框
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 画标签背景和文字
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1, y1), c2, (0, 255, 0), -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return img

    def save_result(self, detect_frame, results, save_path):
        try:
            # 保存图片
            cv2.imwrite(f"{save_path}.jpg", detect_frame)
            # 保存txt
            with open(f"{save_path}.txt", 'w') as f:
                f.write(f"Count: {len(results)}\n")
                for res in results:
                    f.write(f"{res['class']} {res['confidence']:.4f} {res['bbox']}\n")
            return True
        except Exception as e:
            print(f"Save Error: {e}")
            return False