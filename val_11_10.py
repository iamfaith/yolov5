from ultralytics import YOLO
# model = YOLO('yolov10n.pt')
# model = YOLO('yolo11n.pt')
model = YOLO('yolo12n.pt')
# model = YOLO('yolo26n.pt')
metrics = model.val(data='/home/faith/yolov5/data/coco_person.yaml', imgsz=640, batch=16, device='0', single_cls=True)
print(metrics)

############# yolov10n
# results_dict: {'metrics/precision(B)': 0.5011575392511556, 'metrics/recall(B)': 0.5461808529682268, 'metrics/mAP50(B)': 0.45656625613181534, 'metrics/mAP50-95(B)': 0.3251184367267468, 'fitness': 0.3382632186672536}
# save_dir: PosixPath('runs/detect/val3')
# speed: {'preprocess': 0.18641564387517426, 'inference': 3.2956735273076405, 'loss': 0.000411913845149646, 'postprocess': 0.18937238655341632}
# task: 'detect'


############# yolo11n
# results_dict: {'metrics/precision(B)': 0.49006855098485225, 'metrics/recall(B)': 0.5804027094738795, 'metrics/mAP50(B)': 0.44996862433318946, 'metrics/mAP50-95(B)': 0.31756466427911173, 'fitness': 0.3308050602845195}
# save_dir: PosixPath('runs/detect/val4')
# speed: {'preprocess': 0.18315498477565692, 'inference': 2.659837211295435, 'loss': 0.0007454081026899825, 'postprocess': 0.9653740902358469}
# task: 'detect'


############## yolo12n
# results_dict: {'metrics/precision(B)': 0.4770278944724741, 'metrics/recall(B)': 0.5849358791881755, 'metrics/mAP50(B)': 0.4355886849651079, 'metrics/mAP50-95(B)': 0.3068439064127416, 'fitness': 0.3197183842679783}
# save_dir: PosixPath('runs/detect/val6')
# speed: {'preprocess': 0.18224966913618218, 'inference': 3.620122177498364, 'loss': 0.0004459246133026697, 'postprocess': 0.935666549206945}
# task: 'detect'

############# yolo26n
# results_dict: {'metrics/precision(B)': 0.2505685551938135, 'metrics/recall(B)': 0.26083325600816554, 'metrics/mAP50(B)': 0.15026783494475832, 'metrics/mAP50-95(B)': 0.06916900295056053, 'fitness': 0.07727888614998031}
# save_dir: PosixPath('runs/detect/val5')
# speed: {'preprocess': 0.19171652246526497, 'inference': 2.5778451199409487, 'loss': 0.0004641273733277899, 'postprocess': 1.4040420334221067}
# task: 'detect'