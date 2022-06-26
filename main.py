from collections import defaultdict
import json
import argparse
import os
import random

import torch
from PIL import Image
from tqdm import tqdm

from interpreter import *
from executor import *
from methods import *

# 字典中对应三个类
METHODS_MAP = {
    "baseline": Baseline,
    "random": Random,
    "parse": Parse,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="input file with expressions and annotations in jsonlines format")
    parser.add_argument("--image_root", type=str, help="path to images (train2014 directory of COCO)")
    parser.add_argument("--clip_model", type=str, default="RN50x16,ViT-B/32", help="which clip model to use (should use RN50x4, ViT-B/32, or both separated by a comma")
    parser.add_argument("--albef_path", type=str, default=None, help="to use ALBEF (instead of CLIP), specify the path to the ALBEF checkpoint")
    parser.add_argument("--method", type=str, default="parse", help="method to solve expressions")
    parser.add_argument("--box_representation_method", type=str, default="crop,blur", help="method of representing boxes as individual images (crop, blur, or both separated by a comma)")
    parser.add_argument("--box_method_aggregator", type=str, default="sum", help="method of combining box representation scores")
    parser.add_argument("--box_area_threshold", type=float, default=0.0, help="minimum area (as a proportion of image area) for a box to be considered as the answer")
    parser.add_argument("--output_file", type=str, default=None, help="(optional) output path to save results")
    parser.add_argument("--detector_file", type=str, default=None, help="(optional) file containing object detections. if not provided, the gold object boxes will be used.")
    parser.add_argument("--mock", action="store_true", help="(optional) mock CLIP execution.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device to use.")
    # 默认不触发，不触发为false，默认为false，若触发为true
    parser.add_argument("--shuffle_words", action="store_true", help="If true, shuffle words in the sentence")
    # parser.add_argument("--shuffle_words", action="store_false", help="If true, shuffle words in the sentence")
    parser.add_argument("--gradcam_alpha", type=float, nargs='+', help="alpha value to use for gradcam method")
    parser.add_argument("--enlarge_boxes", type=float, default=0.0, help="(optional) whether to enlarge boxes when passing them to the model")
    parser.add_argument("--part", type=str, default=None, help="(optional) specify how many parts to divide the dataset into and which part to run in the format NUM_PARTS,PART_NUM")
    parser.add_argument("--batch_size", type=int, default=1, help="number of instances to process in one model call (only supported for baseline model)")
    # 对于基线，控制是否在表达式的完整表达式和头部名词块上调用模型
    parser.add_argument("--baseline_head", action="store_true", help="For baseline, controls whether model is called on both full expression and head noun chunk of expression")
    parser.add_argument("--mdetr", type=str, default=None, help="to use MDETR as the executor model, specify the name of the MDETR model")
    parser.add_argument("--albef_block_num", type=int, default=8, help="block num for ALBEF gradcam")
    parser.add_argument("--albef_mode", type=str, choices=["itm", "itc"], default="itm")
    parser.add_argument("--expand_position_embedding", action="store_true")
    parser.add_argument("--gradcam_background", action="store_true")
    parser.add_argument("--mdetr_given_bboxes", action="store_true")
    parser.add_argument("--mdetr_use_token_mapping", action="store_true")
    parser.add_argument("--non_square_size", action="store_true")
    parser.add_argument("--blur_std_dev", type=int, default=100, help="standard deviation of Gaussian blur")
    parser.add_argument("--gradcam_ensemble_before", action="store_true", help="Average gradcam maps of different models before summing over the maps")
    parser.add_argument("--cache_path", type=str, default=None, help="cache features")
    # Arguments related to Parse method.
    # 下面 4 个是启发式参数
    parser.add_argument("--no_rel", action="store_true", help="Disable relation extraction.")
    parser.add_argument("--no_sup", action="store_true", help="Disable superlative extraction.")
    parser.add_argument("--no_null", action="store_true", help="Disable null keyword heuristics.")
    parser.add_argument("--ternary", action="store_true", help="Disable ternary relation extraction.")

    parser.add_argument("--baseline_threshold", type=float, default=float("inf"), help="(Parse) Threshold to use relations/superlatives.")
    parser.add_argument("--temperature", type=float, default=1., help="(Parse) Sigmoid temperature.")
    parser.add_argument("--superlative_head_only", action="store_true", help="(Parse) Superlatives only quanntify head predicate.")
    parser.add_argument("--sigmoid", action="store_true", help="(Parse) Use sigmoid, not softmax.")
    parser.add_argument("--no_possessive", action="store_true", help="(Parse) Model extraneous relations as possessive relations.")
    # (Parse模块参数) 扩展名词词块，以包括后代分词token，该token不是其他词块的token祖先
    parser.add_argument("--expand_chunks", action="store_true", help="(Parse) Expand noun chunks to include descendant tokens that aren't ancestors of tokens in other chunks")
    # 只有在表达式中包含关系/最高级关键字时才执行解析过程，默认 false
    parser.add_argument("--parse_no_branch", action="store_true", help="(Parse) Only do the parsing procedure if some relation/superlative keyword is in the expression")
    parser.add_argument("--possessive_no_expand", action="store_true", help="(Parse) Expand ent2 in possessive case")
    args = parser.parse_args()

    # 解析json文件
    with open(args.input_file) as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]

    # 选择模型方法
    device = f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu"
    gradcam = args.method == "gradcam"
    if args.albef_path is not None:
        if args.method.split("_")[0] == "gradcam":
            if len(args.method.split("_")) == 1:
                args.method = "baseline"
            else:
                args.method = args.method.split("_")[1]
            gradcam_alpha = args.gradcam_alpha[0]
            executor = AlbefGradcamExecutor(config_path=os.path.join(args.albef_path, "config.yaml"), checkpoint_path=os.path.join(args.albef_path, "checkpoint.pth"), box_representation_method=args.box_representation_method, device=device, gradcam_alpha=gradcam_alpha, gradcam_mode=args.albef_mode, block_num=args.albef_block_num)
        else:
            executor = AlbefExecutor(config_path=os.path.join(args.albef_path, "config.yaml"), checkpoint_path=os.path.join(args.albef_path, "checkpoint.pth"), box_representation_method=args.box_representation_method, method_aggregator=args.box_method_aggregator, device=device, mode=args.albef_mode, blur_std_dev=args.blur_std_dev, cache_path=args.cache_path)
    elif args.mock:
        executor = MockExecutor()
    elif args.mdetr is not None:
        executor = MdetrExecutor(model_name=args.mdetr, device=device, use_token_mapping=args.mdetr_use_token_mapping, freeform_bboxes=not args.mdetr_given_bboxes)
    else:
        if args.method.split("_")[0] == "gradcam":
            if len(args.method.split("_")) == 1:
                args.method = "baseline"
            else:
                args.method = args.method.split("_")[1]
            executor = ClipGradcamExecutor(clip_model=args.clip_model, box_representation_method=args.box_representation_method, device=device, gradcam_alpha=args.gradcam_alpha, expand_position_embedding=args.expand_position_embedding, square_size=not args.non_square_size, background_subtract=args.gradcam_background, gradcam_ensemble_before=args.gradcam_ensemble_before)
        else:  # 默认执行这个 CLIP 模型, args.method == parser
            # TODO: 核心1，这里只是执行器的配置，并不是真正在执行
            # 下述 ClipExecutor代码 只调用了 ClipExecutor 类的 __init__ 函数进行初始化，并没有执行 __call__ 魔鬼函数进行前向计算
            executor = ClipExecutor(clip_model=args.clip_model, box_representation_method=args.box_representation_method,
                                    method_aggregator=args.box_method_aggregator, device=device,
                                    square_size=not args.non_square_size, expand_position_embedding=args.expand_position_embedding,
                                    blur_std_dev=args.blur_std_dev, cache_path=args.cache_path)

    # TODO：这段代码写得牛批，METHODS_MAP 是一个字典，根据args.method得到字典中对应的类之后，调用类中__init__对args参数进行初始化相应的method
    method = METHODS_MAP[args.method](args)

    correct_count = 0
    total_count = 0
    if args.output_file:
        output_file = open(args.output_file, "w")
    if args.detector_file:
        detector_file = open(args.detector_file)
        detections_list = json.load(detector_file)
        if isinstance(detections_list, dict):
            detections_map = {int(image_id): detections_list[image_id] for image_id in detections_list}
        else:
            detections_map = defaultdict(list)
            for detection in detections_list:
                detections_map[detection["image_id"]].append(detection["box"])
    if args.part is not None:
        num_parts = int(args.part.split(",")[0])
        part = int(args.part.split(",")[1])
        data = data[int(len(data)*part/num_parts):int(len(data)*(part+1)/num_parts)]
    batch_count = 0
    batch_boxes = []
    batch_gold_boxes = []
    batch_gold_index = []
    batch_file_names = []
    batch_sentences = []

    """ 最核心的就是 parse.py, interpreter.py, executor.py(CLIP 计算) """
    # 开始处理数据
    for datum in tqdm(data):  # tqdm是进度条库，对data数据显示进度条, 进度条是按照数据条数来算的，一个进度条要处理 若干个 sentence
        # 解析出当前数据的对应的图片
        if "coco" in datum["file_name"].lower():
            # 去除 file_name 中的尾注：COCO_train2014_000000380440_491042.jpg
            file_name = "_".join(datum["file_name"].split("_")[:-1])+".jpg"
            # print(file_name)  # eg. COCO_train2014_000000380440.jpg
        else:
            file_name = datum["file_name"]
        # 得到图片的索引位置
        img_path = os.path.join(args.image_root, file_name)
        img = Image.open(img_path).convert('RGB')
        # 读取 boundingBox，一幅图片有多个bbox
        gold_boxes = [Box(x=ann["bbox"][0], y=ann["bbox"][1], w=ann["bbox"][2], h=ann["bbox"][3]) for ann in datum["anns"]]
        # 要把标注的序号由数字变成列表
        if isinstance(datum["ann_id"], int) or isinstance(datum["ann_id"], str):
            datum["ann_id"] = [datum["ann_id"]]
        assert isinstance(datum["ann_id"], list)

        # 选取 bounding_box 的标注id 和 标注本身的 id 一致的 id，datum["anns"]是表示bbox
        # gold_index 表示 “sentences” 文本label信息是对应描述 “anns” bbox 框出的物体中（通常会框出好几个物体）的第几个，再将该box去ann_id 中查找
        # TODO: 目前看到的是一张图片只有一个 gold_boxes
        gold_index = [i for i in range(len(datum["anns"])) if datum["anns"][i]["id"] in datum["ann_id"]]
        # print(gold_index)  # 打印出的结果是一个列表，cocog中只看到包含一个数

        # 下面开始对一幅图片中的每一个句子开始处理
        for sentence in datum["sentences"]:
            if args.detector_file:
                boxes = [Box(x=box[0], y=box[1], w=box[2], h=box[3]) for box in detections_map[int(datum["image_id"])]]
                if len(boxes) == 0:
                    boxes = [Box(x=0, y=0, w=img.width, h=img.height)]
            else:
                boxes = gold_boxes

            # TODO: 核心2，配置 执行环境
            # 也只是把 img，boxes，execute 等信息传入，初始化一个env信息，boxes 是 box的列表，datum["image_id"]是图片名后缀如：274266
            env = Environment(img, boxes, executor, (args.mdetr is not None and not args.mdetr_given_bboxes), str(datum["image_id"]))

            # 计算！
            if args.shuffle_words:
                # print("store true")
                words = sentence["raw"].lower().split()
                random.shuffle(words)
                # 传给 execute 的 sentence 已经被随机打乱了，句子没有逻辑关系
                result = method.execute(" ".join(words), env)
            else:
                # print("store false")
                # 默认执行这一条，不shuffle
                # TODO: 核心3，计算！此时只计算1个句子
                '''通过 method 调用并初始化 parser，在parser中调用 execute 开始执行，在parser.execute中调用 Environment.filter 
                   执行Environment类，再在Environment.filter中调用 executor 执行 ClipExecutor'''
                result = method.execute(sentence["raw"].lower(), env)

            boxes = env.boxes
            # print("The Final Result: ", result)
            # print(sentence["raw"].lower())
            # print("boxes: ", boxes)
            """ 举个例子，refcocog_test 第一张图片，6 个 box，2句sentences
                'the man in yellow coat'
                'skiier in red pants.'
            boxes:  [Box(x=228.98, y=41.4, w=141.95, h=270.38), Box(x=374.31, y=65.06, w=136.04, h=201.94), 
                     Box(x=244.21, y=251.3, w=141.07, h=90.63), Box(x=291.38, y=190.07, w=185.19, h=68.27), 
                     Box(x=416.16, y=95.57, w=66.98, h=58.28), Box(x=288.6, y=75.67, w=54.76, h=27.93)]
            """
            correct = False
            # 预测的 bbox 和 groundtruth 中的bbox iou > 0.5 则算正确
            for g_index in gold_index:
                if iou(boxes[result["pred"]], gold_boxes[g_index]) > 0.5:
                    correct = True
                    break

            if correct:
                # print("predicate correct")
                result["correct"] = 1
                correct_count += 1
            else:
                # print("predicate false")
                result["correct"] = 0

            """没有使用，跳过"""
            if args.detector_file:
                argmax_ious = []
                max_ious = []
                for g_index in gold_index:
                    ious = [iou(box, gold_boxes[g_index]) for box in boxes]
                    argmax_iou = -1
                    max_iou = 0
                    if max(ious) >= 0.5:
                        for index, value in enumerate(ious):
                            if value > max_iou:
                                max_iou = value
                                argmax_iou = index
                    argmax_ious.append(argmax_iou)
                    max_ious.append(max_iou)
                argmax_iou = -1
                max_iou = 0
                if max(max_ious) >= 0.5:
                    for index, value in zip(argmax_ious, max_ious):
                        if value > max_iou:
                            max_iou = value
                            argmax_iou = index
                result["gold_index"] = argmax_iou
            else:
                result["gold_index"] = gold_index

            result["bboxes"] = [[box.left, box.top, box.right, box.bottom] for box in boxes]
            result["file_name"] = file_name
            result["probabilities"] = result["probs"]
            result["text"] = sentence["raw"].lower()
            if args.output_file:
                # Serialize numpy arrays for JSON.
                for key in result:
                    if isinstance(result[key], np.ndarray):
                        result[key] = result[key].tolist()
                    if isinstance(result[key], np.int64):
                        result[key] = result[key].item()
                output_file.write(json.dumps(result)+"\n")
            total_count += 1
            # print(f"est_acc: {100 * correct_count / total_count:.3f}")  # 这个是一路测试下来的

    if args.output_file:
        output_file.close()
    print(f"acc: {100 * correct_count / total_count:.3f}")
    stats = method.get_stats()
    if stats:
        pairs = sorted(list(stats.items()), key=lambda tup: tup[0])
        for key, value in pairs:
            if isinstance(value, float):
                print(f"{key}: {value:.5f}")
            else:
                print(f"{key}: {value}")



