from mmdet.apis import init_detector, inference_detector
import os
import cv2

import argparse
import torch
from tqdm import tqdm
from torchvision import transforms
from utils.datasets import create_dataloader
from utils.general import check_img_size, non_max_suppression, scale_coords, colorstr
from utils.torch_utils import select_device
from models.experimental import attempt_load
import os
from utils.reader import get_sort
from utils.recognition import batch_char_recog,batch_char_recog_exe
from utils.vit import vit_base_im96_patch8
import cv2
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel, LoraConfig, get_peft_model
import re
import random
import math
import torchvision.transforms.functional as TF  # 导入transforms.functional
import torch.nn.functional as F
from diffusers import UNet2DModel
from document.tools.build_HDR import build_pipeline
from fontTools.ttLib import TTFont
# from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont
from transformers import set_seed
from opencc import OpenCC
from zhconv import convert
from utils_pipeline import calculate_iou, xyxy2xywh, rank_probability_weighted_fusion, get_topk_multi_tokens, model_init, visualize_boxes, render_char_with_font_T, concatenate_images_vertical, render_char_with_font_L, is_char_in_font, invert_image, restore_image
from scipy.ndimage import binary_dilation, binary_fill_holes
from scipy.ndimage import label as ndimage_label
from det_wrapper import det_model

cc = OpenCC('s2t')
ss = OpenCC('t2s')



def detect(
         opt,
         model,
         data,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         dataloader=None,
         half_precision=True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




    # Load model
    gs = max(int(model.stride.max()), 32)
    imgsz = check_img_size(imgsz, s=gs)  # check img_size

    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()

    # Dataloader
    dataloader = create_dataloader(data, imgsz, batch_size, gs, opt, pad=0.5, rect=True,
                                    prefix=colorstr('val: '), infer=True)[0]
    
    for img, targets, paths, shapes in tqdm(dataloader):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        with torch.no_grad():
            # Run model
            #out, train_out = model(img, augment=False)
            out , train_out= model(img, augment=False)
            # Run NMS
            out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=[], multi_label=True)

        res_dic = {}
        # Statistics per image
        for si, predn in enumerate(out):
            # Predictions
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred
            res = predn[:, :5].cpu().numpy().astype(float).tolist()
            res_dic[paths[si]] = res

    return res_dic

def main(data, opt):

    yield "开始修复...", None

    data_path_api = 'api_test.png'
    data.save(data_path_api)
    data = data_path_api
    opt.data = data

    if opt.seed is not None:
        set_seed(opt.seed)
    
    save_path = './results'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    img_dir = os.path.join(save_path, 'img')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    combined_dir = os.path.join(save_path, 'combined')
    if not os.path.exists(combined_dir):
        os.makedirs(combined_dir)


    yield "加载模型...", None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    # 加载破损检测模型 dino
    model_det_vague = init_detector(opt.vague_det_config, opt.vague_det_weights, device=device)
    dicp = 'ckpt/dic_31524.txt'
    char_dict = open(dicp, encoding='utf-8').read().splitlines()
    det_model = attempt_load(opt.ocr_det_weights, map_location=device)
    # reg_model = vit_base_im96_patch8(num_classes=31524)
    # reg_model = torch.nn.DataParallel(reg_model).to(device)
    # # reg_model.load_state_dict(torch.load('ckpt/ocr_reg.pth')['state_dict'])
    # reg_model.load_state_dict(torch.load('./epoch79.pth')['state_dict'])

    #img = Image.open(data).convert('RGB')
    #img = data if isinstance(data, Image.Image) else Image.open(data).convert('RGB')
    yield "OSTU二值化...", None

    img = Image.open(data).convert('RGB')
    img_invert = invert_image(img)
    img_invert_path = 'tmp_img/api_tmp.jpg'
    img_invert.save(img_invert_path)
    img_invert_gray = Image.open(img_invert_path).convert('L').convert('RGB')

    print('detecting...')
    yield "OCR检测...", None
    ##### 这里是破损检测模型
    # 破损检测模型是灰度输入的
    results = inference_detector(model_det_vague, np.array(img_invert_gray))
    # 获取预测框、分数和标签
    boxes = results.pred_instances.bboxes.cpu().numpy()
    scores = results.pred_instances.scores.cpu().numpy() 
    labels = results.pred_instances.labels.cpu().numpy()
    # 处理每个检测框
    vague_det_bbox = []
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(lambda x: int(round(x)), box)
        if score > 0.3:
            vague_det_bbox.append([x1, y1, x2, y2])

    ##### 这里是ocr检测模型
    # 是RGB输入的
    res_dic = detect(opt,
                     det_model,
                        img_invert_path,
                        opt.det_batch_size,
                        opt.img_size,
                        opt.conf_thres,
                        opt.iou_thres,
                        )
    torch.cuda.empty_cache()
    ocr_det_bbox = []
    for k,v in res_dic.items():
        for coor in v:
            ocr_det_bbox.append([round(coor[0]),round(coor[1]),round(coor[2]),round(coor[3])])

    im = cv2.imread(img_invert_path, 0)

    print('recognizing...')
    yield "OCR识别...", None    

    # 这里首先对OCR检测框进行识别
    char_ims = []
    for line in ocr_det_bbox:
        x1, y1, x2, y2 = [round(float(k)) for k in line]
        char_ims.append(im[y1:y2, x1:x2])

    output_chars, output_probs = batch_char_recog_exe(device, char_dict, char_ims, bs=opt.reg_batch_size)

    # 破损检测框中的字也需要被识别
    char_ims = []
    for line in vague_det_bbox:
        x1, y1, x2, y2 = [round(float(k)) for k in line]
        char_ims.append(im[y1:y2, x1:x2])
    vague_output_chars, vague_output_probs = batch_char_recog_exe(device, char_dict, char_ims, bs=opt.reg_batch_size)

    # # 这一步只在OCR检测不准的时候用
    # chars = {}
    # to_remove_detect_result = set()
    # for idx, (char, prob) in enumerate(zip(output_chars, output_probs)):
    #     chars[idx] = char[0]
    #     # 如果prob[0]小于0.2，则认为这里没有字符，删除detect_result中对应的结果
    #     if prob[0] < 0.7:
    #         # 找到对应的detect_result中的结果
    #         to_remove_detect_result.add(idx)

    # 坐标作为key 字符和置信度作为value 后面要用
    OCR_result = {}
    for idx, (char, prob, detect_box) in enumerate(zip(output_chars, output_probs, ocr_det_bbox)):
        OCR_result[str(detect_box)] = (char, prob)
    
    vague_OCR_result = {}
    for idx, (char, prob, detect_box) in enumerate(zip(vague_output_chars, vague_output_probs, vague_det_bbox)):
        vague_OCR_result[str(detect_box)] = (char, prob)

    # 上面删除了置信度低的识别结果和对应的检测框，现在要合并字符框和破损框
    to_remove = set()
    iou_threshold = 0.5
    for vague_box in vague_det_bbox:
        for idx, detect_box in enumerate(ocr_det_bbox):
            iou = calculate_iou(vague_box, detect_box)
            # 如果IoU大于阈值，说明两者重合
            if iou >= iou_threshold:
                # 重合的时候我只要破损检测框，没啥问题这一步
                to_remove.add(idx)
    
    print('arrange...')
    yield "处理阅读顺序...", None

    # 创建一个新的结果列表来拼接 这个用来做阅读顺序的，只是一个list
    final_results = []
    # 添加degraded_removed_detect_result中的结果
    final_results.extend(vague_det_bbox)

    # 添加未被删除的ocr_det_bbox中的结果
    detect_box_remove = []
    for idx, detect_box in enumerate(ocr_det_bbox):
        if idx not in to_remove:
            detect_box_remove.append(detect_box) # debug可视化用的而已
            final_results.append(detect_box)
    # visualize_boxes('invert_tmp.jpg', vague_det_bbox, detect_box_remove)
    h, w = im.shape[:2]
    out_box_li = get_sort(final_results, h, w)

    chars_list = []
    num_ocr = 0
    num_degraded = 0
    degraded_dict = {}
    extra_num_ocr_prob_dict = {}
    for box in out_box_li:
        if str(box) in OCR_result.keys():
            num_ocr += 1
            chars_list.append(OCR_result[str(box)][0][0])
        elif str(box) in vague_OCR_result.keys():
            prob = vague_OCR_result[str(box)][1][0]
            if prob < 0.9:
                
                chars_list.append(f'<|extra_{num_degraded}|>')
                degraded_dict[f'<|extra_{num_degraded}|>'] = [box]
                box_xywh = xyxy2xywh(box)
                extra_num_ocr_prob_dict[f'<|extra_{num_degraded}|>'] = {
                                                                        'ocr_prob': vague_OCR_result[str(box)][1],
                                                                        'alternatives': vague_OCR_result[str(box)][0],
                                                                        'bbox': tuple(box_xywh),
                                                                        'txt': vague_OCR_result[str(box)][0][0], # 先给一个初始值（ocr top1） 语言模型有可能会少预测这个字，那就会出问题
                                                                        'flag': False,
                                                                    }
                num_degraded += 1
            else:
                num_ocr += 1
                chars_list.append(vague_OCR_result[str(box)][0][0])
                box_xywh = xyxy2xywh(box)
                extra_num_ocr_prob_dict[f'<|ocr_{num_ocr}|>'] = {
                                                                        'ocr_prob': vague_OCR_result[str(box)][1],
                                                                        'alternatives': vague_OCR_result[str(box)][0],
                                                                        'bbox': tuple(box_xywh),
                                                                        'txt': vague_OCR_result[str(box)][0][0],
                                                                        'flag': False,
                                                                    }
        else:
            print('出现了未知的框')
            import pdb; pdb.set_trace()
    
    if num_degraded > 270:
        return None, None
        raise ValueError('Too many degraded characters')


    print(f'识别字符：【{num_ocr}】个，识别破损位置：【{num_degraded}】个')
    char_str = ''.join(chars_list)
    
    yield f'识别字符：【{num_ocr}】个，识别破损位置：【{num_degraded}】个', None
    yield 'OCR识别结果...', None
 
    char_str = convert(char_str, 'zh-cn')
    char_str = cc.convert(char_str)
    yield char_str, None

    del model_det_vague
    del det_model
#    del reg_model

    torch.cuda.empty_cache()

    print('predicting...')
    yield "预测缺失文本...", None
    model_name_or_path = opt.model_name_or_path
    model, tokenizer = model_init(model_name_or_path)

    messages = [
        {
            "role": "system",
            "content": "请帮助恢复古籍中缺失的字。",
        },
        {"role": "user", "content": char_str},
    ]
    messages = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    
    batch_inputs = []
    results = []
    prompt_list = []
    batch_inputs.append(messages)
    prompt_list.append(char_str)

    model_inputs = tokenizer(batch_inputs, padding=True, return_tensors="pt").to(device)
    batch_outputs = model.generate(**model_inputs, 
                                    max_new_tokens=4096,
                                    repetition_penalty=1.0,
                                    pad_token_id=tokenizer.eos_token_id,
                                    eos_token_id=tokenizer.eos_token_id,
                                    return_dict_in_generate=True,
                                    output_scores=True,
                                    )
    batch_scores = tuple(zip(*batch_outputs['scores']))
    
    for pred, scores, prompt_tmp, attention_mask in zip(batch_outputs['sequences'], batch_scores, prompt_list, model_inputs['attention_mask']):
        
        pred_text_ini = tokenizer.decode(pred.cpu(), skip_special_tokens=False)
        
        pred_text = pred_text_ini.split('assistant\n')[1]
        results.append(pred_text)

        pattern = r'(<\|extra_\d+\|>)(.)'
        output_matches = re.findall(pattern, pred_text)

        pred_text_tokens = pred.cpu()
        # 用于记录缺失字在输入序列中的位置
        special_token_positions_dict = {}
        found_assistant_token = False # 用来寻找序列中 assistant\n 中的assistant
        found_beagin_token = False # 用来寻找序列中 assistant\n 中的\n
        input_tokens_length = 0 # 用于记录input序列的长度

        for idx, token_id in enumerate(pred_text_tokens):
            token = tokenizer.decode([token_id])
            if token.startswith('assistant'): # 找到assistant
                found_assistant_token = True

            if not found_beagin_token: # 如果没有找到assistant\n 则表面还在遍历input序列
                input_tokens_length += 1 # 一定要放在 查找 \n 之前，因为 \n 也算一个token
            
            if token.startswith('\n') and found_assistant_token: # 找到assistant之后再找\n
                found_beagin_token = True

            if found_beagin_token and token.startswith('<|extra_') and token.endswith('|>'):
                # import pdb;pdb.set_trace()
                next_token_id = idx + 1
                next_token = tokenizer.decode([pred_text_tokens[next_token_id]])

                num_next_chartoken_count = 0
                token_id_list = []
                scores_id_list = []
                
                # 有些字需要多个token来表示，所以这里要一直拿，直到拿到下一个token
                while True:
                    if next_token.startswith('<|extra_') or next_token == '<|im_end|>':
                        break
                    num_next_chartoken_count += 1
                    next_token = tokenizer.decode([pred_text_tokens[next_token_id+num_next_chartoken_count]])
                    token_id_list.append(idx + num_next_chartoken_count)
                    scores_id_list.append(idx + num_next_chartoken_count - input_tokens_length)
                    # import pdb;pdb.set_trace()
                if token not in special_token_positions_dict.keys():
                    special_token_positions_dict[token] = (scores_id_list, tokenizer.decode([pred_text_tokens[i] for i in token_id_list]))

        top_k = 5
        for key, value in extra_num_ocr_prob_dict.items():
            if 'extra_' in key and key in special_token_positions_dict.keys():
                special_token_idx = special_token_positions_dict[key][0]
                pred_vague_char = special_token_positions_dict[key][1]
                if pred_vague_char == value:
                    correct_count_top1_tl += 1
                    cor_cur_top1_tl += 1

                # scores中间保存了预测出来的token的概率分布 只有assistant token后面的部分
                if len(special_token_idx) == 1:
                    token_scores = scores[special_token_idx[0]]
                    topk_tokens = torch.topk(token_scores, k=top_k).indices
                    topk_chars = [tokenizer.decode([t]) for t in topk_tokens]
                    token_scores = F.softmax(token_scores, dim=0)
                    values = torch.topk(token_scores, k=top_k).values
                    token_scores = [(topk_chars[i], values[i].item()) for i in range(len(values))]
                else:
                    topk_combinations = get_topk_multi_tokens(scores, tokenizer, start_idx=special_token_idx[0], num_tokens=len(special_token_idx), k=top_k)
                    topk_chars = [comb[0] for comb in topk_combinations]
                    token_scores = [(comb[0], comb[1].item()) for comb in topk_combinations]
                        
                try:
                    alternatives_tmp = extra_num_ocr_prob_dict[key]['alternatives']
                    ocr_prob_tmp = extra_num_ocr_prob_dict[key]['ocr_prob']
                    ocr_preds = [(alternatives_tmp[i], ocr_prob_tmp[i]) for i in range(top_k)]
                    best_char, char_score_sorted_dict = rank_probability_weighted_fusion(ocr_preds, token_scores)
                    ocr_llm_topk = [x[0] for x in char_score_sorted_dict][:top_k]
                    extra_num_ocr_prob_dict[key]['txt'] = best_char
                    extra_num_ocr_prob_dict[key]['ocr_llm_topk'] = ocr_llm_topk
                    extra_num_ocr_prob_dict[key]['llm_prob'] = token_scores
                    extra_num_ocr_prob_dict[key]['flag'] = False
                except:
                    # import pdb; pdb.set_trace()
                    best_char = extra_num_ocr_prob_dict[key]['alternatives'][0]
                    extra_num_ocr_prob_dict[key]['txt'] = best_char
                    extra_num_ocr_prob_dict[key]['flag'] = False
                    # print(f"出现多token 解码失败的情况")
                    # import pdb; pdb.set_trace()

    yield '缺失内容预测结果...', None
    yield str(output_matches), None

    print('开始加载修复模型')
    yield "加载修复模型...", None
    del model
    del tokenizer
    torch.cuda.empty_cache()

    ############### 加载修复模型 ###################
    unet = UNet2DModel.from_pretrained(pretrained_model_name_or_path='ckpt/unet')
    unet = unet.to(device)
    pipeline = build_pipeline(
        args=opt,
        unet=unet,)
    generator = torch.Generator(device=pipeline.device).manual_seed(opt.seed)

    print('开始切patches')
    yield "根据破损字符位置对图像切块...", None

    patch_size = 448
    stride = 224
    stride_x = 224
    stride_y = 224
    img_w, img_h = img_invert.size
    patches = []

    # 创建用于可视化的图像副本
    vis_img = img_invert.copy()
    draw = ImageDraw.Draw(vis_img)
    patch_count = 0
    
    count_while_debug = 0
    while True:
        count_while_debug += 1
        # 检查是否还有未修复的字框
        unrepaired_exists = False
        for bbox_info in extra_num_ocr_prob_dict.values():
            if not bbox_info['flag']:
                unrepaired_exists = True
                break
        
        if not unrepaired_exists:  # 所有字框都已修复
            break

        # 找到未修复字框的边界
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')

        for bbox_name, bbox_info in extra_num_ocr_prob_dict.items():
            if bbox_info['flag']:  # 跳过已修复的字框
                continue
            x, y, w, h = bbox_info['bbox']
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)

        # 定义四个角落（不预先减去patch_size）
        corners = [
            ('left_top', (min_x, min_y)),
            ('left_bottom', (min_x, max_y)),
            ('right_top', (max_x, min_y)),
            ('right_bottom', (max_x, max_y))
        ]

        # 计算每个角落patch包含的未修复字框数量
        corner_counts = []
        for corner_name, (corner_x, corner_y) in corners:
            # 根据角落位置计算实际的start位置
            if 'right' in corner_name:
                start_x = min(max(0, corner_x - patch_size), img_w - patch_size)
            else:
                start_x = max(0, corner_x)
                
            if 'bottom' in corner_name:
                start_y = min(max(0, corner_y - patch_size), img_h - patch_size)
            else:
                start_y = max(0, corner_y)
            
            end_x = min(start_x + patch_size, img_w)
            end_y = min(start_y + patch_size, img_h)
            
            # 计算当前patch中未修复的字框数量
            count = 0
            for bbox_name, bbox_info in extra_num_ocr_prob_dict.items():
                if bbox_info['flag']:  # 跳过已修复的字框
                    continue
                x, y, w, h = bbox_info['bbox']
                if (x >= start_x and x + w <= end_x and 
                    y >= start_y and y + h <= end_y):
                    count += 1
            corner_counts.append((corner_name, start_x, start_y, count))

        # 找到包含最少未修复字框的角落
        corner_counts.sort(key=lambda x: x[3])  # 按未修复字框数量升序排序
        best_corner_name, start_x, start_y, _ = corner_counts[0]

        # 根据选择的角落决定滑动方向
        if best_corner_name == 'left_top':
            x_range = range(0, img_w, stride_x)
            y_range = range(0, img_h, stride_y)
        elif best_corner_name == 'left_bottom':
            x_range = range(0, img_w, stride_x)
            y_range = range(img_h - patch_size, -patch_size, -stride_y)
        elif best_corner_name == 'right_top':
            x_range = range(img_w - patch_size, -patch_size, -stride_x)
            y_range = range(0, img_h, stride_y)
        else:  # right_bottom
            x_range = range(img_w - patch_size, -patch_size, -stride_x)
            y_range = range(img_h - patch_size, -patch_size, -stride_y)

        # 从选定的角落开始滑动窗口
        for curr_x in x_range:
            for curr_y in y_range:
                # 计算实际的patch位置
                start_x = max(0, min(curr_x, img_w - patch_size))
                start_y = max(0, min(curr_y, img_h - patch_size))
                end_x = min(start_x + patch_size, img_w)
                end_y = min(start_y + patch_size, img_h)

                # 确保patch大小正确
                if end_x - start_x != patch_size or end_y - start_y != patch_size:
                    if curr_x >= img_w - patch_size:
                        start_x = img_w - patch_size
                        end_x = img_w
                    if curr_y >= img_h - patch_size:
                        start_y = img_h - patch_size
                        end_y = img_h

                # 检查当前patch中完整包含了哪些字框
                contained_bboxes = []
                intersect_bboxes = []
                for bbox_name, bbox_info in extra_num_ocr_prob_dict.items():
                    if bbox_info['flag']:  # 跳过已修复的字框
                        continue
                    x, y, w, h = bbox_info['bbox']
                    if (x >= start_x and x + w <= end_x and 
                        y >= start_y and y + h <= end_y):
                        contained_bboxes.append(bbox_name)
                        extra_num_ocr_prob_dict[bbox_name]['flag'] = True
                    elif (not (x >= end_x or x + w <= start_x or
                            y >= end_y or y + h <= start_y)):
                        intersect_bboxes.append(bbox_name)
                
                if contained_bboxes:
                    patch = img_invert.crop((start_x, start_y, end_x, end_y))
                    patch_info = {
                        'position': (start_x, start_y, end_x, end_y),
                        'contained_bboxes': contained_bboxes,
                        'intersect_bboxes': intersect_bboxes
                    }
                    patches.append((patch, patch_info))
    # print(f'img{i} success, patch num: {len(patches)}')
    print(f'共有{len(patches)}个patch')
    yield f'共有{len(patches)}个patch', None
    # import pdb; pdb.set_trace()

    print_i = 0
    for patch in tqdm(patches):
        
        print_i += 1
        yield f"正在修复第{print_i}个patch...", None
        # 图像块的左上角和右下角坐标
        xmin, ymin, xmax, ymax = patch[1]['position']
        # degraded_image = patch[0]
        degraded_image = img_invert.crop((xmin, ymin, xmax, ymax))

        # 创建mask和content img
        content_image = Image.new('L', (patch_size, patch_size), 255)
        mask_image = Image.new('L', (patch_size, patch_size), 0)

        # 创建整个patch的mask
        combined_mask = np.zeros((patch_size, patch_size), dtype=np.uint8)

        repair_image_dict = {}
        for bbox_name in patch[1]['contained_bboxes']:
            bbox_info = extra_num_ocr_prob_dict[bbox_name]
            # 单字框的左上角和右下角坐标
            bx_min, by_min, bw, bh = bbox_info['bbox']
            bx_max = bx_min + bw; by_max = by_min + bh
            # 计算单字框在图像块中的相对坐标
            rel_x_min = int(bx_min - xmin)
            rel_y_min = int(by_min - ymin)
            rel_x_max = int(bx_max - xmin)
            rel_y_max = int(by_max - ymin)

            combined_mask[rel_y_min:rel_y_max, rel_x_min:rel_x_max] = 255

            # 绘制单字框的mask
            mask = Image.new('L', (rel_x_max - rel_x_min, rel_y_max - rel_y_min), 255)
            mask_image.paste(mask, (rel_x_min, rel_y_min, rel_x_max, rel_y_max))
            # degraded_image.paste(mask, (rel_x_min, rel_y_min, rel_x_max, rel_y_max)) # 考虑要不要inpainting修复
            # import pdb; pdb.set_trace()
            # 渲染单字图片
            single_char_img = render_char_with_font_T(bbox_info['txt'])
            # 将单字图片resize成box的大小
            single_char_img = single_char_img.resize((rel_x_max - rel_x_min, rel_y_max - rel_y_min), Image.Resampling.LANCZOS)
            # 将单字图片粘贴到content图片中
            content_image.paste(single_char_img, (rel_x_min, rel_y_min))


        for bbox_name in patch[1]['intersect_bboxes']:
            bbox_info = extra_num_ocr_prob_dict[bbox_name]
            bx_min, by_min, bw, bh = bbox_info['bbox']
            bx_max = bx_min + bw; by_max = by_min + bh
            
            # 确保坐标在patch范围内
            rel_x_min = max(0, int(bx_min - xmin))
            rel_y_min = max(0, int(by_min - ymin))
            rel_x_max = min(patch_size, int(bx_max - xmin))
            rel_y_max = min(patch_size, int(by_max - ymin))
            
            # 只有当有效区域大于0时才绘制
            if rel_x_max > rel_x_min and rel_y_max > rel_y_min:
                combined_mask[rel_y_min:rel_y_max, rel_x_min:rel_x_max] = 255
                
                # 计算在原始字框中的相对位置
                crop_x = max(0, xmin - bx_min)  # 如果字框超出patch左边界，需要裁剪
                crop_y = max(0, ymin - by_min)  # 如果字框超出patch上边界，需要裁剪
                crop_w = rel_x_max - rel_x_min
                crop_h = rel_y_max - rel_y_min
                
                # 创建裁剪后大小的mask
                mask = Image.new('L', (crop_w, crop_h), 255)
                mask_image.paste(mask, (rel_x_min, rel_y_min, rel_x_max, rel_y_max))


        connected_mask = combined_mask.copy()
        height, width = combined_mask.shape
        
        # 水平方向连接
        for y in range(height):
            white_regions = np.where(combined_mask[y, :] == 255)[0]
            if len(white_regions) > 1:
                for i in range(len(white_regions)-1):
                    gap = white_regions[i+1] - white_regions[i]
                    if 1 < gap < 20:  # 可以调整这个阈值，控制最大连接距离
                        connected_mask[y, white_regions[i]:white_regions[i+1]] = 255
        
        # 垂直方向连接
        for x in range(width):
            white_regions = np.where(combined_mask[:, x] == 255)[0]
            if len(white_regions) > 1:
                for i in range(len(white_regions)-1):
                    gap = white_regions[i+1] - white_regions[i]
                    if 1 < gap < 20:  # 可以调整这个阈值，控制最大连接距离
                        connected_mask[white_regions[i]:white_regions[i+1], x] = 255
        filled_mask = binary_fill_holes(connected_mask).astype(np.uint8) * 255

        # 找出填充区域（原mask和填充后mask的差异区域）
        holes = filled_mask - connected_mask
        
        # 标记连通区域
        labeled_holes, num_features = ndimage_label(holes)
        
        # 计算每个连通区域的大小
        unique_labels, counts = np.unique(labeled_holes, return_counts=True)

        bbox_areas = []
        for bbox_name in patch[1]['contained_bboxes']:
            bbox_info = extra_num_ocr_prob_dict[bbox_name]
            bw, bh = bbox_info['bbox'][2], bbox_info['bbox'][3]  # 宽度和高度
            bbox_areas.append(bw * bh)
        
        if bbox_areas:
            min_char_area = min(bbox_areas)
            area_threshold = min_char_area * 0.8  # 设置为最小字符面积的50%
        else:
            area_threshold = 300
        
        # 创建最终的mask
        final_mask = connected_mask.copy()
        
        # 只填充小于阈值的区域
        for label_idx, count in zip(unique_labels[1:], counts[1:]):  # 跳过背景(label=0)
            if count < area_threshold:
                final_mask[labeled_holes == label_idx] = 255
    
        # # 转换回PIL Image
        # cv2.imwrite('a.png', connected_mask)
        # mask_image = Image.fromarray(final_mask)
        # mask_image.save('a1.png')
        # import pdb; pdb.set_trace()
        degraded_array = np.array(degraded_image)
        mask_array = np.array(final_mask)
        mask_3channel = np.stack([mask_array] * 3, axis=2)
        result_array = np.where(mask_3channel == 255, 255, degraded_array)
        degraded_image = Image.fromarray(result_array)
        # cv2.imwrite('a.jpg', final_mask)
        # import pdb; pdb.set_trace()
    

        repair_image_dict['content_image'] = content_image
        repair_image_dict['mask_image'] = mask_image
        repair_image_dict['degraded_image'] = degraded_image
        repair_image_dict['patch_bbox'] = [xmin, ymin, xmax, ymax]
        repair_image_dict['patch_size'] = patch_size

        
        # 调整patch_image的大小为512x512        
        degraded_image = degraded_image.resize((512, 512), Image.Resampling.LANCZOS)
        content_image = content_image.resize((512, 512), Image.Resampling.LANCZOS)
        mask_image = mask_image.resize((512, 512), Image.Resampling.LANCZOS)

        # 转换成tensor
        degraded_image_tensor = TF.normalize(TF.to_tensor(degraded_image), [0.5], [0.5]).unsqueeze(0)
        mask_image_tensor = TF.normalize(TF.to_tensor(mask_image), [0.5], [0.5]).unsqueeze(0)
        content_image_tensor = TF.normalize(TF.to_tensor(content_image), [0.5], [0.5]).unsqueeze(0)
        degraded_image_tensor = degraded_image_tensor.to(device)
        mask_image_tensor = mask_image_tensor.to(device)
        content_image_tensor = content_image_tensor.to(device)
        # 预测修复结果
        with torch.no_grad():
            image = pipeline(
                    degraded_image=degraded_image_tensor,
                    char_mask_image=mask_image_tensor,
                    content_image=content_image_tensor,
                    image_channel=opt.image_channel,
                    classifier_free=opt.classifier_free,
                    content_mask_guidance_scale=opt.content_mask_guidance_scale,
                    degraded_guidance_scale=opt.degraded_guidance_scale,
                    generator=generator,
                    batch_size=1,
                    num_inference_steps=opt.num_inference_steps,
                    output_type="pil",
                ).images[0]
            

        image = image.resize((patch_size, patch_size), Image.Resampling.LANCZOS)
        img_invert.paste(image, patch[1]['position'])
        # if print_i % 5 == 0:
        #     img_invert.save('patched_image_repaired_1.png')
        # import pdb; pdb.set_trace()
    restore_img = restore_image(img_invert)
    combined = concatenate_images_vertical(restore_img, img)
    print(data)
    # restore_img.save('restored_image.png')
    # combined.save('combined_image_1.png')

    if restore_img is not None:
        restore_img.save(os.path.join(f'{save_path}/img', 'tmp.jpg'))
        combined.save(os.path.join(f'{save_path}/combined', 'tmp.jpg'))

    del pipeline
    del unet
    del generator
    torch.cuda.empty_cache()
    # import pdb; pdb.set_trace()
    yield "修复完成",restore_img

# ###标识

#     yield "正在标识结果...", None
#     image_to_repair_mark = restore_img.copy()
#     draw = ImageDraw.Draw(image_to_repair_mark)

#     try:
#         font = ImageFont.truetype("demo_utils/font/KaiXinSongA.ttf", 20)  # 这里使用系统字体"宋体"
#     except IOError:
#         font = ImageFont.load_default() 

#     colors = ['blue', 'green', 'purple', 'orange', 'brown', 'cyan', 'magenta', 'yellow', 'pink', 'gray']

#     for idx,patch in tqdm(enumerate(patches)): 
#         #图像块坐标
#         xmin, ymin, xmax, ymax = patch[1]['position']
#         color = colors[idx % len(colors)]
#         # 绘制边框
#         draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)
#         label = f"Patch {idx+1} ({patch_size}x{patch_size})"
#         draw.text((xmin, ymin), label, fill=color, font=font)
#         #处理修复字符
#         for bbox_name in patch[1]['intersect_bboxes']:
#             bbox_info = extra_num_ocr_prob_dict[bbox_name]
#             bx_min, by_min, bw, bh = bbox_info['bbox']
#             bx_max = bx_min + bw; by_max = by_min + bh
            
#             # # 确保坐标在patch范围内
#             # rel_x_min = max(0, int(bx_min - xmin))
#             # rel_y_min = max(0, int(by_min - ymin))
#             # rel_x_max = min(patch_size, int(bx_max - xmin))
#             # rel_y_max = min(patch_size, int(by_max - ymin))
            
#             # 只有当有效区域大于0时才绘制
#             if rel_x_max > rel_x_min and rel_y_max > rel_y_min:
#                 combined_mask[rel_y_min:rel_y_max, rel_x_min:rel_x_max] = 255      
#                 #绘制字框
#                 draw.rectangle([bx_min, by_min, bx_max, by_max], outline=colors[(idx+1) % len(colors)], width=2)
#                 #绘制字
#                 text = bbox_info['txt']
#                 draw.text((bx_min, by_min), text, fill='red', font=font)
#     yield "标识完成",None
#     yield "标识完成",image_to_repair_mark
    
# ###
    return "修复完成",restore_img

if __name__ == '__main__':

    config_file = './ckpt/damage_detect.py' # 网络模型py文件
    damage_detect_checkpoint_file = './ckpt/damage_detect.pth'  # 训练好的模型参数
    model_name_or_path = './ckpt/AutoHDR-Qwen2-7B'
    ocr_det_weights = './ckpt/best.pt'

    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--ocr_det_weights', nargs='+', type=str, default=ocr_det_weights, help='model.pt path(s)')
    parser.add_argument('--vague_det_weights', nargs='+', type=str, default=damage_detect_checkpoint_file, help='model.pth path(s)')
    parser.add_argument('--vague_det_config', nargs='+', type=str, default=config_file, help='mmdetection config.py path(s)')
    parser.add_argument('--model_name_or_path', nargs='+', type=str, default=model_name_or_path, help='llm weight path(s)')

    parser.add_argument('--det-batch-size', type=int, default=1, help='size of each image batch')
    parser.add_argument('--reg-batch-size', type=int, default=36, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=2048, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.2, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument("--seed", type=int, default=123)

    # model
    parser.add_argument("--image_channel", type=int, default=3)
    # pipeline setting
    parser.add_argument("--pipeline", type=str, default="DPM-Solver++",
                        choices=['DDPM', 'DPM-Solver', 'DPM-Solver++'])
    parser.add_argument("--classifier_free", action="store_false", \
                        help="Whether to use classifier-free guidance sampling.")
    parser.add_argument(
        "--content_mask_guidance_scale", type=float, default=1.5, help="The guidance scale for contnet and mask image.")
    parser.add_argument(
        "--degraded_guidance_scale", type=float, default=1.2, help="The guidance scale for degraded image.")
    parser.add_argument(
        "--solver_order", type=int, default=2, help="If use DPM-Solver, set this parameter.")
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    parser.add_argument("--prediction_type", type=str, default="sample")
    
    parser.add_argument("--batch_infer", action="store_true", \
                        help="Whether to use batch type inference.")
    # If single image inference, should make sure the image size is 512
    parser.add_argument("--gt_image_path", type=str, default=None)
    # If batch image inference
    parser.add_argument("--data_dir", type=str, default=None, \
                        help="The folder should be consistent with train data dir.")
    parser.add_argument("--degrade_modes", nargs='+', default=['removal', 'hole', 'ink'])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)
    
    opt = parser.parse_args()

    if opt.seed is not None:
        set_seed(opt.seed)
    
    save_path = './results'
    img_path = 'example.jpg'


    if not os.path.exists(save_path):
        os.makedirs(save_path)

    img_dir = os.path.join(save_path, 'img')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    combined_dir = os.path.join(save_path, 'combined')
    if not os.path.exists(combined_dir):
        os.makedirs(combined_dir)

    restore_img, combined = main(data=img_path, opt=opt)
    if restore_img is not None:
        restore_img.save(os.path.join(f'{save_path}/img', img_path))
        combined.save(os.path.join(f'{save_path}/combined', img_path))