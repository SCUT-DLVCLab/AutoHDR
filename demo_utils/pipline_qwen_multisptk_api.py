import argparse
import torch
from tqdm import tqdm
from torchvision import transforms
from utils.datasets import create_dataloader
from utils.general import check_img_size, non_max_suppression, scale_coords, colorstr
from utils.torch_utils import select_device
from models.experimental import attempt_load
import os
from demo_utils.reader import get_sort
from demo_utils.recognition import batch_char_recog
from demo_utils.vit import vit_base_im96_patch8
import cv2

from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel, LoraConfig, get_peft_model
import re
import random
import math

from diffusers import UNet2DModel
from document.tools.build_HDR import build_pipeline
from fontTools.ttLib import TTFont
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont
from transformers import set_seed
import opencc
from mmdet.apis import init_detector, inference_detector

config_file = 'ckpt/damage_detect.py' # 网络模型py文件
damage_detect_checkpoint_file = 'ckpt/damage_detect.pth'  # 训练好的模型参数

def simplify_to_traditional(text):
    converter = opencc.OpenCC('s2t')  # s2t.json 表示简体到繁体
    return converter.convert(text)

def traditional_to_simplified(text):
    converter = opencc.OpenCC('t2s')  # t2s.json 表示繁体到简体
    return converter.convert(text)

font_list = [
    'demo_utils/font/KaiXinSongA.ttf',
    'demo_utils/font/KaiXinSongB.ttf'
]

ImageFont_fonts = []
TTFont_fonts = []
for font_path in font_list:
    font = ImageFont.truetype(font_path, 128)
    ImageFont_fonts.append(font)
    font = TTFont(font_path)
    TTFont_fonts.append(font)

def render_char_with_font_L(ImageFont_font, char, background_color=255, text_color=0):
    try:
        text_width, text_height = ImageFont_font.getsize(char)

        image = Image.new('L', (text_width, text_height), background_color)
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), char, text_color, font=ImageFont_font)
        return image
    except:
        import pdb;pdb.set_trace()

def is_char_in_font(TTFont_font, char):
    cmap = TTFont_font['cmap']
    for subtable in cmap.tables:
        if ord(char) in subtable.cmap:
            return True
    return False

def render_char_with_font_T(char, ImageFont_fonts = ImageFont_fonts, TTFont_font = TTFont_fonts):
    for i, (ImageFont_font, TTFont_font) in enumerate(zip(ImageFont_fonts, TTFont_fonts)):
        if is_char_in_font(TTFont_font, char):
            have_char_ImageFont = ImageFont_font
            break
        if i == len(ImageFont_fonts) - 1:
            print(f"No font can render the character {char}")
            import pdb;pdb.set_trace()
    img = render_char_with_font_L(ImageFont_font=have_char_ImageFont, char=char)
    return img

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

    device = select_device(opt.device, batch_size=batch_size)

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
            out, train_out = model(img, augment=False)
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

def img2bin(img_path, save_path):
    color_image = cv2.imread(img_path)
    if color_image is None:
        print("Image not found or unable to read.")
        exit()

    # 将彩色图像转换为灰度图
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(
            gray_image, 
            0,  # 阈值，但是会被OTSU覆盖
            255, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    cv2.imwrite(save_path, binary_image)

def calculate_iou(boxA, boxB):
    # 计算交集区域
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)  # 交集面积

    # 计算并集区域
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    unionArea = boxAArea + boxBArea - interArea  # 并集面积

    # 计算IoU
    return interArea / unionArea if unionArea > 0 else 0

def main(data, opt):

    yield "开始修复...", None

    data_path_api = 'api_test.png'
    data.save(data_path_api)
    data = data_path_api
    opt.data = data
    
    yield "加载模型...", None
    
    device = select_device('8', batch_size=opt.det_batch_size)
    dicp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dic_31556.txt')
    char_dict = open(dicp, encoding='utf-8').read().splitlines()
    det_model = attempt_load(opt.weights, map_location=device)
    #det_degraded_model = attempt_load(opt.degraded_weights, map_location=device)
    det_degraded_model=init_detector(config_file, damage_detect_checkpoint_file, device=device)
    reg_model = vit_base_im96_patch8(num_classes=31556)
    reg_model = torch.nn.DataParallel(reg_model).to(device)
    reg_model.load_state_dict(torch.load('ckpt/ocr_reg.pth')['state_dict'])

    yield "OSTU二值化...", None

    img_bin_path = data.split('.')[0] + '_bin.png'
    img2bin(data, img_bin_path)

    print('detecting...')
    yield "OCR检测...", None

    # 检测字符，检测破损
    res_dic = detect(opt,
                        det_model,
                        opt.data,
                        opt.det_batch_size,
                        opt.img_size,
                        opt.conf_thres,
                        opt.iou_thres,
                        )
    # 释放显存
    torch.cuda.empty_cache()

    degraded_dic = detect(opt, 
                          det_degraded_model, 
                          img_bin_path, 
                          opt.det_batch_size, 
                          1984, 
                          opt.conf_thres, 
                          opt.iou_thres)
    
    torch.cuda.empty_cache()
    
    h, w = cv2.imread(data).shape[:2]
    detect_result = []
    for k,v in res_dic.items():
        for coor in v:
            detect_result.append([round(coor[0]),round(coor[1]),round(coor[2]),round(coor[3])])

    degraded_detect_result = []
    for k,v in degraded_dic.items():
        for coor in v:
            degraded_detect_result.append([round(coor[0]),round(coor[1]),round(coor[2]),round(coor[3])])
    

    

    # 识别，并删除无效结果
    print('recognizing...')
    yield "OCR识别...", None
    
    im = cv2.imread(data, 0)
    char_ims = []
    for line in detect_result:
        x1, y1, x2, y2 = [round(float(k)) for k in line]
        char_ims.append(im[y1:y2, x1:x2])
    output_chars, output_probs = batch_char_recog(reg_model, device, char_dict, char_ims, bs=opt.reg_batch_size)

    chars = {}
    to_remove_detect_result = set()
    for idx, (char, prob) in enumerate(zip(output_chars, output_probs)):
        chars[idx] = char[0]
        # 如果prob[0]小于0.2，则认为这里没有字符，删除detect_result中对应的结果
        if prob[0] < 0.7:
            # 找到对应的detect_result中的结果
            to_remove_detect_result.add(idx)
    removed_detect_result = []
    OCR_result = {}
    for idx, detect_box in enumerate(detect_result):
        if idx not in to_remove_detect_result:
            removed_detect_result.append(detect_box)
            OCR_result[str(detect_box)] = chars[idx]
    torch.cuda.empty_cache()

    # 上面删除了置信度低的识别结果和对应的检测框，现在要合并字符框和破损框
    to_remove = set()
    iou_threshold = 0.5
    for degraded_box in degraded_detect_result:
        for idx, detect_box in enumerate(removed_detect_result):
            iou = calculate_iou(degraded_box, detect_box)
            if iou >= iou_threshold:
                # 如果IoU很大，则标记要删除的removed_detect_result中的结果
                to_remove.add(idx)
    # 创建一个新的结果列表来拼接
    final_results = []
    # 添加degraded_removed_detect_result中的结果
    final_results.extend(degraded_detect_result)

    # 添加未被删除的detect_result中的结果
    for idx, detect_box in enumerate(removed_detect_result):
        if idx not in to_remove:
            final_results.append(detect_box)

    print('arrange...')
    yield "处理阅读顺序...", None
    out_box_li = get_sort(final_results, h, w)

    chars_list = []
    num_ocr = 0
    num_degraded = 0
    degraded_dict = {}
    for box in out_box_li:
        if str(box) in OCR_result.keys():
            num_ocr += 1
            chars_list.append(OCR_result[str(box)])
        else:
            chars_list.append(f'<|extra_{num_degraded}|>')
            degraded_dict[f'<|extra_{num_degraded}|>'] = [box]
            num_degraded += 1
            

    # extra_nums = list(range(num
    # _degraded))
    # random.shuffle(extra_nums)
    # for i, char in enumerate(chars_list):
    #     if char == '<mask>':
    #         extra_num = extra_nums.pop()
    #         chars_list[i] = f'<|extra_{extra_num}|>'

    char_str = ''.join(chars_list)
    print(f'识别字符：【{num_ocr}】个，识别破损位置：【{num_degraded}】个')
    infomation = f'识别字符：【{num_ocr}】个，识别破损位置：【{num_degraded}】个'

    yield f'识别字符：【{num_ocr}】个，识别破损位置：【{num_degraded}】个', None

    yield 'OCR识别结果...', None
    yield char_str, None
    
    print('predicting...')
    yield "预测缺失文本...", None
    
    model_name_or_path = "llm_model/qwen2_1.5_1021_num_mask-3000"
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True,)
    
    tokenizer.bos_token = '<|im_start|>'
    tokenizer.eos_token = '<|im_end|>'
    tokenizer.pad_token_id = tokenizer.eos_token_id

    char_str_simplified = traditional_to_simplified(char_str)

    pattern = r'(<\|extra_\d+\|>)(.)'

    messages = [
        {
            "role": "system",
            "content": "请帮助恢复古籍中缺失的字。",
        },
        {"role": "user", "content": char_str_simplified},
    ]

    messages = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    model_inputs = tokenizer([messages], return_tensors="pt").to(device)
    # import pdb; pdb.set_trace()
    pred = model.generate(**model_inputs, 
                          max_new_tokens=4096, 
                          repetition_penalty=1.0,
                          pad_token_id=tokenizer.eos_token_id,
                          eos_token_id=tokenizer.eos_token_id,)
    
    pred_text_ini = tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)

    pred_text = pred_text_ini.split('assistant\n')[1]
    pred_text = pred_text.split('<|im_end|>')[0]
    output_traditional = simplify_to_traditional(pred_text)
    output_matches = re.findall(pattern, output_traditional)
    # output_matches_sort = sorted(output_matches, key=lambda x: int(re.search(r'\d+', x[0]).group()))
    yield '缺失内容预测结果...', None

    yield str(output_matches), None

    torch.cuda.empty_cache()

    yield "根据破损字符位置对图像切块...", None

    model_output_reserved_name_list = []
    for i, match in enumerate(output_matches):
        reserved_name = match[0]
        char = match[1]
        model_output_reserved_name_list.append(reserved_name)

    for key in list(degraded_dict.keys()):
        if key not in model_output_reserved_name_list:
            del degraded_dict[key]

    for match in output_matches:
        
        reserved_name = match[0]
        char = match[1]
        if reserved_name in degraded_dict.keys():
            degraded_dict[reserved_name].append(char)
        
    IMAGE_WIDTH = w
    IMAGE_HEIGHT = h
    PATCH_BASE_SIZE = 512
    MARGIN = 1  # 单字框与图像块边缘的最小距离
    SIZE_DELTA = 50        # 大小浮动范围
    MIN_PATCH_SIZE = PATCH_BASE_SIZE - SIZE_DELTA  # 462
    MAX_PATCH_SIZE = PATCH_BASE_SIZE + SIZE_DELTA  # 562

    # 将破损单字框转换为列表
    broken_boxes = []
    for key, value in degraded_dict.items():
        bbox = value[0]
        char = value[1]
        box = {
            'id': key,
            'bbox': bbox,
            'char': char,
            'assigned': False,  # 标记此破损单字框是否已被分配到图像块
            'err' : False
        }
        broken_boxes.append(box)

    # 按x坐标排序，方便处理
    broken_boxes.sort(key=lambda x: x['bbox'][0])  # 按xmin排序

    patches = []  # 存储生成的图像块信息

    def does_patch_cross_broken_boxes(patch_bbox, broken_boxes):
        """
        检查图像块的边缘是否穿过任何破损单字框
        """
        px_min, py_min, px_max, py_max = patch_bbox
        for box in broken_boxes:
            bx_min, by_min, bx_max, by_max = box['bbox']
            # 如果图像块的边缘位于单字框的内部，则边缘穿过了单字框
            # 检查左边缘
            if px_min > bx_min and px_min < bx_max and \
            max(py_min, by_min) < min(py_max, by_max):
                return True
            # 检查右边缘
            if px_max > bx_min and px_max < bx_max and \
            max(py_min, by_min) < min(py_max, by_max):
                return True
            # 检查上边缘
            if py_min > by_min and py_min < by_max and \
            max(px_min, bx_min) < min(px_max, bx_max):
                return True
            # 检查下边缘
            if py_max > by_min and py_max < by_max and \
            max(px_min, bx_min) < min(px_max, bx_max):
                return True
        return False

    def generate_patch_for_box(current_box):
        """
        为当前单字框生成一个符合要求的图像块
        """
        # 获取当前单字框的bbox
        bbox = current_box['bbox']
        bx_min, by_min, bx_max, by_max = bbox

        # 获取包含单字框的最小矩形位置
        min_x = bx_min - MARGIN
        min_y = by_min - MARGIN
        max_x = bx_max + MARGIN
        max_y = by_max + MARGIN

        # 计算所需的最小尺寸
        min_width = max_x - min_x
        min_height = max_y - min_y
        required_size = max(min_width, min_height, MIN_PATCH_SIZE)

        # 确保尺寸不超过最大尺寸
        if required_size > MAX_PATCH_SIZE:
            required_size = MAX_PATCH_SIZE

        # 尝试不同的左上角位置，使得单字框包含在图像块内
        possible_px_min = range(max(0, min_x - (required_size - min_width)), min(min_x, IMAGE_WIDTH - required_size) + 1)
        possible_py_min = range(max(0, min_y - (required_size - min_height)), min(min_y, IMAGE_HEIGHT - required_size) + 1)

        for px_min in possible_px_min:
            for py_min in possible_py_min:
                px_max = px_min + required_size
                py_max = py_min + required_size

                if px_max > IMAGE_WIDTH or py_max > IMAGE_HEIGHT:
                    continue  # 超出图像范围

                patch_bbox = [px_min, py_min, px_max, py_max]

                # 检查图像块的边缘是否穿过任何单字框
                other_boxes = [box for box in broken_boxes if box != current_box]
                crosses = does_patch_cross_broken_boxes(patch_bbox, other_boxes)
                if crosses:
                    continue  # 边缘穿过了其他单字框，尝试下一个位置

                # 检查当前单字框是否完全包含在图像块中
                if (bx_min - MARGIN >= px_min) and (bx_max + MARGIN <= px_max) and \
                (by_min - MARGIN >= py_min) and (by_max + MARGIN <= py_max):
                    # 找到符合要求的图像块
                    return patch_bbox

        # 如果仍未找到，尝试扩大尺寸再试
        for patch_size in range(int(required_size) + 1, MAX_PATCH_SIZE + 1):
            possible_px_min = range(max(0, min_x - (patch_size - min_width)), min(min_x, IMAGE_WIDTH - patch_size) + 1)
            possible_py_min = range(max(0, min_y - (patch_size - min_height)), min(min_y, IMAGE_HEIGHT - patch_size) + 1)

            for px_min in possible_px_min:
                for py_min in possible_py_min:
                    px_max = px_min + patch_size
                    py_max = py_min + patch_size

                    if px_max > IMAGE_WIDTH or py_max > IMAGE_HEIGHT:
                        continue  # 超出图像范围

                    patch_bbox = [px_min, py_min, px_max, py_max]

                    # 检查图像块的边缘是否穿过任何单字框
                    other_boxes = [box for box in broken_boxes if box != current_box]
                    crosses = does_patch_cross_broken_boxes(patch_bbox, other_boxes)
                    if crosses:
                        continue  # 边缘穿过了其他单字框，尝试下一个位置

                    # 检查当前单字框是否完全包含在图像块中
                    if (bx_min - MARGIN >= px_min) and (bx_max + MARGIN <= px_max) and \
                    (by_min - MARGIN >= py_min) and (by_max + MARGIN <= py_max):
                        # 找到符合要求的图像块
                        return patch_bbox

        # 无法找到符合要求的图像块
        return None

    # 主循环，处理所有未分配的破损单字框
    while True:
        # 找到未分配的破损单字框
        unassigned_boxes = [box for box in broken_boxes if not box['assigned']]
        if not unassigned_boxes:
            break  # 所有破损单字框都已分配

        # 选择x最小的未分配破损单字框作为起点
        current_box = unassigned_boxes[0]

        # 尝试为当前单字框生成图像块
        patch_bbox = generate_patch_for_box(current_box)
        if patch_bbox is None:
            print(f"无法为单字框 '{current_box['char']}' 生成符合要求的图像块。")
            current_box['assigned'] = True  # 标记为已处理，避免死循环
            current_box['err'] = True  # 标记为错误
            continue

        # 创建图像块
        patch = {
            'xmin': int(patch_bbox[0]),
            'ymin': int(patch_bbox[1]),
            'xmax': int(patch_bbox[2]),
            'ymax': int(patch_bbox[3]),
            'size': int(patch_bbox[2] - patch_bbox[0]),
            'boxes': [current_box],
        }
        current_box['assigned'] = True

        # 尝试将更多的未分配破损单字框加入当前图像块
        added_boxes = True
        while added_boxes:
            added_boxes = False
            for box in unassigned_boxes:
                if box['assigned']:
                    continue
                # 检查单字框是否完全包含在图像块中，并与边缘保持最小距离
                bx_min, by_min, bx_max, by_max = box['bbox']
                if (bx_min - MARGIN >= patch['xmin']) and (bx_max + MARGIN <= patch['xmax']) and \
                (by_min - MARGIN >= patch['ymin']) and (by_max + MARGIN <= patch['ymax']):
                    # 检查图像块的边缘是否穿过任何破损单字框
                    temp_boxes = patch['boxes'] + [box]
                    crosses = does_patch_cross_broken_boxes([patch['xmin'], patch['ymin'], patch['xmax'], patch['ymax']], temp_boxes)
                    if not crosses:
                        # 可以加入当前图像块
                        patch['boxes'].append(box)
                        box['assigned'] = True
                        added_boxes = True
                        break  # 重新开始检查
            # 如果无法再加入更多单字框，退出循环

        patches.append(patch)

    new_patches = []
    for idx, patch in enumerate(patches):
        px_min = patch['xmin']
        py_min = patch['ymin']
        px_max = patch['xmax']
        py_max = patch['ymax']
        patch_size = patch['size']
        new_boxes = []

        for sptk, boxes in degraded_dict.items():
            bbox = boxes[0]
            char = boxes[1]
            bx_min, by_min, bx_max, by_max = bbox
            if (bx_min >= px_min) and (bx_max <= px_max) and (by_min >= py_min) and (by_max <= py_max):
                # 破损单字框在当前图像块中
                new_boxes.append({'id':sptk, 'bbox': bbox, 'char': char})
        new_patches.append({'patch_id': idx, 'xmin': px_min, 
                            'ymin': py_min, 
                            'xmax': px_max, 'ymax': py_max, 'size': patch_size, 'boxes': new_boxes})
    


    check_patch_repeat_dict = {}
    for patch in new_patches:
        char_list = []
        patche_id = patch['patch_id']
        boxes = patch['boxes']
        # import pdb; pdb.set_trace()
        for box in boxes:
            char_list.append(str(box['id']))
        check_patch_repeat_dict[patche_id] = set(char_list)

    # 用于存储被完全包含的 patch_id
    to_remove = set()

    # 使用一个字典记录每个 patch_id 是否已经处理过完全相等的情况
    handled_equals = set()

    # 双重循环检查完全包含关系
    for patch_id_a, char_set_a in check_patch_repeat_dict.items():
        for patch_id_b, char_set_b in check_patch_repeat_dict.items():
            if patch_id_a != patch_id_b:  # 避免与自己比较
                if char_set_a == char_set_b:
                    # 检查是否已处理完全相等的情况
                    if patch_id_a not in handled_equals and patch_id_b not in handled_equals:
                        # 标记这两个 patch_id 为已处理
                        handled_equals.add(patch_id_a)
                        handled_equals.add(patch_id_b)
                        # 选择一个删除（例如总是删除字典顺序后的一个）
                        to_remove.add(max(patch_id_a, patch_id_b))
                elif char_set_b.issubset(char_set_a):
                    # 如果 char_set_b 是 char_set_a 的子集，记录 char_set_b 的 patch_id
                    to_remove.add(patch_id_b)

    # 移除被完全包含的 patch
    new_patches_remove_repeat = [patch for patch in new_patches if patch['patch_id'] not in to_remove]


    print(f'需要修复 {len(new_patches_remove_repeat)} 个patch。')
    need_repair = f'需要修复 {len(new_patches_remove_repeat)} 个patch。'
    yield f'需要修复 {len(new_patches_remove_repeat)} 个patch', None


    # 创建一个空白的图像
    # image = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), color='white')
    image = Image.open(data)
    draw = ImageDraw.Draw(image)

    # 尝试加载中文字体，用于绘制字符
    try:
        font = ImageFont.truetype("demo_utils/font/KaiXinSongA.ttf", 20)  # 这里使用系统字体"宋体"
    except IOError:
        font = ImageFont.load_default()

    # 绘制破损单字框的边界框和字符
    for box in broken_boxes:
        bbox = box['bbox']
        # 绘制边界框
        draw.rectangle(bbox, outline='red', width=2)
        # 在边界框的左上角绘制字符
        char = box['char']
        draw.text((bbox[0], bbox[1] - 20), char, fill='red', font=font)

    # 不同的图像块使用不同的颜色
    colors = ['blue', 'green', 'purple', 'orange', 'brown', 'cyan', 'magenta', 'yellow', 'pink', 'gray']

    # 绘制图像块及其包含的单字框
    for idx, patch in enumerate(new_patches_remove_repeat):
        color = colors[idx % len(colors)]
        # 绘制图像块的边界框
        draw.rectangle([patch['xmin'], patch['ymin'], patch['xmax'], patch['ymax']], outline=color, width=2)
        # 标注图像块编号和大小
        label = f"Patch {idx+1} ({patch['size']}x{patch['size']})"
        draw.text((patch['xmin'], patch['ymin']), label, fill=color, font=font)
        # 标注每个单字框的所属图像块编号
        for box in patch['boxes']:
            bbox = box['bbox']
            draw.text((bbox[2], bbox[3]), f"P{idx+1}", fill=color, font=font)

    # 保存图像
    patched_image = image
    yield "切patch完成...", patched_image
    image.save('patched_image.png')


    ############### 加载修复模型 ###################
    unet = UNet2DModel.from_pretrained(pretrained_model_name_or_path='model_150/unet')
    unet = unet.to(device)
    pipeline = build_pipeline(
        args=opt,
        unet=unet,)
    generator = torch.Generator(device=pipeline.device).manual_seed(opt.seed)


    image_to_repair = Image.open(data).convert('RGB')

    repair_image_list = []
    for patch in new_patches_remove_repeat:
        # 图像块的左上角和右下角坐标
        xmin, ymin, xmax, ymax = patch['xmin'], patch['ymin'], patch['xmax'], patch['ymax']
        # 图像块的大小
        patch_size = patch['size']

        # 根据patch_bbox从image_to_repair中截取patch
        degraded_image = image_to_repair.crop((xmin, ymin, xmax, ymax))

        # 创建mask和content img
        content_image = Image.new('L', (patch_size, patch_size), 255)
        mask_image = Image.new('L', (patch_size, patch_size), 0)

        repair_image_dict = {}
        for box in patch['boxes']:
            # 单字框的左上角和右下角坐标
            bx_min, by_min, bx_max, by_max = box['bbox']
            # 计算单字框在图像块中的相对坐标
            rel_x_min = bx_min - xmin
            rel_y_min = by_min - ymin
            rel_x_max = bx_max - xmin
            rel_y_max = by_max - ymin
            # 绘制单字框的mask
            mask = Image.new('L', (rel_x_max - rel_x_min, rel_y_max - rel_y_min), 255)
            mask_image.paste(mask, (rel_x_min, rel_y_min, rel_x_max, rel_y_max))
            degraded_image.paste(mask, (rel_x_min, rel_y_min, rel_x_max, rel_y_max))
            # 渲染单字图片
            single_char_img = render_char_with_font_T(box['char'])
            # 将单字图片resize成box的大小
            single_char_img = single_char_img.resize((rel_x_max - rel_x_min, rel_y_max - rel_y_min))
            # 将单字图片粘贴到content图片中
            content_image.paste(single_char_img, (rel_x_min, rel_y_min))
        repair_image_dict['content_image'] = content_image
        repair_image_dict['mask_image'] = mask_image
        repair_image_dict['degraded_image'] = degraded_image
        repair_image_dict['patch_bbox'] = [patch['xmin'], patch['ymin'], patch['xmax'], patch['ymax']]
        repair_image_dict['patch_size'] = patch_size
        repair_image_list.append(repair_image_dict)
    
    

    # 开始修复，一个个patch来修复
    for i, repair_image_dict in enumerate(repair_image_list):
        yield f"正在修复第{i+1}个patch...", None
        content_image = repair_image_dict['content_image']
        mask_image = repair_image_dict['mask_image']
        degraded_image = repair_image_dict['degraded_image']
        patch_bbox = repair_image_dict['patch_bbox']
        patch_size = repair_image_dict['patch_size']
        content_image.save(f'./patches_img/content_image_{i}.png')
        degraded_image.save(f'./patches_img/degraded_image_{i}.png')

        
        # 调整patch_image的大小为512x512        
        degraded_image = degraded_image.resize((512, 512))
        content_image = content_image.resize((512, 512))
        mask_image = mask_image.resize((512, 512))

        # 转换成tensor
        degraded_image_tensor = F.normalize(F.to_tensor(degraded_image), [0.5], [0.5]).unsqueeze(0)
        mask_image_tensor = F.normalize(F.to_tensor(mask_image), [0.5], [0.5]).unsqueeze(0)
        content_image_tensor = F.normalize(F.to_tensor(content_image), [0.5], [0.5]).unsqueeze(0)
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
        
        image = image.resize((patch_size, patch_size))
        image_to_repair.paste(image, patch_bbox)

    image_to_repair.save('patched_image_repaired.png')
    yield "修复结果...", image_to_repair
    torch.cuda.empty_cache()


    image_to_repair_mark = image_to_repair.copy()
    draw = ImageDraw.Draw(image_to_repair_mark)

    # 尝试加载中文字体，用于绘制字符
    try:
        font = ImageFont.truetype("demo_utils/font/KaiXinSongA.ttf", 20)  # 这里使用系统字体"宋体"
    except IOError:
        font = ImageFont.load_default()

    # 绘制破损单字框的边界框和字符
    for box in broken_boxes:
        bbox = box['bbox']
        # 绘制边界框
        draw.rectangle(bbox, outline='red', width=2)
        # 在边界框的左上角绘制字符
        char = box['char']
        draw.text((bbox[0], bbox[1] - 20), char, fill='red', font=font)

    # 不同的图像块使用不同的颜色
    colors = ['blue', 'green', 'purple', 'orange', 'brown', 'cyan', 'magenta', 'yellow', 'pink', 'gray']

    # 绘制图像块及其包含的单字框
    for idx, patch in enumerate(new_patches_remove_repeat):
        color = colors[idx % len(colors)]
        # 绘制图像块的边界框
        draw.rectangle([patch['xmin'], patch['ymin'], patch['xmax'], patch['ymax']], outline=color, width=2)
        # 标注图像块编号和大小
        label = f"Patch {idx+1} ({patch['size']}x{patch['size']})"
        draw.text((patch['xmin'], patch['ymin']), label, fill=color, font=font)
        # 标注每个单字框的所属图像块编号
        for box in patch['boxes']:
            bbox = box['bbox']
            draw.text((bbox[2], bbox[3]), f"P{idx+1}", fill=color, font=font)
    torch.cuda.empty_cache()
    image_to_repair_mark.save('patched_image_repaired_mark.png')
    yield "修复结果标记...", image_to_repair_mark

    yield "修复完成！！！", None

    return patched_image, need_repair, char_str, infomation, output_traditional, image_to_repair, image_to_repair_mark

if __name__ == '__main__':

    config_file = 'demo_utils/ckpt/damage_detect.py' # 网络模型py文件
    damage_detect_checkpoint_file = 'demo_utils/ckpt/damage_detect.pth'  # 训练好的模型参数
    data_path = 'tmp_img/M5_image_0_degraded_crop.png'
    # data_path = 'tmp_img/29.明代刻經_343.png'
    data_path = 'tmp_img/M5_image_0_degraded.png'
    data_path = 'tmp_img/MTH_0001_016_26_16_degraded.png'
    # data_path = 'tmp_img/MTH_0001_016_26_19_degraded.png'
    # data_path = 'temp_img/fssj_bin.png'

    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
    parser.add_argument('--degraded-weights', nargs='+', type=str, default='degraded_best.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default=data_path, help='*.data path')
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
    # import pdb;pdb.set_trace()
    res_dic = main(
        data=data_path, opt=opt,
    )
    txt_path = data_path.replace('.png','.txt')
    # f = open(txt_path,'w', encoding='utf-8')
    # for k,v in res_dic.items():
    #     for cood in v:
    #         f.write(f'{int(cood[0])},{int(cood[1])},{int(cood[2])},{int(cood[3])}\n')
    #     #     print(v)
    #         # import pdb;pdb.set_trace()
    #     # print(k,v)

    # # print(res_dic)
    # f.close()