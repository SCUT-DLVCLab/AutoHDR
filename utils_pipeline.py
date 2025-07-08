
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from transformers import AutoModelForCausalLM, AutoTokenizer
from zhconv import convert
import torch
from typing import List, Tuple
import math

font_list = [
    'demo_utils/font/KaiXinSongA.ttf',
    'demo_utils/font/KaiXinSongB.ttf'
    #'font/TH-Tshyn-P0.ttf',
    #'font/TH-Tshyn-P1.ttf',
    #'font/TH-Tshyn-P2.ttf',
    #'font/TH-Tshyn-P16.ttf',
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
        size = 128
        padding = 0
        pad_size = size + padding
        image = Image.new('L', (size, size), background_color)
        draw = ImageDraw.Draw(image)
        
        # 获取文字边界框
        bbox = ImageFont_font.getbbox(char)
        text_width = bbox[2] - bbox[0]
        
        # 获取字体度量
        metrics = ImageFont_font.getmetrics()
        ascent = metrics[0]
        descent = metrics[1]
        text_height = ascent 
        
        # 计算居中位置（水平和垂直）
        x = (size - text_width) // 2
        # 使用ascent和descent计算垂直偏移
        y = (size - text_height) // 2 - descent // 2
        
        # 绘制居中文字
        draw.text((x, y), char, text_color, font=ImageFont_font)

        # 创建带padding的空白图像
        padded_image = Image.new('L', (pad_size, pad_size), background_color)
        
        # 将原图粘贴到padded_image中心
        paste_x = int(padding/2)
        paste_y = int(padding/2)
        padded_image.paste(image, (paste_x, paste_y))
        padded_image = padded_image.resize((size, size), Image.Resampling.LANCZOS)
        return padded_image
    except:
        import pdb;pdb.set_trace()


def xyxy2xywh(box):
    """
    将 [x1, y1, x2, y2] 格式转换为 [x, y, w, h] 格式
    x, y 是左上角坐标
    w, h 是宽度和高度
    """
    x1, y1, x2, y2 = box
    w = x2 - x1  # 宽度 = 右边界 - 左边界
    h = y2 - y1  # 高度 = 下边界 - 上边界
    return [x1, y1, w, h]


def is_char_in_font(TTFont_font, char):
    cmap = TTFont_font['cmap']
    for subtable in cmap.tables:
        if ord(char) in subtable.cmap:
            return True
    return False

def create_blank_image(size=(100, 100)):
    from PIL import Image
    return Image.new('RGB', size, 'white')

def render_char_with_font_T(char, ImageFont_fonts = ImageFont_fonts, TTFont_font = TTFont_fonts, color=0):
    for i, (ImageFont_font, TTFont_font) in enumerate(zip(ImageFont_fonts, TTFont_fonts)):
        if is_char_in_font(TTFont_font, char):
            have_char_ImageFont = ImageFont_font
            break
        if i == len(ImageFont_fonts) - 1:
            print(f"No font can render the character {char}")
            return create_blank_image(size=(100, 100))
            # import pdb;pdb.set_trace()
    img = render_char_with_font_L(ImageFont_font=have_char_ImageFont, char=char, text_color=color)
    return img

def model_init(model_name_or_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype="auto",
        device_map="cuda",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,)

    special_token_list = []
    if '1_5B' in model_name_or_path:
        for special_token in range(271):
            special_token_list.append(f"<|extra_{special_token}|>")
    else:
        for special_token in range(371):
            special_token_list.append(f"<|extra_{special_token}|>")

    tokenizer.add_special_tokens({"additional_special_tokens": special_token_list})
    tokenizer.padding_side = 'left'
    tokenizer.bos_token = '<|im_start|>'
    tokenizer.eos_token = '<|im_end|>'
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def get_topk_multi_tokens(scores, tokenizer, start_idx, num_tokens, k=5):
    """
    Get the top-k combinations of tokens at multiple positions

    Args:
        scores: Scores output by the model
        tokenizer: Tokenizer
        start_idx: Starting position index
        num_tokens: Number of tokens to combine
        k: Number of top candidates to consider at each position
    """

    position_results = []
    
    # 获取每个位置的topk结果
    for i in range(num_tokens):
        pos_scores = scores[start_idx + i]
        topk = torch.topk(pos_scores, k=k)
        tokens = topk.indices
        probs = torch.softmax(topk.values, dim=-1)
        position_results.append(list(zip(tokens, probs)))

    # 使用递归函数生成所有组合
    def generate_combinations(current_combo=None, current_prob=1.0, pos=0):
        if current_combo is None:
            current_combo = []
            
        if pos == num_tokens:
            # 解码当前组合
            decoded = tokenizer.decode(current_combo)
            if len(decoded.strip()) == 1:  # 确保是一个有效的字
                return [(decoded, current_prob, current_combo)]
            return []

            
        combinations = []
        for token, prob in position_results[pos]:
            new_combo = current_combo + [token]
            new_prob = current_prob * prob
            combinations.extend(generate_combinations(new_combo, new_prob, pos + 1))

        return combinations
    
    # 生成所有组合
    all_combinations = generate_combinations()
    
    # 按联合概率排序
    all_combinations.sort(key=lambda x: x[1], reverse=True)
    
    return all_combinations[:k]



def rank_probability_weighted_fusion(
    ocr_predictions: List[Tuple[str, float]],
    lm_predictions: List[Tuple[str, float]],
    ocr_max_weight: float = 0.6,
    lm_max_weight: float = 0.4,
    rank_score_per_rank: float = 0.05,
    intersection_bonus: float = 1.5
) -> str:
    """
    结合OCR和语言模型置信度，对结果进行预测

    Args:
        ocr_predictions: OCR模型的top预测结果，每个元素为(字符, 概率)元组
        lm_predictions: 语言模型的top预测结果，每个元素为(字符, 概率)元组
        ocr_max_weight: OCR概率的最大权重 (默认: 0.6) 因为OCR挺准的 给大点
        lm_max_weight: 语言模型概率的最大权重 (默认: 0.4)
        rank_score_per_rank: 每提升一个排名增加的分数 (默认: 0.05)
        intersection_bonus: 同时出现在两个模型中，给出的奖励分数 (默认: 1.5)

    Returns:
        str: 最终预测的字符，排名后的概率值字典
    """
    
    def sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(-x))
    
    def compute_weighted_score(info: dict) -> float:
        # 有时候OCR给出的top2距离很近
        # 把权重拉开一点 不然距离太近了
        ocr_weight = sigmoid(info['ocr_prob'] - 0.5) * ocr_max_weight
        lm_weight = sigmoid(info['lm_prob'] - 0.5) * lm_max_weight
        return (info['ocr_prob'] * ocr_weight + info['lm_prob'] * lm_weight)
    
    def compute_rank_score(info: dict) -> float:
        # 排名越靠前，得分越高
        ocr_rank_score = (5 - info['ocr_rank']) * rank_score_per_rank # 写死了，后续也可以传个topk进来
        lm_rank_score = (5 - info['lm_rank']) * rank_score_per_rank
        return ocr_rank_score + lm_rank_score
    
    # 1. 构建候选字符池
    candidates = {}
    
    # 添加OCR预测结果
    for i, (char, prob) in enumerate(ocr_predictions):
        candidates[char] = {
            'ocr_prob': prob,    # OCR预测概率
            'ocr_rank': i,       # OCR预测排名
            'lm_prob': 0.0,      # 初始化语言模型概率为0
            'lm_rank': 5         # 初始化语言模型排名为最低
        }
    
    # 添加语言模型预测结果
    for i, (char, prob) in enumerate(lm_predictions):
        if char in candidates:  # 如果字符已存在于候选池中
            candidates[char]['lm_prob'] = prob
            candidates[char]['lm_rank'] = i
        else:  # 如果是新字符
            candidates[char] = {
                'ocr_prob': 0.0,  # 初始化OCR概率为0
                'ocr_rank': 5,    # 初始化OCR排名为最低
                'lm_prob': prob,  # 语言模型预测概率
                'lm_rank': i      # 语言模型预测排名
            }
    
    # 2. 计算每个候选字符的最终得分
    final_scores = {}
    for char, info in candidates.items():
        # 计算概率加权得分
        weighted_score = compute_weighted_score(info)
        # 计算排名得分
        rank_score = compute_rank_score(info)
        # 检查是否同时出现在两个模型中，是则给予奖励分数
        bonus = intersection_bonus if info['ocr_prob'] > 0 and info['lm_prob'] > 0 else 1.0
        # 计算最终得分
        final_scores[char] = (weighted_score + rank_score) * bonus
    
    # 3. 第一个是给出得分最高的 第二个是给出所有信息
    best_char = max(final_scores.items(), key=lambda x: x[1])[0]
    sorted_dict = dict(sorted(final_scores.items(), key=lambda x: x[1], reverse=True))
    return best_char, sorted_dict


def visualize_boxes(image_path, red_boxes, blue_boxes, output_path='visualization_result.jpg', thickness=2):
    # 读取图片
    image = cv2.imread(image_path)
    
    # 定义颜色 (BGR格式)
    red_color = (0, 0, 255)    # 红色
    blue_color = (255, 0, 0)   # 蓝色
    
    # 绘制红色框
    for i, box in enumerate(red_boxes):
        x1, y1, x2, y2 = box
        # 绘制矩形框
        cv2.rectangle(image, (x1, y1), (x2, y2), red_color, thickness)
        # 添加框的索引编号
        cv2.putText(image, f'R{i}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, red_color, thickness)
    
    # 绘制蓝色框
    for i, box in enumerate(blue_boxes):
        x1, y1, x2, y2 = box
        # 绘制矩形框
        cv2.rectangle(image, (x1, y1), (x2, y2), blue_color, thickness)
        # 添加框的索引编号
        cv2.putText(image, f'B{i}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, blue_color, thickness)
    
    # 保存结果图片
    cv2.imwrite(output_path, image)
    print(f"Result saved to {output_path}")


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


def invert_image(image):
    """
    反转图像的黑白
    :param image: 输入图像 (可以是PIL.Image或cv2读取的图像)
    :return: 反转后的图像 (返回与输入相同类型的图像)
    """
    # 检查是否为PIL图像
    is_pil = isinstance(image, Image.Image)
    
    if is_pil:
        # 将PIL图像转换为numpy数组
        image_array = np.array(image)
        inverted = 255 - image_array
        # 转回PIL图像
        return Image.fromarray(inverted)
    else:
        # 直接处理numpy数组(CV2图像)
        return 255 - image

def restore_image(inverted_image):
    """
    将反转的图像恢复
    :param inverted_image: 反转后的图像
    :return: 恢复后的图像
    """
    # 检查是否为PIL图像
    is_pil = isinstance(inverted_image, Image.Image)
    
    if is_pil:
        image_array = np.array(inverted_image)
        restored = 255 - image_array
        return Image.fromarray(restored)
    else:
        return 255 - inverted_image

def concatenate_images_vertical(image1, image2):
    # 确保两张图片宽度一致
    width = max(image1.size[0], image2.size[0])
    # 如果宽度不一致，将较窄的图片调整至相同宽度
    if image1.size[0] != width:
        image1 = image1.resize((width, int(image1.size[1] * width / image1.size[0])), Image.Resampling.LANCZOS)
    if image2.size[0] != width:
        image2 = image2.resize((width, int(image2.size[1] * width / image2.size[0])), Image.Resampling.LANCZOS)
    
    # 创建新图片
    combined_height = image1.size[1] + image2.size[1]
    combined_image = Image.new('RGB', (width, combined_height))
    
    # 粘贴图片
    combined_image.paste(image1, (0, 0))
    combined_image.paste(image2, (0, image1.size[1]))
    
    return combined_image