import gradio as gr
import cv2
import numpy as np

class BoxState:
    def __init__(self):
        self.boxes = []  # 存储框的列表 [(x1,y1,x2,y2), ...]
        self.drawing = False
        self.start_point = None

box_state = BoxState()

def handle_image(image, evt: gr.SelectData):
    """处理图片点击事件"""
    print("图片被点击！")  # 调试输出
    print(f"点击坐标: {evt.index}")  # 调试输出
    
    if image is None:
        print("没有图片")  # 调试输出
        return image
    
    # 确保图像是numpy数组
    if isinstance(image, dict):
        image = image['image']
    
    x, y = evt.index[0], evt.index[1]
    print(f"处理坐标: ({x}, {y})")  # 调试输出
    
    img = image.copy()
    
    if not box_state.drawing:
        # 第一次点击，记录起始点
        box_state.drawing = True
        box_state.start_point = (int(x), int(y))
        # 画一个点标记起始位置
        cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
        print("标记起始点")  # 调试输出
    else:
        # 第二次点击，画框
        x1, y1 = box_state.start_point
        x2, y2 = int(x), int(y)
        # 确保坐标顺序正确
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        # 画框
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # 保存框
        box_state.boxes.append((x1, y1, x2, y2))
        # 重置状态
        box_state.drawing = False
        box_state.start_point = None
        print("画框完成")  # 调试输出
    
    # 画出所有已有的框
    for box in box_state.boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    return img

def clear_boxes(image):
    """清除所有框"""
    print("清除所有框")  # 调试输出
    box_state.boxes.clear()
    box_state.drawing = False
    box_state.start_point = None
    return image

# 创建Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("# 目标检测框标注工具")
    gr.Markdown("""
    使用说明：
    1. 上传图片
    2. 点击两个点来创建框：
       - 第一次点击确定框的一角
       - 第二次点击确定对角
    3. 点击'清除'按钮删除所有框
    """)
    
    # 图片输入
    image_input = gr.Image(label="点击图片来标注框", type="numpy")
    
    # 清除按钮
    clear_btn = gr.Button("清除")
    
    # 事件绑定
    print("绑定事件...")  # 调试输出
    image_input.select(
        fn=handle_image,
        inputs=image_input,
        outputs=image_input
    )
    
    clear_btn.click(
        fn=clear_boxes,
        inputs=image_input,
        outputs=image_input
    )

print("启动应用...")  # 调试输出
demo.launch()