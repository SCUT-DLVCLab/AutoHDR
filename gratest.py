import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

class BoxAnnotator:
    def __init__(self):
        self.boxes = []  # 存储所有框的信息 [(x1,y1,x2,y2,color),...]
        self.start_pos = None
        
    def start_drawing(self, image, evt: gr.EventData):
        self.start_pos = [evt.index[0], evt.index[1]]
        return image
        
    def end_drawing(self, image, evt: gr.EventData, color):
        if self.start_pos is None or image is None:
            return image
            
        end_pos = [evt.index[0], evt.index[1]]
        
        # 添加新框到列表中
        self.boxes.append((
            self.start_pos[0], 
            self.start_pos[1], 
            end_pos[0], 
            end_pos[1], 
            color
        ))
        
        self.start_pos = None
        return self.draw_all_boxes(image)
    
    def draw_all_boxes(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        else:
            image = Image.open(image)
        
        draw = ImageDraw.Draw(image)
        
        for box in self.boxes:
            x1, y1, x2, y2, color = box
            # 设置框的颜色
            rgb_color = (0, 255, 0) if color == "green" else (255, 0, 0)
            # 绘制框
            draw.rectangle([x1, y1, x2, y2], outline=rgb_color, width=2)
            
        return np.array(image)
    
    def undo(self, image):
        if self.boxes:
            self.boxes.pop()  # 移除最后一个框
        return self.draw_all_boxes(image) if image is not None else None

annotator = BoxAnnotator()

with gr.Blocks() as demo:
    gr.Markdown("## 图像标注工具")
    gr.Markdown("使用说明：\n1. 上传图片\n2. 选择框的颜色（绿色或红色）\n3. 在图片上按住鼠标左键并拖动来画框\n4. 使用撤销按钮可以删除最后画的框")
    
    with gr.Row():
        # 输入图片
        image_input = gr.Image(label="上传图片进行标注", type="numpy")
        # 标注结果
        image_output = gr.Image(label="标注结果", interactive=True)
    
    color_choice = gr.Radio(
        choices=["green", "red"],
        value="green",
        label="选择框的颜色"
    )
    
    undo_btn = gr.Button("撤销上一个框")
    
    # 设置事件处理
    image_input.change(
        fn=lambda x: x,
        inputs=[image_input],
        outputs=[image_output]
    )
    
    # 使用mousedown和mouseup事件进行框的绘制
    image_output.mouse_down(
        fn=annotator.start_drawing,
        inputs=[image_input],
        outputs=[image_output]
    )
    
    image_output.mouse_up(
        fn=annotator.end_drawing,
        inputs=[image_input, color_choice],
        outputs=[image_output]
    )
    
    # 撤销按钮事件
    undo_btn.click(
        fn=annotator.undo,
        inputs=[image_input],
        outputs=[image_output],
    )

demo.launch()