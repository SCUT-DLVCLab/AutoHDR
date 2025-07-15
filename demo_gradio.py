import gradio as gr
import infer_pipeline_api as pipline_qwen_multisptk_api
import argparse


def main_wrapper(image):
    global opt
    status = ""  
    images = []  

    for status_update, new_image in pipline_qwen_multisptk_api.main(image, opt):
        status += status_update + "\n\n"           
        if not images == [] or new_image is not None:  
            if new_image is not None:  
                    images.append(new_image)  
            yield status, images 
        else:
            yield status, None
        yield status, images 

parser = argparse.ArgumentParser(prog='test.py')
parser.add_argument('--ocr_det_weights', nargs='+', type=str, default='./weights/epoch_000.pt', help='model.pt path(s)')
parser.add_argument('--vague_det_weights', nargs='+', type=str, default='./ckpt/damage_detect.pth', help='model.pth path(s)')
parser.add_argument('--vague_det_config', nargs='+', type=str, default='./ckpt/damage_detect.py', help='mmdetection config.py path(s)')
parser.add_argument('--model_name_or_path', nargs='+', type=str, default='./ckpt/AutoHDR-Qwen2-1.5B', help='llm weight path(s)')

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



css = """
body { display: flex; flex-direction: column; align-items: center; }
.component { margin: 10px 0; }
"""

    
with gr.Blocks() as demo:

    top_image = gr.Image(value="images/logo.png", label="顶部图片", show_download_button=False, show_label=False, container=False,)
    image_input = gr.Image(type="pil", label="输入图像")

    gr.Examples(
        examples=[
            ["images/FS_12_138_2.jpg"],
            ["images/FS_12_159_2.jpg"],
            ["images/FS_15_481_1.jpg"],
            ["images/FS_20_361_2.jpg"],
            ["images/FS_25_162_1.jpg"],
        ],
        inputs=image_input,
        label="example"
    )

    submit_button = gr.Button("提交")
    status_output = gr.Textbox(label="处理状态")
    image_output = gr.Gallery(label="图像输出")

    submit_button.click(
        fn=main_wrapper,  
        inputs=[image_input], 
        outputs=[status_output, image_output], 
    )


# 启动 Gradio 界面
demo.launch(server_name='0.0.0.0', server_port=7860, share=True)