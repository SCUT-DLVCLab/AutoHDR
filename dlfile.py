import gradio as gr
import os
import zipfile
import tempfile
from pathlib import Path
import urllib.parse

# 设置根目录（可以修改为你需要的目录）
ROOT_DIR = os.getcwd()

def get_file_size(path):
    """获取文件大小的人类可读格式"""
    try:
        size = os.path.getsize(path)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
    except:
        return "未知"

def is_safe_path(path):
    """检查路径是否安全"""
    try:
        real_path = os.path.realpath(path)
        real_root = os.path.realpath(ROOT_DIR)
        return real_path.startswith(real_root)
    except:
        return False

def browse_directory(current_path=""):
    """浏览目录并返回文件列表"""
    if not current_path:
        current_path = ROOT_DIR
    
    # 安全检查
    if not is_safe_path(current_path):
        current_path = ROOT_DIR
    
    if not os.path.exists(current_path):
        current_path = ROOT_DIR
    
    try:
        items = []
        
        # 添加返回上级目录选项
        if current_path != ROOT_DIR:
            parent_dir = os.path.dirname(current_path)
            if is_safe_path(parent_dir):
                items.append("📁 .. (返回上级目录)")
        
        # 列出目录内容
        for name in sorted(os.listdir(current_path)):
            item_path = os.path.join(current_path, name)
            
            if os.path.isdir(item_path):
                items.append(f"📁 {name}")
            else:
                size = get_file_size(item_path)
                items.append(f"📄 {name} ({size})")
        
        path_info = f"当前路径: {current_path}"
        return path_info, items
        
    except PermissionError:
        return "权限不足，无法访问此目录", []
    except Exception as e:
        return f"错误: {str(e)}", []

def navigate_to_item(current_path, selected_item):
    """导航到选中的项目"""
    if not selected_item:
        path_info, items = browse_directory(current_path)
        return current_path, path_info, items
    
    if selected_item == "📁 .. (返回上级目录)":
        parent_dir = os.path.dirname(current_path)
        if is_safe_path(parent_dir):
            path_info, items = browse_directory(parent_dir)
            return parent_dir, path_info, items
    elif selected_item.startswith("📁 "):
        folder_name = selected_item[2:]  # 移除"📁 "前缀
        new_path = os.path.join(current_path, folder_name)
        if is_safe_path(new_path) and os.path.isdir(new_path):
            path_info, items = browse_directory(new_path)
            return new_path, path_info, items
    
    # 如果是文件或者无法导航，保持当前路径
    path_info, items = browse_directory(current_path)
    return current_path, path_info, items

def download_item(current_path, selected_item):
    """下载选中的项目"""
    if not selected_item:
        return None, "请先选择要下载的项目"
    
    if selected_item == "📁 .. (返回上级目录)":
        return None, "无法下载上级目录链接"
    
    try:
        if selected_item.startswith("📁 "):
            # 下载文件夹（压缩为zip）
            folder_name = selected_item[2:]
            folder_path = os.path.join(current_path, folder_name)
            
            if not is_safe_path(folder_path) or not os.path.exists(folder_path):
                return None, "文件夹不存在或无法访问"
            
            # 创建临时zip文件
            temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=f'_{folder_name}.zip')
            temp_zip.close()
            
            try:
                with zipfile.ZipFile(temp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(folder_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, folder_path)
                            zipf.write(file_path, arcname)
                
                return temp_zip.name, f"文件夹 '{folder_name}' 已压缩为ZIP文件，可以下载"
            except Exception as e:
                os.unlink(temp_zip.name)
                return None, f"压缩失败: {str(e)}"
                
        elif selected_item.startswith("📄 "):
            # 下载单个文件
            filename = selected_item[2:].split(" (")[0]  # 移除"📄 "和文件大小
            file_path = os.path.join(current_path, filename)
            
            if not is_safe_path(file_path) or not os.path.exists(file_path):
                return None, "文件不存在或无法访问"
            
            return file_path, f"文件 '{filename}' 准备下载"
        
        return None, "无效的选择"
        
    except Exception as e:
        return None, f"下载出错: {str(e)}"

# 创建Gradio界面
def create_interface():
    with gr.Blocks(title="展示系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🗂️ 展示系统")
        gr.Markdown(f"**展示路径**: `{ROOT_DIR}`")
        
        # 状态变量
        current_path_state = gr.State(ROOT_DIR)
        
        with gr.Row():
            with gr.Column(scale=2):
                path_info = gr.Textbox(
                    label="展示子路径",
                    value=f"展示子路径: {ROOT_DIR}",
                    interactive=False
                )
                
                file_dropdown = gr.Dropdown(
                    label="选择文件或文件夹",
                    choices=[],
                    value=None,
                    interactive=True
                )
                
            with gr.Column(scale=1):
                navigate_btn = gr.Button("📂 打开展示文件", variant="secondary")
                download_btn = gr.Button("⬇️ 下载展示文件", variant="primary")
                refresh_btn = gr.Button("🔄 刷新", variant="secondary")
        
        with gr.Row():
            download_file = gr.File(
                label="展示文件",
                visible=True
            )
            
            status_msg = gr.Textbox(
                label="状态信息",
                interactive=False,
                lines=3
            )
        
        # 初始化页面
        def init_page():
            path_info_text, items = browse_directory(ROOT_DIR)
            return path_info_text, gr.Dropdown(choices=items, value=None), ROOT_DIR
        
        # 刷新功能
        def refresh_directory(current_path):
            path_info_text, items = browse_directory(current_path)
            return path_info_text, gr.Dropdown(choices=items, value=None), None, ""
        
        # 导航功能
        def navigate_handler(current_path, selected_item):
            new_path, path_info_text, items = navigate_to_item(current_path, selected_item)
            return new_path, path_info_text, gr.Dropdown(choices=items, value=None), None, ""
        
        # 下载功能
        def download_handler(current_path, selected_item):
            file_path, message = download_item(current_path, selected_item)
            return file_path, message
        
        # 绑定事件
        demo.load(
            fn=init_page,
            outputs=[path_info, file_dropdown, current_path_state]
        )
        
        refresh_btn.click(
            fn=refresh_directory,
            inputs=[current_path_state],
            outputs=[path_info, file_dropdown, download_file, status_msg]
        )
        
        navigate_btn.click(
            fn=navigate_handler,
            inputs=[current_path_state, file_dropdown],
            outputs=[current_path_state, path_info, file_dropdown, download_file, status_msg]
        )
        
        download_btn.click(
            fn=download_handler,
            inputs=[current_path_state, file_dropdown],
            outputs=[download_file, status_msg]
        )
    
    return demo

# 启动应用
if __name__ == "__main__":
    interface = create_interface()
    
    # 使用share=True生成公共链接
    interface.launch(
        # server_name="0.0.0.0",
        # server_port=7860,
        share=True,  # 生成公共链接
        # debug=True
    )