import os
import sys

# 禁用CUDA，避免加载CUDA相关的DLL
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_USE_CUDA_DSA"] = "False"

import gradio as gr
import tempfile
from typing import List, Dict, Any, Optional
from datetime import datetime

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入内部模块
from src.models.llm import LocalLLM
from src.models.advanced_report_generator import AdvancedReportGenerator
from src.models.enhanced_report_generator import EnhancedReportGenerator
from src.file_processing.factory import FileProcessorFactory
from src.memory.memory_manager import MemoryManager
from src.utils.prompt_templates import PromptTemplates
from src.utils.word_generator import WordGenerator
from src.utils.report_generator import ContentProcessor
from src.utils.prompt_engineer import PromptEngineer
from src.infrastructure.utils.config_manager import config_manager

print("开始创建单例实例...")
# 创建单例实例
print("创建LocalLLM实例...")
# 默认使用模拟模型，避免模型加载失败
llm = LocalLLM(is_mock_model=False)
print("创建MemoryManager实例...")
memory_manager = MemoryManager()
print("创建WordGenerator实例...")
word_generator = WordGenerator()
print("创建ContentProcessor实例...")
content_processor = ContentProcessor()
print("创建AdvancedReportGenerator实例...")
advanced_report_generator = AdvancedReportGenerator()
print("创建EnhancedReportGenerator实例...")
enhanced_report_generator = EnhancedReportGenerator()
print("创建PromptEngineer实例...")
prompt_engineer = PromptEngineer()

print("单例实例创建完成，开始加载记忆...")
# 加载记忆
memory_manager.load_memory()
print("记忆加载完成")

# 延迟加载模型，只在需要时才加载
print("模型将在首次使用时自动加载")

# 全局变量
selected_conversation = None

class WeeklyReportUI:
    """周报生成助手UI"""
    
    @staticmethod
    def launch_ui():
        """启动UI界面"""
        with gr.Blocks(title="周报生成助手") as demo:
            with gr.Row():
                # 左侧导航栏
                with gr.Column(scale=1, min_width=200):
                    gr.Markdown("# 周报生成助手")
                    gr.Markdown("## 最近对话")
                    
                    conversation_list = gr.Dropdown(
                        choices=[],
                        label="对话记录",
                        interactive=True,
                        elem_id="conversation_list",
                        allow_custom_value=False,
                        show_label=False
                    )
                    
                    clear_btn = gr.Button("清空对话", variant="secondary")
                    
                    gr.Markdown("## 状态")
                    status_text = gr.Textbox("就绪", label="当前状态", interactive=False, lines=3)
                
                # 中间主区域
                with gr.Column(scale=4):
                    # 对话历史显示
                    chat_history = gr.Chatbot(
                        label="对话历史",
                        height=400
                    )
                    
                    # 文件上传区域
                    file_upload = gr.File(
                        label="上传文件 (支持TXT、PDF、DOCX、XLSX、CSV、PNG、JPG)",
                        type="filepath",
                        file_count="multiple",
                        elem_id="file_upload"
                    )
                    
                    # 报告类型选择
                    with gr.Row():
                        report_type = gr.Dropdown(
                            choices=["weekly", "monthly", "quarterly", "annual", "project", "sales"],
                            value="weekly",
                            label="报告类型",
                            interactive=True,
                            elem_id="report_type"
                        )
                    
                    # 分析深度调节
                    with gr.Row():
                        analysis_depth = gr.Slider(
                            minimum=0,
                            maximum=2,
                            step=1,
                            value=1,
                            label="分析深度",
                            interactive=True,
                            elem_id="analysis_depth"
                        )
                        depth_label = gr.Textbox(
                            value="深度分析",
                            label="当前深度",
                            interactive=False,
                            elem_id="depth_label"
                        )
                    
                    # 输入区域和按钮
                    with gr.Row():
                        user_input = gr.Textbox(
                            label="请输入报告要求",
                            placeholder="请描述您的报告要求...",
                            elem_id="user_input",
                            lines=3,
                            scale=5
                        )
                        
                        with gr.Column(scale=1, min_width=120):
                            generate_zh_btn = gr.Button("生成中文报告", variant="primary")
                            generate_en_btn = gr.Button("生成英文报告", variant="secondary")
                    
                    # 生成进度显示
                    progress = gr.Progress(track_tqdm=True)
                    status_bar = gr.Textbox(
                        value="就绪",
                        label="生成状态",
                        interactive=False,
                        elem_id="status_bar"
                    )
                    
                    # 生成的报告下载
                    report_output = gr.File(label="生成的周报", elem_id="report_output")
            
            # 分析深度标签映射
            depth_map = {0: "基础报告", 1: "深度分析", 2: "战略报告"}
            
            # 分析深度变化处理函数
            def update_depth_label(depth_value):
                return depth_map.get(depth_value, "深度分析")
            
            # 生成报告函数
            def generate_report(user_input, files, report_type, analysis_depth, is_chinese=True):
                global selected_conversation
                
                try:
                    # 检查模型是否已加载
                    if not llm.model and not llm.is_mock_model:
                        # 尝试自动加载模型
                        if not llm.load_model():
                            return chat_history.value, gr.update(), "模型加载失败，请确保模型文件已正确下载到指定路径"
                    
                    if not user_input and not files:
                        return chat_history.value, gr.update(), "请输入报告要求或上传文件"
                    
                    # 限制用户输入长度
                    if len(user_input) > memory_manager.max_user_input_length:
                        return chat_history.value, gr.update(), f"用户输入超过长度限制（{memory_manager.max_user_input_length}字）"
                    
                    # 处理上传的文件
                    file_contents = []
                    uploaded_files = []
                    
                    if files:
                        status_bar = "正在处理文件..."
                        
                        try:
                            for file in files:
                                try:
                                    # 获取文件路径
                                    temp_path = file.name
                                    
                                    # 检查文件大小，防止过大文件导致问题
                                    file_size = os.path.getsize(temp_path)
                                    if file_size > 100 * 1024 * 1024:  # 限制文件大小为100MB
                                        return chat_history.value, gr.update(), f"文件 {os.path.basename(temp_path)} 过大（最大支持100MB）"
                                    
                                    # 检查文件类型
                                    file_ext = os.path.splitext(temp_path)[1].lower()
                                    allowed_extensions = ['.txt', '.pdf', '.docx', '.xlsx', '.csv', '.png', '.jpg', '.jpeg']
                                    if file_ext not in allowed_extensions:
                                        return chat_history.value, gr.update(), f"文件类型 {file_ext} 不支持，支持的类型: {', '.join(allowed_extensions)}"
                                    
                                    # 处理文件
                                    processor = FileProcessorFactory()
                                    result = processor.process_file(temp_path)
                                    
                                    if result["success"]:
                                        file_contents.append(f"文件: {os.path.basename(temp_path)}")
                                        file_contents.append(result["text"])
                                        uploaded_files.append(os.path.basename(temp_path))
                                    else:
                                        return chat_history.value, gr.update(), f"处理文件 {os.path.basename(temp_path)} 失败: {result.get('error', '未知错误')}"
                                except Exception as e:
                                    return chat_history.value, gr.update(), f"处理文件 {os.path.basename(temp_path)} 失败: {str(e)}"
                        except Exception as e:
                            return chat_history.value, gr.update(), f"文件处理过程中发生错误: {str(e)}"
                    
                    # 构建提示词
                    status_bar = "正在生成报告..."
                    
                    # 获取上下文
                    context = memory_manager.get_context()
                    
                    # 转换分析深度值
                    depth_str = {0: "basic", 1: "detailed", 2: "strategic"}.get(analysis_depth, "detailed")
                    
                    # 设置prompt工程师语言
                    prompt_engineer.chinese = is_chinese
                    
                    # 构建提示词
                    prompt = prompt_engineer.get_prompt(
                        report_type=report_type,
                        analysis_depth=depth_str,
                        include_context=bool(context)
                    )
                    
                    # 添加用户输入和文件内容
                    prompt += "\n\n【用户要求】\n" if is_chinese else "\n\n[User Requirements]\n"
                    prompt += user_input
                    
                    if file_contents:
                        prompt += "\n\n【文件内容】\n" if is_chinese else "\n\n[File Contents]\n"
                        prompt += "\n\n".join(file_contents)
                    
                    # 生成报告文本
                    report_text = llm.generate(prompt, language="zh" if is_chinese else "en")
                    
                    # 检查AI回复
                    if not report_text or report_text.strip() in ["模型加载失败，请检查模型文件", "生成失败"]:
                        return chat_history.value, gr.update(), f"生成报告文本失败: {report_text}"
                    
                    # 检查AI回复长度
                    if len(report_text) > memory_manager.max_ai_response_length:
                        return chat_history.value, gr.update(), f"AI回复超过长度限制（{memory_manager.max_ai_response_length}字）"
                    
                    # 使用高级报告生成器处理报告内容
                    content_processor.chinese = is_chinese
                    
                    # 检查并移除重复内容
                    duplicate_content = content_processor.check_for_duplicate_content(report_text)
                    if duplicate_content:
                        report_text = content_processor.remove_duplicate_content(report_text)
                        status_bar = f"已移除重复内容（{len(duplicate_content)}处）"
                    
                    # 格式化表格
                    report_text = content_processor.format_tables(report_text)
                    
                    # 尝试使用增强型报告生成器生成更优质的报告
                    try:
                        # 解析用户输入
                        parsed_input = enhanced_report_generator.parse_user_input(user_input)
                        # 使用增强型报告生成器生成报告
                        enhanced_report = enhanced_report_generator.generate_report(parsed_input, is_mock=False)
                        # 使用增强型报告替换原始报告
                        if enhanced_report and len(enhanced_report) > 100:
                            report_text = enhanced_report
                            print("使用增强型报告生成器生成的报告替换了原始报告")
                    except Exception as e:
                        print(f"使用增强型报告生成器时出错: {str(e)}")
                        # 如果增强型报告生成器出错，回退到高级报告生成器
                        try:
                            parsed_input = advanced_report_generator.parse_user_input(user_input)
                            advanced_report = advanced_report_generator.generate_report(parsed_input, is_mock=False)
                            if advanced_report and len(advanced_report) > 100:
                                report_text = advanced_report
                                print("使用高级报告生成器生成的报告替换了原始报告")
                        except Exception as e2:
                            print(f"使用高级报告生成器时也出错: {str(e2)}")
                            # 如果高级报告生成器也出错，仍然使用原始报告
                            pass
                    
                    # 生成Word文档
                    status_bar = "正在生成Word文档..."
                    
                    # 创建临时文件路径
                    temp_dir = tempfile.gettempdir()
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    report_filename = f"{report_type}_report_{timestamp}.docx"
                    report_path = os.path.join(temp_dir, report_filename)
                    
                    # 生成报告
                    # 根据报告类型生成标题
                    if report_type == "weekly":
                        title = "周报" if is_chinese else "Weekly Report"
                    elif report_type == "daily":
                        title = "日报" if is_chinese else "Daily Report"
                    elif report_type == "monthly":
                        title = "月报" if is_chinese else "Monthly Report"
                    else:
                        title = f"{report_type}报告" if is_chinese else f"{report_type} Report"
                    
                    if word_generator.create_report(title, report_text, save_path=report_path):
                        # 添加对话记录
                        memory_manager.add_conversation(
                            user_input=user_input,
                            ai_response=report_text,
                            user_files=uploaded_files,
                            generated_report=report_path
                        )
                        
                        # 保存记忆
                        memory_manager.save_memory()
                        
                        # 更新对话历史
                        # 获取当前聊天历史
                        current_history = chat_history.value
                        # 创建新的历史列表，使用Gradio Chatbot要求的格式
                        new_history = current_history + [
                            {"role": "user", "content": user_input},
                            {"role": "assistant", "content": report_text}
                        ]
                        
                        return new_history, gr.update(value=report_path, visible=True), "报告生成成功！"
                    else:
                        return chat_history.value, gr.update(), "生成Word文档失败"
                except Exception as e:
                    import traceback
                    error_message = f"生成报告失败: {str(e)}\n详细错误: {traceback.format_exc()}"
                    return chat_history.value, gr.update(), error_message
            
            # 分析深度滑块事件处理
            analysis_depth.change(
                fn=update_depth_label,
                inputs=[analysis_depth],
                outputs=[depth_label]
            )
            
            # 生成中文报告按钮点击
            generate_zh_btn.click(
                fn=generate_report,
                inputs=[user_input, file_upload, report_type, analysis_depth],
                outputs=[chat_history, report_output, status_bar]
            )
            
            # 生成英文报告按钮点击
            generate_en_btn.click(
                fn=lambda user_input, files, report_type, analysis_depth: generate_report(user_input, files, report_type, analysis_depth, is_chinese=False),
                inputs=[user_input, file_upload, report_type, analysis_depth],
                outputs=[chat_history, report_output, status_bar]
            )
            
            # 清空对话按钮点击
            def clear_conversation():
                global selected_conversation
                selected_conversation = None
                memory_manager.clear_memory()
                memory_manager.save_memory()
                return [], [], gr.update(value="", visible=False), "对话已清空"
            
            clear_btn.click(
                fn=clear_conversation,
                outputs=[chat_history, conversation_list, report_output, status_bar]
            )
            
            # 选择对话记录
            def select_conversation(conversation_id):
                global selected_conversation
                selected_conversation = conversation_id
                
                # 查找对话
                for conv in memory_manager.memory:
                    if conv["id"] == conversation_id:
                        # 重建对话历史，使用Gradio Chatbot要求的格式
                        chat_history = [
                            {"role": "user", "content": conv["user_input"]},
                            {"role": "assistant", "content": conv["ai_response"]}
                        ]
                        
                        # 如果有生成的报告，显示报告
                        if conv["generated_report"] and os.path.exists(conv["generated_report"]):
                            return chat_history, gr.update(value=conv["generated_report"], visible=True)
                        else:
                            return chat_history, gr.update(value="", visible=False)
                
                return [], gr.update(value="", visible=False)
            
            conversation_list.change(
                fn=select_conversation,
                inputs=[conversation_list],
                outputs=[chat_history, report_output]
            )
        
        # 启动应用
        try:
            print("准备启动Gradio应用...")
            # 优化服务器配置，增加超时设置，解决文件上传问题
            # 使用Gradio 6.x版本支持的配置方式
            demo.launch(
                server_name="127.0.0.1",
                server_port=7861,  # 使用7861端口，避免与其他实例冲突
                share=False,
                inbrowser=True,
                debug=False,  # 关闭调试模式，提高稳定性
                quiet=False,  # 显示详细日志
                theme=gr.themes.Soft()  # 将theme参数移动到这里
            )
        except Exception as e:
            print(f"启动Gradio应用失败: {str(e)}")
            import traceback
            traceback.print_exc()

# 启动UI
if __name__ == "__main__":
    WeeklyReportUI.launch_ui()
