import os
import sys
import logging
from dotenv import load_dotenv

# 配置CUDA环境变量，确保使用正确的CUDA版本（与PyTorch匹配的12.1版本）
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
if cuda_path not in os.environ["PATH"]:
    os.environ["PATH"] = cuda_path + ";" + os.environ["PATH"]

# 禁用cpm_kernels的一些警告
os.environ["CPM_KERNELS_DISABLE_WARNINGS"] = "1"

# 加载环境变量
load_dotenv()

# 导入配置管理
from src.infrastructure.utils.config_manager import config_manager

# 配置日志
logging.basicConfig(
    level=getattr(logging, config_manager.get("log.level")),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config_manager.get("log.file")),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("启动周报生成助手...")
        
        # 检查Python版本
        if sys.version_info < (3, 8):
            logger.error("需要Python 3.8或更高版本")
            sys.exit(1)
            
        # 初始化CUDA
        import torch
        if torch.cuda.is_available():
            try:
                torch.cuda.init()
                torch.cuda.synchronize()
                logger.info(f"CUDA初始化成功，设备数量: {torch.cuda.device_count()}")
                logger.info(f"CUDA版本: {torch.version.cuda}")
                logger.info(f"GPU设备: {torch.cuda.get_device_name(0)}")
            except Exception as e:
                logger.warning(f"CUDA初始化失败: {str(e)}")
        
        # 检查环境变量和配置
        if not os.path.exists(".env"):
            logger.warning(".env文件不存在，将使用默认配置")
            print("警告: .env文件不存在，将使用默认配置")
        
        print("开始导入UI模块...")
        # 延迟导入，减少启动时间
        from src.ui.app import WeeklyReportUI
        print("UI模块导入成功，准备启动UI...")
        WeeklyReportUI.launch_ui()
        print("UI启动完成")
        
    except KeyboardInterrupt:
        logger.info("用户中断程序")
        print("\n程序已中断")
        sys.exit(0)
    except ImportError as e:
        logger.error(f"导入模块失败: {str(e)}", exc_info=True)
        print(f"错误: 导入模块失败 - {str(e)}")
        print("请确保所有依赖包已正确安装")
        print("可以运行 'pip install -r requirements.txt' 安装所有依赖")
        input("按Enter键退出...")
        sys.exit(1)
    except Exception as e:
        logger.error(f"启动失败: {str(e)}", exc_info=True)
        print(f"错误: 启动失败 - {str(e)}")
        print("详细错误信息已记录到日志文件")
        input("按Enter键退出...")
        sys.exit(1)

if __name__ == "__main__":
    main()
