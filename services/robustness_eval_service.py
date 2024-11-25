from dao.robustness_eval_dao import RobustnessEvalDao
import subprocess
import os


class RobustnessEvalService:
    def __init__(self):
        self.dao = RobustnessEvalDao()

    def predict(self, data):
        processed_data = data  # 业务逻辑
        result = self.dao.get_prediction(processed_data)
        return result

    def eval_robustness(self, data):
        # 获取项目根目录
        project_root = os.path.abspath(os.path.dirname(__file__))
        print("项目根目录:", project_root)
        # 获取目标工作目录
        test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model_handlers", "robustness", "test"))
        print("目标工作目录:", test_dir)
        # 虚拟环境路径
        env_path = r"C:\Environment\anaconda3\envs\robustness"
        # 脚本文件路径（相对于工作目录）
        script_name = "testimport_full.py"
        # 参数
        attack_method = "FGSM"
        # 构建命令
        command = [
            os.path.join(env_path, "python.exe"),  # 指定虚拟环境的 Python 解释器
            script_name,
            "--attack_method", attack_method
        ]
        try:
            result = subprocess.run(
                command,
                cwd=test_dir,  # 设置子进程工作目录为 test 文件夹
                stdout=None,  # 子进程标准输出直接显示到主进程
                stderr=None,  # 子进程错误输出直接显示到主进程
                # capture_output=True,  # 捕获标准输出和标准错误
                # text=True,  # 将输出解码为字符串
                check=True  # 如果命令失败，将引发 CalledProcessError
            )
            print("命令执行成功！")
            # print("标准输出:", result.stdout)
            # print("标准错误:", result.stderr)
        except subprocess.CalledProcessError as e:
            print("命令执行失败！")
            # print("返回码:", e.returncode)
            # print("标准输出:", e.stdout)
            # print("标准错误:", e.stderr)
