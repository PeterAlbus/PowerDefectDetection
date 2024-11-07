# from model_handlers.model1_handler import Model1Handler
import subprocess


class RobustnessEvalDao:
    def __init__(self):
        self.model = 0

    def get_prediction(self, data):
        # result = subprocess.run(
        #     ["envs/env_model1/bin/python", "model_handlers/model1_handler.py"],
        #     input=str(data), text=True, capture_output=True
        # )
        return self.model  # 调用模型
