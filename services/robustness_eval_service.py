from dao.robustness_eval_dao import RobustnessEvalDao


class RobustnessEvalService:
    def __init__(self):
        self.dao = RobustnessEvalDao()

    def predict(self, data):
        processed_data = data  # 业务逻辑
        result = self.dao.get_prediction(processed_data)
        return result
