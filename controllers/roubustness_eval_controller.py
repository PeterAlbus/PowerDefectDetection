from flask import Blueprint, request, jsonify

from services.robustness_eval_service import RobustnessEvalService

robustness_bp = Blueprint('robustness', __name__)

robustness_service = RobustnessEvalService()


@robustness_bp.route('/dict', methods=['GET'])
def dict_test():
    if request.method == "GET":
        value = request.args.get("abc")
        result = {'abc': value}
        return result
    return {}


@robustness_bp.route('/list', methods=['GET'])
def list_test():
    result = ['1', 'c']
    return result


@robustness_bp.route('/eval', methods=['GET'])
def eval_robustness():
    robustness_service.eval_robustness({})
    return {}
