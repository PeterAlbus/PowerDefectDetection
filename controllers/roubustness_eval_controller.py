from flask import Blueprint, request, jsonify
# from services.model1_service import Model1Service

robustness_bp = Blueprint('robustness', __name__)


# model1_service = Model1Service()

@robustness_bp.route('/dict', methods=['GET'])
def dict_test():
    if request.method == "GET":
        value = request.args.get("abc")
        result = {'abc': value}
        return result
    return {}


@robustness_bp.route('/list', methods=['GET'])
def list_test():
    result = ['1','c']
    return result
