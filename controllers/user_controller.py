from flask import Blueprint, request, jsonify
from services.user_service import UserService
from common.utils.json_response import JsonResponse

user_bp = Blueprint('user', __name__)

user_service = UserService()


@user_bp.route('/register', methods=['POST'])
def register():
    if request.method == "POST":
        username = request.json.get("username")
        email = request.json.get("email")
        password = request.json.get("password")
        user = user_service.create_user(username, email, password)
        user['password'] = ""
        return user
    return {}


@user_bp.route('/login', methods=['POST'])
def login():
    if request.method == "POST":
        email = request.json.get("email")
        password = request.json.get("password")
        user = user_service.login(email, password)
        if user:
            user['password'] = ""
            return user
        return JsonResponse.error({}, 400,"用户名或密码错误")
    return {}


@user_bp.route('/queryAll', methods=['GET'])
def get_all_users():
    if request.method == "GET":
        users = user_service.get_all_users()
        for user in users:
            user['password'] = ""
        return jsonify(users)
    return {}
