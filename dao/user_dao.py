from models import db
from models.user import User


class UserDao:
    @staticmethod
    def create_user(username, email, password):
        user = User(
            username=username,
            email=email,
            password=password
        )
        db.session.add(user)
        db.session.commit()
        return user.to_dict()

    @staticmethod
    def get_user_by_id(user_id):
        """根据用户ID查询用户"""
        return User.query.get(user_id)

    @staticmethod
    def get_user_by_email(email):
        """根据邮箱查询用户"""
        return User.query.filter_by(email=email).first()

    @staticmethod
    def get_all_users():
        """查询所有用户"""
        users = User.query.all()
        return [user.to_dict() for user in users]

    @staticmethod
    def delete_user(user_id):
        """删除用户"""
        user = User.query.get(user_id)
        if user:
            db.session.delete(user)
            db.session.commit()
            return True
        return False
