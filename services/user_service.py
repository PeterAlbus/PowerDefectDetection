from dao.user_dao import UserDao


class UserService:
    def __init__(self):
        self.dao = UserDao()

    def create_user(self, username, email, password):
        return self.dao.create_user(username, email, password)

    def login(self, email, password):
        user = self.dao.get_user_by_email(email)
        if not user:
            return None
        if user.password == password:
            return user.to_dict()

    def get_all_users(self):
        return self.dao.get_all_users()
