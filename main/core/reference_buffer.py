import random


class ReferenceBank:
    def __init__(self, max_size=20):
        self.data = []
        self.max_size = max_size

    def sample(self):
        if not self.data:
            return None
        return random.choice(self.data)

    def replace_all(self, trajs):
        self.data = trajs[: self.max_size]