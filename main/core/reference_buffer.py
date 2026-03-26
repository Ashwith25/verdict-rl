import random


class ReferenceBank:
    def __init__(self, max_size=100):
        self.data = []
        self.max_size = max_size

    def sample(self):
        if not self.data:
            return None
        return random.choice(self.data)

    def add(self, tau):
        if len(self.data) < self.max_size:
            self.data.append(tau)
        else:
            if random.random() < 0.1:
                idx = random.randint(0, self.max_size - 1)
                self.data[idx] = tau