
class Context2:
    def __init__(self, learning_rate=0.01, regularization_l1=0.0):
        self.tick: int = -1
        self.learning_rate = learning_rate
        self.regularization_l1 = regularization_l1
