class RunningAverage:
    """A simple class that maintains the running average of a quantity
    记录平均损失

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.denominator = 0  # 分母
        self.numerator = 0  # 分子

    def update(self, val1, val2=1):
        self.numerator += val1
        self.denominator += val2

    def __call__(self):
        if self.denominator <= 0:
            return 0.0
        else:
            return self.numerator / float(self.denominator)
