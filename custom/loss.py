import numpy as np

class Loss:

    def mean_squared_error(self, y, t):
        return 0.5 * np.sum((y-t)**2)

    def cross_entropy_error(self, y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
            
        # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
        if t.size == y.size:
            t = t.argmax(axis=1)
                
        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

    def softmax_loss(self, X, t):
        y = self.softmax(X)
        return self.cross_entropy_error(y, t)