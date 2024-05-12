
import numpy as np
from scipy import stats

from icecream import ic

class IQAPerfomanceKonCept:
    def __init__(self, status, *args, **kwargs):
        self.status = status

    def reset(self):
        self._y_pred  = []
        self._y       = []
        self._y_std   = []

    def update(self, output):
        y_pred, y = output
        
        # ic(y_pred, y)
        self._y.extend([t.item() for t in y[0]])
        self._y_std.extend([t.item() for t in y[1]])
        self._y_pred.extend([t.item() for t in y_pred])
        # ic(self._y)
        # ic(self._y_pred)

    def compute(self):
        sq = np.reshape(np.asarray(self._y), (-1,))
        sq_std = np.reshape(np.asarray(self._y_std), (-1,))
        # ic(self._y)
        # ic(self._y_pred)
        self.preds = np.reshape(np.asarray(self._y_pred), (-1,))

        ic(sq)
        ic(self.preds)
        self.SROCC = stats.spearmanr(sq, self.preds)[0]
        self.PLCC = stats.pearsonr(sq, self.preds)[0]
        self.RMSE = np.sqrt(((sq - self.preds) ** 2).mean())
        
        if self.status=='train':
            return {'k': None, 'b': None}

        return {'SROCC': self.SROCC,
                'PLCC': self.PLCC,
                'RMSE': self.RMSE,
                }


class IQAPerformanceLinearity:
    """
    Evaluation of VQA methods using SROCC, PLCC, RMSE.

    `update` must receive output of the form (y_pred, y).
    """
    def __init__(self, status='train', k=[1,1,1], b=[0,0,0], mapping=True):
        # super(IQAPerformanceLinearity, self).__init__()
        self.k = k
        self.b = b
        self.status = status
        self.mapping = mapping


    def reset(self):
        self._y_pred  = []
        self._y_pred1 = []
        self._y_pred2 = []
        self._y       = []
        self._y_std   = []

    def update(self, output):
        y_pred, y = output
        from icecream import ic
        ic(y_pred, y)
        self._y.extend([t.item() for t in y[0]])
        self._y_std.extend([t.item() for t in y[1]])
        self._y_pred.extend([t.item() for t in y_pred[-1]])
        self._y_pred1.extend([t.item() for t in y_pred[0]])
        self._y_pred2.extend([t.item() for t in y_pred[1]])

    def compute(self):
        sq = np.reshape(np.asarray(self._y), (-1,))
        sq_std = np.reshape(np.asarray(self._y_std), (-1,))

        pq_before = np.reshape(np.asarray(self._y_pred), (-1, 1))
        self.preds = self.linear_mapping(pq_before, sq, i=0)
        self.SROCC = stats.spearmanr(sq, self.preds)[0]
        self.PLCC = stats.pearsonr(sq, self.preds)[0]
        self.RMSE = np.sqrt(((sq - self.preds) ** 2).mean())
        
        if self.status=='train':
            return {'k': self.k, 'b': self.b}

        return {'SROCC': self.SROCC,
                'PLCC': self.PLCC,
                'RMSE': self.RMSE,
                }

    def linear_mapping(self, pq, sq, i=0):
        if not self.mapping:
            return np.reshape(pq, (-1,))
        ones = np.ones_like(pq)
        yp1 = np.concatenate((pq, ones), axis=1)
        if self.status == 'train':
            # LSR solution of Q_i = k_1\hat{Q_i}+k_2. One can use the form of Eqn. (17) in the paper. 
            # However, for an efficient implementation, we use the matrix form of the solution here.
            # That is, h = (X^TX)^{-1}X^TY is the LSR solution of Y = Xh,
            # where X = [\hat{\mathbf{Q}}, \mathbf{1}], h = [k_1,k_2]^T, and Y=\mathbf{Q}.
            h = np.matmul(np.linalg.inv(np.matmul(yp1.transpose(), yp1)), np.matmul(yp1.transpose(), sq))
            self.k[i] = h[0].item()
            self.b[i] = h[1].item()
        else:
            h = np.reshape(np.asarray([self.k[i], self.b[i]]), (-1, 1))
        pq = np.matmul(yp1, h)

        return np.reshape(pq, (-1,))