import torch
from torch.autograd import Function
from torch import Tensor
from torch import nn

def sqish(x, alpha, beta, gamma):
    return alpha*x + (1-alpha)*x/torch.sqrt(1+beta*torch.exp(-2*gamma*(1-alpha)*x))

class Sqish_Function(Function):
    @staticmethod
    def forward(ctx, x, alpha, beta, gamma):
        result = sqish(x, alpha, beta, gamma)

        d1 = beta*torch.exp(-2*(1-alpha)*gamma*x) + 1
        d2 = torch.pow(d1, 1.5)

        dx = alpha + (1-alpha)/torch.sqrt(d1) + (1-alpha)*(1-alpha)*beta*gamma*x*(d1-1)/d2
        dalpha = x - x/torch.sqrt(d1) - beta*x*x*gamma*(1-alpha)*(d1-1)/d2
        dbeta = (alpha-1)*x*(d1-1)/(2*d2)
        dgamma = (alpha-1)*(alpha-1)*beta*x*x*(d1-1)/d2 

        ctx.save_for_backward(dx, dalpha, dbeta, dgamma)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        dx, dalpha, dbeta, dgamma = ctx.saved_tensors
        
        # result = grad_output * dx
        return (
                grad_output * dx, 
                grad_output * dalpha, 
                grad_output * dbeta, 
                grad_output * dgamma
            )
    

class Sqish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Sqish, self).__init__()
        self.inplace = inplace
        self.alpha = nn.Parameter(torch.max(torch.tensor(1), torch.randn(1)))
        self.beta = nn.Parameter(torch.max(torch.tensor(1), torch.randn(1)))
        self.gamma = nn.Parameter(torch.max(torch.tensor(1), torch.randn(1)))
    
    def forwar(self, input: Tensor) -> Tensor:
        if self.inplace:
            input = Sqish_Function.apply(input, self.alpha, self.beta, self.gamma)
            return input
        return Sqish_Function.apply(input, self.alpha, self.beta, self.gamma)