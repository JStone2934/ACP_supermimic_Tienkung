import unittest
import torch
from supermimic.acp.train_aiming_fullbody import scale_value_gradient


class ActorValueGradientTests(unittest.TestCase):
    def test_zero_removes_entire_bootstrap_gradient_including_survival(self):
        def evaluate(weight):
            action=torch.tensor([.1,.2],requires_grad=True)
            reward=-(action-.5).square()
            survival=action.sigmoid();value=100*action+10
            objective=reward+scale_value_gradient(survival*value,weight)
            return objective.detach(),torch.autograd.grad(objective.sum(),action)[0]
        zero,g0=evaluate(0.);full,g1=evaluate(1.);half,g5=evaluate(.5)
        torch.testing.assert_close(zero,full,atol=0,rtol=0)
        torch.testing.assert_close(g0,torch.tensor([.8,.6]))
        torch.testing.assert_close(g5,(g0+g1)/2)
        action=torch.tensor([.1,.2],requires_grad=True)
        original=-(action-.5).square()+action.sigmoid()*(100*action+10)
        torch.testing.assert_close(g1,torch.autograd.grad(original.sum(),action)[0])


if __name__=='__main__':unittest.main()
