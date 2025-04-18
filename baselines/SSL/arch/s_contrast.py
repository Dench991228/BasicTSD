import torch.nn as nn
import torch.nn.functional as F
import torch

class ClusterContrast(nn.Module):
    '''Spatial heterogeneity modeling by using a soft-clustering paradigm.
    '''

    def __init__(self, repr_dim: int, nmb_prototype: int, tau=0.5, **kwargs):
        super(ClusterContrast, self).__init__()
        # note 用来标准化的地方
        self.l2norm = lambda x: F.normalize(x, dim=1, p=2)
        # note 用来做聚类的聚类头
        self.prototypes = nn.Linear(repr_dim, nmb_prototype, bias=False)

        self.tau = tau
        self.d_model = repr_dim

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, **kwargs):
        """Compute the contrastive loss of batched data.
        :param z1, z2 (tensor): shape nvc
        :param loss: contrastive loss
        """
        with torch.no_grad():
            w = self.prototypes.weight.data.clone()
            w = self.l2norm(w)
            self.prototypes.weight.copy_(w)

        # l2norm avoids nan of Q in sinkhorn
        zc1 = self.prototypes(self.l2norm(z1.reshape(-1, self.d_model)))  # nd -> nk, assignment q, embedding z
        zc2 = self.prototypes(self.l2norm(z2.reshape(-1, self.d_model)))  # nd -> nk
        print("Shape of Cluster Assignment ", zc1.shape, zc2.shape)
        with torch.no_grad():
            q1 = sinkhorn(zc1.detach())
            q2 = sinkhorn(zc2.detach())
        l1 = - torch.mean(torch.sum(q1 * F.log_softmax(zc2 / self.tau, dim=1), dim=1))
        l2 = - torch.mean(torch.sum(q2 * F.log_softmax(zc1 / self.tau, dim=1), dim=1))
        print("l1 and l2 loss", l1, l2)
        return (l1 + l2)/2


@torch.no_grad()
def sinkhorn(out, epsilon=0.05, sinkhorn_iterations=3):
    Q = torch.exp(out / epsilon).t()  # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1]  # number of samples to assign
    K = Q.shape[0]  # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B  # the colomns must sum to 1 so that Q is an assignment
    return Q.t()
