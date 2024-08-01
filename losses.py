import numpy as np
import jax.numpy as jnp

def laplacian_loss(vertex,faces):
    nv = vertex.shape[0]
    nf = faces.shape[0]
    laplacian = jnp.zeros([nv, nv]).astype(jnp.float32)
    laplacian[faces[:, 0], faces[:, 1]] = -1
    laplacian[faces[:, 1], faces[:, 0]] = -1
    laplacian[faces[:, 1], faces[:, 2]] = -1
    laplacian[faces[:, 2], faces[:, 1]] = -1
    laplacian[faces[:, 2], faces[:, 0]] = -1
    laplacian[faces[:, 0], faces[:, 2]] = -1
    r, c = jnp.diag_indices(laplacian.shape[0])
    laplacian[r, c] = -laplacian.sum(1)

    for i in range(nv):
        laplacian[i, :] /= laplacian[i, i]


class LaplacianLoss():
    def __init__(self, vertex, faces, average=False):
        # super(LaplacianLoss, self).__init__()
        self.nv = vertex.shape[0]
        self.nf = faces.shape[0]
        self.average = average
        self.laplacian = np.zeros([self.nv, self.nv]).astype(np.float32) # [n,n]

        self.laplacian[faces[:, 0], faces[:, 1]] = -1
        self.laplacian[faces[:, 1], faces[:, 0]] = -1
        self.laplacian[faces[:, 1], faces[:, 2]] = -1
        self.laplacian[faces[:, 2], faces[:, 1]] = -1
        self.laplacian[faces[:, 2], faces[:, 0]] = -1
        self.laplacian[faces[:, 0], faces[:, 2]] = -1

        r, c = np.diag_indices(self.laplacian.shape[0])
        self.laplacian[r, c] = -self.laplacian.sum(1)

        for i in range(self.nv):
            self.laplacian[i, :] /= self.laplacian[i, i]

        # self.register_buffer('laplacian', torch.from_numpy(laplacian))

    def forward(self, x):
        batch_size = x.shape[0]
        x = jnp.matmul(self.laplacian, x)
        # dims = tuple(range(x.ndimension())[1:]) # no batch
        x = (x**2).sum()
        if self.average:
            return x.sum() / batch_size
        else:
            return x


class FlattenLoss():
    def __init__(self, faces, average=False):
        # super(FlattenLoss, self).__init__()
        self.nf = faces.shape[0]
        self.average = average

        # faces = faces.detach().cpu().numpy()
        vertices = list(set([tuple(v) for v in np.sort(np.concatenate((faces[:, 0:2], faces[:, 1:3]), axis=0))]))

        self.v0s = np.array([v[0] for v in vertices], 'int32')
        self.v1s = np.array([v[1] for v in vertices], 'int32')
        self.v2s = []
        self.v3s = []
        for v0, v1 in zip(self.v0s, self.v1s):
            count = 0
            for face in faces:
                if v0 in face and v1 in face:
                    v = np.copy(face)
                    v = v[v != v0]
                    v = v[v != v1]
                    if count == 0:
                        self.v2s.append(int(v[0]))
                        count += 1
                    else:
                        self.v3s.append(int(v[0]))
        self.v2s = np.array(self.v2s, 'int32')
        self.v3s = np.array(self.v3s, 'int32')

        # self.register_buffer('v0s', torch.from_numpy(v0s).long())
        # self.register_buffer('v1s', torch.from_numpy(v1s).long())
        # self.register_buffer('v2s', torch.from_numpy(v2s).long())
        # self.register_buffer('v3s', torch.from_numpy(v3s).long())

    def forward(self, vertices, eps=1e-6):
        # make v0s, v1s, v2s, v3s
        batch_size = vertices.shape[0]

        v0s = vertices[:, self.v0s, :]
        v1s = vertices[:, self.v1s, :]
        v2s = vertices[:, self.v2s, :]
        v3s = vertices[:, self.v3s, :]

        a1 = v1s - v0s
        b1 = v2s - v0s
        a1l2 = (a1**2).sum(-1)
        b1l2 = (b1**2).sum(-1)
        a1l1 = jnp.sqrt((a1l2 + eps))
        b1l1 = jnp.sqrt((b1l2 + eps))
        ab1 = (a1 * b1).sum(-1)
        cos1 = ab1 / (a1l1 * b1l1 + eps)
        sin1 = jnp.sqrt((1 - cos1**2 + eps))
        c1 = a1 * (ab1 / (a1l2 + eps))[:, :, None]
        cb1 = b1 - c1
        cb1l1 = b1l1 * sin1

        a2 = v1s - v0s
        b2 = v3s - v0s
        a2l2 = (a2**2).sum(-1)
        b2l2 = (b2**2).sum(-1)
        a2l1 = jnp.sqrt((a2l2 + eps))
        b2l1 = jnp.sqrt((b2l2 + eps))
        ab2 = (a2 * b2).sum(-1)
        cos2 = ab2 / (a2l1 * b2l1 + eps)
        sin2 = jnp.sqrt((1 - cos2**2 + eps))
        c2 = a2 * (ab2 / (a2l2 + eps))[:, :, None]
        cb2 = b2 - c2
        cb2l1 = b2l1 * sin2

        cos = (cb1 * cb2).sum(-1) / (cb1l1 * cb2l1 + eps)

        # dims = tuple(range(cos.ndimension())[1:])
        loss = ((cos + 1)**2).sum()
        if self.average:
            return loss.sum() / batch_size
        else:
            return loss