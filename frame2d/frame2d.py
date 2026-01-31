import numpy as np


class Node2d:
    def __init__(self, x, z, id, fixed, point_mass):
        self.x = x
        self.z = z
        self.id = id
        self.fixed = fixed
        self.point_mass = point_mass


class SemiRigidBeam2d:
    '''
    Semi-rigid beam element with rotational springs at both ends.
    '''
    def __init__(
            self,
            E,
            G,
            m,
            A,
            I,
            i,
            j, 
            ks = 1.0,        # shape factor
            gi = 1.0,        # fixity factor at the i-th end
            gj = 1.0,        # fixity factor at the j-th end
            add_mass = 0.0   # additional mass (kg)
        ):

        self.E = E
        self.G = G
        self.m = m
        self.A = A
        self.I = I
        self.i = int(i)
        self.j = int(j)
        self.ks = ks
        self.gi = gi
        self.gj = gj
        self.add_mass = add_mass

        self.node_i = None
        self.node_j = None

        self.L = None
        self.R = None
        self.Ke = None
        self.Hi = None
        self.Hj = None
        self.mii = None
        self.mij = None
        self.mji = None
        self.mjj = None
        self.kii = None
        self.kij = None
        self.kji = None
        self.kjj = None

    def connect(self, node_i, node_j):
        self.node_i = node_i
        self.node_j = node_j
    
    def make(self, shear = False):
        # angle of element in the global coordinate
        self.L = np.sqrt(
            (self.node_i.x - self.node_j.x)**2 + (self.node_i.z - self.node_j.z)**2
        )
        sina = (self.node_j.z - self.node_i.z) / self.L
        cosa = (self.node_j.x - self.node_i.x) / self.L
        self.R = np.array([[cosa, -sina, 0], [sina, cosa, 0], [0, 0, 1]])

        # elemental mass matrix
        mal = self.m * self.A * self.L + self.add_mass
        self.mii = np.diag([mal / 2, mal / 2, 0])
        self.mjj = np.diag([mal / 2, mal / 2, 0])
        self.mij = np.zeros((3, 3))
        self.mji = np.zeros((3, 3))

        # elemental equilibrium matrices (with a simple-beam form)
        self.Hi = np.array([[-1, 0, 0], [0, -1, 0], [0, -self.L, -1]])
        self.Hj = np.diag([1, 1, 1])
        Hs = np.r_[self.Hi, self.Hj]  

        # elemental stiffness matrix
        EA = self.E * self.A
        EI = self.E * self.I

        if shear:
            gm = 6 * EI * self.ks / (self.G * self.A * self.L**2)
        else:
            gm = 0

        if self.gi == 1.0:
            k1 = 1e+20
        else:
            k1 = 0.75 * self.gi / (1 - self.gi)

        if self.gj == 1.0:
            k2 = 1e+20
        else:
            k2 = 0.75 * self.gj / (1 - self.gj)

        kk = 2 * (k1 + 1) * (k2 + 1) - 1 / 2 + gm * (4 * k1 * k2 + k1 + k2)
        self.Ke = np.array([
            [
                EA / self.L,
                0,
                0
            ],
            [
                0,
                12 * EI / self.L**3 * (4 * k1 * k2 + k1 + k2) / 2 / kk,
                -6 * EI / self.L**2 * (2 * k1 * k2 + k2) / kk
            ],
            [
                0,
                -6 * EI / self.L**2 * (2 * k1 * k2 + k2) / kk,
                4 * EI / self.L * (4 * k1 * k2 * (1 + gm / 2) + 3 * k2) / kk / 2
            ]
        ])

        K = Hs @ self.Ke @ Hs.T
        self.kii = K[:3,:3]
        self.kij = K[:3,3:]
        self.kji = K[3:,:3]
        self.kjj = K[3:,3:]


class Frame2d:
    def __init__(self):
        self.nodes = []
        self.membs = []
        self.n_node = None
        self.n_memb = None
        self.M = None
        self.C = None
        self.K = None
        self.Km = None
        self.M_red = None
        self.K_red = None
        self.K_rec = None
        self.H = None
        self.omeg = None
        self.U = None
        self.U_rec = None
        self.U_ful = None
    
    def add_node(self, node):
        self.nodes.append(node)
    
    def add_memb(self, memb):
        node_i = list(filter(lambda x: x.id == memb.i, self.nodes))[0]
        node_j = list(filter(lambda x: x.id == memb.j, self.nodes))[0]
        memb.connect(node_i, node_j)
        self.membs.append(memb)
    
    def make_MK(self, nodes, membs, shear = False):
        '''
        construct mass & stif matrices.
        '''
        # load nodes & membs
        self.n_node = len(nodes)
        self.n_memb = len(membs)
        self.nodes = []
        self.membs = []
        for i in range(len(nodes)):
            self.add_node(nodes[i])
        for i in range(len(membs)):
            self.add_memb(membs[i])
            self.membs[i].make(shear)
        
        # make global matrices (M, K)
        self.M = np.zeros((3 * self.n_node, 3 * self.n_node))
        self.K = np.zeros((3 * self.n_node, 3 * self.n_node))
        for k in range(self.n_memb):
            istt, iend = 3 * (membs[k].i - 1), 3 * (membs[k].i - 1) + 3
            jstt, jend = 3 * (membs[k].j - 1), 3 * (membs[k].j - 1) + 3

            # mass
            self.M[istt:iend, istt:iend] += membs[k].mii
            self.M[jstt:jend, jstt:jend] += membs[k].mjj

            # stif
            self.K[istt:iend, istt:iend] += membs[k].R @ membs[k].kii @ membs[k].R.T
            self.K[istt:iend, jstt:jend] += membs[k].R @ membs[k].kij @ membs[k].R.T
            self.K[jstt:jend, istt:iend] += membs[k].R @ membs[k].kji @ membs[k].R.T
            self.K[jstt:jend, jstt:jend] += membs[k].R @ membs[k].kjj @ membs[k].R.T
        
        # additional point mass
        for i in range(self.n_node):
            istt, iend = 3 * i, 3 * i + 3
            pm = self.nodes[i].point_mass
            self.M[istt:iend, istt:iend] += np.diag([pm, pm, 0])

        # reflect support conditions
        for i in range(self.n_node):
            if self.nodes[i].fixed == 1:  # fixed support
                istt, iend = 3 * i, 3 * i + 3
                self.M[istt:iend,:] = 0
                self.M[:,istt:iend] = 0
                self.K[istt:iend,:] = 0
                self.K[:,istt:iend] = 0
                self.K[istt:iend, istt:iend] = np.diag([1, 1, 1])

            if self.nodes[i].fixed == 2:  # pin support
                istt, iend = 3 * i, 3 * i + 2
                self.M[istt:iend,:] = 0
                self.M[:,istt:iend] = 0
                self.K[istt:iend,:] = 0
                self.K[:,istt:iend] = 0
                self.K[istt:iend, istt:iend] = np.diag([1, 1])

            if self.nodes[i].fixed == 3:  # roller support
                istt = 3 * i + 1
                self.M[istt,:] = 0
                self.M[:,istt] = 0
                self.K[istt,:] = 0
                self.K[:,istt] = 0
                self.K[istt, istt] = 1
        
        # reduced mass matrix
        dm = np.diag(self.M)
        self.M_red = self.M[dm != 0,:][:,dm != 0]

        # reduced stif matrix
        K11 = self.K[dm != 0,:][:,dm != 0]
        K12 = self.K[dm != 0,:][:,dm == 0]
        K21 = self.K[dm == 0,:][:,dm != 0]
        K22 = self.K[dm == 0,:][:,dm == 0]
        self.K_red = K11 - K12 @ np.linalg.inv(K22) @ K21
        self.K_rec = - np.linalg.inv(K22) @ K21
    
    def moda(self, normalize = 'max'):
        '''
        modal (eigenvalue) analysis.
        '''
        eig, vec = np.linalg.eig(np.linalg.inv(self.M_red) @ self.K_red)
        vec_ful = np.zeros((self.K.shape[0], vec.shape[1]))
        vec_ful[np.diag(self.M) != 0,:] = vec
        vec_ful[np.diag(self.M) == 0,:] = self.K_rec @ vec

        if normalize == 'max':
            for i in range(vec_ful.shape[1]):
                vec_ful[:,i] /= max(abs(vec_ful[:,i]))
                vec_ful[:,i] /= np.sign(vec_ful[np.diag(self.M) != 0, i][0])
        if normalize == 'norm':
            for i in range(vec_ful.shape[1]):
                vec_ful[:,i] /= np.sqrt(np.sum(vec_ful[:,i]**2))
                vec_ful[:,i] /= np.sign(vec_ful[np.diag(self.M) != 0, i][0])

        self.omeg = np.sqrt(eig)
        self.U = vec_ful[np.diag(self.M) != 0,:]
        self.U_ful = vec_ful
    
    def make_C(self, zeta):
        self.moda()
        self.C = 2 * zeta / min(self.omeg) * self.K
    
    def make_H(self):
        '''
        make equilibrium matrix.
        '''
        self.H  = np.zeros((3 * self.n_node, 3 * self.n_memb))
        self.Km = np.zeros((3 * self.n_memb, 3 * self.n_memb))
        for k in range(self.n_memb):
            istt, iend = 3 * (self.membs[k].i - 1), 3 * (self.membs[k].i - 1) + 3
            jstt, jend = 3 * (self.membs[k].j - 1), 3 * (self.membs[k].j - 1) + 3
            kstt, kend = 3 * k, 3 * k + 3
            self.H [istt:iend, kstt:kend] += self.membs[k].R @ self.membs[k].Hi
            self.H [jstt:jend, kstt:kend] += self.membs[k].R @ self.membs[k].Hj
            self.Km[kstt:kend, kstt:kend] += self.membs[k].Ke
    
    def make_B(self):
        '''
        temtative ......
        '''
        self.B = np.zeros((self.Km.shape[0], self.Km.shape[0]))
        for k in range(self.n_memb):
            kstt, kend = 3 * k, 3 * k + 3
            self.B[kstt:kend,:][:,kstt:kend] += np.array(
                [[1, 0, 0], [0, self.membs[k].L, 1], [0, 0, 1]]
            )
    
    def build(self, nodes, membs, zeta, shear = False):
        self.make_MK(nodes, membs, shear)
        self.make_C(zeta)
        self.make_H()
        self.make_B()