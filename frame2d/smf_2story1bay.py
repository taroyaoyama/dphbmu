from frame2d import frame2d as f2d
import numpy as np

def smf_2story1bay(theta):
    # 1B (H-150x75x5x7) -----
    A1B = 17.300e-04
    I1B = 642.00e-08

    # 2B (H-125x60x6x8) -----
    A2B = 16.140e-04
    I2B = 393.80e-08

    # 1C (BOX-125x125x12) -----
    A1C = 54.240e-04
    I1C = 1167.0e-08

    # 2C (BOX-125x125x12) -----
    A2C = 54.240e-04
    I2C = 1167.0e-08

    # material properties -----
    E = 2.05e+11
    G = 7.90e+10
    rho = 0.00

    # rotational stiff. (fixity factors) -----
    g1BE = theta[0]
    g1BW = theta[1]
    g2BE = theta[2]
    g2BW = theta[3]

    # mass -----
    m1 = 3540
    m2 = 3540

    # define nodes -----
    nodes = []
    nodes.append(f2d.Node2d(0.00, 0.00, 1, 1,  0))
    nodes.append(f2d.Node2d(1.60, 0.00, 2, 1,  0))
    nodes.append(f2d.Node2d(0.00, 1.50, 3, 0,  0))
    nodes.append(f2d.Node2d(0.80, 1.50, 4, 0, m1))
    nodes.append(f2d.Node2d(1.60, 1.50, 5, 0,  0))
    nodes.append(f2d.Node2d(0.00, 3.10, 6, 0,  0))
    nodes.append(f2d.Node2d(0.80, 3.10, 7, 0, m2))
    nodes.append(f2d.Node2d(1.60, 3.10, 8, 0,  0))

    # define membs -----
    membs = []
    membs.append(f2d.SemiRigidBeam2d(E, G, rho, A1C, I1C, 1, 3, 2.0,  0.0,  1.0, 0.0))  # 1CE
    membs.append(f2d.SemiRigidBeam2d(E, G, rho, A1C, I1C, 2, 5, 2.0,  0.0,  1.0, 0.0))  # 1CW
    membs.append(f2d.SemiRigidBeam2d(E, G, rho, A1B, I1B, 3, 4, 2.4, g1BE,  1.0, 0.0))  # 1BE
    membs.append(f2d.SemiRigidBeam2d(E, G, rho, A1B, I1B, 4, 5, 2.4,  1.0, g1BW, 0.0))  # 1BW
    membs.append(f2d.SemiRigidBeam2d(E, G, rho, A2C, I2C, 3, 6, 2.0,  1.0,  1.0, 0.0))  # 2CE
    membs.append(f2d.SemiRigidBeam2d(E, G, rho, A2C, I2C, 5, 8, 2.0,  1.0,  1.0, 0.0))  # 2CW
    membs.append(f2d.SemiRigidBeam2d(E, G, rho, A2B, I2B, 6, 7, 2.4, g2BE,  1.0, 0.0))  # 2BE
    membs.append(f2d.SemiRigidBeam2d(E, G, rho, A2B, I2B, 7, 8, 2.4,  1.0, g2BW, 0.0))  # 2BW

    # build -----
    frame = f2d.Frame2d()
    frame.make_MK(nodes, membs, False)
    frame.make_H()
    frame.make_B()
    frame.moda()

    # modal frequencies -----
    num_mode = 1
    w_sim = np.sort(frame.omeg)[:num_mode]
    idx_mode = np.argsort(frame.omeg)[:num_mode]

    # static analysis -----
    idx_d = [9, 18]
    idx_r = [2, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23]
    mat_r = frame.B @ frame.Km @ frame.H.T @ frame.U_ful
    d_sim = frame.U_ful[idx_d,:][:,idx_mode]
    r_sim = mat_r[idx_r,:][:,idx_mode] / 1e+6

    # normalize -----
    for n in range(num_mode):
        rat = np.sqrt(np.sum(d_sim[:,n]**2))
        d_sim[:,n] /= rat
        r_sim[:,n] /= rat

    return np.concatenate([w_sim, d_sim[:,0], r_sim[:,0]])