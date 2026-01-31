from frame2d import frame2d as f2d
import numpy as np

def smf_3story2bay(theta):

    # G1 (H-500x200x10x16)
    AG1 = 112.3e-04
    IG1 = 46800e-08

    # G2 (H-500x200x10x16)
    AG2 = 112.3e-04
    IG2 = 46800e-08

    # G3 (H-400x200x8x13)
    AG3 = 83.37e-04
    IG3 = 23500e-08

    # C1 (BOX-300x300x19)
    AC1 = 204.3e-04
    IC1 = 26200e-08

    # C2 (BOX-300x300x12)
    AC2 = 134.5e-04
    IC2 = 18300e-08

    # C3 (BOX-300x300x12)
    AC3 = 134.5e-04
    IC3 = 18300e-08

    # material properties -----
    E = 2.05e+11
    G = 7.90e+10
    rho = 7.85e+03

    # mass
    m1 = 180e+03 / 9.8  # kg
    m2 = 270e+03 / 9.8  # kg

    # nodes -----
    nodes = []
    nodes.append(f2d.Node2d( 0.0,  0.00,  1, 1, m1))
    nodes.append(f2d.Node2d( 7.2,  0.00,  2, 1, m2))
    nodes.append(f2d.Node2d(14.4,  0.00,  3, 1, m1))
    nodes.append(f2d.Node2d( 0.0,  3.50,  4, 0, m1))
    nodes.append(f2d.Node2d( 7.2,  3.50,  5, 0, m2))
    nodes.append(f2d.Node2d(14.4,  3.50,  6, 0, m1))
    nodes.append(f2d.Node2d( 0.0,  6.75,  7, 0, m1))
    nodes.append(f2d.Node2d( 7.2,  6.75,  8, 0, m2))
    nodes.append(f2d.Node2d(14.4,  6.75,  9, 0, m1))
    nodes.append(f2d.Node2d( 0.0, 10.00, 10, 0, m1))
    nodes.append(f2d.Node2d( 7.2, 10.00, 11, 0, m2))
    nodes.append(f2d.Node2d(14.4, 10.00, 12, 0, m1))

    # membs -----
    membs = []

    # columns
    membs.append(f2d.SemiRigidBeam2d(E, G, rho, AC1, IC1,  1,  4, 2.0, 1.0, 1.0, 0.0))
    membs.append(f2d.SemiRigidBeam2d(E, G, rho, AC1, IC1,  2,  5, 2.0, 1.0, 1.0, 0.0))
    membs.append(f2d.SemiRigidBeam2d(E, G, rho, AC1, IC1,  3,  6, 2.0, 1.0, 1.0, 0.0))
    membs.append(f2d.SemiRigidBeam2d(E, G, rho, AC2, IC2,  4,  7, 2.0, 1.0, 1.0, 0.0))
    membs.append(f2d.SemiRigidBeam2d(E, G, rho, AC2, IC2,  5,  8, 2.0, 1.0, 1.0, 0.0))
    membs.append(f2d.SemiRigidBeam2d(E, G, rho, AC2, IC2,  6,  9, 2.0, 1.0, 1.0, 0.0))
    membs.append(f2d.SemiRigidBeam2d(E, G, rho, AC3, IC3,  7, 10, 2.0, 1.0, 1.0, 0.0))
    membs.append(f2d.SemiRigidBeam2d(E, G, rho, AC3, IC3,  8, 11, 2.0, 1.0, 1.0, 0.0))
    membs.append(f2d.SemiRigidBeam2d(E, G, rho, AC3, IC3,  9, 12, 2.0, 1.0, 1.0, 0.0))

    # beams
    membs.append(f2d.SemiRigidBeam2d(E, G, rho, AG1, IG1,  4,  5, 2.4, theta[ 0], theta[ 1], 0.0))
    membs.append(f2d.SemiRigidBeam2d(E, G, rho, AG1, IG1,  5,  6, 2.4, theta[ 1], theta[ 2], 0.0))
    membs.append(f2d.SemiRigidBeam2d(E, G, rho, AG2, IG2,  7,  8, 2.4, theta[ 3], theta[ 4], 0.0))
    membs.append(f2d.SemiRigidBeam2d(E, G, rho, AG2, IG2,  8,  9, 2.4, theta[ 4], theta[ 5], 0.0))
    membs.append(f2d.SemiRigidBeam2d(E, G, rho, AG3, IG3, 10, 11, 2.4, theta[ 6], theta[ 7], 0.0))
    membs.append(f2d.SemiRigidBeam2d(E, G, rho, AG3, IG3, 11, 12, 2.4, theta[ 7], theta[ 8], 0.0))

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
    idx_d = [
        12, 21, 30
    ]
    idx_r = [
         1,  2,  4,  5,  7,  8,
        10, 11, 13, 14, 16, 17,
        19, 20, 22, 23, 25, 26,
    ]

    mat_r = frame.B @ frame.Km @ frame.H.T @ frame.U_ful
    d_sim = frame.U_ful[idx_d,:][:,idx_mode]
    r_sim = mat_r[idx_r,:][:,idx_mode] / 1e+6

    # normalize -----
    for n in range(num_mode):
        rat = np.sqrt(np.sum(d_sim[:,n]**2))
        d_sim[:,n] /= rat
        r_sim[:,n] /= rat

    return np.concatenate([w_sim, d_sim[:,0], r_sim[:,0]])