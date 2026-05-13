import numpy as np
import matplotlib.pyplot as plt

def downsample_images(dataset):
    """
    Reducing images from 28x28 to 14x14 using block averaging.
    Input: dataset (nimg, 784)
    Output: dataset_reduced (nimg, 196)
    """
    nimg = dataset.shape[0]
    imgs = dataset.reshape(nimg, 28, 28)
    
    # Block mean: trasforming (nimg, 14, 2, 14, 2) and averaging over 2x2 blocks
    reduced = imgs.reshape(nimg, 14, 2, 14, 2).mean(axis=(2, 4))
    
    return reduced.reshape(nimg, 196)

def get_active_pixels(dataset,threshold=0.05):

    vars=np.var(dataset,axis=0)
    active_idxs=np.where(vars > threshold)[0]
    plt.imshow((vars > 0.01).reshape(14, 14), cmap='gray')
    plt.show()

    return active_idxs.min(), active_idxs.max()


def mappixels(img):
    """
    Mapping pixels:
    phi(x) = [cos(pi/2 * x), sin(pi/2 * x)]
    """
    phi0 = np.cos(np.pi / 2 * img)
    phi1 = np.sin(np.pi / 2 * img)
    
    return np.stack((phi0, phi1), axis=1)

def onehot(y, numclass):
    """One-hot encoding del label."""
    vecy = np.zeros(numclass)
    vecy[y] = 1.0
    return vecy

def canonize_chain(chain, L, initpos):
    """
    Shifts the orthogonality center to 'initpos' using QR decomposition.
    Left-sites (i < initpos) become left-isometries; right-sites (i > initpos) 
    become right-isometries. Residual norms are absorbed into the center.
"""
    # Left-side chain
    for i in range(0, initpos):
        dL, dP, dR = chain[i].shape[:3]
        mat = chain[i].reshape(dL * dP, dR) #(2M,M)
        #QR factorization (Q^t * Q = 1)
        Q, R = np.linalg.qr(mat) # Q(2M, M); R(M,M)
    
        # Normalizing: Avoids norm decay
        #R_norm = np.linalg.norm(R)
        #if R_norm > 0:
         #   R /= R_norm 

        chain[i] = Q.reshape(dL, dP, -1)
        if chain[i+1].ndim == 3:
            chain[i+1] = np.einsum('ab, bcd -> acd', R, chain[i+1])
        else:
            chain[i+1] = np.einsum('ab, bcdl -> acdl', R, chain[i+1])

    # Right-side chain
    for i in range(L - 1, initpos, -1):
        dL, dP, dR = chain[i].shape[:3]
        mat = chain[i].reshape(dL, dP * dR).T
        Q, R = np.linalg.qr(mat)
    
        # Normalizing : avoids norm decay
        #R_norm = np.linalg.norm(R)
        #if R_norm > 0:
         #   R /= R_norm

        chain[i] = Q.T.reshape(-1, dP, dR)
        RT = R.T
        if chain[i-1].ndim == 3:
            chain[i-1] = np.einsum('abc, cd -> abd', chain[i-1], RT)
        else:
            chain[i-1] = np.einsum('abcl, cd -> abdl', chain[i-1], RT)


    total_norm = np.linalg.norm(chain[initpos])
    gain = 250.0 # calibrate on test (logits ~1)
    
    if total_norm > 0:
        chain[initpos] = (chain[initpos] / total_norm) * gain

def snake_flatten(img_1d):
    # Reshape 2D, applies snake flattening and back to 1D
    img = img_1d.reshape(14, 14).copy()
    img[1::2, :] = img[1::2, ::-1] 
    return img.flatten()