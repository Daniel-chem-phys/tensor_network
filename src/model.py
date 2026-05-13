import numpy as np

def init_single_tensor(dL, dP, dR, num_class=None):
    shape = (dL, dP, dR) if num_class is None else (dL, dP, dR, num_class)
    tensor = np.zeros(shape)
    
    # |0 > -> (1, 0) (Black pixel)
    # If black, info passes trough inaltered
    for i in range(min(dL, dR)):
        if num_class is None:
            tensor[i, 0, i] = 1.0  # Identity on black channel
        else:
            for s in range(num_class): # same probability on each class
                tensor[i, 0, i, s] = 1.0 / np.sqrt(num_class)
    
    # Small noise only to break simmetry on white pixels (index 1)
    tensor += np.random.normal(0, 0.001, tensor.shape)
    return tensor


def contr_lr_tens(chain, vec, vec_l, act_pos):

    """
    Contracting MPS with img vec..
    Only chain[act_pos] has 4 indexes (L, P, R, S).
    Other sites have 3 indexes (L, P, R).
    """
    # === Left chain (0 -> act_pos-1) ===

    l_vec = np.einsum('lpr, p -> lr', chain[0], vec[0]).reshape(-1) 

    # Initial normalizing
    norm = np.linalg.norm(l_vec)
    if norm > 1e-16:
        l_vec /= norm

    for m in range(1, act_pos):
        # Contracting site m: (M_in, P, M_out) with pixel P
        matrix_m = np.einsum('lpr, p -> lr', chain[m], vec[m])
        # Matrix/matrix mul to move forward
        l_vec = l_vec @ matrix_m

        norm = np.linalg.norm(l_vec)
        if norm > 1e-16:
            l_vec /= norm

    # === Right chain (L-1 -> act_pos+1) ===

    r_vec = np.einsum('lpr, p -> lr', chain[vec_l-1], vec[vec_l-1]).reshape(-1)

    norm = np.linalg.norm(r_vec)
    if norm > 1e-16:
        r_vec /= norm

    for m in range(vec_l-2, act_pos, -1):
        matrix_m = np.einsum('lpr, p -> lr', chain[m], vec[m])
        r_vec = matrix_m @ r_vec

        norm = np.linalg.norm(r_vec)
        if norm > 1e-16:
            r_vec /= norm

    return l_vec, r_vec

def contrfin(lvec, rvec, tcentre, pcentre):
    
    # Contraction: (L) * (P) * (R) * (L, P, R, S) -> (S)
    pred_vec = np.einsum('l, p, r, lprs -> s', lvec, pcentre, rvec, tcentre)
    
    pred_shifted = pred_vec - np.max(pred_vec) # all logits=<0

    #softmax
    pred_exp = np.exp(pred_shifted) # all logits in [0,1]
    soft_vec = pred_exp / np.sum(pred_exp) # sum to 1 (-->probabilities) 
    
    return soft_vec, pred_vec
  

def sweeping(actpos, direction, imin, imax):
    """
    Sweeping:
    Moving active site b & f between imin, imax.
    """
    newpos = actpos + direction
    if newpos >= imax:
        newpos = imax - 1
        direction = -1
    elif newpos <= imin:
        newpos = imin + 1
        direction = 1
    
    return newpos, direction

def shift_label(chain, c_index, direction, M, imin, imax, i):
    if direction == 1: # Right sweep (moving label to right)
        if c_index >= imax: return chain

        dL, dP, dR, dS = chain[c_index].shape
        
        #SVD: truncating to M-dim correlation space: discarding low correlations
        # S: singolar values vector (descending order): importance of channel correlations
        # U,V left/right-label site orthonormal basis
        A_2d = chain[c_index].transpose(0, 1, 3, 2).reshape(dL*dP, dS*dR)
        U, S, V = np.linalg.svd(A_2d, full_matrices=False) 

        limit = min(len(S), M)

        U = U[:, :limit]; S = S[:limit]; V = V[:limit, :]
       
        chain[c_index] = U.reshape(dL, dP, limit)
        M_mat = (np.diag(S) @ V).reshape(limit, dS, dR)
        
        chain[c_index+1] = np.einsum('lsr, rpk -> lprs', M_mat, chain[c_index+1])

    else: # Left sweep (moving label to left)
        if c_index <= imin: return chain
        dL, dP, dR, dS = chain[c_index].shape
        
        A_2d = chain[c_index].transpose(0, 3, 1, 2).reshape(dL*dS, dP*dR)
        U, S, V = np.linalg.svd(A_2d, full_matrices=False)
        
        limit = min(len(S), M)
         
        U = U[:, :limit]; S = S[:limit]; V = V[:limit, :]
        
        chain[c_index] = V.reshape(limit, dP, dR)
        M_mat = (U @ np.diag(S)).reshape(dL, dS, limit)
        
        chain[c_index-1] = np.einsum('lpk, ksr -> lprs', chain[c_index-1], M_mat)

    return chain

