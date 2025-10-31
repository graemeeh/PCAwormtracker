import cv2
import numpy as np
from fbpca import pca # this is slower than sklearn
import pandas as pd
import scipy
from sklearn.utils.extmath import randomized_svd
from threading import Thread # I want to add this at some point
from queue import Queue

FILENAME = "N2.avi"
SCALE = 0.5
MAX_ITER = 10
RANK = 3
TOL=1e-2
BATCH_SIZE = 20

def resizr(image, scale):
    h, w = image.shape[:2]
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

# source for below functions up to/including PCP: https://github.com/fastai/numerical-linear-algebra/blob/master/nbs/3.%20Background%20Removal%20with%20Robust%20PCA.ipynb
def shrink(M, tau):
    S = np.abs(M) - tau
    return np.sign(M) * np.where(S > 0, S, 0)

def _svd(M, rank):
    return randomized_svd(M, min(rank, min(M.shape)), n_iter=2)

def norm_op(M):
    return _svd(M, 1)[1][0]

def svd_reconstruct(M, rank, min_sv):
    u, s, v = _svd(M, rank)
    s -= min_sv
    nnz = (s > 0).sum()
    return u[:, :nnz] @ np.diag(s[:nnz]) @ v[:nnz], nnz

def pcp(X, maxiter, rank, k = 1):
    m, n = X.shape
    lamda = 1 / np.sqrt(m)
    op_norm = norm_op(X)
    Y = np.copy(X) / max(op_norm, np.linalg.norm(X, np.inf) / lamda)
    mu = k*1.25 / op_norm
    rho = k*1.5
    L = np.zeros_like(X)
    d_norm = np.linalg.norm(X, 'fro')
    for i in range(maxiter):
        X2 = X + Y / mu
        S = shrink(X2 - L, lamda / mu)
        L, svp = svd_reconstruct(X2 - S, rank, 1 / mu)
        rank = svp + (1 if svp < rank else round(0.05 * n))
        Z = X - L - S
        Y += mu * Z
        mu *= rho
        err = np.linalg.norm(Z, 'fro') / d_norm
        print(f"err:", err)
        if err < TOL:
            break
    return L, S

def chunks(n, min, thresh):
    while n % min < thresh:
        min += 1
    return min

def worms_track(edges, min: int, max: int, time: int):
    labeled_image, num_labels = scipy.ndimage.label(edges)
    slices = scipy.ndimage.find_objects(labeled_image)  # chops it up into slices based on labels
    contour_data = []
    highlighted = np.zeros_like(edges, dtype=np.uint8)
    for i, slice in enumerate(slices, 1):  # start from 1 because 0 is background
        contour_mask = (labeled_image[slice] == i)
        area = np.sum(contour_mask)
        if min < area < max:
            cx, cy = np.mean(np.column_stack(np.where(contour_mask)), axis=0)
            highlighted[slice][contour_mask] = 255
            contour_data.append({
                'index': i,
                'frame': time,
                'centroid_x': int(cx + slice[1].start),
                'centroid_y': int(cy + slice[0].start),
                'area': area,
                'convex_hull_area': int(scipy.spatial.ConvexHull(np.column_stack(np.where(contour_mask))).volume),
                'height': slice[0].stop - slice[0].start,
                'width': slice[1].stop - slice[1].start})
    return pd.DataFrame(contour_data), highlighted

def vid_setup(c, s):
    ret, frame = c.read()
    n_frames = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
    gray = resizr(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), s)
    h, w = gray.shape[:2]
    return n_frames, h, w

def load_vid(path, scale, batch_size, dtype=np.float32, **pcp_kwargs):
    vid = cv2.VideoCapture(path)
    n_frames, h, w = vid_setup(vid, scale)
    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
    pf = 0
    startframe = 0
    df_l = []
    while pf < n_frames:
        chunk_l = []
        count = 0
        for _ in range(chunks(n_frames, batch_size, 3*batch_size/4)):
            ret, frame = vid.read()
            if not ret:
                break
            chunk_l.append(resizr(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scale).flatten())
            count += 1
        if not chunk_l:
            break
        X = np.array(chunk_l, dtype=dtype).T
        endframe = startframe + count
        print(f"frames {startframe}-{endframe - 1}")
        Lb, Sb = pcp(X, **pcp_kwargs)
        df_l.append(play_vid(np.abs(Sb), h, w, index = startframe, path=path, scale=scale))
        pf += count
        startframe = endframe
    vid.release()

    return pd.concat(df_l)

def play_vid(M, h, w, index, scale, path, title="Video"):
    vid = cv2.VideoCapture(path)
    df_l = []
    for i in range(M.shape[1]):
        frame = M[:, i].reshape((h, w))
        _, frame = cv2.threshold(cv2.resize(np.uint8(cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)), (int(w/scale), int(h/scale))), thresh = 1, type=cv2.THRESH_BINARY, maxval=255)
        df, f = worms_track(frame, min=100, max=1000, time=i)
        df_l.append(df)
        f = cv2.resize(f, (1024, 768))
        vid.set(cv2.CAP_PROP_POS_FRAMES, i+index)
        _, b = vid.read()
        b = cv2.resize(b, (1024, 768))
        f = cv2.merge((f, np.zeros_like(f), f))
        cv2.imshow(title, cv2.addWeighted(b, 0.5, f, 0.9, 0))
        if cv2.waitKey(300) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    return pd.concat(df_l)

df = load_vid(FILENAME, scale=SCALE, batch_size=BATCH_SIZE, maxiter=MAX_ITER, rank=RANK)
df.to_csv(FILENAME+'_worms.csv')
