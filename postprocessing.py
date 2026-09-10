import numpy as np
from skimage.filters import threshold_otsu


def normalize_crack_map(v):
    """Convert AT phase field v into a normalized crack map in [0, 1]."""
    crack_map_np = (1.0 - v).detach().cpu().numpy()[0, 0]

    minimum = crack_map_np.min()
    denominator = crack_map_np.max() - minimum

    if denominator <= 1e-8:
        return np.zeros_like(crack_map_np)

    return (crack_map_np - minimum) / denominator


def custom_threshold(crack_map_norm, mode = 99.0):
    """Threshold a normalized crack map with Otsu or a percentile."""
    if isinstance(mode, str):
        mode_string = mode.strip().lower()

        if mode_string == "otsu":
            threshold = threshold_otsu(crack_map_norm)
            return (crack_map_norm > threshold).astype(np.float32)

        try:
            percentile = float(mode_string)
        except ValueError as error:
            raise ValueError(
                "mode must be 'otsu' or a numeric percentile"
            ) from error
    else:
        percentile = float(mode)

    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be between 0 and 100")

    threshold = np.percentile(crack_map_norm, percentile)
    return (crack_map_norm > threshold).astype(np.float32)


def extract_tiles(img_np, tile_size=512, overlap=384):
    if not 0 <= overlap < tile_size:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_size")
        
    H, W, _ = img_np.shape
    # If image is smaller than tile, pad it first
    pad_h = max(0, tile_size - H)
    pad_w = max(0, tile_size - W)

    if pad_h > 0 or pad_w > 0:
        pad_mode = "reflect" if H > 1 and W > 1 else "edge"
        img_np = np.pad(
            img_np,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode=pad_mode
        )

    padded_h, padded_w, _ = img_np.shape
    step = tile_size - overlap
    y_starts = list(range(0, padded_h - tile_size + 1, step))
    x_starts = list(range(0, padded_w - tile_size + 1, step))

    if y_starts[-1] != padded_h - tile_size:
        y_starts.append(padded_h - tile_size)
    if x_starts[-1] != padded_w - tile_size:
        x_starts.append(padded_w - tile_size)

    tiles = []
    for y in y_starts:
        for x in x_starts:
            tile = img_np[y:y + tile_size, x:x + tile_size]
            tiles.append((x, y, tile))
    
    return tiles, H, W, padded_h, padded_w


def stitch_soft_maps(tiles_out, H, W, tile_size=512):
    acc = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)
    for x, y, crack_map in tiles_out:
        acc[y:y + tile_size, x:x + tile_size] += crack_map
        weight[y:y + tile_size, x:x + tile_size] += 1.0
    return acc / np.maximum(weight, 1e-8)


def stitch_binary_maps(tiles_out, H, W, tile_size=512):
    out = np.zeros((H, W), dtype=np.float32)
    for x, y, crack_bin in tiles_out:
        out[y:y + tile_size, x:x + tile_size] = np.maximum(out[y:y + tile_size, x:x + tile_size], crack_bin)
    
    return out


class Postprocessor:
    ''' 
    Postprocessor takes a raw crack v as input (AT perspective: v = 0 means crack, v = 1 means no crack) and 
    returns an inverted and normalized crack_map_norm and an additionally thresholded crack_bin via a previously defined custom threshold function
    where in both cases 0 means no crack, 1 means crack.
    mode is either a percentage or "otsu".
    '''

    def __init__(self, mode):
        self.mode = mode

    def __call__(self, v):
        # normalize to [0,1]
        crack_map_norm = normalize_crack_map(v)

        # thresholding
        crack_bin = custom_threshold(
            crack_map_norm,
            self.mode,
        )

        return crack_map_norm, crack_bin
