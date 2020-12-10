
def save_vol(fn, vol):
    import mrcfile

    f = mrcfile.new(fn, overwrite=True)
    f.set_data(vol)
    f.close()


def paste_slices(tup):
    pos, w, max_w = tup
    wall_min = max(pos, 0)
    wall_max = min(pos+w, max_w)
    block_min = -min(pos, 0)
    block_max = max_w-max(pos+w, max_w)
    block_max = block_max if block_max != 0 else None
    
    return slice(wall_min, wall_max), slice(block_min, block_max)

def paste(wall, block, loc):
    loc_zip = zip(loc, block.shape, wall.shape)
    wall_slices, block_slices = zip(*map(paste_slices, loc_zip))
    wall[wall_slices] += block[block_slices]


def split_vol(vol, n=[8, 8, 4]):
    import numpy as np

    ##pad to square
    # shape_vec = np.array(vol.shape)
    # pad_vec = -shape_vec + shape_vec.max()
    # vol = np.pad(vol, ((0, pad_vec[0]), (0, pad_vec[1]), (0, pad_vec[2])), 'constant')

    ##########
    #paste center
    # shape_vec = np.array(vol.shape)
    # wall = np.zeros((shape_vec.max(), shape_vec.max(), shape_vec.max())).astype(np.float32)
    # offset = (np.array(wall.shape) - shape_vec)/2
    # paste(wall, vol, (int(offset[0]), int(offset[1]), int(offset[2])))
    # vol = wall


    x_split = np.array_split(vol, n[0], axis=0)
    y_split = []
    for sv in x_split:
        y_split.extend(np.array_split(sv, n[1], axis=1))

    z_split = []
    for sv in y_split:
        z_split.extend(np.array_split(sv, n[2], axis=2))

    #for sv in z_split:
    #   print('shape: ' + str(sv.shape))

    return z_split

def merge_vol(vol_list, n=[8, 8, 4]):
    import numpy as np

    merge_n = int(len(vol_list)/(n[0]*n[1]))
    z_merge = []
    for i in range(n[0]*n[1]):
        sub_list = vol_list[i*merge_n:(i+1)*merge_n]
        z_merge.append(np.concatenate(sub_list, axis=2))

    merge_n = int(len(z_merge)/n[1])
    y_merge = []
    for i in range(n[1]):
        sub_list = z_merge[i*merge_n:(i+1)*merge_n]
        y_merge.append(np.concatenate(sub_list, axis=1))

    full_vol = np.concatenate(y_merge, axis=0)

    return full_vol

    