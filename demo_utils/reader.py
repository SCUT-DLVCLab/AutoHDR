"""
Merge characters into lines based on heuritic rules
"""
import numpy as np
from shapely.geometry import Polygon
from demo_utils.utils_reader import calc_poly2, add_array, point_inside_quad
import cv2



def argsort(x):
    d = dict(enumerate(x))
    r = dict(sorted(d.items(), key=lambda x:x[1]))
    return list(r.keys())

def cat(list1, list2, list3):
    out = []
    for i in range(list1.shape[0]):
        temp1 = list1[i].astype('str').tolist()
        temp2 = list2[i].astype('str')
        temp3 = list3[i].astype('str')
        temp1.append(temp2)
        temp1.append(temp3)
        out.append(temp1)
    return out

def read_coors(coor_chars):
    coors = np.array(coor_chars)[:,:4]
    coors = coors.astype('float64')
    return coors

def read_chars(coor_chars):
    chars = coor_chars[:,4]
    return chars


def inside(chars, scores, coors, lines, img_shape):
    hor_lines, ver_lines = lines
    h, w = img_shape
    # sort hor_lines->top to bottom, sort ver_lines->right to left
    if len(hor_lines):
        tmp = [val[0][1]+val[1][1] for val in hor_lines]
        index = argsort(tmp)
        sort_hor_lines = [hor_lines[val] for val in index]
        hor_lines = sort_hor_lines
    hor_lines.insert(0, [(0,0), (w,0)])
    hor_lines.append([(0,h), (w,h)])
    if len(ver_lines):
        tmp = [-(val[0][0]+val[1][0]) for val in ver_lines]
        index = argsort(tmp)
        sort_ver_lines = [ver_lines[val] for val in index]
        ver_lines = sort_ver_lines
    ver_lines.insert(0, [(w,0), (w,h)])
    ver_lines.append([(0,0), (0,h)])

    regions_chars = []; regions_scores = [];  regions_coors = []
    regions_idx = []


    for id1 in range(len(hor_lines)-1):
        for id2 in range(len(ver_lines)-1):
            # clockwise lines
            # lines = [hor_lines[id1], ver_lines[id2], hor_lines[id1+1], ver_lines[id2+1]]
            # region_idx = pt_in_rect(lines, np.array(coors)[:, [0,1,2,1,2,3,0,3]],)
            hl = [hor_lines[id1], hor_lines[id1+1]]
            vl = [ver_lines[id2], ver_lines[id2+1]]

            region_idx = point_inside_quad(np.array(coors), hl, vl)
            
            regions_idx.append(region_idx)

    return regions_idx




def sort_coor(region_coors, h, w):

    # !!! Trick: first sort the characters using width
    index = (-(region_coors[:,2]-region_coors[:,0])).argsort()
    coors = region_coors[index]

    # 1. coarse result
    merge_coor_chars_1 = merge_1(coors)
    merge_polys_1 = []
    for val in merge_coor_chars_1:
        col_coor = val[:,:4].astype(np.int16)
        poly = calc_poly2(col_coor, img_shape=(h, w))
        merge_polys_1.append(poly)


    # 2. merge region
    merge_coor_chars_2, merge_polys_2 = merge_coor_chars_1, merge_polys_1
    for i in range(5):
        update_coor_chars_2 = merge_2(merge_coor_chars_2, merge_polys_2)
        update_polys_2 = []
        for val in update_coor_chars_2:
            col_coor = val[:,:4].astype(np.int16)
            poly = calc_poly2(col_coor, img_shape=(h, w))
            update_polys_2.append(poly)
        if len(update_coor_chars_2)==len(merge_coor_chars_2):
            break
        merge_coor_chars_2, merge_polys_2 = update_coor_chars_2, update_polys_2
    merge_coor_chars_2, merge_polys_2 = update_coor_chars_2, update_polys_2

    # visualize
    # vis = overlay_mask(vis_img.copy(), merge_polys_2)
    # cv2.imwrite('1.jpg', vis)

    # 3. merge inside region
    merge_coor_chars_3 = merge_3(merge_coor_chars_2, merge_polys_2)
    merge_polys_3 = []
    for val in merge_coor_chars_3:
        col_coor = val[:,:4].astype(np.int16)
        poly = calc_poly2(col_coor, img_shape=(h, w))
        merge_polys_3.append(poly)

    # visualize
    # vis = overlay_mask(vis_img.copy(), merge_polys_3)
    # cv2.imwrite('2.jpg', vis)

    # reading from output
    region_cols, region_each_cols = read_col(merge_coor_chars_3, merge_polys_3, img_shape=(h, w))

    return region_cols, region_each_cols

def merge_1(coor_chars, thresh=10):
    req1 = lambda cur_l,  xmin, thresh:abs(cur_l-xmin)<=thresh
    req2 = lambda cur_r,  xmax, thresh:abs(cur_r-xmax)<=thresh
    req3 = lambda cur_t,  ymax, thresh:abs(cur_t-ymax)<=thresh
    req4 = lambda cur_bt, ymin, thresh:abs(cur_bt-ymin)<=thresh

    coors = read_coors(coor_chars)
    num = len(coor_chars)
    mark = dict(zip(range(num), [False]*num))

    merge_coor_chars = []
    for i in range(num):
        if mark[i]:     continue
        mark[i] = True
        same_col = []
        same_col.append(coor_chars[i])

        adding = True
        while adding:
            adding = False
            for j in range(num):
                if mark[j]:     continue
                same_col_coors = read_coors(same_col)
                xmin = min(same_col_coors[:,0])
                xmax = max(same_col_coors[:,2])
                ymin = min(same_col_coors[:,1])
                ymax = max(same_col_coors[:,3])
                cur_l, cur_t, cur_r, cur_bt = coors[j][0], coors[j][1], coors[j][2], coors[j][3]
                if (req1(cur_l,xmin,thresh) or req2(cur_r,xmax,thresh)) \
                and (req3(cur_t,ymax,thresh) or req4(cur_bt,ymin,thresh)):
                    same_col.append(coor_chars[j])
                    mark[j] = True
                    adding  = True
        merge_coor_chars.append(np.array(same_col))
    return merge_coor_chars

def merge_2(region_coor_chars, region_polys):
    # merge poly
    def req1(poly1, poly2):
        # poly1: bottom edge;  poly2: top edge
        inter = max(0, min(poly1[4], poly2[2])-max(poly1[6], poly2[0]))
        l1 = poly1[4]-poly1[6]
        l2 = poly2[2]-poly2[0]
        inside = max(inter/l1, inter/l2)
        flag1 = True if inside>0.4 else False

        # poly1: top edge;  poly2: bottom edge
        inter = max(0, min(poly1[2], poly2[4])-max(poly1[0], poly2[6]))
        l1 = poly1[2]-poly1[0]
        l2 = poly2[4]-poly2[6]
        inside = max(inter/l1, inter/l2)
        flag2 = True if inside>0.4 else False
        return flag1 or flag2

    def req2(poly1, poly2):
        flag1 = abs(poly1[1]-poly2[5])<300
        flag2 = abs(poly1[5]-poly2[1])<300
        return flag1 or flag2

    polys = region_polys
    inter = []
    for idx1 in range(len(polys)):
        cur = [idx1]
        for idx2 in range(idx1+1, len(polys)):
            if req1(polys[idx1], polys[idx2]) and req2(polys[idx1], polys[idx2]):
                cur.append(idx2)
        inter.append(cur)
    merge_coor_chars = func(inter, region_coor_chars)
    return merge_coor_chars

def merge_3(region_coor_chars, region_polys):
    def req1(poly1, poly2):
        # poly1: bottom edge;  poly2: top edge
        inter = max(0, min(poly1[4], poly2[2])-max(poly1[6], poly2[0]))
        l1 = poly1[4]-poly1[6]
        l2 = poly2[2]-poly2[0]
        inside = max(inter/l1, inter/l2)
        flag1 = True if inside>0.7 else False

        # poly1: top edge;  poly2: bottom edge
        inter = max(0, min(poly1[2], poly2[4])-max(poly1[0], poly2[6]))
        l1 = poly1[2]-poly1[0]
        l2 = poly2[4]-poly2[6]
        inside = max(inter/l1, inter/l2)
        flag2 = True if inside>0.7 else False
        return flag1 or flag2

    def req2(poly1, poly2):
        poly1 = Polygon(poly1.reshape(4,2)).convex_hull
        poly2 = Polygon(poly2.reshape(4,2)).convex_hull
        inter_area = poly1.intersection(poly2).area
        union_area = poly1.area + poly2.area - inter_area
        inside = max(inter_area/poly1.area, inter_area/poly2.area)
        flag1 = True if inside>0.3 else False   # inside polygon

        ver_ovr = inter_area/min(poly1.area, poly2.area)  # vertical overlap
        flag2 = True if ver_ovr>0.1 else False
        return flag1 or flag2

    polys = region_polys
    inter = []
    for idx1 in range(len(polys)):
        cur = [idx1]
        for idx2 in range(idx1+1, len(polys)):
            if req1(polys[idx1], polys[idx2]) and req2(polys[idx1], polys[idx2]):
                cur.append(idx2)
        inter.append(cur)
    merge_coor_chars = func(inter, region_coor_chars)
    return merge_coor_chars


def func(inter, region_coor_chars):
    # 将有相同元素的lists组在一个集合里;
    while True:
        res = []
        mark = [False] * len(inter)
        for idx1 in range(len(inter)):
            if mark[idx1]:  continue
            mark[idx1] = True
            cur = inter[idx1]
            for idx2 in range(idx1+1, len(inter)):
                if mark[idx2]:  continue
                for v in inter[idx2]:
                    if v in cur:
                        cur.extend(inter[idx2])
                        mark[idx2] = True
                        break
            res.append(list(set(cur)))
        num1 = sum([len(val) for val in res])
        sig_li = []
        for val in res:
            sig_li.extend(val)
        sig_li = list(set(sig_li))
        if len(sig_li)==num1:
            break

        inter = res


    merge_coor_chars = []
    for v1 in res:
        cur_chars = []
        for v in v1:
            cur_chars.append(region_coor_chars[v])

        cur_chars = np.vstack(cur_chars)
        coors = np.vstack([read_coors(v) if len(v.shape)!=1 else read_coors(v[np.newaxis, :]) for v in cur_chars])
        temp, idx = np.unique(coors, axis=0, return_index=True)
        merge_coor_chars.append(cur_chars[idx])
    return merge_coor_chars



def read_col(region_coor_chars, region_polys, img_shape):
    def req_1(cur_coor, width, ct_x):
        flag1 = (cur_coor[2]-cur_coor[0])/width>0.6
        flag2 = (cur_coor[0]<=ct_x) and (cur_coor[2]>=ct_x)
        flag3 = 2/5< (ct_x-cur_coor[0])/(cur_coor[2]-ct_x+1e-3) <5/2
        flag4 = (cur_coor[2]-cur_coor[0])/width>=0.45
        flag = flag1 or (flag2 and flag3 and flag4)
        return flag

    xmin = [min(read_coors(val)[:,0]) for val in region_coor_chars]
    index  = np.argsort(xmin)[::-1].tolist()
    sort_coor_chars = np.array(region_coor_chars, dtype=object)[index]
    sort_polys = np.array(region_polys, dtype=object)[index]
    # sort from top to bottom
    sort_coor_chars_2 = []
    for coor_char in sort_coor_chars:
        coors = read_coors(coor_char)
        dd = coors[:,1].argsort()
        sort_coor_chars_2.append(coor_char[dd])
    sort_coor_chars = sort_coor_chars_2

    region_cols, region_each_cols = [], []
    '''
    region_cols: list of ordered output from each line
    region_each_cols: list of each line segments, which distinguish small segments
    '''
    nums = len(region_polys)
    for i in range(nums):
        cur_coor_chars  = sort_coor_chars[i]
        width = sort_polys[i][2]-sort_polys[i][0]
        pt_1, pt_2, pt_3, pt_4 = sort_polys[i].reshape(-1,2)

        flag_sm = False
        col_sm, col_large, col_chars, line_lists = [], [], [], []
        '''
        col_sm:     tmp lists storing small chars
        col_large:  tmp lists storing large chars
        col_chars:  list storing ordered chars from each line
        line_lists: list storing column segments
        '''
        for j in range(cur_coor_chars.shape[0]):
            cur = cur_coor_chars[j]
            cur_coor = cur[:4].astype(np.int16)
            if j!=cur_coor_chars.shape[0]-1:
                next_coor = cur_coor_chars[j+1][:4].astype(np.int16)
            else:
                next_coor = np.array([])

            # line equation: (x-x1)/(x2-x1) = (y-y1)/(y2-y1)
            ct_x = (cur_coor[1]-pt_1[1])/(pt_4[1]-pt_1[1]+1e-4)*(pt_4[0]-pt_1[0])+pt_1[0]+width/2
            ct_x = int(ct_x)

            if req_1(cur_coor, width, ct_x) or \
              (len(next_coor)!=0 and len(col_sm)==0 and req_1(next_coor, width, ct_x)) or (len(next_coor)==0 and len(col_sm)==0):
                if len(col_sm)!=0:
                    col_sm, col_chars = output_sm_char(col_sm, col_chars, img_shape)
                    line_lists = add_array(col_sm, line_lists)

                    col_sm = []
                    col_chars.append(cur)
                    col_large.append(cur)
                else:
                    col_chars.append(cur)
                    col_large.append(cur)
            else:
                flag_sm = True
                if len(col_large):
                    line_lists.append(np.array(col_large))
                    col_large = []
                col_sm.append(cur)
                if j==cur_coor_chars.shape[0]-1:
                    col_sm, col_chars = output_sm_char(col_sm, col_chars, img_shape)
                    line_lists = add_array(col_sm, line_lists)

        if len(col_large):
            line_lists.append(np.array(col_large))

        region_cols.append(np.vstack(col_chars))
        region_each_cols.append(line_lists)
    return region_cols, region_each_cols


def output_sm_char(col_sm, col_chars, img_shape):
    col_sm = np.array(col_sm)
    mark = dict(zip(range(col_sm.shape[0]), [False]*col_sm.shape[0]))
    out = []
    for i in range(col_sm.shape[0]):
        if mark[i]: continue
        mark[i] = True
        temp = []
        temp.append(col_sm[i])
        mark, same_col = find_same_col_v2(col_sm, temp, mark, img_shape, thresh=5)
        out.append(np.array(same_col))

    poly_region = []
    for col in out:
        coors = col[:,:4].astype(np.int16)
        pts = calc_poly2(coors, img_shape)    # [l,t,r,t,r,bt,l,bt]
        poly_region.append(pts)

    out = merge_4(out, poly_region, img_shape)

    # sort from right->left(for two small columns)
    xmin = [min(read_coors(col)[:,0]) for col in out]
    index = sorted(range(len(xmin)), key=lambda k: xmin[k])[::-1]
    out = np.array(out, dtype=object)[index]
    # sort from top->bottom
    sort_out_2 = []
    for i in range(len(out)):
        dd = read_coors(out[i])[:,1].argsort()
        sort_out_2.append(out[i][dd])
    out = sort_out_2

    # adding
    for i in range(len(out)):
        for j in range(out[i].shape[0]):
            col_chars.append(out[i][j])
    return out, col_chars

def find_same_col_v2(search_, same_col, mark, img_shape, thresh):
    """
    only consider x direction
    """
    coors = read_coors(search_)
    adding = True
    req1 = lambda array, xmin, thresh:abs(array[0]-xmin)<=thresh
    req2 = lambda array, xmax, thresh:abs(array[2]-xmax)<=thresh
    while adding:
        adding = False
        for i in range(len(search_)):
            if not mark[i]:
                same_col_coors = read_coors(same_col)
                xmin = min(same_col_coors[:,0])
                xmax = max(same_col_coors[:,2])
                ymin = min(same_col_coors[:,1])
                ymax = max(same_col_coors[:,3])

                if (req1(coors[i],xmin,thresh) or req2(coors[i],xmax,thresh)):
                    same_col.append(search_[i])
                    mark[i] = True
                    adding = True
    return mark, same_col

def merge_4(region_coor_chars, poly_region, img_shape, vis_img=None):
    def req1(array1, array2, bp=False):
        # array1: bottom edge;  array2: top edge
        inter = max(0, min(array1[4], array2[2])-max(array1[6], array2[0]))
        l1 = array1[4]-array1[6]
        l2 = array2[2]-array2[0]
        inside = max(inter/l1, inter/l2)
        flag1 = True if inside>0.7 else False

        # array1: top edge;  array2: bottom edge
        inter = max(0, min(array1[2], array2[4])-max(array1[0], array2[6]))
        l1 = array1[2]-array1[0]
        l2 = array2[4]-array2[6]
        inside = max(inter/l1, inter/l2)
        flag2 = True if inside>0.7 else False
        return flag1 or flag2

    def req2(array1, array2):
        poly1 = Polygon(array1.reshape(4,2)).convex_hull
        poly2 = Polygon(array2.reshape(4,2)).convex_hull
        inter_area = poly1.intersection(poly2).area
        union_area = poly1.area + poly2.area - inter_area
        inside = max(inter_area/poly1.area, inter_area/poly2.area)
        flag1 = True if inside>0.3 else False

        iou = inter_area/union_area
        flag2 = True if iou>0.2 else False
        return flag1 or flag2

    inter = []
    polys = poly_region
    for idx1 in range(len(polys)):
        cur = [idx1]
        for idx2 in range(idx1+1, len(polys)):
            if req1(polys[idx1], polys[idx2]) or req2(polys[idx2], polys[idx1]):\
                cur.append(idx2)
        inter.append(cur)

    merge_coor_chars = func(inter, region_coor_chars)
    return merge_coor_chars



def get_sort(coors, h, w):
    regions_coors = np.array(coors)
    _, page_each_cols = sort_coor(regions_coors, h, w)
    out_box_li = []
    for cols in page_each_cols:
        for col in cols:
            col = col[:, :5].tolist()
            out_box_li.extend(col)
    return out_box_li

def arrange_single(imp, coors):
    h, w = cv2.imread(imp).shape[:2]
    out_box_li = get_sort(coors, h, w)
    return out_box_li


if __name__ == '__main__':
    imp = '/home/zyy/dec_reg/29.明代刻經_343_inverted.png'
    txtp = '/home/zyy/dec_reg/29.明代刻經_343_inverted.txt'

    h, w = cv2.imread(imp).shape[:2]
    rf = open(txtp).read().splitlines()
    coors = []
    for line in rf:
        x1, y1, x2, y2 = [round(float(k)) for k in line.split(',')[:4]]
        coors.append([x1, y1, x2, y2])
    import pdb; pdb.set_trace()
    out_box_li = get_sort(coors, h, w)
    wf = open('/home/zyy/dec_reg/29.明代刻經_343_inverted_arranged.txt', 'w')
    for box in out_box_li:
        wf.write(','.join(map(str, box)) + '\n')