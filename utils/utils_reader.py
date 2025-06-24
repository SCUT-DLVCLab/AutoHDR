import numpy as np
import cv2
from shapely.geometry import Polygon, MultiPoint 

def read_coors(coor_chars):
    coors = np.array(coor_chars)[:,:4]
    coors = coors.astype('float64')
    return coors

def add_array(coor_chars, lists):
    out = lists
    for i in range(len(coor_chars)):
        out.append(coor_chars[i])
    return out

def argsort(x):
    d = dict(enumerate(x))
    r = dict(sorted(d.items(), key=lambda x:x[1]))
    return list(r.keys())

def order_pt(array):
    sum_ = np.sum(array, 0)
    centroid = sum_ / 4
    theta = np.arctan2((array[:,1]-centroid[1]), (array[:,0]-centroid[0]))
    sort_pts = array[np.argsort(theta)]
    return sort_pts

def calc_poly(coors, img_shape):
    h, w = img_shape
    mask = np.zeros((h,w))
    cv2.fillPoly(mask, [coors[:,[0,1,2,1,2,3,0,3]].astype(np.int32).reshape(-1,1,2)], color=(1,))
    # for val in coors:
    #     pts = val[[0,1,2,1,2,3,0,3]].astype(np.int32)
    #     cv2.fillPoly(mask, [pts.reshape(-1,1,2)], color=(1,))
    points = np.where(mask==1)
    points = np.concatenate([points[1][:, np.newaxis], points[0][:, np.newaxis]], 1)
    rect = cv2.minAreaRect(np.array([points]))
    box = np.int0(cv2.boxPoints(rect))
    poly = order_pt(box).reshape(-1)
    return poly

def calc_poly2(coors, img_shape):
    # resize for speed up computation
    s = 4 
    h, w = img_shape
    mask = np.zeros( (int(h/s), int(w/s)) )
    cv2.fillPoly(mask, [(coors[:,[0,1,2,1,2,3,0,3]]/s).astype(np.int32).reshape(-1,1,2)], color=(1,))
    points = np.where(mask==1)
    points = np.concatenate([points[1][:, np.newaxis], points[0][:, np.newaxis]], 1)
    rect = cv2.minAreaRect(np.array([points]))
    box = np.int0(cv2.boxPoints(rect))
    poly = order_pt(box).reshape(-1)
    return poly*s

def inside_nms(coors, probs):
    areas = (coors[:,3]-coors[:,1])*(coors[:,2]-coors[:,0])
    order = np.argsort(areas)[::-1]
    temp = []
    thresh = 0.5
    print(coors.shape[0])
    while len(order)>0:
        i = order[0]
        temp.append(i)
        xx1 = np.maximum(coors[i][0].reshape(-1), coors[order[1:]][:,0])
        yy1 = np.maximum(coors[i][1].reshape(-1), coors[order[1:]][:,1])
        xx2 = np.minimum(coors[i][2].reshape(-1), coors[order[1:]][:,2])
        yy2 = np.minimum(coors[i][3].reshape(-1), coors[order[1:]][:,3])
        w = np.maximum(0, (xx2-xx1))
        h = np.maximum(0, (yy2-yy1))
        inter = np.maximum(0., w*h)
        inside_iou = inter/(np.minimum(areas[i], areas[order[1:]]))
        inds = np.where(inside_iou <= thresh)[0]
        order = order[inds + 1]
    print("len temp: ", len(temp))
    import pdb; pdb.set_trace()

    return coors, probs

def nms_combine_det(detc_output, pagenet_output, thresh=0.5):
    '''
    combine detectron2 and pagenet output
    以pageNet输出排第一，过滤掉detectron2中已经在pageNet中有输出的部分
    dets1: detectron2 output;  dets2: pagenet output
    '''
    dets1 = detc_output[:, 1:]
    dets2 = pagenet_output[:, 1:]

    x1_1 = dets1[:, 0] - dets1[:, 2] / 2
    y1_1 = dets1[:, 1] - dets1[:, 3] / 2
    x1_2 = dets1[:, 0] + dets1[:, 2] / 2
    y1_2 = dets1[:, 1] + dets1[:, 3] / 2
    dets1_temp = np.concatenate((x1_1.reshape(-1,1), y1_1.reshape(-1,1), x1_2.reshape(-1,1), y1_2.reshape(-1,1)), axis=1)

    x2_1 = dets2[:, 0] - dets2[:, 2] / 2
    y2_1 = dets2[:, 1] - dets2[:, 3] / 2
    x2_2 = dets2[:, 0] + dets2[:, 2] / 2
    y2_2 = dets2[:, 1] + dets2[:, 3] / 2
    dets2_temp = np.concatenate((x2_1.reshape(-1,1), y2_1.reshape(-1,1), x2_2.reshape(-1,1), y2_2.reshape(-1,1)), axis=1)
    keep = []

    for idx,det1 in enumerate(dets1_temp):
        repeat_det1 = np.repeat(det1[np.newaxis, :], dets2_temp.shape[0], axis=0)
        max_l =  np.max(np.hstack((repeat_det1[:,0].reshape(-1,1), dets2_temp[:,0].reshape(-1,1))), axis=1)
        min_r =  np.min(np.hstack((repeat_det1[:,2].reshape(-1,1), dets2_temp[:,2].reshape(-1,1))), axis=1)
        max_t =  np.max(np.hstack((repeat_det1[:,1].reshape(-1,1), dets2_temp[:,1].reshape(-1,1))), axis=1)
        min_bt = np.min(np.hstack((repeat_det1[:,3].reshape(-1,1), dets2_temp[:,3].reshape(-1,1))), axis=1)
        area1 = int((det1[2]-det1[0])*(det1[3]-det1[1]))
        area1 = np.repeat(np.array([area1])[np.newaxis, :], dets2_temp.shape[0], axis=0)[:,0]
        area2 = (dets2_temp[:,2]-dets2_temp[:,0])*(dets2_temp[:,3]-dets2_temp[:,1])

        w = np.max(np.hstack((np.zeros((dets2_temp.shape[0],1)),(min_r-max_l+1).reshape(-1,1))), axis=1)
        h = np.max(np.hstack((np.zeros((dets2_temp.shape[0],1)),(min_bt-max_t+1).reshape(-1,1))), axis=1)
        inter = w * h
        union = area1+area2-inter
        inter_iou  = inter/union
        inside_iou = inter/np.minimum(area1, area2)
        judge1 = len(np.where(inter_iou>thresh)[0])
        judge2 = len(np.where(inside_iou>thresh)[0])
        if (not judge1) and (not judge2):
            keep.append(idx)
    keep_dets1 = detc_output[keep]
    concat_output = np.vstack((pagenet_output, keep_dets1,))
    return concat_output

def refine_recog_result(char_dict, page_cols):
    # 字符间有较大间距则添加空格
    output = []
    for regions in page_cols:
        for lines in regions:
            char_hs = (lines[:,3]-lines[:,1])
            avg_dist = np.sum(char_hs)/len(char_hs)
            recog_res = []
            for idx, val in enumerate(lines):
                l, t, r, bt, cls_ = val
                if idx>0 and (t-lines[idx-1][3]>1.5*avg_dist):
                    recog_res.append(' ')
                recog_res.append(char_dict[cls_])
            output.append(recog_res)
    return output

def merge_line_chars(line_results, output, pred_label, img_shape):
    '''
    pagenet result and detectron2 result are combined;
    chars may appear inside line;
    merge chars into lines when this situation happens
    '''
    in_line  = list(set(sum([val.tolist() for val in line_results], [])))
    out_line = list(set(list(np.arange(output.shape[0]))).difference(set(in_line)))
    inter = []
    in_line_char = []
    for line_result in line_results:
        line_dets = output[line_result][:,:4]
        poly1 = calc_poly2(line_dets, (img_shape[0], img_shape[1]))
        poly1 = Polygon(poly1.reshape(4,2)).convex_hull
        cur = line_result.tolist()
        for idx in out_line:
            poly2 = output[idx][[0,1,2,1,2,3,0,3]]
            poly2 = Polygon(poly2.reshape(4,2)).convex_hull
            inter_area = poly1.intersection(poly2).area
            inside = max(inter_area/poly1.area, inter_area/poly2.area)
            if inside>0.8:
                cur.append(idx)
                in_line_char.append(idx)
        inter.append(cur)

    out_line_char = list(set(out_line).difference(set(in_line_char)))
    lines, lines_labels, lines_polys = [], [], []
    for line_result in inter:
        line_dets   = output[line_result][:,:4]
        line_labels = pred_label[line_result]
        lines.append(line_dets)
        lines_labels.append(line_labels)
        # calc outer poly
        val = calc_poly2(line_dets, (img_shape[0], img_shape[1]))
        lines_polys.append(val.reshape(-1))

    lines.extend(output[out_line_char][:,:4].tolist())
    lines_labels.extend(pred_label[out_line_char][:,np.newaxis].tolist())
    lines_polys.extend(output[out_line_char][:,[0,1,2,1,2,3,0,3]].astype(np.int32).tolist())
    return lines, lines_labels, lines_polys

def calc_intersect(line1, line2):
    '''
    input param:
    line1, (pt1, pt2);  line2, (pt3, pt4)
    output param:
    intersect point
    '''
    x1, y1 = line1[0];      x2, y2 = line1[1]
    x3, y3 = line2[0];      x4, y4 = line2[1]
    if x1==x2:  k1=None
    else:  k1 = (y2-y1)/(x2-x1);    b1 = (x2*y1-x1*y2)/(x2-x1)
    if x3==x4:  k2=None
    else:  k2 = (y4-y3)/(x4-x3);    b2 = (x4*y3-x3*y4)/(x4-x3)

    if k1!=None and k2!=None:
        x = int((b1-b2)/(k2-k1))
        y = int(k1*x+b1)
    elif k1==None and k2!=None:
        x = x1
        y = int(k2*x+b2)
    elif k1!=None and k2==None:
        x = x3
        y = int(k1*x+b1)
    return (x,y)

def pt_in_rect(lines, coors):
    cross_pts = []
    # order: (lefttop, righttop, rightbt, leftbt)
    for index in range(len(lines)):
        cross_pts.append(calc_intersect(lines[(index-1)%4], lines[index]))
    pts_ABCD = np.array(cross_pts)
    pts_BCDA = np.roll(np.array(cross_pts), -1, axis=0)
    vec = pts_BCDA-pts_ABCD
    x_AB = vec[0][0];   y_AB = vec[0][1]
    x_BC = vec[1][0];   y_BC = vec[1][1]
    x_CD = vec[2][0];   y_CD = vec[2][1]
    x_DA = vec[3][0];   y_DA = vec[3][1]

    coors = np.array(coors)
    coors_ctr_0 = np.sum(coors[:, [0,2,4,6]], axis=1)/4
    coors_ctr_1 = np.sum(coors[:, [1,3,5,7]], axis=1)/4
    ct_pts = np.concatenate((coors_ctr_0.reshape(-1,1), coors_ctr_1.reshape(-1,1)), axis=1)

    a = (ct_pts[:,1]-pts_ABCD[0][1])*x_AB-(ct_pts[:,0]-pts_ABCD[0][0])*y_AB
    b = (ct_pts[:,1]-pts_ABCD[1][1])*x_BC-(ct_pts[:,0]-pts_ABCD[1][0])*y_BC
    c = (ct_pts[:,1]-pts_ABCD[2][1])*x_CD-(ct_pts[:,0]-pts_ABCD[2][0])*y_CD
    d = (ct_pts[:,1]-pts_ABCD[3][1])*x_DA-(ct_pts[:,0]-pts_ABCD[3][0])*y_DA
    """
    https://my.oschina.net/u/4380991/blog/3359197
    https://www.cnblogs.com/mqxs/p/8385895.html

    分别计算向量 AB与AO、BC与BO、CD与CO、DA与DO的叉乘，所得结果的 z 轴分量同号则说明点 O 在四边形 ABCD 内部。
    """

    ind1 = (a>0).astype(np.int32)*(b>0).astype(np.int32)*(c>0).astype(np.int32)*(d>0).astype(np.int32)
    ind2 = (a<0).astype(np.int32)*(b<0).astype(np.int32)*(c<0).astype(np.int32)*(d<0).astype(np.int32)

    cand = []
    cand.extend(np.where(ind1>0)[0].tolist())
    cand.extend(np.where(ind2>0)[0].tolist())
    cand = list(set(cand))
    return cand

def overlay_mask(vis_img, polys):
    bg = np.array([0,0,255])
    for val in polys:
        mask = np.zeros((vis_img.shape[0], vis_img.shape[1]))
        cv2.fillPoly(mask, [np.array(val).reshape(-1,1,2)], (1,))
        vis_img[mask==1] = vis_img[mask==1]*0.7+bg*0.3
    return vis_img



def point_inside_quad(coors, hl, vl):
    # 求两条直线的交点
    def line_intersection(l1, l2):
        [x1, y1], [x2, y2] = l1
        [x3, y3], [x4, y4] = l2
        if y1 == y2:
            y = y1
            x = (y - y3) * (x4 - x3) / (y4 - y3) + x3
            return x, y
        if y3 == y4:
            y = y3
            x = (y - y1) * (x2 - x1) / (y2 - y1) + x1
            return x, y
        if x1 == x2:
            x = x1
            y = (x - x3) * (y4 - y3) / (x4 - x3) + y3
            return x, y
        if x3 == x4:
            x = x3
            y = (x - x1) * (y2 - y1) / (x2 - x1) + y1
            return x, y

        m1 = (y2 - y1) / (x2 - x1)
        b1 = y1 - m1 * x1
        m2 = (y4 - y3) / (x4 - x3)
        b2 = y3 - m2 * x3

        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1
        return x, y


    # 中心点坐标
    pts_x = (coors[:, 0] + coors[:, 2]) / 2
    pts_y = (coors[:, 1] + coors[:, 3]) / 2
    pts = np.array([pts_x, pts_y]).T

    quad = []
    # 遍历求交点
    for i in hl:
        for j in vl:
            x, y = line_intersection(i, j)
            quad.append([x, y])


    x1, y1 = quad[0]
    x2, y2 = quad[2]
    x3, y3 = quad[3]
    x4, y4 = quad[1]
    # 判断点是否在四边形内部

    # 判断点是否在ABCD形成的三角形内
    # point_inside_triangle改成np pts是np.array，有很多点
    def points_inside_triangle(pts, x1, y1, x2, y2, x3, y3):
        area_ABC = 0.5 * abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
        area_PBC = 0.5 * abs(pts[:,0]*(y2-y3) + x2*(y3-pts[:,1]) + x3*(pts[:,1]-y2))
        area_PAC = 0.5 * abs(x1*(pts[:,1]-y3) + pts[:,0]*(y3-y1) + x3*(y1-pts[:,1]))
        area_PAB = 0.5 * abs(x1*(y2-pts[:,1]) + x2*(pts[:,1]-y1) + pts[:,0]*(y1-y2))
        return abs(area_ABC - area_PBC - area_PAC - area_PAB) < 1e-6
    
    # 判断点是否在四边形内部
    res = points_inside_triangle(pts, x1, y1, x2, y2, x3, y3) | points_inside_triangle(pts, x1, y1, x3, y3, x4, y4)
    return np.where(res==True)[0].tolist()

