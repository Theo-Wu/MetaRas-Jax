import numpy as np
import jax.numpy as jnp
from numpy import dot
import math
from math import sqrt

def compute_line_magnitude(x1, y1, x2, y2):
    lineMagnitude = jnp.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return lineMagnitude

def point_to_line_distance_parallel(points, p1, p2):
    px = points[:,:,0]
    py = points[:,:,1]
    h,w,_ = points.shape
    x1 = p1[:,0]
    y1 = p1[:,1]
    x2 = p2[:,0]
    y2 = p2[:,1]
    num_tri = len(x1)
    # x1 = np.tile(x1,(1,h,w))
    # y1 = np.tile(y1,(1,h,w))
    # x2 = np.tile(x2,(1,h,w))
    # y2 = np.tile(y2,(1,h,w))
    x1 = x1[:,None,None]
    y1 = y1[:,None,None]
    x2 = x2[:,None,None]
    y2 = y2[:,None,None]
    px = jnp.tile(px,(num_tri,1,1))
    py = jnp.tile(py,(num_tri,1,1))

    line_magnitude = compute_line_magnitude(x1, y1, x2, y2)

    # Don't need to consider segment is too short

    u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))

    u = u1 / (line_magnitude * line_magnitude)

    # The projection of the point is inside the line segment
    ix = x1 + u * (x2 - x1)
    iy = y1 + u * (y2 - y1)
    distance = compute_line_magnitude(px, py, ix, iy)

    # If the projection of the point to the line is not in the line segment
    d1 = compute_line_magnitude(px, py, x1, y1)
    d2 = compute_line_magnitude(px, py, x2, y2)
    mask = ((u<1e-5) + (u>1))
    out_distance = jnp.fmin(d1,d2) * mask
    in_distance = distance * (1. - mask)
    distance = in_distance + out_distance
    # distance = distance.at[u<1e-5].set(jnp.fmin(d1,d2)[u<1e-5])
    # distance = distance.at[u>1].set(jnp.fmin(d1,d2)[u>1])

    return distance 


    if line_magnitude.any() < 0.00000001: #if the line segment is too short
        line_magnitude = line_magnitude.at[line_magnitude<0.00000001].set(10000)
    else:
        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
 
        u = u1 / (line_magnitude * line_magnitude)

        # The projection of the point is inside the line segment
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        distance = compute_line_magnitude(px, py, ix, iy)

        # If the projection of the point to the line is not in the line segment
        d1 = compute_line_magnitude(px, py, x1, y1)
        d2 = compute_line_magnitude(px, py, x2, y2)
        mask = ((u<1e-5) + (u>1))
        out_distance = jnp.fmin(d1,d2) * mask
        in_distance = distance * (1. - mask)
        distance = in_distance + out_distance
        # distance = distance.at[u<1e-5].set(jnp.fmin(d1,d2)[u<1e-5])
        # distance = distance.at[u>1].set(jnp.fmin(d1,d2)[u>1])

        return distance 
 
def point_to_line_distance(point, p1, p2):
    px, py = point
    x1, y1 = p1
    x2, y2 = p2
    line_magnitude = compute_line_magnitude(x1, y1, x2, y2)
    if line_magnitude < 0.00000001: #if the line segment is too short
        return 10000
    else:
        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (line_magnitude * line_magnitude)
        if (u < 0.00001) or (u > 1):
            # If the projection of the point to the line is not in the line segment
            d1 = compute_line_magnitude(px, py, x1, y1)
            d2 = compute_line_magnitude(px, py, x2, y2)
            distance = min(d1,d2)
        else:
            # The projection of the point is inside the line segment
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            distance = compute_line_magnitude(px, py, ix, iy)
        return distance

def barycentric_coordinate(point,polygon_position):
    w = np.zeros(3).astype('float')
    x,y = point
    w[0] = polygon_position[0][0] * x + polygon_position[0][1] * y + polygon_position[0][2]
    w[1] = polygon_position[1][0] * x + polygon_position[1][1] * y + polygon_position[1][2]
    w[2] = polygon_position[2][0] * x + polygon_position[2][1] * y + polygon_position[2][2]
    return w

def forward_barycentric_p2f_distance(w):
    dis = (w[2] if w[1] > w[2] else w[1]) if w[0] > w[1] else (w[2] if w[0] > w[2] else w[0])
    #dis = pow(dis, 2) if dis > 0 else -pow(dis, 2)
    dis = dis**2
    return dis

def lineseg_dists(p, a, b):
    # normalized tangent vectors
    d_ba = b - a
    d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1]).reshape(-1, 1)))

    # signed parallel distance components
    # rowwise dot products of 2D vectors
    s = np.multiply(a - p, d).sum(axis=1)
    t = np.multiply(p - b, d).sum(axis=1)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros(len(s))])

    # perpendicular distance component
    # rowwise cross products of 2D vectors  
    d_pa = p - a
    c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]

    return np.hypot(h, c)

def pointTriangleDistance(TRI, P):
    # function [dist,PP0] = pointTriangleDistance(TRI,P)
    # calculate distance between a point and a triangle in 3D
    # SYNTAX
    #   dist = pointTriangleDistance(TRI,P)
    #   [dist,PP0] = pointTriangleDistance(TRI,P)
    #
    # DESCRIPTION
    #   Calculate the distance of a given point P from a triangle TRI.
    #   Point P is a row vector of the form 1x3. The triangle is a matrix
    #   formed by three rows of points TRI = [P1;P2;P3] each of size 1x3.
    #   dist = pointTriangleDistance(TRI,P) returns the distance of the point P
    #   to the triangle TRI.
    #   [dist,PP0] = pointTriangleDistance(TRI,P) additionally returns the
    #   closest point PP0 to P on the triangle TRI.
    #
    # Author: Gwolyn Fischer
    # Release: 1.0
    # Release date: 09/02/02
    # Release: 1.1 Fixed Bug because of normalization
    # Release: 1.2 Fixed Bug because of typo in region 5 20101013
    # Release: 1.3 Fixed Bug because of typo in region 2 20101014

    # Possible extention could be a version tailored not to return the distance
    # and additionally the closest point, but instead return only the closest
    # point. Could lead to a small speed gain.

    # Example:
    # %% The Problem
    # P0 = [0.5 -0.3 0.5]
    #
    # P1 = [0 -1 0]
    # P2 = [1  0 0]
    # P3 = [0  0 0]
    #
    # vertices = [P1; P2; P3]
    # faces = [1 2 3]
    #
    # %% The Engine
    # [dist,PP0] = pointTriangleDistance([P1;P2;P3],P0)
    #
    # %% Visualization
    # [x,y,z] = sphere(20)
    # x = dist*x+P0(1)
    # y = dist*y+P0(2)
    # z = dist*z+P0(3)
    #
    # figure
    # hold all
    # patch('Vertices',vertices,'Faces',faces,'FaceColor','r','FaceAlpha',0.8)
    # plot3(P0(1),P0(2),P0(3),'b*')
    # plot3(PP0(1),PP0(2),PP0(3),'*g')
    # surf(x,y,z,'FaceColor','b','FaceAlpha',0.3)
    # view(3)

    # rewrite triangle in normal form
    B = TRI[0, :]
    E0 = TRI[1, :] - B
    # E0 = E0/sqrt(sum(E0.^2)); %normalize vector
    E1 = TRI[2, :] - B
    # E1 = E1/sqrt(sum(E1.^2)); %normalize vector
    D = B - P
    a = dot(E0, E0)
    b = dot(E0, E1)
    c = dot(E1, E1)
    d = dot(E0, D)
    e = dot(E1, D)
    f = dot(D, D)

    #print "{0} {1} {2} ".format(B,E1,E0)
    det = a * c - b * b
    s = b * e - c * d
    t = b * d - a * e

    # Terible tree of conditionals to determine in which region of the diagram
    # shown above the projection of the point into the triangle-plane lies.
    if (s + t) <= det:
        if s < 0.0:
            if t < 0.0:
                # region4
                if d < 0:
                    t = 0.0
                    if -d >= a:
                        s = 1.0
                        sqrdistance = a + 2.0 * d + f
                    else:
                        s = -d / a
                        sqrdistance = d * s + f
                else:
                    s = 0.0
                    if e >= 0.0:
                        t = 0.0
                        sqrdistance = f
                    else:
                        if -e >= c:
                            t = 1.0
                            sqrdistance = c + 2.0 * e + f
                        else:
                            t = -e / c
                            sqrdistance = e * t + f

                            # of region 4
            else:
                # region 3
                s = 0
                if e >= 0:
                    t = 0
                    sqrdistance = f
                else:
                    if -e >= c:
                        t = 1
                        sqrdistance = c + 2.0 * e + f
                    else:
                        t = -e / c
                        sqrdistance = e * t + f
                        # of region 3
        else:
            if t < 0:
                # region 5
                t = 0
                if d >= 0:
                    s = 0
                    sqrdistance = f
                else:
                    if -d >= a:
                        s = 1
                        sqrdistance = a + 2.0 * d + f;  # GF 20101013 fixed typo d*s ->2*d
                    else:
                        s = -d / a
                        sqrdistance = d * s + f
            else:
                # region 0
                invDet = 1.0 / det
                s = s * invDet
                t = t * invDet
                sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f
    else:
        if s < 0.0:
            # region 2
            tmp0 = b + d
            tmp1 = c + e
            if tmp1 > tmp0:  # minimum on edge s+t=1
                numer = tmp1 - tmp0
                denom = a - 2.0 * b + c
                if numer >= denom:
                    s = 1.0
                    t = 0.0
                    sqrdistance = a + 2.0 * d + f;  # GF 20101014 fixed typo 2*b -> 2*d
                else:
                    s = numer / denom
                    t = 1 - s
                    sqrdistance = s * (a * s + b * t + 2 * d) + t * (b * s + c * t + 2 * e) + f

            else:  # minimum on edge s=0
                s = 0.0
                if tmp1 <= 0.0:
                    t = 1
                    sqrdistance = c + 2.0 * e + f
                else:
                    if e >= 0.0:
                        t = 0.0
                        sqrdistance = f
                    else:
                        t = -e / c
                        sqrdistance = e * t + f
                        # of region 2
        else:
            if t < 0.0:
                # region6
                tmp0 = b + e
                tmp1 = a + d
                if tmp1 > tmp0:
                    numer = tmp1 - tmp0
                    denom = a - 2.0 * b + c
                    if numer >= denom:
                        t = 1.0
                        s = 0
                        sqrdistance = c + 2.0 * e + f
                    else:
                        t = numer / denom
                        s = 1 - t
                        sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f

                else:
                    t = 0.0
                    if tmp1 <= 0.0:
                        s = 1
                        sqrdistance = a + 2.0 * d + f
                    else:
                        if d >= 0.0:
                            s = 0.0
                            sqrdistance = f
                        else:
                            s = -d / a
                            sqrdistance = d * s + f
            else:
                # region 1
                numer = c + e - b - d
                if numer <= 0:
                    s = 0.0
                    t = 1.0
                    sqrdistance = c + 2.0 * e + f
                else:
                    denom = a - 2.0 * b + c
                    if numer >= denom:
                        s = 1.0
                        t = 0.0
                        sqrdistance = a + 2.0 * d + f
                    else:
                        s = numer / denom
                        t = 1 - s
                        sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f

    # account for numerical round-off error
    if sqrdistance < 0:
        sqrdistance = 0

    dist = sqrt(sqrdistance)

    PP0 = B + s * E0 + t * E1
    return dist, PP0