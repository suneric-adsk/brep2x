
import numpy as np
from math import cos, sin, pi
from OCC.Core.gp import (
    gp_Trsf, 
    gp_Vec, 
    gp_Pnt, 
    gp_Quaternion, 
    gp_Ax2, 
    gp_Dir, 
    gp_Ax3,
    gp_Circ, 
    gp_Lin,
    gp_Pln, 
    gp_Pnt2d,
)
from OCC.Core.BRep import BRep_Tool
from OCC.Core.Geom import Geom_Plane, Geom_Line
from OCC.Core.GeomAPI import GeomAPI_IntCS, GeomAPI_ProjectPointOnSurf
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_SOLID, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.TopoDS import topods, TopoDS_Face, TopoDS_Edge
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_Transform, 
    BRepBuilderAPI_MakePolygon, 
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_Sewing,
    BRepBuilderAPI_MakeSolid,
    BRepBuilderAPI_MakeVertex,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_MakeEdge,
)
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Extend.ShapeFactory import make_edge
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.BRepFeat import BRepFeat_MakePrism
from OCC.Core.GC import GC_MakeArcOfCircle, GC_MakeSegment
from OCC.Core.TopTools import TopTools_ListIteratorOfListOfShape

from occwl.edge import Edge
from occwl.uvgrid import uvgrid
from occwl.graph import face_adjacency
from sklearn.preprocessing import KBinsDiscretizer

PT_Tol = 1e-6

def position_discretizer(nBins=257):
    """position values are in range [-1.0,1.0] converted to [0,255] levels
    """
    x = np.linspace(-1.0, 1.0, nBins).reshape(-1,1)
    disBin = KBinsDiscretizer(n_bins=nBins, encode='ordinal', strategy='uniform')
    disBin.fit(x)
    # test with the values from paper
    # x1 = np.array([-1.0,-0.711,-0.477,-0.398,0.0,0.039,0.156,0.391,0.711,1.0]).reshape(-1,1)
    # yt = np.array([0,37,67,77,128,133,148,178,219,256]).reshape(-1,1)
    # y1 = disBin.transform(x1)
    # print(y1.squeeze(1))
    return disBin

def parameter_discretizer(nBins=257):
    """parameter values are in range [0,2.0] converted to [0,256]-1 levels
    -1 for invalid parameter
    """
    x = np.linspace(0.0, 2.0, nBins).reshape(-1,1)
    disBin = KBinsDiscretizer(n_bins=nBins, encode='ordinal', strategy='uniform')
    disBin.fit(x)
    # x1 = np.array([0,0.164,0.289,0.43,0.555,0.641,0.711,0.789,2.0]).reshape(-1,1)
    # yt = np.array([0,20,36,54,70,81,90,100,255]).reshape(-1,1)
    # y1 = disBin.transform(x1)
    # print(y1.squeeze(1))
    return disBin

POS_Disctretizer = position_discretizer()
PAR_Disctretizer = parameter_discretizer()

def print_transformation(face):
    loc = TopLoc_Location()
    BRep_Tool.Surface(face, loc)
    mat = np.eye(4)
    for i in range(3):
        for j in range(4):
            mat[i,j] = loc.Transformation().Value(i+1,j+1)
    print(mat)

def position_level(continuous_value):
    """convert continuous position value to discrete level"""
    continuous_value = np.array(continuous_value).reshape(-1,1)
    discrete_value = POS_Disctretizer.transform(continuous_value)
    return discrete_value.squeeze(1)[0]

def parameter_level(continuous_value):
    """convert continuous parameter value to discrete level"""
    continuous_value = np.array(continuous_value).reshape(-1,1)
    discrete_value = PAR_Disctretizer.transform(continuous_value)
    return discrete_value.squeeze(1)[0]-1

def position_value(discrete_value):
    """convert discrete position level to continuous value"""
    discrete_value = np.array(discrete_value).reshape(-1,1)
    continuous_value = POS_Disctretizer.inverse_transform(discrete_value)
    return continuous_value.squeeze(1)[0]

def parameter_value(discrete_value):
    """convert discrete parameter level to continuous value"""
    if discrete_value < 0:
        return -1.0
    discrete_value = np.array(discrete_value+1).reshape(-1,1)
    continuous_value = PAR_Disctretizer.inverse_transform(discrete_value)
    return continuous_value.squeeze(1)[0]

def make_solid_and_upgrade(shell):
    ms = BRepBuilderAPI_MakeSolid()
    ms.Add(shell)
    if ms.IsDone():
        su = ShapeUpgrade_UnifySameDomain(ms.Solid(), True, True, True)
        su.Build()
        return su.Shape()
    return shell

def update_prism_shape(shape):
    """ 
    The prism contain a face with translation part due to the extrucsion
    Bake the translation to the face geometry and set the location to identity
    """
    sewing = BRepBuilderAPI_Sewing()
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = explorer.Current()
        loc = TopLoc_Location()
        BRep_Tool.Surface(face, loc)
        if loc.IsIdentity():
            sewing.Add(face)
        else:
            # print_transformation(face)
            builder = BRepBuilderAPI_Transform(face, loc.Transformation(), True)
            new_face = builder.Shape()
            new_face.Location(TopLoc_Location())
            sewing.Add(new_face)
        explorer.Next()
    sewing.Perform()
    return make_solid_and_upgrade(sewing.SewedShape())
    
def solid_count(shape):
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    count = 0
    while explorer.More():
        count += 1
        explorer.Next()
    return count

def solid_edges(shape):
    edges = []
    for edge in TopologyExplorer(shape).edges():
        edges.append(edge)
    return edges

def make_2d_polygon_points(n, r):
    points = []
    for i in range(n):
        x = r * cos(i*2*pi/n)
        y = r * sin(i*2*pi/n)
        points.append(gp_Pnt2d(x, y))
    return points

def middle_point(pt1, pt2):
    return gp_Pnt((pt1.X() + pt2.X())/2, (pt1.Y() + pt2.Y())/2, (pt1.Z() + pt2.Z())/2)

def calculate_point(s, dir, len):
    return gp_Pnt(s.X() + len*dir.X(), s.Y() + len*dir.Y(), s.Z() + len*dir.Z())

def find_closet_edge_on_face(face, pt1, pt2):
    ref = make_edge(pt1, pt2)
    closest_edge = None
    min_dist = float("inf")
    for edge in TopologyExplorer(face).edges():
        dist = BRepExtrema_DistShapeShape(ref, edge).Value()
        if dist < min_dist:
            min_dist = dist
            closest_edge = edge
    return closest_edge

def edge_direction(edge):
    crv, fp, lp = BRep_Tool.Curve(edge)
    geom = Geom_Line.DownCast(crv)
    if geom is None:
        print("Edge is not a line")
        return None
    line = geom.Line()
    return line.Direction()

def find_closet_vertex_on_face(face, pt1, pt2):
    ref = make_edge(pt1, pt2)
    closest_vertex = None
    min_dist = float("inf")
    for v in TopologyExplorer(face).vertices():
        dist = BRepExtrema_DistShapeShape(ref, v).Value()
        if dist < min_dist:
            min_dist = dist
            closest_vertex = v
    return BRep_Tool.Pnt(closest_vertex)

def find_closet_face(shape, pt1, pt2):
    closest_face = None
    min_dist = float("inf")
    ref = make_edge(pt1, pt2)
    for face in TopologyExplorer(shape).faces():
        dist = BRepExtrema_DistShapeShape(ref, face).Value()
        if dist < min_dist:
            min_dist = dist
            closest_face = face
    return closest_face

def find_closet_parallel_edge(shape, pt1, pt2):
    closest_edge = None
    min_dist = float("inf")
    dir = gp_Dir(gp_Vec(pt1, pt2))
    ref = make_edge(pt1, pt2)
    for edge in TopologyExplorer(shape, True).edges():
        pts = []
        for v in TopologyExplorer(edge).vertices():
            pts.append(BRep_Tool.Pnt(v))

        if len(pts) != 2:
            continue
        edge_dir = gp_Dir(gp_Vec(pts[0], pts[1]))
        if edge_dir.IsParallel(dir, PT_Tol):
            dist = BRepExtrema_DistShapeShape(ref, edge).Value()
            if dist < min_dist:
                min_dist = dist
                closest_edge = edge
    return closest_edge

def find_intersection_point(edge, pt, dir):
    plane = Geom_Plane(gp_Ax3(pt, dir))
    curve, fp, lp = BRep_Tool.Curve(edge)
    line = Geom_Line.DownCast(curve)
    if line is None:
        print("Edge is not a line")
        return None

    intersector = GeomAPI_IntCS(line, plane)
    if intersector.NbPoints() == 0:
        print("No intersection found")
        return None
    return intersector.Point(1)

def map_face_after(base, feat_mk):
    fmap = {}
    for face in TopologyExplorer(base).faces():
        if feat_mk.IsDeleted(face):
            continue

        fmap[face] = []
        modified = feat_mk.Modified(face)
        if modified.Size() == 0:
            fmap[face].append(face)
            continue

        iter = TopTools_ListIteratorOfListOfShape(modified)
        while iter.More():
            a_shape = iter.Value()
            assert a_shape.ShapeType() == TopAbs_FACE
            fmap[face].append(topods.Face(a_shape))
            iter.Next()
    return fmap

def same_shape_in_list(shape, slist):
    hash_code = shape.__hash__()
    for a_shape in slist:
        if a_shape.__hash__() == hash_code:
            return a_shape
    return None

def face_label_map(fmap, old_map, new_shape, new_name):
    new_map = {}

    new_faces = []
    for face in TopologyExplorer(new_shape).faces():
        new_faces.append(face)

    for oldf in fmap:
        old_name = old_map[oldf]
        for samef in fmap[oldf]:
            samef = same_shape_in_list(samef, new_faces)
            if samef is None:
                continue
            new_map[samef] = old_name
            new_faces.remove(samef)

    for n_face in new_faces:
        new_map[n_face] = new_name

    return new_map


def apply_feature(base, label_map, feat_name, feat_face, depth_dir, fuse=False):
    fm = BRepFeat_MakePrism()
    fm.Init(base, feat_face, TopoDS_Face(), depth_dir, fuse, False)
    fm.Build()
    fm.Perform(np.linalg.norm(depth_dir))
    shape = fm.Shape()
    fmap = map_face_after(base, fm)
    new_labels = face_label_map(fmap, label_map, shape, feat_name)
    return fm.Shape(), new_labels

def transform_to_3d_point(pt2d, c, dir):
    plane = gp_Ax3(c, dir)
    x = plane.XDirection()
    y = plane.YDirection()
    return gp_Pnt(c.X() + pt2d.X() * x.X() + pt2d.Y() * y.X(),
                  c.Y() + pt2d.X() * x.Y() + pt2d.Y() * y.Y(),
                  c.Z() + pt2d.X() * x.Z() + pt2d.Y() * y.Z())

def transform_to_3d_points(pt2ds, c, dir):
    pt3ds = []
    for pt2d in pt2ds:
        pt3d = transform_to_3d_point(pt2d, c, dir)
        pt3ds.append(pt3d)
    return pt3ds

def make_face_circle(c, r, dir):
    circ = gp_Circ(gp_Ax2(c, dir), r)
    edge = BRepBuilderAPI_MakeEdge(circ, 0., 2*pi).Edge()
    wire = BRepBuilderAPI_MakeWire(edge).Wire()
    return BRepBuilderAPI_MakeFace(wire).Face()

def make_face_oring(c, r, dir, ratio=0.5):
    outer_circ = gp_Circ(gp_Ax2(c, dir), r)
    inner_circ = gp_Circ(gp_Ax2(c, dir), r*ratio)
    outer_edge = BRepBuilderAPI_MakeEdge(outer_circ, 0., 2*pi).Edge()
    inner_edge = BRepBuilderAPI_MakeEdge(inner_circ, 0., 2*pi).Edge()
    outer_wire = BRepBuilderAPI_MakeWire(outer_edge).Wire()
    inner_wire = BRepBuilderAPI_MakeWire(inner_edge).Wire()
    fm = BRepBuilderAPI_MakeFace(outer_wire)
    fm.Add(inner_wire)
    return fm.Face()

def make_face_polygon(points):
    wm = BRepBuilderAPI_MakeWire()
    for i in range(len(points)):
        j = (i+1)%len(points)
        wm.Add(BRepBuilderAPI_MakeEdge(points[i], points[j]).Edge())
    return BRepBuilderAPI_MakeFace(wm.Wire()).Face()

def get_offset_points(points, depth):
    # compute the normal of the plane
    vec1 = gp_Vec(points[0], points[1])
    vec2 = gp_Vec(points[0], points[2])
    dir = gp_Dir(vec1.Crossed(vec2).Normalized())
    plane = gp_Pln(points[0], dir)
    # offset this plane by depth
    trans_vec = gp_Vec(dir).Multiplied(depth)
    base_loc = plane.Location()
    offset_loc = base_loc.Translated(trans_vec)
    offset = gp_Pln(offset_loc, dir)

    offset_points = []
    for pt in points:
        projector = GeomAPI_ProjectPointOnSurf(pt, Geom_Plane(offset))
        if projector.NbPoints() == 0:
            continue
        offset_points.append(projector.Point(1))
    return offset_points, dir

def make_face_fan(c, s, e):
    vec1 = gp_Vec(c, s)
    vec2 = gp_Vec(c, e)
    r = vec1.Magnitude()
    dir = gp_Dir(vec1.Crossed(vec2).Normalized()) 
    cir = gp_Circ(gp_Ax2(c, dir), r)
    sm = [GC_MakeSegment(c, s), GC_MakeArcOfCircle(cir, s, e, True), GC_MakeSegment(e, c)]
    wm = BRepBuilderAPI_MakeWire()
    for seg in sm:
        wm.Add(BRepBuilderAPI_MakeEdge(seg.Value()).Edge())
    return BRepBuilderAPI_MakeFace(wm.Wire()).Face()

def make_face_circle_x(c, w, h, dir):
    r = w/2
    if abs(r-h) < PT_Tol:
        pt0 = transform_to_3d_point(gp_Pnt2d(r, 0), c, dir)
        pt1 = transform_to_3d_point(gp_Pnt2d(-r, 0), c, dir)
        circ = gp_Circ(gp_Ax2(c, dir), r)
        edge0 = BRepBuilderAPI_MakeEdge(pt0, pt1).Edge()
        edge1 = BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(circ, pt1, pt0, True).Value()).Edge()
        wm = BRepBuilderAPI_MakeWire(edge0, edge1)
        return BRepBuilderAPI_MakeFace(wm.Wire()).Face()
    
    sm = []
    pt0 = transform_to_3d_point(gp_Pnt2d(r, 0), c, dir)
    pt1 = transform_to_3d_point(gp_Pnt2d(-r, 0), c, dir)
    if h < r:
        angle = np.arctan2(h,r)
        x, y = r*cos(angle), r*sin(angle)
        pt2 = transform_to_3d_point(gp_Pnt2d(-x, -y), c, dir)
        pt3 = transform_to_3d_point(gp_Pnt2d(x, -y), c, dir)
        circ = gp_Circ(gp_Ax2(c, dir), r)
        sm = [GC_MakeSegment(pt0, pt1), GC_MakeArcOfCircle(circ, pt1, pt2, True), 
              GC_MakeSegment(pt2, pt3), GC_MakeArcOfCircle(circ, pt3, pt0, True)]
    else:
        pt2 = transform_to_3d_point(gp_Pnt2d(-r,-(h-r)), c, dir)
        pt3 = transform_to_3d_point(gp_Pnt2d(r, -(h-r)), c, dir)
        c23 = transform_to_3d_point(gp_Pnt2d(0, -(h-r)), c, dir)
        cir23 = gp_Circ(gp_Ax2(c23, dir), r)
        sm = [GC_MakeSegment(pt0, pt1), GC_MakeSegment(pt1, pt2), 
               GC_MakeArcOfCircle(cir23, pt2, pt3, True), GC_MakeSegment(pt3, pt0)]
        
    wm = BRepBuilderAPI_MakeWire()
    for seg in sm:
        wm.Add(BRepBuilderAPI_MakeEdge(seg.Value()).Edge())
    fm = BRepBuilderAPI_MakeFace(wm.Wire())
    return fm.Face()

def make_face_key_hole(c, w, h, dir):
    if abs(w-h) < PT_Tol:
        return make_face_circle(w/2, c, dir)
    
    r = min(w,h)/2
    pt0 = transform_to_3d_point(gp_Pnt2d((w-h)/2,-h/2), c, dir)
    pt1 = transform_to_3d_point(gp_Pnt2d((w-h)/2,h/2), c, dir)
    pt2 = transform_to_3d_point(gp_Pnt2d((h-w)/2,h/2), c, dir)
    pt3 = transform_to_3d_point(gp_Pnt2d((h-w)/2,-h/2), c, dir)
    c01 = transform_to_3d_point(gp_Pnt2d((w-h)/2,0), c, dir)
    c23 = transform_to_3d_point(gp_Pnt2d((h-w)/2,0), c, dir)
    if w < h:
        pt0 = transform_to_3d_point(gp_Pnt2d(w/2,(h-w)/2), c, dir)
        pt1 = transform_to_3d_point(gp_Pnt2d(-w/2,(h-w)/2), c, dir)
        pt2 = transform_to_3d_point(gp_Pnt2d(-w/2,(w-h)/2), c, dir)
        pt3 = transform_to_3d_point(gp_Pnt2d(w/2,(w-h)/2), c, dir)
        c01 = transform_to_3d_point(gp_Pnt2d(0,(h-w)/2), c, dir)
        c23 = transform_to_3d_point(gp_Pnt2d(0,(w-h)/2), c, dir)
    cir01 = gp_Circ(gp_Ax2(c01, dir), r)
    cir23 = gp_Circ(gp_Ax2(c23, dir), r)
    sm = [GC_MakeArcOfCircle(cir01, pt0, pt1, True), GC_MakeSegment(pt1,pt2),
          GC_MakeArcOfCircle(cir23, pt2, pt3, True), GC_MakeSegment(pt3,pt0)]
    wm = BRepBuilderAPI_MakeWire()
    for seg in sm:
        wm.Add(BRepBuilderAPI_MakeEdge(seg.Value()).Edge())
    return BRepBuilderAPI_MakeFace(wm.Wire()).Face()

def display_uv_net(v, solid):
    g = face_adjacency(solid)
    bbox = solid.box()
    point_radius = bbox.max_box_length() * 0.03
    arrow_radius = point_radius * 0.85
    arrow_length = arrow_radius * 4

    face_grids = {}
    for face_idx in g.nodes:
        face = g.nodes[face_idx]["face"]
        points = uvgrid(face, num_u=10, num_v=10, method="point")
        mask = uvgrid(face, num_u=10, num_v=10, method="inside")
        normals = uvgrid(face, num_u=10, num_v=10, method="normal")
        face_grids[face_idx] = {"points": points, "normals": normals, "mask": mask}

    print(f"Number of nodes (faces): {len(g.nodes)}")
    print(f"Number of edges: {len(g.edges)}")

    # Get the points at each face's center for visualizing edges
    face_centers = {}
    for face_idx in g.nodes():
        # Display a sphere for each UV-grid point
        face = g.nodes[face_idx]["face"]
        grid = face_grids[face_idx]
        # Display points
        face_points = grid["points"].reshape((-1, 3))
        face_mask = grid["mask"].reshape(-1)
        face_points = face_points[face_mask, :]
        v.display_points(face_points, marker="point", color="GREEN")
        # Display normals
        face_normals = grid["normals"].reshape((-1, 3))
        face_normals = face_normals[face_mask, :]
        lines = [Edge.make_line_from_points(pt, pt + arrow_length * nor) for pt, nor in zip(face_points, face_normals)]
        for l in lines:
            v.display(l, color="RED")
        face_centers[face_idx] = grid["points"][4, 4]

    for fi, fj in g.edges():
        pt1 = face_centers[fi]
        pt2 = face_centers[fj]
        dist = np.linalg.norm(pt2 - pt1)
        if dist > 1e-3:
            v.display(Edge.make_line_from_points(pt1, pt2), color=(51.0 / 255.0, 0, 1))

def bounding_box(input):
    pts = input[...,:3].reshape((-1,3))
    mask = input[...,6].reshape(-1)
    point_indices_inside_faces = mask == 1
    pts = pts[point_indices_inside_faces,:]
    x, y, z = pts[:,0], pts[:,1], pts[:,2]
    return np.array([[x.min(), y.min(), z.min()],[x.max(), y.max(), z.max()]])

def center_and_scale_uvgrid(input, bbox=None):
    # bounding box
    if bbox is None:
        bbox = bounding_box(input)
    # center and scale
    scale = 2.0 / np.linalg.norm(bbox[1] - bbox[0])
    center = 0.5 * (bbox[0] + bbox[1])
    input[..., :3] -= center
    input[..., :3] *= scale
    return input, center, scale

def softmax(values):
    exp_values = np.exp(values - np.max(values))
    return exp_values / exp_values.sum()

def d2_a3_distance(face0, face1, diag_len, size, num_sample=16):
    """
    statistical shape function based on the distance between two randomly chosen points on the face
    """
    num_tot = num_sample*num_sample
    pts0 = uvgrid(face0, num_u=num_sample, num_v=num_sample, method="point")
    pts1 = uvgrid(face1, num_u=num_sample, num_v=num_sample, method="point")
    pts0 = pts0.reshape((num_tot, 3))
    pts1 = pts1.reshape((num_tot, 3))
    # random select points
    dists, angles = [], []
    for _ in range(4*size):
        idx = np.random.randint(0, num_tot, (2, 3))
        dist = np.linalg.norm(pts0[idx[0,0]]-pts1[idx[1,0]])/diag_len
        dists.append(dist)
        dir0 = np.cross(pts0[idx[0,1]]-pts0[idx[0,0]], pts0[idx[0,2]]-pts0[idx[0,0]])
        dir1 = np.cross(pts1[idx[1,1]]-pts1[idx[1,0]], pts1[idx[1,2]]-pts1[idx[1,0]])
        ang = np.arccos(np.dot(dir0, dir1)/(np.linalg.norm(dir0)*np.linalg.norm(dir1)+1e-6))
        angles.append(ang)
    dist_values, _ = np.histogram(dists, bins=np.linspace(0,1,size+1))
    angle_values, _ = np.histogram(angles, bins=np.linspace(0,np.pi,size+1))
    return softmax(dist_values), softmax(angle_values)