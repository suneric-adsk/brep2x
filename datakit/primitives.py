from datakit.utils import *
from OCC.Core.BRepPrimAPI import (
    BRepPrimAPI_MakeBox, 
    BRepPrimAPI_MakePrism, 
    BRepPrimAPI_MakeCylinder, 
    BRepPrimAPI_MakeCone, 
    BRepPrimAPI_MakeSphere,
)
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.BRepCheck import BRepCheck_Analyzer

PRIMITIVE_NAMES = ["box", "cylinder", "prism", "cone", "sphere"]

class Primitive:
    """
    Base class for all primitive shapes
    """
    def __init__(self, type, param):
        self.type = type
        self.param = param
        self.prim_shape = None

    def rotation(self):
        qw = position_value(int(self.param["Q_0"]))
        qx = position_value(int(self.param["Q_1"]))
        qy = position_value(int(self.param["Q_2"]))
        qz = position_value(int(self.param["Q_3"]))
        print("Rotation: qw {:.2f}, qx {:.2f}, qy {:.2f}, qz {:.2f}".format(qw, qx, qy, qz))
        trns = gp_Trsf()
        trns.SetRotation(gp_Quaternion(qx,qy,qz,qw))
        return trns
    
    def translation(self):
        tx = position_value(int(self.param["T_x"]))
        ty = position_value(int(self.param["T_y"]))
        tz = position_value(int(self.param["T_z"]))
        trns = gp_Trsf()
        trns.SetTranslation(gp_Vec(tx, ty, tz))
        return trns

    def transform_shape(self, shape, translation, rotation = None):
        trns = translation
        if rotation is not None:
            trns = trns.Multiplied(rotation)
        shape_trans = BRepBuilderAPI_Transform(shape, trns, True)
        shape_trans.Build()
        return shape_trans.Shape()
    
    def shape(self):
        raise NotImplementedError("Subclass must implement abstract method")
    
    def fuse_primitive(self, base):
        """
        fuse current primitive shape with the existing shape
        """
        shape = self.shape()
        if shape is None:
            print("no primitive shape to fuse")
            return

        if base is None:
            return shape
        
        fused = BRepAlgoAPI_Fuse(base, shape).Shape()
        # sew the fused shape and upgrade the faces (merge them into same domain)
        if solid_count(fused) == 1:
            sewing = BRepBuilderAPI_Sewing()
            sewing.Add(fused)    
            sewing.Perform()
            fused = make_solid_and_upgrade(sewing.SewedShape())
        return fused
    
class PrimitiveBox(Primitive):
    def __init__(self, type, param):
        super().__init__(type, param)

    def shape(self):
        if self.prim_shape is None:
            l1 = parameter_value(int(self.param["L1"]))
            l2 = parameter_value(int(self.param["L2"]))
            l3 = parameter_value(int(self.param["L3"]))
            box = BRepPrimAPI_MakeBox(l1, l2, l3).Solid()
            t = self.translation()
            r = self.rotation()
            print("Box: length {:.2f}, width {:.2f}, height {:.2f}, translation ({:.2f},{:.2f},{:.2f})"
                .format(l1, l2, l3, t.TranslationPart().X(), t.TranslationPart().Y(), t.TranslationPart().Z()))  
            self.prim_shape = self.transform_shape(box, t, r)
        return self.prim_shape

class PrimitiveCylinder(Primitive):
    def __init__(self, type, param):
        super().__init__(type, param)

    def shape(self):
        if self.prim_shape is None:
            l1 = parameter_value(int(self.param["L1"]))
            l2 = parameter_value(int(self.param["L2"]))
            cylinder = BRepPrimAPI_MakeCylinder(l1, l2).Solid()
            t = self.translation()
            r = self.rotation()
            print("Cylinder: radius {:.2f}, height {:.2f}, translation ({:.2f},{:.2f},{:.2f})"
                .format(l1, l2, t.TranslationPart().X(), t.TranslationPart().Y(), t.TranslationPart().Z()))
            self.prim_shape = self.transform_shape(cylinder, t, r)
        return self.prim_shape
    
class PrimitiveCone(Primitive):
    def __init__(self, type, param):
        super().__init__(type, param)

    def shape(self):
        if self.prim_shape is None:
            l1 = parameter_value(int(self.param["L1"]))
            l2 = parameter_value(int(self.param["L2"]))
            cone = BRepPrimAPI_MakeCone(l1,0.0,l2).Solid()
            t = self.translation()
            r = self.rotation()  
            print("Cone: radius {:.2f}, height {:.2f}, translation ({:.2f},{:.2f},{:.2f})"
                .format(l1, l2, t.TranslationPart().X(), t.TranslationPart().Y(), t.TranslationPart().Z()))
            self.prim_shape = self.transform_shape(cone, t, r) 
        return self.prim_shape   
        
class PrimitiveSphere(Primitive):
    def __init__(self, type, param):
        super().__init__(type, param)

    def shape(self):
        if self.prim_shape is None:
            l1 = parameter_value(int(self.param["L1"]))
            sphere = BRepPrimAPI_MakeSphere(l1).Solid()
            t = self.translation()
            print("Sphere: radius {:.2f}, translation ({:.2f},{:.2f},{:.2f})"
                .format(l1, t.TranslationPart().X(), t.TranslationPart().Y(), t.TranslationPart().Z()))
            self.prim_shape = self.transform_shape(sphere, t)
        return self.prim_shape

class PrimitivePrism(Primitive):
    def __init__(self, type, param):
        super().__init__(type, param)

    def shape(self):
        if self.prim_shape is None:
            l1 = parameter_value(int(self.param["L1"]))
            l2 = parameter_value(int(self.param["L2"]))
            e = int(self.param["E"])
            vec = gp_Vec(0, 0, l2)
            pt2ds = make_2d_polygon_points(e, l1)
            pt3ds = transform_to_3d_points(pt2ds, gp_Pnt(0, 0, 0), gp_Dir(vec))
            face = make_face_polygon(pt3ds)
            prism = BRepPrimAPI_MakePrism(face, vec).Shape()
            t = self.translation()
            r = self.rotation()
            print("Prism: radius {:.2f}, height {:.2f}, side {}, translation ({:.2f},{:.2f},{:.2f})"
                .format(l1, l2, e, t.TranslationPart().X(), t.TranslationPart().Y(), t.TranslationPart().Z()))  
            shape = self.transform_shape(prism, t, r)
            self.prim_shape = update_prism_shape(shape)
        return self.prim_shape