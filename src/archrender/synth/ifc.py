"""Synthetic plans as IFC4 (IfcOpenShell), one or more storeys, with exact ground truth.

Each storey of a :class:`PlanSpec` becomes an ``IfcBuildingStorey`` with:
- ``IfcWall``s: an 'Axis' representation (the centerline: a line, or a polyline for an arc wall),
  a 'Body' (extruded wall band) and an ``IfcMaterialLayerSetUsage`` whose layer thickness is the
  wall thickness;
- ``IfcOpeningElement``s voiding their host wall (``IfcRelVoidsElement``) and filled by an
  ``IfcDoor``/``IfcWindow`` (``IfcRelFillsElement``) with ``Tag``, ``OverallWidth`` and
  ``OverallHeight``; the opening's local placement sits on the wall axis at the opening centre
  (x along the wall, y towards the side a door leaf swings to) and its sill height;
- ``IfcSpace``s with the room number as ``Name``, the room name as ``LongName``, the net floor
  polygon as an extruded footprint and ``Qto_SpaceBaseQuantities.NetFloorArea``.

Project length unit: millimetre (as in most exports), so readers must apply the unit scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import ifcopenshell
import ifcopenshell.api.aggregate
import ifcopenshell.api.context
import ifcopenshell.api.feature
import ifcopenshell.api.geometry
import ifcopenshell.api.material
import ifcopenshell.api.project
import ifcopenshell.api.pset
import ifcopenshell.api.root
import ifcopenshell.api.spatial
import ifcopenshell.api.unit
import ifcopenshell.util.element
import numpy as np

from archrender.synth.plan import PlanSpec

MM = 1000.0  # project units per metre


@dataclass
class IfcStorey:
    spec: PlanSpec
    name: str
    elevation: float


def _matrix(origin: tuple[float, float, float], x_axis: tuple[float, float]) -> np.ndarray:
    ux, uy = x_axis
    m = np.eye(4)
    m[:3, 0] = (ux, uy, 0.0)
    m[:3, 1] = (-uy, ux, 0.0)
    m[:3, 2] = (0.0, 0.0, 1.0)
    m[:3, 3] = origin
    return m


def _polyline_profile(f: ifcopenshell.file, ring: list[tuple[float, float]]) -> Any:
    pts = [f.createIfcCartesianPoint((x * MM, y * MM)) for x, y in ring]
    pts.append(pts[0])
    return f.createIfcArbitraryClosedProfileDef("AREA", None, f.createIfcPolyline(pts))


def _extrusion(f: ifcopenshell.file, context: Any, profile: Any, height: float) -> Any:
    solid = f.createIfcExtrudedAreaSolid(
        profile,
        f.createIfcAxis2Placement3D(f.createIfcCartesianPoint((0.0, 0.0, 0.0))),
        f.createIfcDirection((0.0, 0.0, 1.0)),
        height * MM,
    )
    return f.createIfcShapeRepresentation(context, "Body", "SweptSolid", [solid])


def plan_ifc(storeys: list[IfcStorey], *, project_name: str = "ArchRender sentetik ofis") -> bytes:
    f = ifcopenshell.api.project.create_file(version="IFC4")
    project = ifcopenshell.api.root.create_entity(f, ifc_class="IfcProject", name=project_name)
    mm = ifcopenshell.api.unit.add_si_unit(f, unit_type="LENGTHUNIT", prefix="MILLI")
    m2 = ifcopenshell.api.unit.add_si_unit(f, unit_type="AREAUNIT")
    ifcopenshell.api.unit.assign_unit(f, units=[mm, m2])
    model = ifcopenshell.api.context.add_context(f, context_type="Model")
    body = ifcopenshell.api.context.add_context(
        f, context_type="Model", context_identifier="Body", target_view="MODEL_VIEW", parent=model
    )
    axis_ctx = ifcopenshell.api.context.add_context(
        f, context_type="Model", context_identifier="Axis", target_view="GRAPH_VIEW", parent=model
    )
    site = ifcopenshell.api.root.create_entity(f, ifc_class="IfcSite", name="Arsa")
    building = ifcopenshell.api.root.create_entity(f, ifc_class="IfcBuilding", name="Bina")
    ifcopenshell.api.aggregate.assign_object(f, products=[site], relating_object=project)
    ifcopenshell.api.aggregate.assign_object(f, products=[building], relating_object=site)
    for st in storeys:
        storey = ifcopenshell.api.root.create_entity(f, ifc_class="IfcBuildingStorey", name=st.name)
        storey.Elevation = st.elevation * MM
        ifcopenshell.api.geometry.edit_object_placement(
            f, product=storey, matrix=_matrix((0.0, 0.0, st.elevation), (1.0, 0.0))
        )
        ifcopenshell.api.aggregate.assign_object(f, products=[storey], relating_object=building)
        _storey(f, st, storey, body, axis_ctx)
    text: str = f.to_string()  # type: ignore[no-untyped-call]
    return text.encode("utf-8")


def _storey(f: ifcopenshell.file, st: IfcStorey, storey: Any, body: Any, axis_ctx: Any) -> None:
    spec, z = st.spec, st.elevation
    walls: list[Any] = []
    materials: dict[float, Any] = {}
    for i, w in enumerate(spec.walls):
        wall = ifcopenshell.api.root.create_entity(f, ifc_class="IfcWall", name=f"Duvar {i + 1}")
        wall.PredefinedType = "STANDARD" if not w.exterior else "SOLIDWALL"
        if w.is_arc:
            # axis: polyline in plan coordinates; body: the arc band as an extruded profile
            ifcopenshell.api.geometry.edit_object_placement(
                f, product=wall, matrix=_matrix((0.0, 0.0, z), (1.0, 0.0))
            )
            pts = w.polyline()
            axis_rep = f.createIfcShapeRepresentation(
                axis_ctx,
                "Axis",
                "Curve2D",
                [
                    f.createIfcPolyline(
                        [f.createIfcCartesianPoint((x * MM, y * MM)) for x, y in pts]
                    )
                ],
            )
            left = [
                (x + nx * w.thickness / 2, y + ny * w.thickness / 2)
                for (x, y), (nx, ny) in (
                    (w.point(w.length * k / 48), w.normal(w.length * k / 48)) for k in range(49)
                )
            ]
            right = [
                (x - nx * w.thickness / 2, y - ny * w.thickness / 2)
                for (x, y), (nx, ny) in (
                    (w.point(w.length * k / 48), w.normal(w.length * k / 48)) for k in range(49)
                )
            ]
            band = left + right[::-1]
            body_rep = _extrusion(f, body, _polyline_profile(f, band), spec.storey_height)
        else:
            ux, uy = (w.b[0] - w.a[0]) / w.chord, (w.b[1] - w.a[1]) / w.chord
            ifcopenshell.api.geometry.edit_object_placement(
                f, product=wall, matrix=_matrix((w.a[0], w.a[1], z), (ux, uy))
            )
            axis_rep = ifcopenshell.api.geometry.add_axis_representation(
                f, context=axis_ctx, axis=((0.0, 0.0), (w.chord, 0.0))
            )
            body_rep = ifcopenshell.api.geometry.add_wall_representation(
                f,
                context=body,
                length=w.chord,
                height=spec.storey_height,
                thickness=w.thickness,
                offset=-w.thickness / 2,
            )
        ifcopenshell.api.geometry.assign_representation(f, product=wall, representation=axis_rep)
        ifcopenshell.api.geometry.assign_representation(f, product=wall, representation=body_rep)
        layer_set = materials.get(w.thickness)
        if layer_set is None:
            layer_set = ifcopenshell.api.material.add_material_set(
                f, name=f"Duvar {round(w.thickness * 100)} cm", set_type="IfcMaterialLayerSet"
            )
            mat = ifcopenshell.api.material.add_material(f, name="Tuğla", category="brick")
            layer = ifcopenshell.api.material.add_layer(f, layer_set=layer_set, material=mat)
            layer.LayerThickness = w.thickness * MM
            materials[w.thickness] = layer_set
        ifcopenshell.api.material.assign_material(
            f, products=[wall], type="IfcMaterialLayerSetUsage", material=layer_set
        )
        # the layers start half a thickness to the right of the axis: the axis is the centerline
        usage = ifcopenshell.util.element.get_material(wall)
        assert usage is not None
        usage.OffsetFromReferenceLine = -w.thickness / 2 * MM
        usage.DirectionSense = "POSITIVE"
        ifcopenshell.api.spatial.assign_container(f, products=[wall], relating_structure=storey)
        walls.append(wall)

    for o in spec.openings:
        w = spec.walls[o.wall]
        cx, cy = w.point(o.t)
        ux, uy = w.direction(o.t)
        side = o.swing if o.kind == "door" else 1
        x_axis = (ux, uy) if side > 0 else (-ux, -uy)  # local +y towards the leaf's swing side
        opening = ifcopenshell.api.root.create_entity(
            f, ifc_class="IfcOpeningElement", name=f"Boşluk {o.tag}"
        )
        ifcopenshell.api.geometry.edit_object_placement(
            f, product=opening, matrix=_matrix((cx, cy, z + o.sill), x_axis)
        )
        rep = ifcopenshell.api.geometry.add_wall_representation(
            f,
            context=body,
            length=o.width,
            height=o.height,
            thickness=w.thickness + 0.1,
            offset=-(w.thickness + 0.1) / 2,
        )
        # the representation starts at local x = 0: shift it so the opening is centred
        loc = rep.Items[0].Position.Location.Coordinates
        rep.Items[0].Position.Location.Coordinates = (loc[0] - o.width / 2 * MM, loc[1], loc[2])
        ifcopenshell.api.geometry.assign_representation(f, product=opening, representation=rep)
        ifcopenshell.api.feature.add_feature(f, feature=opening, element=walls[o.wall])
        cls = "IfcDoor" if o.kind == "door" else "IfcWindow"
        filling = ifcopenshell.api.root.create_entity(f, ifc_class=cls, name=o.tag)
        filling.Tag = o.tag
        filling.OverallWidth = o.width * MM
        filling.OverallHeight = o.height * MM
        if o.kind == "door":
            # hinge at the wall-start side of the gap; in local axes that is −x when x runs along
            # the wall, +x when the placement is flipped
            hinge_left = (o.hinge == "start") == (side > 0)
            filling.OperationType = "SINGLE_SWING_LEFT" if hinge_left else "SINGLE_SWING_RIGHT"
        else:
            filling.PartitioningType = "SINGLE_PANEL"
        ifcopenshell.api.geometry.edit_object_placement(
            f, product=filling, matrix=_matrix((cx, cy, z + o.sill), x_axis)
        )
        ifcopenshell.api.feature.add_filling(f, opening=opening, element=filling)
        ifcopenshell.api.spatial.assign_container(f, products=[filling], relating_structure=storey)

    for r in spec.rooms:
        face = spec.faces[r.id]
        space = ifcopenshell.api.root.create_entity(f, ifc_class="IfcSpace", name=r.number)
        space.LongName = r.name
        ifcopenshell.api.geometry.edit_object_placement(
            f, product=space, matrix=_matrix((0.0, 0.0, z), (1.0, 0.0))
        )
        ring = list(face.exterior.coords)[:-1]
        rep = _extrusion(f, body, _polyline_profile(f, ring), r.ceiling)
        ifcopenshell.api.geometry.assign_representation(f, product=space, representation=rep)
        ifcopenshell.api.aggregate.assign_object(f, products=[space], relating_object=storey)
        qto = ifcopenshell.api.pset.add_qto(f, product=space, name="Qto_SpaceBaseQuantities")
        ifcopenshell.api.pset.edit_qto(
            f,
            qto=qto,
            properties={"NetFloorArea": round(face.area, 3), "Height": r.ceiling * MM},
        )
