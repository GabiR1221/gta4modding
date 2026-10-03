bl_info = {
    "name": "GTA IV Blender Tools (ODR/WDD)",
    "author": "Tu si AI-ul",
    "version": (1, 2, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > GTA IV Tool",
    "description": "Importa si exporta modele GTA IV (OpenFormats)",
    "category": "Import-Export",
}

import bpy
import os
import math
import json
import re
from bpy.props import StringProperty, BoolProperty
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper
from mathutils.kdtree import KDTree

SKEL_BONES = []

GTA_BOUNDS = {
    "head_000_r": (-0.09708900, -0.08554300, 0.42933100, 0.09705700, 0.15447900, 0.81180500),
    "hand_000_r": (-0.60209100, -0.03384500, -0.17887400, 0.60211300, 0.07412200, 0.04113800),
    "teef_000_u": (-0.03496700, 0.04197500, 0.60124200, 0.03497400, 0.12686900, 0.64089200),
    "hair_000_u": (-0.08865200, -0.02647200, 0.70185700, 0.09288900, 0.12498800, 0.80639000),
    "feet_000_u": (-0.21986300, -0.08528100, -1.00257700, 0.22450200, 0.25103200, -0.88427900),
    "feet_001_u": (-0.21862800, -0.08968500, -1.00111600, 0.22458200, 0.25101900, -0.85925600),
    "uppr_000_u": (-0.52524100, -0.17419300, -0.06019000, 0.52940500, 0.19806800, 0.59548200),
    "uppr_001_u": (-0.51927100, -0.17414200, -0.03597900, 0.52304100, 0.20494600, 0.58847600),
    "uppr_002_u": (0.13242800, -0.14967300, 0.37146100, 0.17440300, -0.13382300, 0.45940200),
    "lowr_000_u": (-0.22715700, -0.13745200, -0.95996000, 0.22947900, 0.16745400, 0.04506700),
    "lowr_001_u": (-0.22414200, -0.13857100, -0.95130000, 0.22564400, 0.17427200, 0.04853700),
}

def clean_name(name):
    return re.sub(r'\.\d{3}$', '', name)


def decode_gta_weights(packed_weights):
    """Decode GTA IV's cumulative 4-byte skin-weight representation.

    The values in an OpenFormats mesh are cumulative, not four independent
    weights.  For example ``171 214 255 255`` means 171, 43, 41 and 0.
    Treating them as independent weights (or replacing them with 255 0 0 0)
    changes the bind pose and makes heads/hands detach in-game.
    """
    result = []
    previous = 0
    for value in packed_weights[:4]:
        current = max(previous, min(255, int(value)))
        result.append((current - previous) / 255.0)
        previous = current
    return result + [0.0] * (4 - len(result))


def encode_gta_weights(influences):
    """Encode normalized ``(bone_index, weight)`` pairs for GTA IV meshes."""
    influences = [(index, weight) for index, weight in influences if weight > 0.0]
    influences.sort(key=lambda item: item[1], reverse=True)
    influences = influences[:4]
    total = sum(weight for _, weight in influences)
    if total <= 0.0:
        return [0, 0, 0, 0], [0, 0, 0, 0]

    # GTA uses cumulative bytes.  Rounding each independent weight first can
    # leave a total other than 255, so correct the largest influence.
    discrete = [int(round(weight / total * 255.0)) for _, weight in influences]
    discrete[0] += 255 - sum(discrete)
    bone_indices = [index for index, _ in influences]
    while len(bone_indices) < 4:
        bone_indices.append(bone_indices[-1])
        discrete.append(0)

    packed = []
    cumulative = 0
    for weight in discrete:
        cumulative = max(cumulative, min(255, cumulative + weight))
        packed.append(cumulative)
    packed[-1] = 255
    return bone_indices, packed


def has_valid_gta_weights(weights):
    """Return whether a packed GTA IV weight vector can be used as-is.

    A valid packed vector always terminates at 255.  Old versions of this
    addon wrote rigid data such as ``255 0 0 0``; preserving that metadata on a
    later export silently keeps the detached-head bug alive.
    """
    return len(weights) == 4 and all(0 <= int(weight) <= 255 for weight in weights) and weights[-1] == 255


def has_non_rigid_ped_skin(mesh_name, obj, name_to_idx):
    """Ensure head/hand meshes were not inherited from the old rigid export."""
    required_bone = {
        "head_000_r": "Char_Head",
        "hand_000_r": "Char_R_Hand",
    }.get(mesh_name)
    if required_bone is None:
        return True

    required_idx = name_to_idx.get(required_bone)
    for vertex in obj.data.vertices:
        influences = [
            obj.vertex_groups[group.group].name
            for group in vertex.groups
            if group.weight > 0.0 and obj.vertex_groups[group.group].name in name_to_idx
        ]
        # At least one vertex must blend with a bone other than the rigid root.
        if any(name_to_idx[name] != required_idx for name in influences):
            return True
    return False


def mesh_name_from_object(obj):
    """Return the OpenFormats mesh name used for an object."""
    raw_name = clean_name(obj.name)
    return raw_name.replace("_high", "").replace(".mesh", "")


def blender_copy_index(name):
    """Return Blender's numeric copy suffix, or -1 for the original object."""
    match = re.search(r'\.(\d{3})$', name)
    return int(match.group(1)) if match else -1


def unique_export_meshes(objects):
    """Choose one object per output mesh name.

    ``head_000_r`` and ``head_000_r.001`` both write ``head_000_r.mesh``.
    Exporting both also writes duplicate gtaDrawable entries to the ODD.  Prefer
    the highest Blender copy suffix, which is normally the edited replacement.
    """
    chosen = {}
    for obj in objects:
        mesh_name = mesh_name_from_object(obj)
        current = chosen.get(mesh_name)
        if current is None or blender_copy_index(obj.name) > blender_copy_index(current.name):
            chosen[mesh_name] = obj
    return list(chosen.values())


def find_reference_skin_source(target, mesh_name):
    """Find an untouched imported copy of the same GTA mesh in the scene.

    Data Transfer can alter Blender group values and group order.  When the
    original imported head is still present, its raw OpenFormats skin entries
    are the authoritative data.  We transfer those entries by closest vertex
    at export time, avoiding a lossy Blender-group round trip.
    """
    for candidate in bpy.data.objects:
        if candidate == target or candidate.type != 'MESH':
            continue
        if mesh_name_from_object(candidate) != mesh_name:
            continue
        data = candidate.data
        try:
            indices = json.loads(data["gta_bone_indices"])
            weights = json.loads(data["gta_bone_weights"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if (len(indices) == len(data.vertices) and len(weights) == len(data.vertices)
                and all(has_valid_gta_weights(entry) for entry in weights)):
            return candidate, indices, weights
    return None, None, None


def build_reference_skin_tree(reference_vertices):
    """Build a nearest-vertex index once instead of scanning per target vertex."""
    tree = KDTree(len(reference_vertices))
    for index, vertex in enumerate(reference_vertices):
        tree.insert(vertex.co, index)
    tree.balance()
    return tree


def nearest_reference_skin(target_vertex, reference_tree, reference_indices, reference_weights):
    """Return exact raw skin data from the closest reference vertex."""
    _, closest_index, _ = reference_tree.find(target_vertex.co)
    return reference_indices[closest_index], reference_weights[closest_index]

def parse_skel_file(filepath):
    bones = []
    stack = []
    current_bone = None
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("bone "):
            parts = line.split(" ")
            if len(parts) > 1:
                name = parts[1]
                parent_name = stack[-1] if stack else None
                current_bone = {"name": name, "parent": parent_name, "head": (0.0, 0.0, 0.0)}
                bones.append(current_bone)
        elif line.startswith("WorldOffset ") and current_bone:
            parts = line.split(" ")
            if len(parts) == 4:
                current_bone["head"] = (float(parts[1]), float(parts[2]), float(parts[3]))
        elif line.startswith("Children "):
            if current_bone:
                stack.append(current_bone["name"])
        elif line == "}":
            if stack:
                stack.pop()
            current_bone = next((b for b in bones if b["name"] == stack[-1]), None) if stack else None
    return bones


def parse_mesh_file(filepath):
    vertices, faces, uvs = [], [], []
    bone_indices, bone_weights = [], []
    tangents, colors = [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    mode = None
    idx_list = []
    for line in lines:
        line = line.strip()
        if not line or line in "{}":
            continue
        if line.startswith("Idx "):
            mode = "IDX"
            continue
        elif line.startswith("Verts "):
            mode = "VERTS"
            continue
        if mode == "IDX":
            try:
                idx_list.extend([int(x) for x in line.split()])
            except ValueError:
                pass
        elif mode == "VERTS":
            parts = line.split("/")
            if len(parts) >= 7:
                vertices.append([float(x) for x in parts[0].strip().split()])
                colors.append([float(x) for x in parts[2].strip().split()])
                bone_indices.append([int(x) for x in parts[3].strip().split()])
                bone_weights.append([int(x) for x in parts[4].strip().split()])
                tangents.append([float(x) for x in parts[5].strip().split()])
                uv = [float(x) for x in parts[6].strip().split()]
                uvs.append((uv[0], 1.0 - uv[1]))
    for i in range(0, len(idx_list), 3):
        faces.append((idx_list[i], idx_list[i + 1], idx_list[i + 2]))
    return vertices, faces, uvs, bone_indices, bone_weights, tangents, colors


def create_armature(bones, name="GTA_Skeleton"):
    arm_data = bpy.data.armatures.new(name)
    arm_obj = bpy.data.objects.new(name, arm_data)
    bpy.context.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')
    edit_bones = {}
    for b in bones:
        eb = arm_data.edit_bones.new(b["name"])
        eb.head = b["head"]
        eb.tail = (b["head"][0] + 0.1, b["head"][1], b["head"][2])
        edit_bones[b["name"]] = eb
    for b in bones:
        if b["parent"] and b["parent"] in edit_bones:
            edit_bones[b["name"]].parent = edit_bones[b["parent"]]
    for b in bones:
        children = [c for c in bones if c["parent"] == b["name"]]
        if children:
            edit_bones[b["name"]].tail = edit_bones[children[0]["name"]].head
    bpy.ops.object.mode_set(mode='OBJECT')
    return arm_obj


class GTA4_OT_ImportSkel(Operator, ImportHelper):
    bl_idname = "gta4.import_skel"
    bl_label = "Import .skel"
    filename_ext = ".skel"
    filter_glob: StringProperty(default="*.skel", options={'HIDDEN'})

    def execute(self, context):
        global SKEL_BONES
        SKEL_BONES = parse_skel_file(self.filepath)
        create_armature(SKEL_BONES, name=os.path.basename(self.filepath).split(".")[0])
        self.report({'INFO'}, f"Schelet: {len(SKEL_BONES)} oase. Primul: {SKEL_BONES[0]['name']}")
        # Debug: print bone indices for known bones
        for i, b in enumerate(SKEL_BONES):
            if b["name"] in ("Char_Head", "Char_Neck", "Char_Spine3", "Char_R_Hand", "Char_L_Foot", "Char_R_Foot"):
                print(f"[GTA4 SKEL] idx={i} bone={b['name']} pos={b['head']}")
        return {'FINISHED'}


class GTA4_OT_ImportMesh(Operator, ImportHelper):
    bl_idname = "gta4.import_mesh"
    bl_label = "Import .mesh (Skinning)"
    filename_ext = ".mesh"
    filter_glob: StringProperty(default="*.mesh", options={'HIDDEN'})

    def execute(self, context):
        global SKEL_BONES
        filepath = self.filepath
        vertices, faces, uvs, bone_indices, bone_weights, tangents, colors = parse_mesh_file(filepath)
        mesh_name = os.path.basename(filepath).split(".")[0]
        mesh = bpy.data.meshes.new(mesh_name)
        mesh.from_pydata(vertices, [], faces)
        mesh.update()
        uv_layer = mesh.uv_layers.new()
        for loop in mesh.loops:
            if loop.vertex_index < len(uvs):
                uv_layer.data[loop.index].uv = uvs[loop.vertex_index]
        obj = bpy.data.objects.new(mesh_name, mesh)
        context.collection.objects.link(obj)

        mesh["gta_bone_indices"] = json.dumps(bone_indices)
        mesh["gta_bone_weights"] = json.dumps(bone_weights)
        mesh["gta_tangents"] = json.dumps(tangents)
        mesh["gta_colors"] = json.dumps(colors)

        if SKEL_BONES:
            idx_to_name = {i: b["name"] for i, b in enumerate(SKEL_BONES)}
            for b in SKEL_BONES:
                obj.vertex_groups.new(name=b["name"])
            for v_idx, (b_idx, raw_wgt) in enumerate(zip(bone_indices, bone_weights)):
                decoded_weights = decode_gta_weights(raw_wgt)
                weights_by_bone = {}
                for i, idx in enumerate(b_idx):
                    if idx in idx_to_name and i < len(decoded_weights) and decoded_weights[i] > 0.0:
                        weights_by_bone[idx] = weights_by_bone.get(idx, 0.0) + decoded_weights[i]
                for idx, weight in weights_by_bone.items():
                    obj.vertex_groups[idx_to_name[idx]].add([v_idx], weight, 'REPLACE')
            arm_obj = next((o for o in bpy.data.objects if o.type == 'ARMATURE'), None)
            if arm_obj:
                obj.parent = arm_obj
                mod = obj.modifiers.new(name="Armature", type='ARMATURE')
                mod.object = arm_obj
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        context.view_layer.objects.active = obj
        return {'FINISHED'}


class GTA4_OT_ExportWDD(Operator, ImportHelper):
    bl_idname = "gta4.export_wdd"
    bl_label = "Export .mesh & .odd"
    filename_ext = ".odd"
    filter_glob: StringProperty(default="*.odd", options={'HIDDEN'})

    def execute(self, context):
        global SKEL_BONES
        if not SKEL_BONES:
            self.report({'ERROR'}, "Importa .skel mai intai!")
            return {'CANCELLED'}

        selected = [o for o in context.selected_objects if o.type == 'MESH']
        if not selected:
            self.report({'ERROR'}, "Selecteaza cel putin un mesh!")
            return {'CANCELLED'}
        export_meshes = unique_export_meshes(selected)
        if len(export_meshes) != len(selected):
            skipped = len(selected) - len(export_meshes)
            self.report({'WARNING'}, f"{skipped} duplicat(e) ignorat(e): export un singur mesh pentru fiecare nume GTA.")
            print(f"[GTA4 EXPORT] Ignor duplicate Blender: {len(selected)} selectate -> {len(export_meshes)} mesh-uri GTA")

        filepath = self.filepath
        if filepath.endswith(('/', '\\')) or os.path.isdir(filepath):
            filepath = os.path.join(filepath, "ig_roman.odd")
        if not filepath.lower().endswith(".odd"):
            filepath += ".odd"
        base_dir = os.path.dirname(filepath)
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        mesh_dir = os.path.join(base_dir, base_name)
        if not os.path.exists(mesh_dir):
            os.makedirs(mesh_dir)

        name_to_idx = {b["name"]: i for i, b in enumerate(SKEL_BONES)}
        skel_name = "ig_roman"

        # Debug: verify Char_Head exists in skeleton
        if "Char_Head" not in name_to_idx:
            self.report({'ERROR'}, f"Char_Head NU EXISTA in schelet! Bones: {list(name_to_idx.keys())[:20]}")
            return {'CANCELLED'}
        char_head_idx = name_to_idx["Char_Head"]
        self.report({'INFO'}, f"Char_Head are index {char_head_idx} in schelet")
        print(f"[GTA4 EXPORT] Char_Head index = {char_head_idx}")

        SHADER_MAP = {
            "head_000_r": "gta_ped.sps head_diff_000_a_whi head_normal_000 35.00000000 0.20000000 1.00000000;0.00000000;0.00000000 head_spec_000 1.00000000",
            "hand_000_r": "gta_ped.sps hand_diff_000_a_whi hand_normal_000 35.00000000 0.20000000 1.00000000;0.00000000;0.00000000 hand_spec_000 1.00000000",
            "teef_000_u": "gta_ped.sps teef_diff_000_a_uni teef_normal_000 35.00000000 0.20000000 1.00000000;0.00000000;0.00000000 teef_spec_000 1.00000000",
            "hair_000_u": "gta_ped.sps givemechecker givemechecker 35.00000000 0.20000000 1.00000000;0.00000000;0.00000000 givemechecker 1.00000000",
            "uppr_000_u": "gta_ped.sps uppr_diff_000_a_uni uppr_normal_000 35.00000000 0.20000000 1.00000000;0.00000000;0.00000000 uppr_spec_000 1.00000000",
            "uppr_001_u": "gta_ped.sps uppr_diff_001_a_uni uppr_normal_001 35.00000000 0.20000000 1.00000000;0.00000000;0.00000000 uppr_spec_001 1.00000000",
            "uppr_002_u": "gta_ped.sps uppr_diff_002_a_uni uppr_normal_002 35.00000000 0.20000000 1.00000000;0.00000000;0.00000000 uppr_spec_002 1.00000000",
            "lowr_000_u": "gta_ped.sps lowr_diff_000_a_uni lowr_normal_000 35.00000000 0.20000000 1.00000000;0.00000000;0.00000000 lowr_spec_000 1.00000000",
            "lowr_001_u": "gta_ped.sps lowr_diff_001_a_uni lowr_normal_001 35.00000000 0.20000000 1.00000000;0.00000000;0.00000000 lowr_spec_001 1.00000000",
            "feet_000_u": "gta_ped.sps feet_diff_000_a_uni feet_normal_000 35.00000000 0.20000000 1.00000000;0.00000000;0.00000000 feet_spec_000 1.00000000",
            "feet_001_u": "gta_ped.sps feet_diff_001_a_uni feet_normal_001 35.00000000 0.20000000 1.00000000;0.00000000;0.00000000 feet_spec_001 1.00000000",
        }

        odd_content = "Version 110 12\n{\n"

        for obj in export_meshes:
            raw_name = clean_name(obj.name)
            mesh_name = raw_name.replace("_high", "").replace(".mesh", "")
            mesh_filename = f"{raw_name}.mesh" if not raw_name.endswith(".mesh") else raw_name
            mesh_filepath = os.path.join(mesh_dir, mesh_filename)

            # IMPORTANT: aplicam transformarile obiectului inainte de a citi vertexii
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            context.view_layer.objects.active = obj
            try:
                bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
            except Exception as e:
                self.report({'WARNING'}, f"Nu am putut aplica transformarile: {e}")

            mesh_data = obj.data
            mesh_data.calc_loop_triangles()

            print(f"\n[GTA4 EXPORT] ====== Procesez obiect: '{obj.name}' -> mesh_name='{mesh_name}' ======")
            print(f"[GTA4 EXPORT] Numar vertices: {len(mesh_data.vertices)}")

            # AUTO-ALIGN
            xs = [v.co.x for v in mesh_data.vertices]
            ys = [v.co.y for v in mesh_data.vertices]
            zs = [v.co.z for v in mesh_data.vertices]
            cur_min_x, cur_max_x = min(xs), max(xs)
            cur_min_y, cur_max_y = min(ys), max(ys)
            cur_min_z, cur_max_z = min(zs), max(zs)
            print(f"[GTA4 EXPORT] Bounds BEFORE align: X[{cur_min_x:.4f}, {cur_max_x:.4f}] Y[{cur_min_y:.4f}, {cur_max_y:.4f}] Z[{cur_min_z:.4f}, {cur_max_z:.4f}]")

            if mesh_name in GTA_BOUNDS:
                tgt_min_x, tgt_min_y, tgt_min_z, tgt_max_x, tgt_max_y, tgt_max_z = GTA_BOUNDS[mesh_name]
                cur_size_x = cur_max_x - cur_min_x
                cur_size_y = cur_max_y - cur_min_y
                cur_size_z = cur_max_z - cur_min_z
                tgt_size_x = tgt_max_x - tgt_min_x
                tgt_size_y = tgt_max_y - tgt_min_y
                tgt_size_z = tgt_max_z - tgt_min_z
                if cur_size_x > 0 and cur_size_y > 0 and cur_size_z > 0:
                    sx = tgt_size_x / cur_size_x
                    sy = tgt_size_y / cur_size_y
                    sz = tgt_size_z / cur_size_z
                    for v in mesh_data.vertices:
                        v.co.x = (v.co.x - cur_min_x) * sx + tgt_min_x
                        v.co.y = (v.co.y - cur_min_y) * sy + tgt_min_y
                        v.co.z = (v.co.z - cur_min_z) * sz + tgt_min_z
                    mesh_data.update()
                    print(f"[GTA4 EXPORT] AUTO-ALIGNAT! Scale: ({sx:.3f}, {sy:.3f}, {sz:.3f})")
                    self.report({'INFO'}, f"Auto-aliniat {mesh_name} (scale {sx:.2f})")
                else:
                    print(f"[GTA4 EXPORT] WARN: dimensiune 0, skip auto-align")
            else:
                print(f"[GTA4 EXPORT] WARN: '{mesh_name}' NU e in GTA_BOUNDS, SKIP auto-align!")
                self.report({'WARNING'}, f"{mesh_name} nu e in GTA_BOUNDS - auto-align SKIP!")

            # Recalculeaza bounds
            xs = [v.co.x for v in mesh_data.vertices]
            ys = [v.co.y for v in mesh_data.vertices]
            zs = [v.co.z for v in mesh_data.vertices]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            min_z, max_z = min(zs), max(zs)
            print(f"[GTA4 EXPORT] Bounds AFTER align: X[{min_x:.4f}, {max_x:.4f}] Y[{min_y:.4f}, {max_y:.4f}] Z[{min_z:.4f}, {max_z:.4f}]")

            center_str = f"{(min_x + max_x) / 2:.8f} {(min_y + max_y) / 2:.8f} {(min_z + max_z) / 2:.8f}"
            aabbmin_str = f"{min_x:.8f} {min_y:.8f} {min_z:.8f}"
            aabbmax_str = f"{max_x:.8f} {max_y:.8f} {max_z:.8f}"
            radius_str = f"{math.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2 + (max_z - min_z) ** 2) / 2:.8f}"

            # Recuperare date originale
            stored_indices = None
            stored_weights = None
            stored_tangents = None
            stored_colors = None
            try:
                if "gta_bone_indices" in mesh_data.keys():
                    stored_indices = json.loads(mesh_data["gta_bone_indices"])
                    stored_weights = json.loads(mesh_data["gta_bone_weights"])
                    stored_tangents = json.loads(mesh_data["gta_tangents"])
                    stored_colors = json.loads(mesh_data["gta_colors"])
            except Exception as e:
                self.report({'WARNING'}, f"Eroare: {e}")

            # Preserve raw data for an untouched imported mesh.  For edited
            # geometry Blender vertex groups are the source of truth: edit
            # operations interpolate them, while old per-vertex metadata no
            # longer matches the changed vertex count.
            has_vertex_group_weights = any(
                any(group.weight > 0.0 and obj.vertex_groups[group.group].name in name_to_idx
                    for group in vertex.groups)
                for vertex in mesh_data.vertices
            )
            use_original = (stored_indices is not None
                            and len(stored_indices) == len(mesh_data.vertices)
                            and all(has_valid_gta_weights(weights) for weights in stored_weights))

            reference_obj = None
            reference_indices = None
            reference_weights = None
            head_reference_obj = None
            # A GTA IV facial head needs its original per-vertex skin mapping.
            # Data Transfer cannot reproduce that mapping reliably when the
            # replacement has a different vertex count/topology.
            if not use_original and mesh_name != "head_000_r":
                reference_obj, reference_indices, reference_weights = find_reference_skin_source(obj, mesh_name)
            elif not use_original and mesh_name == "head_000_r":
                head_reference_obj, _, _ = find_reference_skin_source(obj, mesh_name)
                if (head_reference_obj is not None
                        and len(head_reference_obj.data.vertices) != len(mesh_data.vertices)):
                    self.report({'ERROR'},
                        "head_000_r: topologie diferita. Porneste din capul original, NU adauga/sterge vertex-uri, apoi modeleaza-l.")
                    print("[GTA4 EXPORT] STOP head_000_r: "
                          f"{len(mesh_data.vertices)} verts vs referinta {len(head_reference_obj.data.vertices)}. "
                          "Skin-ul facial GTA IV necesita topologia originala.")
                    return {'CANCELLED'}
            use_reference_skin = reference_obj is not None
            has_safe_ped_skin = has_non_rigid_ped_skin(mesh_name, obj, name_to_idx)

            if use_original:
                print(f"[GTA4 EXPORT] Folosesc metadata originala ({len(stored_indices)} verts)")
            elif use_reference_skin:
                print(f"[GTA4 EXPORT] Transfer skin raw din '{reference_obj.name}' ({len(reference_indices)} verts)")
            elif has_vertex_group_weights and has_safe_ped_skin:
                print("[GTA4 EXPORT] Folosesc vertex groups Blender si encodez weights cumulative GTA IV")
            else:
                if mesh_name in ("head_000_r", "hand_000_r") and not has_safe_ped_skin:
                    self.report({'ERROR'}, f"{mesh_name}: skinning rigid detectat. Importa mesh-ul ORIGINAL, transfera weights, apoi exporta.")
                else:
                    self.report({'ERROR'}, f"{mesh_name}: mesh fara skin weights. Importa mesh-ul original si pastreaza/transfera vertex groups.")
                return {'CANCELLED'}

            reference_skin_tree = None
            if use_reference_skin:
                reference_skin_tree = build_reference_skin_tree(reference_obj.data.vertices)

            with open(mesh_filepath, 'w', encoding='utf-8', newline='\r\n') as f:
                f.write("Version 11 13\n{\n\tSkinned 1\n\tMtl 0\n\t{\n\t\tPrim 0\n\t\t{\n")
                tri_count = len(mesh_data.loop_triangles)
                f.write(f"\t\t\tIdx {tri_count * 3}\n\t\t\t{{\n")
                buffer = []
                for tri in mesh_data.loop_triangles:
                    buffer.extend([tri.vertices[0], tri.vertices[1], tri.vertices[2]])
                    if len(buffer) >= 15:
                        f.write("\t\t\t\t" + " ".join(str(x) for x in buffer) + "\n")
                        buffer = []
                if buffer:
                    f.write("\t\t\t\t" + " ".join(str(x) for x in buffer) + "\n")
                f.write("\t\t\t}\n")
                f.write(f"\t\t\tVerts {len(mesh_data.vertices)}\n\t\t\t{{\n")
                uv_layer = mesh_data.uv_layers.active.data if mesh_data.uv_layers.active else None

                for v_idx, v in enumerate(mesh_data.vertices):
                    pos = f"{v.co.x:.8f} {v.co.y:.8f} {v.co.z:.8f}"
                    norm = f"{v.normal.x:.8f} {v.normal.y:.8f} {v.normal.z:.8f}"

                    if use_original:
                        b_idx = stored_indices[v_idx]
                        b_wgt = stored_weights[v_idx]
                        tang = stored_tangents[v_idx]
                        col = stored_colors[v_idx]
                    elif use_reference_skin:
                        b_idx, b_wgt = nearest_reference_skin(
                            v, reference_skin_tree, reference_indices, reference_weights)
                        tang = [0.0, 0.0, 0.0, 1.0]
                        col = [1.0, 1.0, 1.0, 0.0]
                    else:
                        vg_weights = []
                        for g in v.groups:
                            if g.weight > 0.0:
                                gname = obj.vertex_groups[g.group].name
                                if gname in name_to_idx:
                                    vg_weights.append((name_to_idx[gname], g.weight))
                        b_idx, b_wgt = encode_gta_weights(vg_weights)
                        tang = [0.0, 0.0, 0.0, 1.0]
                        col = [1.0, 1.0, 1.0, 0.0]

                    col_str = f"{col[0]:.8f} {col[1]:.8f} {col[2]:.8f} {col[3]:.8f}"
                    bones_str = f"{b_idx[0]} {b_idx[1]} {b_idx[2]} {b_idx[3]}"
                    weights_str = f"{b_wgt[0]} {b_wgt[1]} {b_wgt[2]} {b_wgt[3]}"
                    tang_str = f"{tang[0]:.8f} {tang[1]:.8f} {tang[2]:.8f} {tang[3]:.8f}"

                    uv = (0.0, 0.0)
                    if uv_layer:
                        for loop_idx, loop in enumerate(mesh_data.loops):
                            if loop.vertex_index == v_idx:
                                uv = uv_layer[loop_idx].uv
                                break
                    uv_str = f"{uv[0]:.8f} {1.0 - uv[1]:.8f}"

                    f.write(f"\t\t\t\t{pos} / {norm} / {col_str} / {bones_str} / {weights_str} / {tang_str} / {uv_str} / 0.0 0.0\n")
                f.write("\t\t\t}\n\t\t}\n\t}\n}\n")

            print(f"[GTA4 EXPORT] Scris: {mesh_filepath}")

            shader_line = SHADER_MAP.get(mesh_name, "gta_ped.sps givemechecker givemechecker 35.00000000 0.20000000 1.00000000;0.00000000;0.00000000 givemechecker 1.00000000")
            odd_content += f"\tgtaDrawable {mesh_name}\n\t{{\n"
            odd_content += "\t\tshadinggroup\n\t\t{\n\t\t\tShaders 1\n\t\t\t{\n"
            odd_content += f"\t\t\t\t{shader_line}\n"
            odd_content += "\t\t\t}\n\t\t}\n"
            odd_content += f"\t\tskel\n\t\t{{\n\t\t\tskel {skel_name}\\{skel_name}.skel\n\t\t}}\n"
            odd_content += "\t\tlodgroup\n\t\t{\n"
            odd_content += f"\t\t\thigh 1 {base_name}\\{mesh_filename} 0 9999.00000000\n"
            odd_content += "\t\t\tmed none 9999.00000000\n\t\t\tlow none 9999.00000000\n\t\t\tvlow none 9999.00000000\n"
            odd_content += f"\t\t\tcenter {center_str}\n"
            odd_content += f"\t\t\tAABBMin {aabbmin_str}\n"
            odd_content += f"\t\t\tAABBMax {aabbmax_str}\n"
            odd_content += f"\t\t\tradius {radius_str}\n"
            odd_content += "\t\t}\n\t}\n"
        odd_content += "}\n"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(odd_content)
        self.report({'INFO'}, f"Exportat in {base_dir}! Verifica consola Blender pt detalii.")
        return {'FINISHED'}


class GTA4_PT_MainPanel(bpy.types.Panel):
    bl_label = "GTA IV Tools"
    bl_idname = "GTA4_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'GTA IV Tool'

    def draw(self, context):
        layout = self.layout
        layout.label(text="1. Import:")
        layout.operator(GTA4_OT_ImportSkel.bl_idname, icon='ARMATURE_DATA')
        layout.operator(GTA4_OT_ImportMesh.bl_idname, icon='MESH_DATA')
        layout.separator()
        layout.label(text="2. Export:")
        layout.operator(GTA4_OT_ExportWDD.bl_idname, icon='EXPORT')
        layout.separator()
        layout.label(text="Vezi consola pt debug!")


def register():
    bpy.utils.register_class(GTA4_OT_ImportSkel)
    bpy.utils.register_class(GTA4_OT_ImportMesh)
    bpy.utils.register_class(GTA4_OT_ExportWDD)
    bpy.utils.register_class(GTA4_PT_MainPanel)


def unregister():
    bpy.utils.unregister_class(GTA4_OT_ImportSkel)
    bpy.utils.unregister_class(GTA4_OT_ImportMesh)
    bpy.utils.unregister_class(GTA4_OT_ExportWDD)
    bpy.utils.unregister_class(GTA4_PT_MainPanel)


if __name__ == "__main__":
    register()
