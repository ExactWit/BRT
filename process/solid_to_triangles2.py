import argparse
import logging
import os
import pathlib
import time
from collections import deque
from itertools import repeat
from multiprocessing import Process, Queue, get_context
from multiprocessing.pool import Pool
import signal
import torch
import numpy as np
from occwl.graph import face_adjacency
from OCC.Core.BRep import BRep_Tool
from occwl.face import Face
from occwl.edge import Edge
from occwl.geometry import geom_utils
from tqdm import tqdm
from OCC.Core.GeomConvert import GeomConvert_BSplineSurfaceToBezierSurface as Converter, geomconvert
from OCC.Core.GeomConvert import GeomConvert_BSplineCurveToBezierCurve
from OCC.Core.Geom import Geom_BSplineSurface
from OCC.Core.TColStd import TColStd_Array1OfReal as ArrReal
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRepTools import breptools
from OCC.Core.Geom import Geom_RectangularTrimmedSurface, Geom_TrimmedCurve
from occwl.compound import Compound
from utils.triangle import Triangle
from utils.sampling import randn_uvgrid, ugrid
from triangles3 import Rectangle,splitRectangle,HandleLeaves,CollectTrisInLine,make_rect,HandleLeavesRectangle,CollectRectangles
import triangles3
from solid_to_brt import build_data as build_BRT, build_data_no_label as build_BRT_no_label
import uuid
from numpy.linalg import norm
from mechcad_failure import append_failure_record, make_failure_record


def rotation_matrix_to_z_axis(v):
    assert np.isclose(norm(v), 1.0), "v must be unit vector"

    target = np.array([0, 0, 1])
    dot = np.dot(v, target)

    # 如果 v 已经与 target 同向，无需旋转
    if np.isclose(dot, 1.0):
        return np.eye(3)

    # 如果 v 与 target 反向，旋转 180 度（绕任意垂直轴）
    if np.isclose(dot, -1.0):
        # 选择一个垂直于 v 的轴（如 [1, 0, 0] 或 [0, 1, 0]）
        if not np.isclose(v[0], 1.0):
            axis = np.array([1, 0, 0])
        else:
            axis = np.array([0, 1, 0])
        axis = axis - np.dot(axis, v) * v
        axis = axis / norm(axis)
        return rotation_matrix_from_axis_angle(axis, np.pi)

    # 一般情况：计算旋转轴和角度
    axis = np.cross(v, target)
    axis = axis / norm(axis)
    angle = np.arccos(dot)
    return rotation_matrix_from_axis_angle(axis, angle)


def rotation_matrix_from_axis_angle(axis, angle):
    """用 Rodrigues 公式从旋转轴和角度构造旋转矩阵"""
    axis = axis / norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


def edge_fn(e):
    return convertEdgeToBeziers2(e, max_knots=150, sampling=False)[0]


def convertFaceToTriangles(
    face: Face, num_sample_points=256, normalize=True, trim=True, rotated_and_normalized=True, **kwargs
):
    """
    Convert a face to a graph
    Args:
        face (occwl.face.Face): Face
    Returns:
        dgl graph
    """

    surface, loc = getNURBS(face)

    # knot insertion
    doKnotInsertion(surface, num_max_knots=5)

    converter = Converter(surface)

    uNumPatches = converter.NbUPatches()
    vNumPatches = converter.NbVPatches()

    if uNumPatches == 0 or vNumPatches == 0:
        raise RuntimeError("no patches")

    uKnots = ArrReal(1, uNumPatches + 1)
    vKnots = ArrReal(1, vNumPatches + 1)

    converter.UKnots(uKnots)
    converter.VKnots(vKnots)

    rects = []
    for u in range(uNumPatches):
        for v in range(vNumPatches):
            rect = Rectangle()
            rect.points = [
                (uKnots[u], vKnots[v]),
                (uKnots[u + 1], vKnots[v]),
                (uKnots[u], vKnots[v + 1]),
                (uKnots[u + 1], vKnots[v + 1]),
            ]
            rects.append(rect)

    if trim:
        crvs = []
        for wire in face.wires():
            for edge in wire.ordered_edges():
                edge: Edge
                crv, interval = pcurve(face, edge)
                crvs.append((crv, interval))
        tris = []
        with triangles3.suppress_subdivsion_err():
            for rect in rects:
                splitRectangle(face, rect, crvs, max_split=5)
                HandleLeaves(face, rect, surface, loc)
                CollectTrisInLine(rect, tris, face, surface, loc)
    else:
        tris = []
        for rect in rects:
            rect.leaf_info = make_rect(face, rect, surface, loc)
            CollectTrisInLine(rect, tris, face, surface, loc)

    def getTriNormal(tri: Triangle):
        x = (tri.v1[0] + tri.v2[0] + tri.v3[0]) / 3
        y = (tri.v1[1] + tri.v2[1] + tri.v3[1]) / 3
        return face.normal([x, y])

    nodes = [tri.control_points if type(tri) is not tuple else tri[1].control_points for tri in tris]
    nodes = np.stack(nodes)

    mask = [type(tri) is not tuple for tri in tris]
    in_mask = np.array(mask)

    tri_normals = [getTriNormal(tri) if type(tri) is not tuple else getTriNormal(tri[1]) for tri in tris]
    tri_normals = np.stack(tri_normals)

    if rotated_and_normalized:
        new_feature = np.zeros((len(tri_normals), 7), dtype=tri_normals.dtype)
        new_feature[:, :3] = tri_normals
        for i in range(len(nodes)):
            R = rotation_matrix_to_z_axis(tri_normals[i])
            nodes[i][..., :3] = (R @ nodes[i][..., :3].T).T

            x = nodes[..., 0]
            y = nodes[..., 1]
            z = nodes[..., 2]
            bbox = [[x.min(), y.min(), z.min()], [x.max(), y.max(), z.max()]]
            bbox = np.array(bbox)

            diag = bbox[1] - bbox[0]
            scale = 2.0 / max(diag[0], diag[1], diag[2])
            center = 0.5 * (bbox[0] + bbox[1])

            nodes[..., :3] -= center
            if not np.isnan(scale).any():
                nodes[..., :3] *= scale
            else:
                scale = 1
            new_feature[:, 3:6] = center
            new_feature[:,6] = scale
        tri_normals = new_feature

    # sample some points as label
    points, uv_values = randn_uvgrid(
        face,
        method="point",
        num=num_sample_points,
        bounds=[uKnots[0], uKnots[uNumPatches], vKnots[0], vKnots[vNumPatches]],
    )

    normals = randn_uvgrid(
        face,
        method="normal",
        num=num_sample_points,
        bounds=[uKnots[0], uKnots[uNumPatches], vKnots[0], vKnots[vNumPatches]],
        given_uvs=uv_values,
        uvs=False,
    )

    visibility_status = randn_uvgrid(
        face,
        method="visibility_status",
        num=num_sample_points,
        bounds=[uKnots[0], uKnots[uNumPatches], vKnots[0], vKnots[vNumPatches]],
        given_uvs=uv_values,
        uvs=False,
    )
    mask = np.logical_or(visibility_status == 0, visibility_status == 2)  # 0: Inside, 1: Outside, 2: On boundary

    # normalize
    if normalize:
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        bbox = [[x.min(), y.min(), z.min()], [x.max(), y.max(), z.max()]]
        bbox = np.array(bbox)

        diag = bbox[1] - bbox[0]
        scale = 2.0 / max(diag[0], diag[1], diag[2])
        center = 0.5 * (bbox[0] + bbox[1])

        points -= center
        points *= scale

        nodes[..., :3] -= center
        nodes[..., :3] *= scale
    else:
        scale = 1.0
        center = np.zeros(3)

    points = torch.from_numpy(points)
    uv_values = torch.from_numpy(uv_values)
    vis_mask = torch.from_numpy(mask)
    scale = torch.tensor(scale)
    normals = torch.tensor(normals)

    nodes = torch.from_numpy(nodes)
    in_mask = torch.from_numpy(in_mask)
    tri_normals = torch.from_numpy(tri_normals)

    return nodes, in_mask, tri_normals, points, normals, vis_mask, uv_values, scale


#####################################################################################
def convertFaceToRectangleBeziers(face:Face,num_sample_points=256,normalize=True,trim=True,rotated_and_normalized=True,**kwargs):
    surface, loc = getNURBS(face)

    # knot insertion
    doKnotInsertion(surface,num_max_knots=5)

    converter = Converter(surface)

    uNumPatches = converter.NbUPatches()
    vNumPatches = converter.NbVPatches()

    if uNumPatches==0 or vNumPatches==0:
        raise RuntimeError('no patches')


    uKnots = ArrReal(1, uNumPatches+1)
    vKnots = ArrReal(1, vNumPatches+1)

    converter.UKnots(uKnots)
    converter.VKnots(vKnots)

    rects=[]
    for u in range(uNumPatches):
        for v in range(vNumPatches):
            rect=Rectangle()
            rect.points=[(uKnots[u],vKnots[v]),(uKnots[u+1],vKnots[v]),(uKnots[u],vKnots[v+1]),(uKnots[u+1],vKnots[v+1])]
            rects.append(rect)

    if trim:
        crvs=[]
        for wire in face.wires():
            for edge in wire.ordered_edges():
                edge:Edge 
                crv,interval=pcurve(face,edge)
                crvs.append((crv,interval))
        output_lst=[]
        with triangles3.suppress_subdivsion_err():
            for rect in rects:
                splitRectangle(face,rect,crvs,max_split=5)
                HandleLeavesRectangle(face,rect,surface,loc) 
                CollectRectangles(rect,output_lst,face,surface,loc)
    else:
        raise NotImplementedError


    nodes=[r[1] for r  in output_lst]
    nodes=np.stack(nodes)
    nodes=nodes.reshape(nodes.shape[0],-1,nodes.shape[-1])

    mask=[True for r in output_lst]
    in_mask=np.array(mask)

    rec_normals=[r[0] for r in output_lst]
    rec_normals=np.stack(rec_normals)

    if rotated_and_normalized:
        new_feature=np.zeros((len(rec_normals),7),dtype=rec_normals.dtype)
        new_feature[:,:3]=rec_normals
        for i in range(len(nodes)):
            R=rotation_matrix_to_z_axis(rec_normals[i])
            nodes[i][...,:3]= (R@nodes[i][...,:3].T).T

            x = nodes[..., 0]
            y = nodes[..., 1]
            z = nodes[..., 2]
            bbox = [[x.min(), y.min(), z.min()], [x.max(), y.max(), z.max()]]
            bbox = np.array(bbox)

            diag = bbox[1] - bbox[0]
            scale = 2.0 / max(diag[0], diag[1], diag[2])
            center = 0.5 * (bbox[0] + bbox[1])

            nodes[...,:3] -= center
            if not np.isnan(scale).any():
                nodes[...,:3] *= scale
            else:
                scale=1
            new_feature[:,3:6]=center
            new_feature[:,6]=scale
        rec_normals=new_feature

    points, uv_values = randn_uvgrid(
        face, method="point",num=num_sample_points,
        bounds=[uKnots[0], uKnots[uNumPatches], vKnots[0], vKnots[vNumPatches]]
    )

    normals = randn_uvgrid(
        face, method="normal",num=num_sample_points,
        bounds=[uKnots[0], uKnots[uNumPatches], vKnots[0], vKnots[vNumPatches]],given_uvs=uv_values,uvs=False
    )

    visibility_status = randn_uvgrid(
        face, method="visibility_status",num=num_sample_points,
        bounds=[uKnots[0], uKnots[uNumPatches], vKnots[0], vKnots[vNumPatches]],given_uvs=uv_values,uvs=False
    )
    mask = np.logical_or(visibility_status == 0, visibility_status == 2)  # 0: Inside, 1: Outside, 2: On boundary

    # normalize
    if normalize:
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        bbox = [[x.min(), y.min(), z.min()], [x.max(), y.max(), z.max()]]
        bbox = np.array(bbox)

        diag = bbox[1] - bbox[0]
        scale = 2.0 / max(diag[0], diag[1], diag[2])
        center = 0.5 * (bbox[0] + bbox[1])

        points -= center
        points *= scale

        nodes[...,:3] -= center
        nodes[...,:3] *= scale
    else:
        scale=1.0
        center=np.zeros(3)

    points = torch.from_numpy(points)
    uv_values = torch.from_numpy(uv_values)
    vis_mask=torch.from_numpy(mask)
    scale = torch.tensor(scale)
    normals=torch.tensor(normals)
    
    nodes=torch.from_numpy(nodes)
    in_mask=torch.from_numpy(in_mask)
    rec_normals=torch.from_numpy(rec_normals)

    return nodes,in_mask,rec_normals,points,normals,vis_mask,uv_values, scale

def convertEdgeToBeziers2(edge: Edge, degree=10, max_knots=100, sampling=True):
    crvdata = BRep_Tool.Curve(edge.topods_shape())
    if len(crvdata) == 2:
        return None
    else:
        crv, first, last = crvdata
    vertex_0, vertex_1 = geom_utils.gp_to_numpy(crv.Value(first)), geom_utils.gp_to_numpy(crv.Value(last))
    crv = Geom_TrimmedCurve(crv, first, last)
    crv = geomconvert.CurveToBSplineCurve(crv)

    n_uknots = crv.NbKnots()
    if n_uknots <= 1:
        raise RuntimeError("bad nurbs")

    deg = crv.Degree()
    torch._assert(deg <= degree, f"degree {deg} too high")
    if deg < degree:
        crv.IncreaseDegree(degree)
    converter = GeomConvert_BSplineCurveToBezierCurve(crv)
    NbArcs = converter.NbArcs()
    control_points = np.zeros((NbArcs, degree + 1, 4), dtype=np.float32)

    for i in range(NbArcs):
        arc = converter.Arc(i + 1)
        for j in range(degree + 1):
            p = arc.Pole(j + 1)
            W = arc.Weight(j + 1)
            control_points[i, j] = p.X(), p.Y(), p.Z(), W

    if not sampling:
        return control_points, vertex_0, vertex_1, None, None

    sampled_points, us = ugrid(crv, (crv.Knot(1), crv.Knot(crv.NbKnots())), method="point", num_u=100, us=True)
    sampled_normals = ugrid(crv, (crv.Knot(1), crv.Knot(crv.NbKnots())), method="tangent", num_u=100)

    vertex_0 = sampled_points[0]
    vertex_1 = sampled_points[-1]
    if (
        np.linalg.norm(vertex_0 - control_points[0, 0, :3]) > 1e-3
        or np.linalg.norm(vertex_1 - control_points[-1, -1, :3]) > 1e-3
    ):
        logging.info("bad vertex: " + str(vertex_0) + " " + str(vertex_1) + " " + str(control_points))

    return control_points, vertex_0, vertex_1, sampled_points, sampled_normals, us


def getNURBS(face: Face):

    loc = TopLoc_Location()
    surface_brep = BRep_Tool().Surface(face.topods_shape(), loc)
    bound_box = breptools().UVBounds(face.topods_shape())

    # determine the face bound
    face_bound = surface_brep.Bounds()
    true_bound = [
        max(bound_box[0], face_bound[0]),
        min(bound_box[1], face_bound[1]),
        max(bound_box[2], face_bound[2]),
        min(bound_box[3], face_bound[3]),
    ]

    bound_flag = True
    if bound_box[1] <= face_bound[0]:
        bound_flag = False

    if bound_box[0] >= face_bound[1]:
        bound_flag = False

    if bound_box[3] <= face_bound[2]:
        bound_flag = False

    if bound_box[2] >= face_bound[3]:
        bound_flag = False

    if bound_flag:
        surface_box = Geom_RectangularTrimmedSurface(surface_brep, *true_bound)
    else:
        surface_box = surface_brep

    surface = geomconvert.SurfaceToBSplineSurface(surface_box)
    return surface, loc


def pcurve(face: Face, edge):
    """
    Get the given edge's curve geometry as a 2D parametric curve
    on this face

    Args:
        edge (occwl.edge.Edge): Edge

    Returns:
        Geom2d_Curve: 2D curve
        Interval: domain of the parametric curve
    """
    crv, umin, umax = BRep_Tool().CurveOnSurface(edge.topods_shape(), face.topods_shape())
    return crv, (umin, umax)


def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def append_skip_log(skip_log, fn, reason):
    if not skip_log:
        return
    path = pathlib.Path(skip_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{pathlib.Path(fn).resolve()}\t{reason}\n")


def _subprocess_entry(fn_str, args, queue):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        ok = process_one_file((pathlib.Path(fn_str), args))
        queue.put(("ok", bool(ok)))
    except Exception as exc:
        queue.put(("err", repr(exc)))


def _kill_process(proc):
    proc.terminate()
    proc.join(5)
    if proc.is_alive():
        proc.kill()
        proc.join()


def process_parallel_with_timeout(work_files, args):
    ctx = get_context("fork")
    pending = deque(work_files)
    active = []
    ok_count = 0
    timeout = args.file_timeout
    skip_log = args.skip_log

    with tqdm(total=len(work_files)) as bar:
        while pending or active:
            while pending and len(active) < args.num_processes:
                fn = pending.popleft()
                if (pathlib.Path(args.output) / f"{fn.stem}.bin").exists():
                    bar.update(1)
                    continue
                queue = ctx.Queue()
                proc = ctx.Process(
                    target=_subprocess_entry,
                    args=(str(fn), args, queue),
                    daemon=True,
                )
                proc.start()
                active.append(
                    {"proc": proc, "fn": fn, "queue": queue, "start": time.monotonic()}
                )

            still_active = []
            for job in active:
                proc = job["proc"]
                fn = job["fn"]
                elapsed = time.monotonic() - job["start"]

                if not proc.is_alive():
                    proc.join()
                    if not job["queue"].empty():
                        status, detail = job["queue"].get()
                        if status == "ok" and detail:
                            ok_count += 1
                        elif status == "err":
                            append_skip_log(skip_log, fn, detail)
                            logging.warning("Failed %s: %s", fn, detail)
                    bar.update(1)
                    continue

                if elapsed >= timeout:
                    logging.warning("Timeout after %ss: %s", timeout, fn)
                    append_skip_log(skip_log, fn, f"timeout>{timeout}s")
                    _kill_process(proc)
                    bar.update(1)
                    continue

                still_active.append(job)
            active = still_active
            if pending or active:
                time.sleep(0.2)

    print(f"Processed {ok_count} files.")


def process_sequential_with_timeout(work_files, args):
    ctx = get_context("fork")
    ok_count = 0
    timeout = args.file_timeout
    skip_log = args.skip_log

    for fn in tqdm(work_files):
        if (pathlib.Path(args.output) / f"{fn.stem}.bin").exists():
            continue
        queue = ctx.Queue()
        proc = ctx.Process(
            target=_subprocess_entry,
            args=(str(fn), args, queue),
            daemon=True,
        )
        proc.start()
        proc.join(timeout)
        if proc.is_alive():
            logging.warning("Timeout after %ss: %s", timeout, fn)
            append_skip_log(skip_log, fn, f"timeout>{timeout}s")
            _kill_process(proc)
            continue
        if queue.empty():
            append_skip_log(skip_log, fn, "no_result")
            continue
        status, detail = queue.get()
        if status == "ok" and detail:
            ok_count += 1
        elif status == "err":
            append_skip_log(skip_log, fn, detail)
            logging.warning("Failed %s: %s", fn, detail)

    print(f"Processed {ok_count} files.")


def process(args):
    input_path = pathlib.Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder {input_path} does not exist.")
    output_path = pathlib.Path(args.output)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    if args.list is not None:
        work_files = []
        with open(args.list, "r") as f:
            for line in f:
                line = line.strip()
                work_files.append(pathlib.Path(line))

    else:
        work_files = list(input_path.rglob("*.st*p"))

    if args.file_timeout and args.file_timeout > 0:
        if args.num_processes == 1:
            process_sequential_with_timeout(work_files, args)
        else:
            process_parallel_with_timeout(work_files, args)
        return

    process_func = process_one_file

    if args.num_processes == 1:
        ok_count = 0
        for fn in tqdm(work_files):
            if process_func((fn, args)):
                ok_count += 1
        print(f"Processed {ok_count} files.")
    else:
        pool = Pool(
            processes=args.num_processes,
            initializer=initializer,
            maxtasksperchild=1,
        )
        try:
            results = list(
                tqdm(pool.imap_unordered(process_func, zip(work_files, repeat(args))), total=len(work_files))
            )
            print(f"Processed {len([valid for valid in results if valid])} files.")
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()


def main(cmd=None):
    if cmd is not None:
        print(cmd)
    arg_parser = argparse.ArgumentParser(
        "Convert faces into triangular beziers or convert solid models into brt structures"
    )
    arg_parser.add_argument("input", type=str, help="Input folder of STEP files")
    arg_parser.add_argument("output", type=str, help="Output folder of DGL graph BIN files")
    arg_parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of processes to use",
    )

    arg_parser.add_argument(
        "--no_random_name",
        default=False,
        help="randomize the name of the output file",
        action="store_true",
    )

    arg_parser.add_argument(
        "--no_label",
        default=False,
        help="no label",
        action="store_true",
    )

    arg_parser.add_argument("--list", type=str, help="only process files in list")

    arg_parser.add_argument("--method", type=int, help="method version: 1: old tri,2:new tri,3:uvgrid,4:sample points")

    arg_parser.add_argument(
        "--file_timeout",
        type=int,
        default=900,
        help="Per-file timeout in seconds (0 disables; uses isolated subprocess + kill on hang)",
    )

    arg_parser.add_argument(
        "--skip_log",
        type=str,
        default=None,
        help="Append timed-out or failed STEP paths to this TSV log",
    )

    arg_parser.add_argument(
        "--failure_log",
        type=str,
        default=None,
        help="Append structured JSONL failure records (stage, face_index, occwl/OCC layer)",
    )

    arguments = arg_parser.parse_args(cmd)

    if arguments.method==8:
        arguments.build_fn= build_triangles
    elif arguments.method==10:
        if arguments.no_label:
            arguments.build_fn= build_brt_data_no_label
        else:
            arguments.build_fn= build_brt_data
    elif arguments.method==18:
        arguments.build_fn=build_triangles
    
    if arguments.method==8:
        arguments.sub_fn=convertFaceToTriangles
    elif arguments.method==18:
        arguments.sub_fn=convertFaceToRectangleBeziers

    print(f"process number: {arguments.num_processes}")
    if arguments.file_timeout and arguments.file_timeout > 0:
        print(f"file timeout: {arguments.file_timeout}s")
    if arguments.skip_log:
        print(f"skip log: {arguments.skip_log}")
    if arguments.failure_log:
        print(f"failure log: {arguments.failure_log}")

    process(arguments)


def build_triangles(solid, save_path, filename, **kwargs):
    """
    compared to build_graph_faces,
    it save multiple faces in one file.
    """
    if kwargs["no_random_name"]:
        target = f"{filename}.bin"
    else:
        target = f"{filename}_{uuid.uuid4()}.bin"
    if os.path.exists(str(save_path / target)):
        logging.info(f"{save_path/target} already exists")
        return False

    step_path = kwargs.get("_step_path")
    failure_log = kwargs.get("_failure_log")

    try:
        graph = face_adjacency(solid)
    except Exception as exc:
        if failure_log and step_path is not None:
            append_failure_record(
                failure_log,
                make_failure_record(
                    step_path=pathlib.Path(step_path),
                    stage="face_adjacency",
                    exc=exc,
                ),
            )
        raise

    nodes_lst = []
    in_mask_lst = []
    points_lst = []
    normals_lst = []
    uv_values_lst = []
    vis_mask_lst = []
    scale_lst = []
    tri_normals_list = []
    fn = kwargs["sub_fn"]

    for face_idx in graph.nodes:
        face = graph.nodes[face_idx]["face"]
        try:
            nodes, in_mask, tri_normals, points, normals, vis_mask, uv_values, scale = fn(
                face, normalize=False, **kwargs
            )
        except Exception as exc:
            if failure_log and step_path is not None:
                append_failure_record(
                    failure_log,
                    make_failure_record(
                        step_path=pathlib.Path(step_path),
                        stage="face_triangulation",
                        exc=exc,
                        face_index=int(face_idx),
                    ),
                )
            raise

        nodes_lst.append(nodes)
        in_mask_lst.append(in_mask)
        points_lst.append(points)
        normals_lst.append(normals)
        uv_values_lst.append(uv_values)
        scale_lst.append(scale)
        vis_mask_lst.append(vis_mask)
        tri_normals_list.append(tri_normals)

    try:
        points_lst = torch.stack(points_lst)
        uv_values_lst = torch.stack(uv_values_lst)
        normals_lst = torch.stack(normals_lst)
        scale_lst = torch.stack(scale_lst)
        vis_mask_lst = torch.stack(vis_mask_lst)
    except Exception as exc:
        if failure_log and step_path is not None:
            append_failure_record(
                failure_log,
                make_failure_record(
                    step_path=pathlib.Path(step_path),
                    stage="save",
                    exc=exc,
                ),
            )
        raise

    torch.save(
        {
            "nodes": nodes_lst,
            "in_mask": in_mask_lst,
            "tri_normals": tri_normals_list,
            "points": points_lst,
            "uvs": uv_values_lst,
            "vis": vis_mask_lst,
            "scale": scale_lst,
            "normal": normals_lst,
        },
        str(save_path / target),
    )
    return True


def build_brt_data(solid, save_path, filename, shape_att, **kwargs):
    data = build_BRT(
        solid,
        shape_att=shape_att,
        edge_fn=lambda e: convertEdgeToBeziers2(e, max_knots=150, sampling=False)[0],
        **kwargs,
    )

    if kwargs["no_random_name"]:
        target = f"{filename}.bin"
    else:
        target = f"{filename}_{uuid.uuid4()}.bin"
    torch.save(data, str(save_path / target))
    return True


def build_brt_data_no_label(solid, save_path, filename, **kwargs):
    data = build_BRT_no_label(
        solid, edge_fn=lambda e: convertEdgeToBeziers2(e, max_knots=150, sampling=False)[0], **kwargs
    )

    if kwargs["no_random_name"]:
        target = f"{filename}.bin"
    else:
        target = f"{filename}_{uuid.uuid4()}.bin"
    torch.save(data, str(save_path / target))
    return True


def process_one_file(arguments):
    fn, args = arguments

    fn_stem = fn.stem
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    output_path = pathlib.Path(args.output)

    target_file = output_path / f"{fn_stem}.bin"

    if target_file.exists():
        return False

    try:
        if args.no_label:
            compound = Compound.load_from_step(fn)
        else:
            compound, shape_att = Compound.load_step_with_attributes(fn)
            args.shape_att = shape_att

    except Exception as exc:
        if getattr(args, "failure_log", None):
            append_failure_record(
                args.failure_log,
                make_failure_record(step_path=fn, stage="load_step", exc=exc),
            )
        logging.exception(f"Read Step Error in {fn}")
        return False

    graph = False
    build_kwargs = {**vars(args), "_step_path": fn, "_failure_log": getattr(args, "failure_log", None)}
    solid_index = 0
    try:
        for idx, solid in enumerate(compound.solids()):
            solid_index = idx
            build_fn = args.build_fn
            graph = build_fn(solid, output_path, fn_stem, **build_kwargs)
            break

    except ValueError as exc:
        if getattr(args, "failure_log", None):
            append_failure_record(
                args.failure_log,
                make_failure_record(
                    step_path=fn,
                    stage="face_triangulation",
                    exc=exc,
                    solid_index=solid_index,
                ),
            )
        logging.exception(f"Found Value Error in {fn.stem}")
        return False
    except Exception as exc:
        logging.exception(f"Build Error in {fn.stem}")
        return False

    return graph


MECHCAD_CATEGORIES = [
    "shaft",
    "pulley",
    "nut",
    "gear",
    "bracket",
    "bearing",
    "bolt",
    "coupling",
    "flange",
    "screw",
]


def _mechcad_category_cmd(
    input_path,
    output_path,
    category,
    method,
    target,
    process_num,
    file_timeout,
    skip_log,
    no_label=True,
):
    cmd = [
        f"{input_path}/{category}",
        f"{output_path}/{target}/{category}",
        "--num_processes",
        str(process_num),
        "--no_random_name",
        "--method",
        str(method),
    ]
    if no_label:
        cmd.append("--no_label")
    if file_timeout:
        cmd.extend(["--file_timeout", str(file_timeout)])
    if skip_log:
        cmd.extend(["--skip_log", skip_log])
    return cmd


def doKnotInsertion(spline_surf: Geom_BSplineSurface, num_max_knots=3):
    n_uknots = spline_surf.NbUKnots()
    n_vknots = spline_surf.NbVKnots()

    if n_uknots <= 1 or n_vknots <= 1:
        raise RuntimeError("bad nurbs")

    if n_uknots < num_max_knots:
        insert_num = num_max_knots - n_uknots

        insert_list = np.linspace(spline_surf.UKnot(1), spline_surf.UKnot(n_uknots), insert_num + 2)
        insert_list = insert_list[1:-1]
        for knot in insert_list:
            spline_surf.InsertUKnot(knot, 1, 1e-6)

    if n_vknots < num_max_knots:
        insert_num = num_max_knots - n_vknots

        insert_list = np.linspace(spline_surf.VKnot(1), spline_surf.VKnot(n_vknots), insert_num + 2)
        insert_list = insert_list[1:-1]
        for knot in insert_list:
            spline_surf.InsertVKnot(knot, 1, 1e-6)


def process_main(
    input_path,
    output_path,
    method=10,
    dataset="tmcad",
    target="brt",
    process_num=30,
    file_timeout=900,
    skip_log=None,
    categories=None,
):

    if dataset == "tmcad":
        cats = categories or MECHCAD_CATEGORIES
        for category in cats:
            print(f"[mechcad] category={category} target={target}")
            main(
                _mechcad_category_cmd(
                    input_path,
                    output_path,
                    category,
                    method,
                    target,
                    process_num,
                    file_timeout,
                    skip_log,
                )
            )


if __name__ == "__main__":

    logging.basicConfig(
        filename="logs/solid_to_triangles",
        filemode="w",
        format=" %(asctime)s :: %(levelname)-8s :: %(message)s",
        level=logging.INFO,
    )

    process_main(method=8, dataset="tmcad", target="triangles", process_num=30)
