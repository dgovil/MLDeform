"""
Functions to write out data for the Machine Learning algorithm

"""
import csv
import json
import logging
import math
import os

import maya.api.OpenMaya as om
import maya.api.OpenMayaAnim as oma
import maya.cmds as mc

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from . import skinning

DEFAULT_LOCATION = os.path.join(mc.internalVar(userAppDir=True), 'training')


def write(path, target, skin=None, outdir=None, start=None, end=None):
    """
    Write out data for the machine learning algorithm to train from.

    :param path: The path to the mesh we're writing data for.
    :param target: The target mesh to compare the vertices to.
    :param skin: The skin cluster to read weights from.
    :param outdir: The directory to write to. If no directory is provided, uses training directory.
    :param start: The start frame to write from.
    :param end: The end frame to write to
    :return: The path to the written data.
    """
    # Make sure we can write out the data
    if not outdir:
        logger.warning('No output directory specified. Using default: %s', DEFAULT_LOCATION)
        outdir = DEFAULT_LOCATION
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # Figure out the start and end range
    if start is None:
        start = mc.playbackOptions(minTime=True, query=True)
    if end is None:
        end = mc.playbackOptions(maxTime=True, query=True)
    start = int(math.floor(start))
    end = int(math.ceil(end))
    currentTime = mc.currentTime(query=True)

    # Get the meshes
    sel = om.MSelectionList()
    sel.add(skinning.get_mesh(path))
    sel.add(skinning.get_mesh(target))

    mesh = om.MFnMesh(sel.getDagPath(0))
    target_mesh = om.MFnMesh(sel.getDagPath(1))

    # Get the skin cluster
    if not skin:
        skin = skinning.get_skincluster(mesh.fullPathName())
    sel.add(skin)
    skin_node = sel.getDependNode(2)
    skin_cluster = oma.MFnSkinCluster(skin_node)

    # Get the weights
    vertices = range(mesh.numVertices)
    influence_objects = skin_cluster.influenceObjects()
    joints = [i.fullPathName() for i in influence_objects]
    joint_transforms = [om.MFnTransform(j) for j in influence_objects]
    influence_indexes, vertex_cmpt, weights = skinning.get_weights(mesh, skin_cluster, vertices)
    stride = len(influence_indexes)

    # Store the weight associations for vertices
    weight_map = []
    joint_map = []
    for i in range(stride):
        joint_map.append(list())

    for vtx in vertices:
        vtx_weights = weights[vtx * stride: (vtx * stride) + stride]
        for i, weight in enumerate(vtx_weights):
            if weight > 0:
                weight_map.append(i)
                joint_map[i].append(vtx)
                break

    # Prepare data for joints
    frame_data = {}
    for joint in joints:
        frame_data[joint] = []

    # Go through every frame and write out the data for all the joints and the vertexes
    for frame in range(start, end + 1):
        logger.debug('Processing frame %s', frame)
        mc.currentTime(frame)

        points = mesh.getPoints()
        target_points = target_mesh.getPoints()

        for jidx, vertices in enumerate(joint_map):
            xform = joint_transforms[jidx]
            rotation = xform.rotation(om.MSpace.kWorld, asQuaternion=True)
            translate = xform.translation(om.MSpace.kWorld)
            data = []
            data += rotation
            data += translate
            for vtx in vertices:
                mpoint = points[vtx]
                tpoint = target_points[vtx]

                displacement = tpoint - mpoint
                data += displacement

            frame_data[joints[jidx]].append(data)

    # Write the data out to csv files
    csv_files = []
    for joint in joints:
        data = frame_data[joint]
        filename = '%s.csv' % joint.replace('|', '_')
        if filename.startswith('_'):
            filename = filename[1:]
        filename = os.path.join(outdir, filename)
        logger.info('Wrote data for %s to %s', joint, filename)
        heading = ['rx', 'ry', 'rz', 'rw', 'tx', 'ty', 'tz']
        verts = joint_map[joints.index(joint)]
        for v in verts:
            heading.extend(['vtx%sx' % v, 'vtx%sy' % v, 'vtx%sz' % v])

        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(heading)
            writer.writerows(data)

        csv_files.append(filename)

    mc.currentTime(currentTime)
    map_data = {
        'joint_names': joints,
        'joint_indexes': [i for i in influence_indexes],
        'weights': weight_map,
        'csv_files': csv_files,
        'joint_map': joint_map,
        'input_fields': ['rx', 'ry', 'rz', 'rw', 'tx', 'ty', 'tz']
    }

    # Finally write out the data
    map_file = os.path.join(outdir, 'input_data.json')
    with open(map_file, 'w') as f:
        json.dump(map_data, f)
    logger.info('Wrote Weight Map to %s', map_file)
    return outdir


def test_writer():
    mc.file('/Users/dhruv/Projects/MLDeform/TestScenes/Cylinder.ma', open=True, force=True)
    mc.currentTime(1)
    mesh = skinning.clone_mesh('Tube')
    skinning.skin_mesh(mesh)
    skinning.simplify_weights(mesh, fast=True)
    mc.setAttr('Tube.visibility', False)
    return write(mesh, 'Tube', outdir=DEFAULT_LOCATION)
