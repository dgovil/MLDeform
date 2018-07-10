"""
This module consists of functions to handle skinning and weighting on meshes to prepare them for the machine learning system.
"""

import logging
import math

import maya.api.OpenMaya as om
import maya.api.OpenMayaAnim as oma
import maya.cmds as mc

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def clone_mesh(mesh, group='MLDeform'):
    """
    Clone a given mesh and put it under a group
    :param mesh:
    :param group:
    :return:
    """
    if not mc.objExists(group):
        group = mc.group(name=group, empty=True)

    nodes = mc.ls(mesh, l=True)
    if not nodes:
        raise RuntimeError('Could not find mesh')

    mesh = nodes[0]
    if mc.objectType(mesh) != 'transform':
        mesh = mesh.rpartition('|')[0]

    if mc.objectType(mesh) != 'transform':
        raise RuntimeError("Could not find a suitable transform to clone")

    clone = mc.duplicate(mesh)
    clone = mc.parent(clone, group)[0]

    clone = mc.ls('%s|%s' % (group, clone), l=True)[0]

    return clone


def get_mesh(path):
    """
    For a given mesh, get the path to the mesh
    :param path:
    :return:
    """
    nodes = mc.ls(path, l=True)
    if len(nodes) == 0:
        raise RuntimeError("No object matches %s" % path)
    if len(nodes) > 1:
        raise RuntimeError("More than one object matches %s" % path)

    node = nodes[0]
    if mc.objectType(node, isa='transform'):
        children = mc.listRelatives(node, type='mesh')
        if not children:
            raise RuntimeError('%s has no mesh children' % path)

        node = children[0]
    return node


def get_skincluster(mesh):
    """
    For a given mesh, get a skin cluster that applies to it.
    :param mesh: The mesh to find the skin cluster for
    :return: The name of the skin cluster
    :rtype: str
    """
    history = mc.listHistory(mesh, pruneDagObjects=True)
    clusters = mc.ls(history, type='skinCluster')
    if not clusters:
        return None

    if len(clusters) > 1:
        logger.warning('More than one skin cluster found: %s', ' , '.join(clusters))
    return clusters[0]


def get_joints(joints=None):
    """
    For a series of joints, get their full paths
    and remove any that are children of the other joints
    :param joints:
    :return:
    """
    if joints:
        joints = mc.ls(joints, type='joint', l=True)
    else:
        joints = mc.ls(type='joint', l=True)

    roots = []
    for joint in joints:
        for other in joints:
            if joint == other:
                continue
            if joint.startswith(other):
                break
        else:
            roots.append(joint)

    return roots


def skin_mesh(mesh, joints=None, clone=False):
    """
    Skin the mesh with given joints.
    :param mesh: the mesh to skin
    :param joints: The joints to use as influences. Uses all joints if not provided.
    :param clone: Whether to clone the mesh or not
    :return: The name of the skinCluster
    """
    joints = get_joints(joints=joints)
    if clone:
        mesh = clone_mesh(mesh)

    args = joints + [mesh]
    return mc.skinCluster(*args)


def get_weights(mesh, skin, vertices):
    """Get weights from a skin cluster for a given mesh
    Returns the indexes of the influences, the vertex components and weights
    """
    single_cmpt = om.MFnSingleIndexedComponent()
    vertex_cmpt = single_cmpt.create(om.MFn.kMeshVertComponent)
    single_cmpt.addElements(vertices)
    influence_objects = skin.influenceObjects()
    influence_indexes = om.MIntArray(len(influence_objects), 0)
    for i, influence in enumerate(influence_objects):
        influence_indexes[i] = int(skin.indexForInfluenceObject(influence))
    weights = skin.getWeights(mesh.dagPath(), vertex_cmpt, influence_indexes)
    return influence_indexes, vertex_cmpt, weights


def simplify_weights(path, target=None, fast=True, deformer=None, start=None, end=None, steps=1):
    """
    Simplify weighting on the mesh so that each vertice may only have a single influence.

    :param path: The path to the mesh to simplify the weights on
    :param target: The target mesh to compare weights against. Defaults to the current mesh.
    :param fast: Use the fast mode (less accurate) or not (iterates frame range but better results)
    :param deformer: The skin cluster to simplify weights for
    :param start: The start frame to use for the non-fast path.
    :param end: The end frame to use for the non-fast path.
    :param steps: The intervals between frames to use for the non fast path
    """
    sel = om.MSelectionList()
    sel.add(get_mesh(path))
    dag = sel.getDagPath(sel.length() - 1)
    mesh = om.MFnMesh(dag)

    if target:
        sel.add(get_mesh(target))
        target = sel.getDagPath(sel.length() - 1)
        target_mesh = om.MFnMesh(target)
    else:
        target = dag
        target_mesh = mesh

    if not deformer:
        deformer = get_skincluster(mesh.fullPathName())

    sel.add(deformer)
    skin_node = sel.getDependNode(sel.length() - 1)
    skin_cluster = oma.MFnSkinCluster(skin_node)

    if fast:
        __simplify_weights_fast(mesh, skin_cluster)
    else:
        __simplify_weights(mesh, target_mesh, skin_cluster, start=start, end=end, steps=steps)


def __simplify_weights_fast(mesh, skin):
    """
    Sets each vertex weight to follow its maximum influence and ignore the rest.
    :param mesh: MFnMesh of the mesh to change the weights on
    :param skin: MFnSkinCluster of the skinCluster driving this mesh
    """
    # Reference: https://gist.github.com/utatsuya/a95afe3c5523ab61e61b
    vertices = range(mesh.numVertices)
    influence_indexes, vertex_cmpt, weights = get_weights(mesh, skin, vertices)
    stride = len(influence_indexes)
    for vtx in vertices:
        vtx_weights = weights[vtx * stride: (vtx * stride) + stride]
        idx, weight = max(enumerate(vtx_weights), key=lambda x: x[1])
        for i, w in enumerate(vtx_weights):
            vtx_weights[i] = 1.0 if idx == i else 0

        weights[vtx * stride: (vtx * stride) + stride] = vtx_weights

    skin.setWeights(mesh.dagPath(), vertex_cmpt, influence_indexes, weights)


def __simplify_weights(mesh, target, skin, start=None, end=None, steps=1):
    """
    Sets each vertex to follow the single joint that produces the least difference
    in position over the given frame range


    :param mesh:
    :param target:
    :param skin:
    :param start:
    :param end:
    :param steps:
    :return:
    """
    # Verify we have the right number of vertices
    numVerts = mesh.numVertices
    if target.numVertices != numVerts:
        raise RuntimeError("Target Mesh has a different vertex count")

    # Figure out the start and end range
    if start is None:
        start = mc.playbackOptions(minTime=True, query=True)
    if end is None:
        end = mc.playbackOptions(maxTime=True, query=True)
    start = int(math.floor(start))
    end = int(math.ceil(end))
    currentTime = mc.currentTime(query=True)

    same_mesh = mesh.fullPathName() == target.fullPathName()
    space = om.MSpace.kObject

    # Lets get the weights
    vertices = range(mesh.numVertices)
    influence_indexes, vertex_cmpt, weights = get_weights(mesh, skin, vertices)

    deltas = [0.0 for x in weights]
    stride = len(influence_indexes)

    # We'll calculate the pattern of the list before hand so we aren't calculating
    # it on every frame
    preweights = []
    for i in range(stride):
        preweights.append([float(x == i) for x in range(stride)])

    # Now go through the frame range
    for frame in range(start, end + 1, steps):
        logger.info('Processing frame %s', frame)
        mc.currentTime(frame)

        # If we have the same mesh, restore the default weights on each frame before we continue
        if same_mesh:
            skin.setWeights(mesh.dagPath(), vertex_cmpt, influence_indexes, weights)
        target_points = mesh.getPoints(space)

        # Set each vertex weight to full then measure the distance and add to the deltas
        for i in range(stride):
            skin.setWeights(mesh.dagPath(), vertex_cmpt, influence_indexes,
                            om.MDoubleArray(preweights[i] * numVerts))

            points = mesh.getPoints()
            for vtx, point in enumerate(points):
                # Then calculate the distance between the vertexes.
                tpoint = target_points[vtx]
                distance = math.sqrt(
                    ((point.x - tpoint.x) ** 2) +
                    ((point.y - tpoint.y) ** 2) +
                    ((point.z - tpoint.z) ** 2)
                )

                deltas[(vtx * stride) + i] += distance

    for vtx in vertices:
        vtx_deltas = deltas[vtx * stride: (vtx * stride) + stride]
        idx = min(enumerate(vtx_deltas), key=lambda x: x[1])[0]
        weights[vtx * stride: (vtx * stride) + stride] = preweights[idx] * numVerts

    skin.setWeights(mesh.dagPath(), vertex_cmpt, influence_indexes, weights)
    mc.currentTime(currentTime)


def test_simplify(steps=1):
    mc.file('/Users/dhruv/Projects/MLDeform/TestScenes/Cylinder.ma', open=True, force=True)
    mc.currentTime(1)
    mesh = clone_mesh('Tube')
    mesh2 = clone_mesh('Tube')
    skin_mesh(mesh)
    skin_mesh(mesh2)
    simplify_weights(mesh, fast=False, steps=steps)
    simplify_weights(mesh2, fast=True)
    mc.setAttr('Tube.visibility', False)
    return mesh
