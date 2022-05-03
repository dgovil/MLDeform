import json
import logging
import os
from collections import namedtuple

import numpy as np
import tensorflow as tf
from maya import OpenMaya as om
from maya import OpenMayaMPx as ompx
from maya import cmds as mc
from tensorflow import Session, Graph

logging.basicConfig()
logger = logging.getLogger('MLDeformer')
logger.setLevel(logging.DEBUG)

inputAttr = ompx.cvar.MPxGeometryFilter_input
inputGeomAttr = ompx.cvar.MPxGeometryFilter_inputGeom
outputGeomAttr = ompx.cvar.MPxGeometryFilter_outputGeom
envelopeAttr = ompx.cvar.MPxGeometryFilter_envelope

TFModel = namedtuple('TFModel',
                     ['graph', 'session', 'input_tensor', 'output_tensor', 'vertices',
                      'normalized', 'trans_max', 'trans_min', 'verts_max', 'verts_min'])


class MLDeformerNode(ompx.MPxDeformerNode):
    name = 'mldeformer'
    id = om.MTypeId(0x0012d5c1)

    # Attributes
    data_loc = om.MObject()
    in_mats = om.MObject()

    def __init__(self, *args, **kwargs):
        super(MLDeformerNode, self).__init__(*args, **kwargs)
        self.data_location = None
        self.location_changed = True
        self.models = []  # Type: list[TFModel]

    @classmethod
    def creator(cls):
        return ompx.asMPxPtr(cls())

    @staticmethod
    def initialize():
        tattr = om.MFnTypedAttribute()
        mattr = om.MFnMatrixAttribute()

        # Path to the training data outputData.json
        MLDeformerNode.data_loc = tattr.create('trainingData', 'ta', om.MFnData.kString)
        tattr.setHidden(False)
        tattr.setKeyable(True)
        MLDeformerNode.addAttribute(MLDeformerNode.data_loc)
        MLDeformerNode.attributeAffects(MLDeformerNode.data_loc, outputGeomAttr)

        # List of input joints.
        MLDeformerNode.in_mats = mattr.create('matrix', 'mat')
        mattr.setKeyable(True)
        mattr.setArray(True)
        MLDeformerNode.addAttribute(MLDeformerNode.in_mats)
        MLDeformerNode.attributeAffects(MLDeformerNode.in_mats, outputGeomAttr)

        # Finally let us paint weights so we can control where the deformer applies.
        mc.makePaintable(
            MLDeformerNode.name,
            'weights',
            attrType='multiFloat',
            shapeMode='deformer'
        )

    def getInputMesh(self, data, idx):
        """
        Return the mesh object from the input mesh so we
        can can query values directly from it.
        """
        inputHandle = data.outputArrayValue(inputAttr)
        inputHandle.jumpToElement(idx)
        mesh = inputHandle.outputValue().child(inputGeomAttr).asMesh()
        return mesh

    def normalize_transform(self, array, model):
        """
        Normalize the transform values so we can feed them in with the same range as we trained the model with.
        :param array: The transform array
        :param model: The TFModel object
        """
        t_max = model.trans_max
        t_min = model.trans_min
        trans = array[-3:]

        trans = (trans - t_min) / (t_max - t_min)
        trans = np.nan_to_num(trans)

        array[-3:] = trans
        return array

    def denormalize_prediction(self, prediction, model):
        """
        Denormalize the prediction values from the training models range to real world ranges.
        :param prediction: The prediction values.
        :param model: The TFModel object.
        """
        v_max = model.verts_max
        v_min = model.verts_min

        prediction = (prediction * (v_max - v_min)) + v_min
        prediction = np.nan_to_num(prediction)
        return prediction

    def deform(self, data, iterator, world, geometryIndex):
        # Check if we need to reload up the training data.
        if self.location_changed:
            locHandle = data.inputValue(self.data_loc)
            loc = locHandle.asString()
            self.location_changed = False
            self.loadModels(loc)

        envelopeHandle = data.inputValue(envelopeAttr)
        envelope = envelopeHandle.asFloat()

        matricesHandle = data.inputArrayValue(self.in_mats)
        matricesCount = matricesHandle.elementCount()

        numModels = len(self.models)

        mesh = self.getInputMesh(data, geometryIndex)
        meshFn = om.MFnMesh(mesh)

        # Get the mesh so we can prepopulate the deltas array
        # Deltas are stored as expanded float3.
        deltas = np.zeros(3 * meshFn.numVertices())

        for i in range(matricesCount):
            # If we've got more input joints than models, we will skip them.
            if i >= numModels:
                continue

            # Some joints don't have a model in the json so skip them.
            model = self.models[i]
            if not model:
                continue

            # Get the matrix
            matricesHandle.jumpToElement(i)
            matrixHandle = matricesHandle.inputValue()
            matrix = matrixHandle.asMatrix()

            # Get the matrix from the joint
            transform = om.MTransformationMatrix(matrix)

            # Get the rotation values as quats. These are already normalized by their nature.
            rotation = transform.rotation()

            # Get the translate values
            translate = transform.translation(om.MSpace.kWorld)
            values = np.array([rotation.x, rotation.y, rotation.z, rotation.w,
                               translate.x, translate.y, translate.z])

            # If the model was normalized, so must these transform values
            if model.normalized:
                values = self.normalize_transform(values, model)

            # Keras trains with the regular shape, but tensorflow expects the transposed version.
            array = np.array([[v] for v in values]).T

            # Set the graph and session, and guess the values.
            with model.graph.as_default():
                with model.session.as_default():
                    results = model.session.run(
                        model.output_tensor,
                        feed_dict={model.input_tensor: array})
                    prediction = results[0]

                    # If the model was normalized, then lets denormalize the prediction
                    if model.normalized:
                        prediction = self.denormalize_prediction(prediction, model)

            # Finally save the predicted deltas
            for index, vtx in enumerate(model.vertices):
                deltas[vtx * 3:(vtx * 3) + 3] = prediction[index * 3:(index * 3) + 3]

        # Now iterate through the mesh and set the offsets.
        while not iterator.isDone():
            index = iterator.index()
            pos = iterator.position()
            weight = self.weightValue(data, geometryIndex, index)

            delta = deltas[index * 3:(index * 3) + 3]
            delta = [d * envelope * weight for d in delta]
            pos.x += delta[0]
            pos.y += delta[1]
            pos.z += delta[2]

            iterator.setPosition(pos)
            iterator.next()

    def preEvaluation(self, context, evaluationNode):
        """Check if certain values, specifically the training location, have changed to cause a recache"""
        if evaluationNode.dirtyPlugExists(MLDeformerNode.data_loc):
            self.location_changed = True
        return super(MLDeformerNode, self).preEvaluation(context, evaluationNode)

    def setDependentsDirty(self, plug, plugArray):
        """Check if certain values, specifically the training location, have changed to cause a recache"""
        if plug == MLDeformerNode.data_loc:
            self.location_changed = True
        return super(MLDeformerNode, self).setDependentsDirty(plug, plugArray)

    def loadModels(self, path):
        """Load models from the json file given."""
        logger.info('Loading models from %s', path)
        self.data_location = None
        self.models = []

        if path == self.data_location:
            return

        if not os.path.exists(path):
            logger.error('Could not find file: %s', path)

        self.data_location = path
        with open(path, 'r') as f:
            data = json.load(f)

        if 'models' not in data:
            logger.error('No models defined in data')
            return

        models = data['models']
        for i, model in enumerate(models):
            if not model:
                self.models.append(None)
                continue
            vertices = data['joint_map'][i]
            graph = Graph()
            with graph.as_default():
                session = Session()
                with session.as_default():
                    meta = model.get('meta')
                    root = model.get('root')
                    saver = tf.train.import_meta_graph(meta)
                    saver.restore(session, tf.train.latest_checkpoint(root))

                    in_tensor = session.graph.get_tensor_by_name(model['input'])
                    out_tensor = session.graph.get_tensor_by_name(model['output'])

                    normalized = model['normalized']
                    verts_max, verts_min, trans_max, trans_min = None, None, None, None
                    if normalized:
                        trans_max = np.array(model['trans_max'])
                        trans_min = np.array(model['trans_min'])
                        verts_max = np.array(model['verts_max'])
                        verts_min = np.array(model['verts_min'])

                    tfmodel = TFModel(graph=session.graph,
                                      session=session,
                                      input_tensor=in_tensor,
                                      output_tensor=out_tensor,
                                      vertices=vertices,
                                      normalized=normalized,
                                      trans_max=trans_max,
                                      trans_min=trans_min,
                                      verts_max=verts_max,
                                      verts_min=verts_min)

                    self.models.append(tfmodel)


def initializePlugin(plugin):
    pluginFn = ompx.MFnPlugin(plugin, 'Dhruv Govil', '0.1')
    try:
        pluginFn.registerNode(
            MLDeformerNode.name,
            MLDeformerNode.id,
            MLDeformerNode.creator,
            MLDeformerNode.initialize,
            ompx.MPxNode.kDeformerNode
        )
    except:
        om.MGlobal.displayError('Failed to register node: %s' % MLDeformerNode.name)
        raise


def uninitializePlugin(plugin):
    pluginFn = ompx.MFnPlugin(plugin)

    try:
        pluginFn.deregisterNode(MLDeformerNode.id)
    except:
        om.MGlobal.displayError('Failed to unregister node: %s' % MLDeformerNode.name)
        raise


def load_plugin():
    """Load this file as a plugin directly."""
    plugins = mc.pluginInfo(query=True, listPlugins=True)
    for plugin in plugins:
        path = mc.pluginInfo(plugin, query=True, path=True)
        if path == __file__ or path == __file__[:-1]:
            try:
                logger.warning('Unloading %s', plugin)
                mc.unloadPlugin(plugin)
            except:
                logger.error('Failed to unload %s', plugin)

    mc.loadPlugin(__file__, name=MLDeformerNode.name)


def test_deformer():
    mc.file('/Users/dhruv/Projects/MLDeform/TestScenes/Cylinder_setup.ma', open=True, force=True)
    load_plugin()
    mc.currentTime(1)

    deformer = mc.deformer('Tube1', type=MLDeformerNode.name)[0]
    mc.select(deformer)
    data_file = '/Users/dhruv/Library/Preferences/Autodesk/maya/training/output_data.json'
    mc.setAttr(deformer + '.trainingData', data_file, type='string')

    with open(data_file, 'r') as f:
        data = json.load(f)

    joint_names = data.get('joint_names')
    for i, joint in enumerate(joint_names):
        mc.connectAttr('%s.worldMatrix' % joint, '%s.matrix[%s]' % (deformer, i))
