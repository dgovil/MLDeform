# MLDeform

A library for machine learning of skeletal deformations on a skinned mesh.

This will try and make the logic as DCC agnostic as possible but the main target implementation will be
Autodesk Maya 2018 and above.

I encourage others to help improve this repo or to add implementations for other applications.

## Overview

This is a quick ELI5 overview of the system and the methodology used.

Let's say you have a production rig that has lots of complex pose driven deformations. These can be things like complex skinning where each vertice has multiple joints influencing it, or things like pose driven blendshapes, deltamushes and other deformers of that nature.

The cost to process all of these adds up and your character rig can get very slow.

Instead what if the computer could guess where the vertices should be based on the position of the joints?
This is where machine learning can be useful.

The first step is to reduce our complexity down. We do this by making sure that each vertex is influenced by only one joint and has no other deformers. This already speeds up our deformation speeds of the rig but leaves us with some very ugly deformations. The joints however help give stable deformations and allow us to have direct control over the rig itself.

Next, we make a record for each vertex on how far it is from where it's meant to be. We repeat this for each frame of our sample animation, and also record the rotation and translation values of our joints.

Now we have all the data needed for training our machine learning model.
For each joint, it is given the rotation and translation values of the joints as an input, and it's given the vertex offsets as the output. Based on that, it figures out the best way to go from the input to the output to guess the vertex offset values.

Finally we save the trained model out and can load it back in to our scene. If everything worked correctly it will be able to move the vertices to their correct positions based on the joint positions.

This can be incredibly quick, giving us a very good approximation of our initial complex rig, while being much quicker.

In fact, with some extra work, you can take this trained model and put it on a mobile device like an iPad and have it running faster than the initial rig could have on a super powered desktop, all while giving reasonably close approximations of the deformations we wanted.

As animators animate more scenes, we can feed more data into the machine learning model and it will get better over time too.
    
## Usage

### Installation

If you have the dependencies above installed, you can install this package by placing the MLDeform directory anywhere
on your `PYTHONPATH`.

If you're using Maya you can place the MLDeform directory in your scripts directory.

### Simplifying Skinning

This method requires that each joint have a single influence.
We'll still use standard skinning with joints to provide stability to the deformation.
However since each vertice only has one influence and no other deformations, the skinning is much faster and efficient.

To simplify the skinning, in Maya you can run the following:

```python

from MLDeform import skinning

# Get a target mesh.
target = 'Tube'

# Clone it (you can do this yourself too)
mesh = skinning.clone_mesh(target)

# Skin the clone. You can also handle this yourself too.
skinning.skin_mesh(mesh)

# Finally simplify the weights
skinning.simplify_weights(mesh, fast=False)

```

There are two types of `simplify_weights` methods.

* Fast: The fast method simply assigns the vertex to the joint with the maximum influence on it. This is pretty quick to run, but may not be the best choice if other deformations are used.

* Not-Fast: This method samples the scene across all the frames, testing each vertex against every joint it is weighted by. This lets us find the joint that provides the closest deformation to the target shape. It can be slower, but will give much better results.

### Writing Data

We need to write data to train the machine learning system

```python
from MLDeform import writer

# Write out the data to this location
# Will default to your Maya directory if none is given
outdir = 'Path/To/Write/To'

# Write the data out to the above location
path = writer.write(mesh, target, outdir=outdir)

```

### Training Models

Now we train the models!

```python
from MLDeform import train
training_data = train.train(path) # If you don't have matplotlib, set plot=False
print(training_data)

```

### Deform

You can load the deformer by running:

```python
from MLDeform import deformer
deformer.load_plugin()

```

Additionally you can add `MLDeform._maya` to your `MAYA_PLUGIN_PATH` so Maya can always find the plugin.
Otherwise you will need to run this every time.

Create the deformer using

```python
from maya import cmds
deformer = cmds.deformer(mesh, type='mldeformer')

```

Finally set the location of the `output_data.json` on the deformer and connect the joints up.
Take a look at the `test_deformer` function to see how to set this up on a sample scene.

The deformer will now load up the Tensorflow models we wrote earlier and predict values based on the
transforms of the joints it has been given.

## Notes

This repo is not very mature.
Known issues:

* Normalization causes deformation issues. Still need to fix it.
* No C++ deformer for Maya yet
* Data structure may change to be lighter.

## Reference Reading

Here are projects used as references for this


* [Fast and Deep Deformation Approximations](http://graphics.berkeley.edu/papers/Bailey-FDD-2018-08/index.html)
 ( Stephen W. Bailey, Dave Otte, Paul Dilorenzo, and James F. O'Brien. )
 
 * ['Fast and Deep Deformation Approximations’ Implementation](http://3deeplearner.com/fdda-implementation/)
 ( [3DeepLearner](http://3deeplearner.com/) )
 
 
## Dependencies
 
 There are a few Python depencies you will need.
 
### Required
 
* **six**
   
 Needed for supporting Python2 and Python3

* **tensorflow**
 
  Required for the actual training and deformers.
    
  **NOTE:** Some platforms have issues importing Tensorflow into Maya.
    
  To workaround this, you need to add a file called `__init__.py` to the `google` package so that it can be imported properly.
    
  Find the google package by running `from google import protobuf;print(protobuf.__file__)`.
  This gives you the location of the protobuf folder.
  The parent directory will be the google package.
 
 * **pandas**

    Necessary for efficient processing of data objects
    
### Optional
  
  
* **matplotlib**

  If you intend to display training plots, this is an optional requirement.
  
## Related Projects

* [MeshCompare](https://github.com/dgovil/MeshCompare)

  MeshCompare is a set of scripts and plugins for Maya that will color each vertex according to how far it is from the target mesh.
  This is really useful in visually seeing how far the machine learning predictions are from the target mesh.
