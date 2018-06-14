import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets.resnet_utils import Block, conv2d_same, subsample
from tensorflow.contrib.slim.nets.resnet_v1 import bottleneck, resnet_v1


#TODO:
# 1. basic block features
# 2. bottleneck features
# 3. resnet_v1 features

# block for resnet-18/34
@slim.add_arg_scope
def basic_block_features(inputT):
  pass


# bottleneck for resnet-50/101/152
@slim.add_arg_scope
def bottleneck_features(inputT,
                        depth,
                        depth_bottleneck,
                        stride,
                        rate=1,
                        outputs_collections=None,
                        scope=None):
  """
  simplified bottleneck_v1 with features

  Args:
    inputT: A tensor of size [batch, height, width, channel]
    depth: output depth
    depth_bottleneck: depth of bottleneck
    stride: Amount of downsampling of the output to input
    rate: An integer, rate for atrous convolution
    outputs_collections: Collection to add the output
    scope: optional variable scope
  """
  with tf.variable_scope(scope, 'bottleneck_v1', [inputT]) as sc:
    depth_in = slim.utils.last_dimension(inputT.get_shape(), min_rank=4)
    
    # skip-connection part
    if depth == depth_in:
      shortcut = subsample(inputT, stride, scope='shortcut')
    else:
      shortcut = slim.conv2d(
          inputT,
          depth, [1, 1],
          stride=stride,
          activation_fn=None,
          scope='shortcut')

    # residual part
    residual = slim.conv2d(inputT, depth_bottleneck, [1, 1], stride=1,
                           scope='conv1')
    residual = conv2d_same(residual, depth_bottleneck, 3, stride,
                                        rate=rate, scope='conv2')
    conv_3 = slim.conv2d(residual, depth, [1, 1], stride=1,
                           activation_fn=None, normalizer_fn=None, scope='conv3')
    residual = slim.batch_norm(conv_3)


    output = tf.nn.relu(shortcut + residual)

    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.name,
                                            output)


# resnet_v1 features
def resnet_v1_features(inputT):
  pass
