# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Contains the definition of the Inception Resnet V1 architecture.
As described in http://arxiv.org/abs/1602.07261.
  Inception-v4, Inception-ResNet and the Impact of Residual Connections
    on Learning
  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


# Inception-Resnet-A (论文Figure 10)
def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 35x35 resnet block."""
    with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
        # 下面3个平行的结构都采用默认padding参数也就是SAME, 确保size统一, 便于后面concat
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
            tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv2_1 = slim.conv2d(tower_conv2_0, 32, 3, scope='Conv2d_0b_3x3')
            tower_conv2_2 = slim.conv2d(tower_conv2_1, 32, 3, scope='Conv2d_0c_3x3')
        # 在最后一个维度上合并. 上面计算的conv2d返回值都是4个维度: f[l] x f[l] x nC[l-1] x nC[l]
        # https://www.tensorflow.org/api_docs/python/tf/concat
        mixed = tf.concat([tower_conv, tower_conv1_1, tower_conv2_2], 3)

        up = slim.conv2d(mixed,
                         net.get_shape()[3],  # 确保输出channel与输入channel不变
                         1,
                         normalizer_fn=None,
                         activation_fn=None,  # 这个CONV没有activation函数
                         scope='Conv2d_1x1')
        # residual的跳跃叠加步骤
        # 与原resnet论文不同, 这里还增加了scale来调整residual增加的量, 用来应对filters太多导致训练失败的问题.(论文段落3.3)
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net


# Inception-Resnet-B (Figure 11)
def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 17x17 resnet block."""
    with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net,
                                     128,
                                     1,
                                     scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net,
                                        128,
                                        1,
                                        scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0,
                                        128,
                                        [1, 7], # filter的大小不再是 f x f 的正方形了, 而是拆分成了两层, 分别是 1 x f 和 f x 1
                                        scope='Conv2d_0b_1x7')
            tower_conv1_2 = slim.conv2d(tower_conv1_1,
                                        128,
                                        [7, 1],
                                        scope='Conv2d_0c_7x1')
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        up = slim.conv2d(mixed,
                         net.get_shape()[3],
                         1,
                         normalizer_fn=None,
                         activation_fn=None,
                         scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net


# Inception-Resnet-C (Figure 13)
def block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 8x8 resnet block."""
    with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net,
                                     192,
                                     1,
                                     scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net,
                                        192,
                                        1,
                                        scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0,
                                        192,
                                        [1, 3],
                                        scope='Conv2d_0b_1x3')
            tower_conv1_2 = slim.conv2d(tower_conv1_1,
                                        192,
                                        [3, 1],
                                        scope='Conv2d_0c_3x1')
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        up = slim.conv2d(mixed,
                         net.get_shape()[3],
                         1,
                         normalizer_fn=None,
                         activation_fn=None,
                         scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net


# Figure 7, Table 1
def reduction_a(net, k, l, m, n):
    with tf.variable_scope('Branch_0'):
        tower_conv = slim.conv2d(net,
                                 n,
                                 3,
                                 stride=2,
                                 padding='VALID',
                                 scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_1'):
        tower_conv1_0 = slim.conv2d(net,
                                    k,
                                    1,
                                    scope='Conv2d_0a_1x1')
        tower_conv1_1 = slim.conv2d(tower_conv1_0,
                                    l,
                                    3,
                                    scope='Conv2d_0b_3x3')
        tower_conv1_2 = slim.conv2d(tower_conv1_1,
                                    m,
                                    3,
                                    stride=2,
                                    padding='VALID',
                                    scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_2'):
        tower_pool = slim.max_pool2d(net,
                                     3,  # for pooling layer must cN[l] = cN[l-1], 所以无需设置, 这里设置的是f
                                     stride=2,
                                     padding='VALID',
                                     scope='MaxPool_1a_3x3')
    net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)
    return net


# Figure 12
def reduction_b(net):
    with tf.variable_scope('Branch_0'):
        tower_conv = slim.conv2d(net,
                                 256,
                                 1,
                                 scope='Conv2d_0a_1x1')
        tower_conv_1 = slim.conv2d(tower_conv,
                                   384,
                                   3,
                                   stride=2,
                                   padding='VALID',
                                   scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_1'):
        tower_conv1 = slim.conv2d(net,
                                  256,
                                  1,
                                  scope='Conv2d_0a_1x1')
        tower_conv1_1 = slim.conv2d(tower_conv1,
                                    256,
                                    3,
                                    stride=2,
                                    padding='VALID',
                                    scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_2'):
        tower_conv2 = slim.conv2d(net,
                                  256,
                                  1,
                                  scope='Conv2d_0a_1x1')
        tower_conv2_1 = slim.conv2d(tower_conv2,
                                    256,
                                    3,
                                    scope='Conv2d_0b_3x3')
        tower_conv2_2 = slim.conv2d(tower_conv2_1,
                                    256,
                                    3,
                                    stride=2,
                                    padding='VALID',
                                    scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_3'):
        tower_pool = slim.max_pool2d(net,
                                     3,
                                     stride=2,
                                     padding='VALID',
                                     scope='MaxPool_1a_3x3')
    net = tf.concat([tower_conv_1, tower_conv1_1, tower_conv2_2, tower_pool], 3)
    return net


# This module must define a function inference(images, ...), where images is a placeholder for the input images
# (dimensions <?,160,160,3> in the case of Inception-ResNet-v1) and returns a reference to the embeddings variable.
def inference(images, keep_probability, phase_train=True,
              bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
    }

    # Stores the default arguments for the given set of list_ops.
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        # 没有使用He或者tf.contrib.layers.xavier_initializer, 而是使用了简单的initializer和tf.zeros_initializer()类似
                        # truncated_normal_initializer与random_normal_initializer的区别:
                        # https://stackoverflow.com/questions/41704484/what-is-difference-between-tf-truncated-normal-and-tf-random-normal
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        return inception_resnet_v1(images, is_training=phase_train,
              dropout_keep_prob=keep_probability, bottleneck_layer_size=bottleneck_layer_size, reuse=reuse)


def inception_resnet_v1(inputs,
                        is_training=True,
                        dropout_keep_prob=0.8,
                        bottleneck_layer_size=128,
                        reuse=None,
                        scope='InceptionResnetV1'):
    """Creates the Inception Resnet V1 model.
    Args:
      inputs: a 4-D tensor of size [batch_size, height, width, 3].
      is_training: whether is training or not.
      dropout_keep_prob: float, the fraction to keep before final layer.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.
    Returns:
      logits: the logits outputs of the model.
      end_points: the set of end_points from the inception model.
    """
    end_points = {}

    # 给后面定义的variable自动加上scope(namespace)
    # https://www.tensorflow.org/api_docs/python/tf/variable_scope
    with tf.variable_scope(scope, 'InceptionResnetV1', [inputs], reuse=reuse):
        # 定义了函数的默认参数
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            # 定义了函数的默认参数
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
                # 网络结构参考论文C Szegedy - ‎2016 - Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning‎

                # 总体结构参考Figure 15

                # Stem (Figure 14)

                # 论文里的输入inputs维度是 299 x 299 x 3
                # 149 x 149 x 32, 此标记代表输出维度, 下同
                net = slim.conv2d(inputs,
                                  32,  # nC, 输出的channel数, 也就是filters的个数
                                  3,  # f, 每个filter的维度是f x f x 输入的nC
                                  stride=2,
                                  padding='VALID',  # 输出图片大小不保证与输入大小相同
                                  scope='Conv2d_1a_3x3')
                end_points['Conv2d_1a_3x3'] = net

                # 147 x 147 x 32
                net = slim.conv2d(net,
                                  32,
                                  3,
                                  padding='VALID',
                                  scope='Conv2d_2a_3x3')
                end_points['Conv2d_2a_3x3'] = net

                # 147 x 147 x 64
                net = slim.conv2d(net,
                                  64,
                                  3,
                                  scope='Conv2d_2b_3x3')
                end_points['Conv2d_2b_3x3'] = net

                # 73 x 73 x 64
                net = slim.max_pool2d(net,
                                      3,
                                      stride=2,
                                      padding='VALID',
                                      scope='MaxPool_3a_3x3')
                end_points['MaxPool_3a_3x3'] = net

                # 73 x 73 x 80
                net = slim.conv2d(net,
                                  80,
                                  1,
                                  padding='VALID',  # 这里与论文不同, 论文中是SAME, 但是由于1x1不会改变size, 所以也就无所谓了
                                  scope='Conv2d_3b_1x1')
                end_points['Conv2d_3b_1x1'] = net

                # 71 x 71 x 192
                net = slim.conv2d(net,
                                  192,
                                  3,
                                  padding='VALID',
                                  scope='Conv2d_4a_3x3')
                end_points['Conv2d_4a_3x3'] = net

                # 35 x 35 x 256
                net = slim.conv2d(net,
                                  256,
                                  3,
                                  stride=2,
                                  padding='VALID',
                                  scope='Conv2d_4b_3x3')
                end_points['Conv2d_4b_3x3'] = net

                # 5 x Inception-resnet-A (Figure 10)
                net = slim.repeat(net, 5, block35, scale=0.17)
                end_points['Mixed_5a'] = net

                # Reduction-A (Figure 7)
                with tf.variable_scope('Mixed_6a'):
                    # (Table 1)
                    net = reduction_a(net, 192, 192, 256, 384)
                end_points['Mixed_6a'] = net

                # 10 x Inception-Resnet-B (Figure 11)
                net = slim.repeat(net, 10, block17, scale=0.10)
                end_points['Mixed_6b'] = net

                # Reduction-B (Figure 12)
                with tf.variable_scope('Mixed_7a'):
                    net = reduction_b(net)
                end_points['Mixed_7a'] = net

                # 5 x Inception-Resnet-C (Figure 13)
                net = slim.repeat(net, 5, block8, scale=0.20)
                end_points['Mixed_8a'] = net

                # TODO: 这是干什么的?
                net = block8(net, activation_fn=None)
                end_points['Mixed_8b'] = net

                with tf.variable_scope('Logits'):
                    end_points['PrePool'] = net

                    # Average Pooling
                    # pylint: disable=no-member
                    net = slim.avg_pool2d(net,
                                          net.get_shape()[1:3],
                                          padding='VALID',
                                          scope='AvgPool_1a_8x8')
                    net = slim.flatten(net)

                    # Dropout
                    net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='Dropout')

                    end_points['PreLogitsFlatten'] = net

                # 这是干什么的?
                # https://github.com/davidsandberg/facenet/issues/605
                net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None, scope='Bottleneck', reuse=False)

    return net, end_points
