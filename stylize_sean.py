import os

import numpy as np
import tensorflow as tf
from PIL import Image
import vgg19
from vgg19_mat import *
import scipy.io
import scipy.misc
import time

VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'
VGG_MEAN = [123.68, 116.779, 103.939]
IMAGE_W = 800
IMAGE_H = 600
MEAN_VALUES = np.array([123, 117, 104]).reshape((1,1,1,3))

def read_image(path):
    image = scipy.misc.imread(os.path.join('images', path))
    image = scipy.misc.imresize(image,(IMAGE_H,IMAGE_W))
    image = image[np.newaxis,:,:,:]
    image = image - MEAN_VALUES
    return image

def write_image(path, image):
    image = image + MEAN_VALUES
    image = image[0]
    image = np.clip(image, 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)
    image_file = Image.fromarray(image)
    return image_file

def load_image(filename, new_width):
    """
    Load an image and conver from rgb to bgr.
    Resize image if new_width is set
    """
    rgb = Image.open(os.path.join('images', filename))

    # Resize image if new_width is set
    if new_width:
        original_size = rgb.size
        new_height = int(rgb.size[1] * new_width / rgb.size[0])
        rgb = rgb.resize((new_width, new_height), Image.ANTIALIAS)

        msg = '{} was resized from {} to {}'
        print(msg.format(filename, original_size, rgb.size))

    rgb = np.float32(rgb) - np.array(VGG_MEAN).reshape((1, 1, 3))

    bgr = rgb[:, :, ::-1]  # rgb to bgr
    bgr = bgr.reshape((1,) + bgr.shape)
    return bgr


def generate_noisey_image(content, noise_ratio):
    """Add some noise to the content image"""
    noise = np.random.uniform(low=-20, high=20, size=content.shape)
    image = noise*noise_ratio + content*(1 - noise_ratio)
    return image


def save_image(filename, bgr):
    """Convert an image from bgr to rgb and save to a file"""
    bgr = bgr[0]

    rgb = bgr[:, :, ::-1]  # bgr to rgb
    rgb += np.array(VGG_MEAN).reshape((1, 1, 3))
    rgb = np.clip(rgb, 0, 255).astype('uint8')

    image = Image.fromarray(rgb)
    image.save(os.path.join('images', filename))
    return image


def calculate_content_loss(vgg_image, vgg_content):
    """define loss function for content"""
    layer = 'conv4_2'
    # data_shape = vgg_image[layer].get_shape().as_list()
    data_shape = vgg_image[layer].shape
    M = data_shape[1] * data_shape[2]
    N = data_shape[3]
    # loss = tf.nn.l2_loss(vgg_image[layer] - vgg_content[layer])
    loss = (1./(2* N**0.5 * M**0.5 )) * tf.reduce_sum(tf.pow((vgg_image[layer] - vgg_content[layer]), 2))
    return loss


def calculate_style_loss(vgg_image, vgg_style):
    """define loss function for style"""
    def _gram_matrix(conv):
        n = tf.shape(conv)[3]
        m = tf.shape(conv)[1] * tf.shape(conv)[2]

        conv = tf.reshape(conv, (m, n))
        gram = tf.matmul(tf.transpose(conv), conv)
        return gram

    layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    ws = [1/5]*5
    total_loss = 0



    for layer, w in zip(layers, ws):

        # data_shape = vgg_image[layer].get_shape().as_list()
        data_shape = vgg_image[layer].shape
        M = data_shape[1] * data_shape[2]
        N = data_shape[3]
        gram_style = _gram_matrix(vgg_style[layer])
        gram_image = _gram_matrix(vgg_image[layer])

        loss = (1./(4 * N**2 * M**2)) * tf.nn.l2_loss(gram_style - gram_image)
        total_loss += loss*w

    return total_loss


def apply(content_file, style_file, learning_rate,
          iterations, alpha, beta, noise_ratio, new_width=None):
    start_time = time.time()

    """Apply neural style transfer to content image"""
    vgg_npy_file = 'vgg19.npy'

    # Create output_file as output_content_style.jpg
    without_ext = lambda f: os.path.splitext(f)[0]  # noqa E731
    output_file = 'output_{content}_{style}.jpg'.format(
        content=without_ext(content_file), style=without_ext(style_file))

    # Load images and construct a noisy image
    # content_image = load_image(content_file, new_width)
    # style_image = load_image(style_file, new_width)
    # image = generate_noisey_image(content_image, noise_ratio=noise_ratio)

    # Create tensorflow objects
    # Content and style images as constant and output as variable
    # tf_content_image = tf.constant(content_image, dtype=tf.float32)
    # tf_style_image = tf.constant(style_image, dtype=tf.float32)
    # tf_image = tf.Variable(image, dtype=tf.float32)

    # Create 3 VGG models - for content, style, and output images
    # vgg_content = vgg19.build(vgg_npy_file, tf_content_image)
    # vgg_style = vgg19.build(vgg_npy_file, tf_style_image)
    # vgg_image = vgg19.build(vgg_npy_file, tf_image)

    net = build_vgg19(VGG_MODEL)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    noise_img = np.random.uniform(-20, 20, (1, IMAGE_H, IMAGE_W, 3)).astype('float32')
    content_img = read_image(content_file)
    style_img = read_image(style_file)

    sess.run([net['input'].assign(content_img)])
    cost_content = calculate_content_loss(sess.run(net), net)

    sess.run([net['input'].assign(style_img)])
    cost_style = calculate_style_loss(sess.run(net), net)

    cost_total = cost_content + beta * cost_style
    optimizer = tf.train.AdamOptimizer(learning_rate)

    train = optimizer.minimize(cost_total)
    sess.run(tf.global_variables_initializer())
    sess.run(net['input'].assign(noise_ratio * noise_img + (1. - noise_ratio) * content_img))


    # content_img = read_image(content_file)
    # vgg_content = build_vgg19(VGG_MODEL)
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    # sess.run([vgg_content['input'].assign(content_img)])
    #
    # style_img = read_image(style_file)
    # vgg_style = build_vgg19(VGG_MODEL)
    # sess.run([vgg_style['input'].assign(style_img)])
    #
    # noise_img = np.random.uniform(-20, 20, (1, IMAGE_H, IMAGE_W, 3)).astype('float32')
    # vgg_image = build_vgg19(VGG_MODEL)
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    # sess.run([vgg_image['input'].assign(noise_ratio* noise_img + (1.-noise_ratio) * content_img)])


    # Calculate content, style and total loss
    # content_loss = calculate_content_loss(vgg_image, vgg_content)
    # style_loss = calculate_style_loss(vgg_image, vgg_style)
    # total_loss = alpha*content_loss + beta*style_loss

    # step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    # msg = 'Iteration:{}, total loss:{}, content loss:{}, style loss:{}'
    # init = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     sess.run(init)
    #
    #     for i in range(iterations):
    #         sess.run(train)
    #         # tf_outputs = sess.run([step, total_loss, content_loss, style_loss])
    #         # tf_outputs[0] = i  # Replace step with iteration
    #
    #         if i % 100 == 0:
    #             result_img = sess.run(net['input'])
    #             print(sess.run(cost_total))
    #             # print(msg.format(*tf_outputs))
    #             # image = save_image(output_file, tf_image.eval())
    #             image = save_image(output_file, result_img)
    for i in range(iterations):
        sess.run(train)
        if i % 100 == 0:
            result_img = sess.run(net['input'])
            print(sess.run(cost_total))
            # image = save_image(output_file, result_img)
            image = write_image(output_file, result_img)

    print("--- %s seconds ---" % (time.time() - start_time))
    return image
