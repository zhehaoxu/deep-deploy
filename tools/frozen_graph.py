import tensorflow.compat.v1 as tf

path = './yolov3_coco.pb'
output_tensor = [
    'pred_sbbox/concat_2', 'pred_mbbox/concat_2', 'pred_lbbox/concat_2'
]

with tf.Session() as sess:
    with open(path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        input = tf.placeholder(tf.float32, (1, 416, 416, 3), name="input")
        output = tf.import_graph_def(graph_def,
                                     input_map={'input/input_data:0': input},
                                     name='')
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, output_tensor)
        with tf.gfile.GFile('yolov3.pb', mode='wb') as fs:
            fs.write(constant_graph.SerializeToString())
