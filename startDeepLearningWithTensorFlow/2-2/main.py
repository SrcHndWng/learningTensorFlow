import tensorflow as tf

def create_graph():
    a = tf.constant(3, name='const1')
    b = tf.Variable(0, name="val1")
    add = tf.add(a, b)

    assign = tf.assign(b, add)
    c = tf.placeholder(tf.int32, name='input')
    mul = tf.multiply(assign, c)
    return c, mul

def main():
    c, mul = create_graph()

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(3):
            result = sess.run(mul, feed_dict={c:2})
            print("result = {}".format(result))

if __name__ == "__main__":
    main()
