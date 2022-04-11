import matplotlib.pyplot as plt
import io
import tensorflow as tf


def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)

    face = tf.image.decode_png(buf.getvalue(), channels=4)
    face = tf.expand_dims(face, 0)

    return face