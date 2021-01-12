import tensorflow as tf

def copy_config(model, name=None, index=None):
    layer = model.get_layer(name=name, index=index)
    cls_name = layer.__class__.__name__
    layer_cls = getattr(tf.keras.layers, cls_name)
    config = layer.get_config()
    print(cls_name, config)
    if cls_name == 'Conv2D':
        filters = config.pop('filters')
        kernel_size = config.pop('kernel_size')
        new_layer = layer_cls(filters, kernel_size, **config)
    else:
        new_layer = layer_cls().from_config(config)
    return new_layer


class VGG16Base(tf.keras.Model):
    def __init__(self, vgg_orig):
        super().__init__()

        self.conv1_1 = copy_config(vgg_orig, name='block1_conv1', index=1)
        self.conv1_2 = copy_config(vgg_orig, name='block1_conv2', index=2)
        self.conv1_3 = copy_config(vgg_orig, name='block1_conv3', index=3)


class InitRepNet(tf.keras.Model):
    def __init__(self, vgg_orig, out_ids, out_attribs):
        super().__init__()
        
        self.out_ids = out_ids
        self.out_attribs = out_attribs
        print(f"=> out_ids: {self.out_ids}, out_attribs: {self.out_attribs}")

        print(vgg_orig.summary())
        print(vgg_orig.layers[:3])
        # Conv1
        self.conv1_1 = vgg_orig.get_layer(name='block1_conv1', index=1)
        self.conv1_2 = vgg_orig.get_layer(name='block1_conv2', index=2)
        self.conv1_3 = vgg_orig.get_layer(name='block1_pool', index=3)

        self.conv1 = tf.keras.Sequential([
            self.conv1_1,
            self.conv1_2,
            self.conv1_3
        ])

        # Conv2
        self.conv2_1 = vgg_orig.get_layer(name='block2_conv1', index=4)
        self.conv2_2 = vgg_orig.get_layer(name='block2_conv2', index=5)
        self.conv2_3 = vgg_orig.get_layer(name='block2_pool', index=6)
        
        self.conv2 = tf.keras.Sequential([
            self.conv2_1,
            self.conv2_2,
            self.conv2_3
        ])

        # Conv3
        self.conv3_1 = vgg_orig.get_layer(name='block3_conv1', index=7)
        self.conv3_2 = vgg_orig.get_layer(name='block3_conv2', index=8)
        self.conv3_3 = vgg_orig.get_layer(name='block3_conv3', index=9)
        self.conv3_4 = vgg_orig.get_layer(name='block3_pool', index=10)

        self.conv3 = tf.keras.Sequential([
            self.conv3_1,
            self.conv3_2,
            self.conv3_3,
            self.conv3_4
        ])

        # Conv4_1
        self.conv4_1_1 = vgg_orig.get_layer(name='block4_conv1', index=11)
        self.conv4_1_2 = vgg_orig.get_layer(name='block4_conv2', index=12)
        self.conv4_1_3 = vgg_orig.get_layer(name='block4_conv3', index=13)
        self.conv4_1_4 = copy_layer(vgg_orig, name='block4_pool', index=14)
        print(self.conv4_1_3)
        # print(self.conv4_1_4)

        self.conv4_1 = tf.keras.Sequential([
            self.conv4_1_1,
            self.conv4_1_2,
            self.conv4_1_3,
            self.conv4_1_4
        ])

        # Conv4_2
        self.conv4_2_1 = copy_layer(vgg_orig, name='block4_conv1', index=11)
        self.conv4_2_2 = copy_layer(vgg_orig, name='block4_conv2', index=12)
        self.conv4_2_3 = copy_layer(vgg_orig, name='block4_conv3', index=13)
        self.conv4_2_4 = copy_layer(vgg_orig, name='block4_pool', index=14)
        print(self.conv4_2_3)
        # print(self.conv4_2_4)

        self.conv4_2 = tf.keras.Sequential([
            self.conv4_2_1,
            self.conv4_2_2,
            self.conv4_2_3,
            self.conv4_2_4
        ])

def copy_layer(model, name=None, index=None):
    layer = model.get_layer(name=name, index=index)
    cls_name = layer.__class__.__name__
    layer_cls = getattr(tf.keras.layers, cls_name)
    config = layer.get_config()
    print(cls_name, config)
    if cls_name == 'Conv2D':
        filters = config.pop('filters')
        kernel_size = config.pop('kernel_size')
        new_layer = layer_cls(filters, kernel_size, **config)
    else:
        new_layer = layer_cls().from_config(config)
    weights = layer.get_weights()
    new_layer.set_weights(weights)
    return new_layer

def test():
    vgg_orig = tf.keras.applications.VGG16(weights='imagenet')
    out_ids = None
    out_attribs = None
    net = InitRepNet(vgg_orig, out_ids, out_attribs)
    print(net)


if __name__ == '__main__':
    test()
