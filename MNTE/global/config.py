import warnings


class DefaultConfig(object):

    # data
    data_name = 'f30k'  # f30k, coco
    data_path = '/data/data_yl/TERAN/'
    images_path = '/data/flickr30k/flickr30k-images/'  # path of images
    # images_path = '/data/COCO/'  # path of images
    restval = True

    # training
    resume = ''
    log_step = 10
    val_step = 100000000000000
    test_step = 100000000
    logger_name = 'runs/f30k/global'

    num_epochs = 30
    bs = 30
    workers = 10

    lr = 0.00001
    lr_update = 15
    scheduler = 'steplr'
    gamma = 0.1
    step_size = 20
    warmup = 'linear'
    warmup_period = 1000

    measure = 'dot'
    grad_clip = 2.0
    margin = 0.2
    max_violation =True  # IMPORTANT

    # model
    embed_size = 1024
    dropout = 0.1
    layers = 2

    # image-model
    image_crop_size = 224
    image_model_type = 'resnet101'
    image_finetune = False
    image_feat_dim = 2048

    # text-model
    text_model_name = 'bert'
    bert_config = '/data/data_yl/modeling_bert/bert-base-uncased-config.json'
    bert_text = '/data/data_yl/modeling_bert/bert-base-uncased-vocab.txt'
    bert_model = '/data/data_yl/modeling_bert/bert-base-uncased-pytorch_model.bin'
    text_extraction_hidden_layer = 6
    text_word_dim = 768
    text_dropout = 0.1
    text_finetune = True


    def parse(self, kwargs):
        """
        update configuration by kwargs.
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Waning: opt has no attribute %s" % k)
            setattr(self, k, v)

        print('User config:')
        for k, v in self.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))