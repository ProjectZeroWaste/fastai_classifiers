class args:
    seed = 42
    lr = 1e-3
    validate_ratio = 0.2
    label_smoothing_eps = 0.1
    lookahead_steps = 6
    batch_size = 32
    epochs = 40
    generator = TRAIN_IMAGE_GENERATOR

    model_name = 'efficientnet-b0'  # b0-b7
    # activation = 'relu'             # TODO: relu / mish
    # optimizer = 'ranger'            # TODO: ranger / radam / adam / rmsprop

    load = 'b0-trained-for-40'
    save = False # 'b0-trained-for-40'

    train = True
    visualise = False
    use_gpu = True

    num_classes = 45