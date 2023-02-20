import torch


class Config:
    # Training
    seed = 17
    epochs = 5
    learning_rate = 0.001
    num_classes = 2     # 0 - background, 1 - pnevmoniya
    model_dir = 'model'
    model_path = 'model/yolov7'
    model_weights = 'model/yolov7.pt'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Data preparation
    train_data_path = 'data/stage_2_train_images/'
    test_data_path = 'data/stage_2_test_images/'
    train_data_labels = 'data/stage_2_train_labels.csv'
    train_data_labels_preprocessed = 'data/stage_2_train_labels_preprocessed.csv'
    valid_fraction = 0.2
    batch_size = 8
    num_workers = 4
    
    # Data transformation
    resize_to = (512, 512)
    img_size = (448, 448)
    random_scale = (0.8, 1.0)
    flip_probability = 0.2
    mean = 0.125
    std = 0.60