"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import data_setup, engine, model_builder, utils

from torchvision import transforms

if __name__ == '__main__':
        
    # setup hyperparameters
    NUM_EPOCHS = 5
    BATCH_SIZE = 32
    HIDDEN_UNITS = 10
    LEARNING_RATE = 0.001
    
    # setup directories
    train_dir = '../data/pizza_steak_sushi/train/'
    test_dir = '../data/pizza_steak_sushi/test/'
    
    # setup target device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # create transforms
    data_transform = transforms.Compose([
        transforms.Resize(size=(64,64)),
        transforms.ToTensor()
    ])
    
    # create DataLoaders with help from data_setup.py
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir = test_dir,
        transform=data_transform,
        batch_size=BATCH_SIZE
    )
    # class_names = next(enumerate(train_dataloader))
    # create model with help from model_builder.py
    model = model_builder.TinyVGG(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names)
    ).to(device)
    
    #set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=LEARNING_RATE)
    
    # start training with the help of engine.py
    engine.train(model=model,
                 train_dataloader=train_dataloader,
                 test_dataloader=test_dataloader,
                 loss_fn=loss_fn,
                 optimizer=optimizer,
                 epochs=NUM_EPOCHS,
                 device=device)
    
    # save the model with the help of utils.py
    utils.save_model(model=model,
                     target_dir='models',
                     model_name='05_going_modular_script_made_tinyvgg_model.pth')
