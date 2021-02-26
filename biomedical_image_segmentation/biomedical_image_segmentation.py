import sys
import os
import mlflow
import mlflow.pytorch
import click
import torch

#import torch.optim as optim

import time
import tempfile

from tensorboardX import SummaryWriter
from rich import traceback

from mlf_core.mlf_core import log_sys_intel_conda_env, set_general_random_seeds

from models.unet_3d_models import create_model, create_parallel_model
#from training.train import train, test
from training.train import train_net

#from data_loading.data_loader import load_train_test_data

import numpy as np
###########

@click.command()
@click.option('--cuda', type=click.Choice(['True', 'False']), default='True', help='Enable or disable CUDA support.')
@click.option('--epochs', type=int, default=5, help='Number of epochs to train')
@click.option('--general-seed', type=int, default=0, help='General Python, Python random and Numpy seed.')
@click.option('--pytorch-seed', type=int, default=0, help='Pytorch specific random seed.')
@click.option('--training-batch-size', type=int, default=8, help='Input batch size for training')
@click.option('--test-batch-size', type=int, default=8, help='Input batch size for testing')

@click.option('--learning-rate', type=float, default=0.0001, help='Initial learning rate')
@click.option('--lr-step-size', type=int, default=10000, help='Learning rate step size')
@click.option('--lr-gamma', type=float, default=0.1, help='Learning rate gamma value')

@click.option('--class-weights', type=str, default='0.2, 1.0, 2.5', help='Class weights for cross-entropy loss')
@click.option('--test-percent', type=float, default=0.15, help='Percent of dataset to be sampled for testing')
@click.option('--test-epochs', type=int, default=10, help='Number of training epochs before testing')

@click.option('--dataset-path', type=str, default='/mnt/datasets/lits/ds/', help='Path to dataset')
@click.option('--checkpoint-path', type=str, default='checkpoints/', help='Path to checkpoints')

@click.option('--dataset-size', type=int, default=131, help='Dataset size')
@click.option('--n-channels', type=int, default=1, help='Number of input channels')
@click.option('--n-class', type=int, default=3, help='Number of classes')
@click.option('--dropout-rate', type=float, default=0.25, help='Dropout rate')

def start_training(cuda, epochs, general_seed, pytorch_seed,
                   training_batch_size, test_batch_size,
                   learning_rate, lr_step_size, lr_gamma,
                   class_weights, test_percent, test_epochs,
                   dataset_path, checkpoint_path,
                   dataset_size, n_channels, n_class, dropout_rate):
    
    try:
        # Set GPU settings
        use_cuda = (True if cuda == 'True' else False) and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        if use_cuda and torch.cuda.device_count() > 0:
            click.echo(click.style(f'Using {torch.cuda.device_count()} GPUs!', fg='blue'))

        # Set all random seeds and possibly turn of GPU non determinism
        set_general_random_seeds(general_seed)
        set_pytorch_random_seeds(pytorch_seed, use_cuda=use_cuda)

        # set numpy seed for determinism
        np.random.seed(0)

        # Load training and testing data
        #train_loader, test_loader = load_train_test_data(training_batch_size, test_batch_size)

        # Define model, device and optimizer
        if use_cuda and torch.cuda.device_count() > 1:
            model = create_parallel_model(n_channels, n_class, dropout_val=dropout_rate)
        else:
            model = create_model(n_channels, n_class, dropout_val=dropout_rate)
        model.to(device)
        
        #optimizer = optim.Adam(model.parameters())
        #optimizer.step() #??

        with mlflow.start_run():
            # Create a SummaryWriter to write TensorBoard events locally
            events_output_dir = tempfile.mkdtemp()
            writer = SummaryWriter(events_output_dir)
            click.echo(click.style(f'Writing TensorBoard events locally to {events_output_dir}\n', fg='blue'))

            # Start training
            runtime = time.time()
            #for epoch in range(1, epochs + 1):
            #    train(use_cuda, model, epoch, optimizer, log_interval, train_loader, writer)
            #    test(use_cuda, model, epoch, test_loader, writer)

            train_net(net=model,
                    epochs=epochs,
                    batch_size=training_batch_size, #(64 + 4), #args.batchsize, +4
                    lr=learning_rate, #0.0001, #args.lr,
                    lr_step_size=lr_step_size,
                    lr_gamma=lr_gamma,
                    gpu=use_cuda,
                    class_weights=class_weights,
                    test_percent=test_percent, #0.15
                    test_epochs=test_epochs,
                    dataset_size=dataset_size,
                    n_class=n_class,
                    dataset_path=dataset_path,
                    checkpoint_path=checkpoint_path,
                    device=device,
                    writer=writer)
            
            device = 'GPU' if use_cuda else 'CPU'
            click.echo(click.style(f'{device} Run Time: {str(time.time() - runtime)} seconds', fg='green'))

            # Closing writer to allow for the model to be logged
            writer.close()

            # Log the model to mlflow
            click.echo(click.style('Logging model to mlflow...', fg='blue'))
            mlflow.pytorch.log_model(model, 'models')

            # Log hardware and software
            log_sys_intel_conda_env()

            # Upload the TensorBoard event logs as a run artifact
            click.echo(click.style('Uploading TensorBoard events as a run artifact...', fg='blue'))
            mlflow.log_artifacts(events_output_dir, artifact_path='events')
            click.echo(click.style(f'\nLaunch TensorBoard with:\ntensorboard --logdir={os.path.join(mlflow.get_artifact_uri(), "events")}', fg='blue'))

    except KeyboardInterrupt:
        torch.save(net.state_dict(), checkpoint_path + 'INTERRUPTED.pth')
        print('Saved interrupt')

        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


def set_pytorch_random_seeds(seed, use_cuda):
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multiGPU
        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_deterministic(True)
        

if __name__ == '__main__':
    traceback.install()
    click.echo(click.style(f'Num GPUs Available: {torch.cuda.device_count()}', fg='blue'))

    start_training()
