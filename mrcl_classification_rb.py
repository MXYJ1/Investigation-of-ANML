import argparse
import logging

import numpy as np
import torch
from tensorboardX import SummaryWriter

import datasets.datasetfactory as df
import datasets.task_sampler as ts
import model.modelfactory as mf
import utils.utils as utils
from experiment.experiment import experiment
from model.meta_learner import MetaLearingClassification
#from google.colab import files

logger = logging.getLogger('experiment')

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def main(args):
    # Introduce replay buffer for inputs and outputs
    replay_buff_x = torch.empty((20,1,3,28,28))
    replay_buff_y = torch.empty((20,1))
    sample_coeff = 1.5
    #sample_prob = []
    #size_of_buffer = 0
    utils.set_seed(args.seed)

    my_experiment = experiment(args.name, args, "../results/", commit_changes=args.commit)
    writer = SummaryWriter(my_experiment.path + "tensorboard")

    logger = logging.getLogger('experiment')

    # Using first 963 classes of the omniglot as the meta-training set
    args.classes = list(range(963))

    dataset = df.DatasetFactory.get_dataset(args.dataset, background=True, train=True, all=True)
    dataset_test = df.DatasetFactory.get_dataset(args.dataset, background=True, train=False, all=True)

    # Iterators used for evaluation
    iterator_test = torch.utils.data.DataLoader(dataset_test, batch_size=5, # from pytorch
                                                shuffle=True, num_workers=1)

    iterator_train = torch.utils.data.DataLoader(dataset, batch_size=5,
                                                 shuffle=True, num_workers=1)

    sampler = ts.SamplerFactory.get_sampler(args.dataset, args.classes, dataset, dataset_test)

    config = mf.ModelFactory.get_model(args.treatment, args.dataset)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    maml = MetaLearingClassification(args, config, args.treatment).to(device)
    find_step = 0 # if no checkpoint
    if args.checkpoint:
        checkpoint = torch.load(args.saved_model, map_location='cpu')
        step = np.load('my_step.npy') # load in current step
        find_step = step[0]
        # print(checkpoint.parameters()[0].data)
        for idx in range(15):
            maml.net.parameters()[idx].data = checkpoint.parameters()[idx].data
    print('number of steps completed', find_step)
    maml = maml.to(device)

    utils.freeze_layers(args.rln, maml)
    # save step
    current_step = np.array([0])
    rb_counter = 0
    for step in range(find_step,args.steps):
        current_step[0] = step # save current time step
        np.save('my_step.npy', current_step)
        t1 = np.random.choice(args.classes, args.tasks, replace=False)

        d_traj_iterators = []
        for t in t1:
            d_traj_iterators.append(sampler.sample_task([t]))
            #print('LOOOK!!',sampler.sample_task([t]))

        d_rand_iterator = sampler.get_complete_iterator()
        # spt is 20 samples for each class, qry includes samples from remember set
        x_spt, y_spt, x_qry, y_qry = maml.sample_training_data(d_traj_iterators, d_rand_iterator,
                                                               steps=args.update_step, reset=not args.no_reset)
        
        
        ################################################################
        # if no past data has been seen
        if rb_counter == 0:
          replay_buff_x = x_spt
          replay_buff_y = y_spt

          x_spt_rb = x_spt
          y_spt_rb = y_spt

          x_qry_rb = x_qry
          y_qry_rb = y_qry

          rb_counter += 1
          #size_of_buffer = 20
        # sample 5 instances of past data to include in new training data
        else:
          # create probability vec
          #positions = [float(i) for i in range(1, replay_buff_x.shape[0]+1)] # for non-uniform sampling
          #positions.reverse() # for non-uniform sampling
          print('length of buffer',replay_buff_x.shape[0])
          #sample_prob = sample_coeff * np.array(positions) # for non-uniform sampling
          #sample_prob = softmax(sample_prob) # for non-uniform sampling
          rb_idx = np.random.choice(replay_buff_x.shape[0], 10, replace=False, p = sample_prob) # for non-uniform sampling
          rb_idx = np.random.choice(replay_buff_x.shape[0], 10, replace=False)
          #rb_idx = np.random.choice(replay_buff_x.shape[0], 5, replace=False)
          rb_samples_x = replay_buff_x[rb_idx,:]
          rb_samples_y = replay_buff_y[rb_idx,:]

          # create new training data which combines these samples
          x_spt_rb = torch.vstack((rb_samples_x, x_spt))
          y_spt_rb = torch.vstack((rb_samples_y, y_spt))

          x_qry_rb = torch.cat( (torch.reshape(rb_samples_x, (1, 10, 3, 28, 28)), x_qry), 1 )
          y_qry_rb = torch.cat( (torch.reshape(rb_samples_y, (1, 10)), y_qry), 1 )

          # re-shuffle data before using it for training 
          shuffle_idx = np.random.choice(x_spt_rb.shape[0], x_spt_rb.shape[0], replace=False)
          shuffle_idx_qry = np.random.choice(x_qry_rb.shape[0], x_qry_rb.shape[0], replace=False)
          x_spt_rb = x_spt_rb[shuffle_idx,:]
          y_spt_rb = y_spt_rb[shuffle_idx,:]

          x_qry_rb = x_qry_rb[shuffle_idx_qry,:]
          y_qry_rb = y_qry_rb[shuffle_idx_qry,:]

          # add new training data into replay buffer
          replay_buff_x = torch.vstack((replay_buff_x, x_spt))
          replay_buff_y = torch.vstack((replay_buff_y, y_spt))

          # shuffle replay buffer
          #shuffle_idx_rb = np.random.choice(replay_buff_x.shape[0], replay_buff_x.shape[0], replace=False)
          #replay_buff_x = replay_buff_x[shuffle_idx_rb,:]
          #replay_buff_y = replay_buff_y[shuffle_idx_rb,:]
        
        if torch.cuda.is_available():
            x_spt_rb, y_spt_rb, x_qry_rb, y_qry_rb = x_spt_rb.cuda(), y_spt_rb.cuda(), x_qry_rb.cuda(), y_qry_rb.cuda()

        accs, loss = maml(x_spt_rb, y_spt_rb, x_qry_rb, y_qry_rb)#, args.tasks)

        # Evaluation during training for sanity checks
        if step % 40 == 0:
            #writer.add_scalar('/metatrain/train/accuracy', accs, step)
            logger.info('step: %d \t training acc %s', step, str(accs))
        if step % 100 == 0 or step == 19999:
            torch.save(maml.net, args.model_name)
            #print('shape of parameters',maml.net.parameters())
            #torch.jit.save(maml.net, args.model_name)
            #files.download(args.model_name)
        if step % 2000 == 0 and step != 0:
            utils.log_accuracy(maml, my_experiment, iterator_test, device, writer, step)
            utils.log_accuracy(maml, my_experiment, iterator_train, device, writer, step)

#
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--steps', type=int, help='epoch number', default=40000)
    argparser.add_argument('--model_name', help='Name of model to be saved', default='my_sweet_model.net')
    argparser.add_argument('--treatment', help='Neuromodulation or OML', default='Neuromodulation')
    argparser.add_argument('--checkpoint', help='Use a checkpoint model', action='store_true')
    argparser.add_argument('--saved_model', help='Saved model to load', default='my_model.net')
    argparser.add_argument('--seed', type=int, help='Seed for random', default=10000)
    argparser.add_argument('--seeds', type=int, nargs='+', help='n way', default=[10])
    argparser.add_argument('--tasks', type=int, help='meta batch size, namely task num', default=1)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-2)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=20)
    argparser.add_argument('--name', help='Name of experiment', default="mrcl_classification")
    argparser.add_argument('--dataset', help='Name of experiment', default="omniglot")
    argparser.add_argument("--commit", action="store_true")
    argparser.add_argument("--no-reset", action="store_true")
    argparser.add_argument("--rln", type=int, default=9)
    argparser.add_argument('--model', type=str, help='epoch number', default="none")
    args = argparser.parse_args()

    args.name = "/".join([args.dataset, str(args.meta_lr).replace(".", "_"), args.name])
    print(args)
    main(args)
