"""
Training a linear controller on latent + recurrent state
with CMAES.
This is a bit complex. num_workers slave threads are launched
to process a queue filled with parameters to be evaluated.
"""
import sys
from os.path import join, exists
from os import mkdir, unlink, listdir, getpid
from time import sleep
from torch.multiprocessing import Process, Queue
import torch
import cma
from src.models.controller import CONTROLLER
from tqdm import tqdm
import numpy as np
from src.env.rollout_generator import RolloutGenerator, load_parameters, flatten_parameters
from src.common.utils import PROJECT_ROOT, get_env
import hydra
from omegaconf import DictConfig, OmegaConf

################################################################################
#                           Thread routines                                    #
################################################################################
def slave_routine(p_queue, r_queue, e_queue, p_index,tmp_dir,hparams):
    """ Thread routine.
    Threads interact with p_queue, the parameters queue, r_queue, the result
    queue and e_queue the end queue. They pull parameters from p_queue, execute
    the corresponding rollout, then place the result in r_queue.
    Each parameter has its own unique id. Parameters are pulled as tuples
    (s_id, params) and results are pushed as (s_id, result).  The same
    parameter can appear multiple times in p_queue, displaying the same id
    each time.
    As soon as e_queue is non empty, the thread terminate.
    When multiple gpus are involved, the assigned gpu is determined by the
    process index p_index (gpu = p_index % n_gpus).
    :args p_queue: queue containing couples (s_id, parameters) to evaluate
    :args r_queue: where to place results (s_id, results)
    :args e_queue: as soon as not empty, terminate
    :args p_index: the process index
    """
    # init routine
    gpu = p_index % torch.cuda.device_count()
    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')

    # redirect streams
    sys.stdout = open(join(tmp_dir, str(getpid()) + '.out'), 'a')
    sys.stderr = open(join(tmp_dir, str(getpid()) + '.err'), 'a')

    with torch.no_grad():
        r_gen = RolloutGenerator(hparams, device)

        while e_queue.empty():
            if p_queue.empty():
                sleep(.1)
            else:
                s_id, params = p_queue.get()
                r_queue.put((s_id, r_gen.rollout(params)))


################################################################################
#                           Evaluation                                         #
################################################################################
def evaluate(solutions, results, p_queue, r_queue, rollouts=100):
    # https://blog.otoro.net/2017/10/29/visual-evolution-strategies/
    """ Give current controller evaluation.
    Evaluation is minus the cumulated reward averaged over rollout runs.
    :args solutions: CMA set of solutions
    :args results: corresponding results
    :args rollouts: number of rollouts
    :returns: minus averaged cumulated reward (to get the max of f(x) we compute the min of -f(x))
    """
    index_min = np.argmin(results)
    best_guess = solutions[index_min]
    restimates = []

    for s_id in range(rollouts):
        p_queue.put((s_id, best_guess))

    print("Evaluating...")
    for _ in tqdm(range(rollouts)):
        while r_queue.empty():
            sleep(.1)
        restimates.append(r_queue.get()[1])

    return best_guess, np.mean(restimates), np.std(restimates)

# intersting reading: https://blog.otoro.net/2017/10/29/visual-evolution-strategies/
@hydra.main(version_base=None, config_path=PROJECT_ROOT / "conf/hparams", config_name="config")
def train(cfg: DictConfig):
    hparams = cfg
    num_workers = hparams.controller.n_workers
    # population size lambda, number of candidate
    pop_size = hparams.controller.pop_size
    target_return = hparams.controller.target_return
    n_samples = hparams.controller.n_samples

    ################################################################################
    #                Define queues and start workers                               #
    ################################################################################
    p_queue = Queue()
    r_queue = Queue()
    e_queue = Queue()
    # create tmp dir if non existent and clean it if existent
    log_dir = get_env('LOG_FOLDER')
    tmp_dir = join(log_dir, 'tmp')
    if not exists(tmp_dir):
        mkdir(tmp_dir)
    else:
        for fname in listdir(tmp_dir):
            unlink(join(tmp_dir, fname))
    for p_index in range(num_workers):
        Process(target=slave_routine, args=(p_queue, r_queue, e_queue, p_index,tmp_dir, hparams)).start()
    controller = CONTROLLER(hparams.controller)  # dummy instance
    ctrl_file = hparams.controller.pth_folder
    # define current best and load parameters
    cur_best = None
    print("Attempting to load previous best...")
    if exists(ctrl_file):
        # Map tensors from GPU 0 to cpu
        state = torch.load(ctrl_file, map_location={'cuda:0': 'cpu'})
        cur_best = - state['reward']
        controller.load_state_dict(state['state_dict'])
        print("Previous best was {}...".format(-cur_best))
    parameters = controller.parameters()
    # initial standard deviation. The problem variables should have been scaled,
    # such that a single standard deviation on all variables is useful and the 
    # optimum is expected to lie within about x0 +- 3*sigma0.
    sigma = 0.1 
    es = cma.CMAEvolutionStrategy(flatten_parameters(parameters), sigma, {'popsize': pop_size})
    # http://www.cmap.polytechnique.fr/~nikolaus.hansen/html-pythoncma/
    epoch = 0
    log_step = 3
    while not es.stop():
        if cur_best is not None and - cur_best > target_return:
            print("Already better than target, breaking...")
            break

        r_list = [0] * pop_size  # result list
        solutions = es.ask()
        # push parameters to queue
        for s_id, s in enumerate(solutions):
            for _ in range(n_samples):
                p_queue.put((s_id, s))

        for _ in range(pop_size * n_samples):
            while r_queue.empty():
                # print("empty")
                sleep(.1)
            r_s_id, r = r_queue.get()
            r_list[r_s_id] += r / n_samples

        es.tell(solutions, r_list)
        es.disp()

        # evaluation and saving
        if epoch % log_step == log_step - 1:
            best_params, best, std_best = evaluate(solutions, r_list, p_queue, r_queue)
            print("Current evaluation: {}".format(best))
            if not cur_best or cur_best > best:
                cur_best = best
                print("Saving new best with value {}+-{}...".format(-cur_best, std_best))
                load_parameters(best_params, controller)
                torch.save(
                    {'epoch': epoch,
                    'reward': - cur_best,
                    'state_dict': controller.state_dict()},
                    ctrl_file)
            if - best > target_return:
                print("Terminating controller training with value {}...".format(best))
                break
        epoch += 1

    es.result_pretty()
    e_queue.put('EOP')

if __name__ == "__main__":
    train()