import sys
from os.path import join, exists
from os import mkdir, unlink, listdir, getpid
from time import sleep, time as current_time
from torch.multiprocessing import Process, Queue, Manager
import cma
sys.path.append('/home/siyao/project/rlPractice/dlai_project')
from src.models.controller import CONTROLLER
from tqdm import tqdm
from src.env.rollout_generator import RolloutGenerator, load_parameters, flatten_parameters
from src.common.utils import PROJECT_ROOT, get_env
import hydra
from omegaconf import DictConfig
import numpy as np
import random
import torch

np.random.seed(0)
random.seed(0)
torch.cuda.manual_seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def log_process_action(p_index, action, s_id):
    print(f"Process {p_index}: {action} for {s_id}")

def handle_task(p_index, p_queue, r_queue, r_gen):
    if not p_queue.empty():
        s_id, params = p_queue.get()
        log_process_action(p_index, "Received params", s_id)
        result = r_gen.rollout(params)
        r_queue.put((s_id, result))
        log_process_action(p_index, "Sent result", s_id)
        # Clear CUDA cache
        torch.cuda.empty_cache()

def slave_routine(p_queue, r_queue, e_queue, p_index, tmp_dir, hparams):
    gpu = p_index % torch.cuda.device_count()
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

    # redirect streams
    sys.stdout = open(join(tmp_dir, str(getpid()) + '.out'), 'a')
    sys.stderr = open(join(tmp_dir, str(getpid()) + '.err'), 'a')

    with torch.no_grad():
        r_gen = RolloutGenerator(hparams, device)
        try:
            while e_queue.empty():
                handle_task(p_index, p_queue, r_queue, r_gen)
                sleep(.1)
        except Exception as e:
            print(f"Process {p_index} encountered an error: {e}")

def evaluate(solutions, results, p_queue, r_queue, rollouts=200):
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
        # Clear CUDA cache
        torch.cuda.empty_cache()

    return best_guess, np.mean(restimates), np.std(restimates)

@hydra.main(version_base=None, config_path=PROJECT_ROOT / "conf/hparams", config_name="config")
def train(cfg: DictConfig):
    hparams = cfg
    num_workers = hparams.controller.n_workers
    pop_size = hparams.controller.pop_size
    target_return = hparams.controller.target_return
    n_samples = hparams.controller.n_samples

    manager = Manager()
    p_queue = manager.Queue()
    r_queue = manager.Queue()
    e_queue = manager.Queue()


    log_dir = get_env('LOG_FOLDER')
    tmp_dir = join(log_dir, 'tmp')
    if not exists(tmp_dir):
        mkdir(tmp_dir)
    else:
        for fname in listdir(tmp_dir):
            unlink(join(tmp_dir, fname))

    processes = []
    for p_index in range(num_workers):
        p = Process(target=slave_routine, args=(p_queue, r_queue, e_queue, p_index, tmp_dir, hparams))
        p.start()
        processes.append(p)

    controller = CONTROLLER(hparams.controller)
    ctrl_file = hparams.controller.pth_folder
    cur_best = None
    print("Attempting to load previous best...")
    if exists(ctrl_file):
        state = torch.load(ctrl_file, map_location={'cuda:0': 'cpu'})
        cur_best = -state['reward']
        controller.load_state_dict(state['state_dict'])
        print("Previous best was {}...".format(-cur_best))
    parameters = controller.parameters()
    sigma = hparams.controller.sigma
    es = cma.evolution_strategy.CMAEvolutionStrategy(flatten_parameters(parameters), sigma, 
                                                     {'popsize': pop_size, 
                                                    'tolfun': 1e-6,
                                                    'tolx': 1e-6,
                                                     })
    epoch = 0
    log_step = 3
    max_epoch = hparams.controller.n_epochs
    while not es.stop() and epoch < max_epoch:
        if cur_best is not None and -cur_best > target_return:
            print("Already better than target, breaking...")
            break

        r_list = [0] * pop_size
        solutions = es.ask_geno()
        
        for s_id, s in enumerate(solutions):
            for _ in range(n_samples):
                p_queue.put((s_id, s))
                # print(f"main generate s_id: {s_id}")

        processed_tasks = set()
        for _ in range(pop_size * n_samples):
            start_time = current_time()
            while r_queue.empty():
                if current_time() - start_time > 10:  # timeout after 10 seconds
                    print("Timeout waiting for results.")
                    break
                sleep(.1)
            if not r_queue.empty():
                r_s_id, r = r_queue.get()
                if r_s_id not in processed_tasks:
                    r_list[r_s_id] += r / n_samples
                    processed_tasks.add(r_s_id)
            # Clear CUDA cache
            torch.cuda.empty_cache()

        es.tell(solutions, r_list)
        es.logger.add()
        es.disp()

        if epoch % log_step == log_step - 1:
            best_params, best, std_best = evaluate(solutions, r_list, p_queue, r_queue)
            print("Current evaluation: {} vs current best {}".format(best, cur_best))
            if not cur_best or cur_best > best:
                cur_best = best
                print("Saving new best with value {}+-{}...".format(-cur_best, std_best))
                load_parameters(best_params, controller)
                torch.save(
                    {'epoch': epoch,
                     'reward': -cur_best,
                     'state_dict': controller.state_dict()},
                    ctrl_file)
            if -best > target_return:
                print("Terminating controller training with value {}...".format(best))
                break
        epoch += 1
    e_queue.put('EOP')
    for p in processes:
        p.join(timeout=60)
        if p.is_alive():
            print(f"Process {p.pid} did not terminate within the timeout.")
            p.terminate()
    es.result_pretty()
    es.logger.plot(fontsize=3)
    print("Saving results...")
    cma.s.figsave(log_dir + '/result_of_controller_train.svg')

if __name__ == "__main__":
    train()
