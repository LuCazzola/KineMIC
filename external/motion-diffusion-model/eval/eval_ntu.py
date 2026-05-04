import torch
from datetime import datetime
from collections import OrderedDict

from diffusion import logger
from train.train_platforms import WandBPlatform  # required for the eval operation

from data_loaders.unified.scripts.motion_process import *
from data_loaders.unified.utils.utils import *
from data_loaders.unified.motion_loaders.model_motion_loaders import get_mdm_loader

from eval.evaluators.metrics import * # 

torch.multiprocessing.set_sharing_strategy('file_system')

def get_activations(eval_wrapper, motion_loaders):
    
    activation_dict = OrderedDict({})
    for motion_loader_name, motion_loader in motion_loaders.items():
        out = {'embedding' : [], 'logits' : [], 'gt' : []}
        with torch.no_grad():
            for motions, m_lens, action, _ in motion_loader:
                motions = motions.to(eval_wrapper.device) # [Batch, Joints, xyz, Time]
                m_lens = m_lens.to(eval_wrapper.device)

                embedding, logits = eval_wrapper.get_motion_embeddings(
                    x=motions,
                    y={'lengths':m_lens}
                )

                out['embedding'].append(embedding.cpu().numpy())
                out['logits'].append(logits.cpu().numpy())
                out['gt'].append(action.squeeze().cpu().numpy())
            
            out['embedding'] = np.concatenate(out['embedding'], axis=0)
            out['logits'] = np.concatenate(out['logits'], axis=0)
            out['gt'] = np.concatenate(out['gt'], axis=0)

        activation_dict[motion_loader_name] = out
    
    return activation_dict


def evaluate_fid(eval_wrapper, groundtruth_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print('========== Evaluating FID ==========')
    with torch.no_grad():
        for motions, m_lens, _, _ in groundtruth_loader:
            motions = motions.to(eval_wrapper.device)
            m_lens = m_lens.to(eval_wrapper.device)
            motion_embeddings, _ = eval_wrapper.get_motion_embeddings(
                x=motions,
                y={'lengths':m_lens}
            )
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    for model_name in activation_dict.keys():
        motion_embeddings = activation_dict[model_name]['embedding']
        mu, cov = calculate_activation_statistics(motion_embeddings)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        print(f'---> [{model_name}] FID: {fid:.4f}')
        print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
        eval_dict[model_name] = fid
    return eval_dict


def evaluate_diversity(activation_dict, file, diversity_times):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    for model_name in activation_dict.keys():
        motion_embeddings = activation_dict[model_name]['embedding']
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
        print(f'---> [{model_name}] Diversity: {diversity:.4f}', file=file, flush=True)
    return eval_dict

def evaluate_accuracy(activation_dict, file, top_k=(1, 2, 3)):
    eval_dict = OrderedDict({})
    print('========== Evaluating Accuracy ==========')
    for model_name in activation_dict.keys():
        logits = torch.from_numpy(activation_dict[model_name]['logits'])
        gt = torch.from_numpy(activation_dict[model_name]['gt'])
        
        _, sorted_preds = torch.topk(logits, k=max(top_k), dim=1)
        sorted_preds_np = sorted_preds.cpu().numpy()
        gt_np = gt.cpu().numpy()
        
        accuracy_scores = []
        for k in top_k:
            correct = np.equal(sorted_preds_np[:, :k], gt_np[:, np.newaxis]).any(axis=1)
            accuracy = correct.mean() * 100.0
            accuracy_scores.append(accuracy)

        acc_str = ""
        for i, score in enumerate(accuracy_scores):
            acc_str += f'Top-{top_k[i]}: {score:.4f} '
        
        print(f'---> [{model_name}] Acc: {acc_str.strip()}')
        print(f'---> [{model_name}] Acc: {acc_str.strip()}', file=file, flush=True)

        eval_dict[model_name] = np.array(accuracy_scores)
    return eval_dict


def evaluate_multimodality(eval_wrapper, mm_motion_loaders, file, mm_num_times):
    eval_dict = OrderedDict({})
    print('========== Evaluating MultiModality ==========')
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        with torch.no_grad():
            for motions, m_lens in mm_motion_loader:
                # (1, mm_replications, dim_pos):
                motions = motions.to(eval_wrapper.device)
                m_lens = m_lens.to(eval_wrapper.device)
                motion_embedings, _ = eval_wrapper.get_motion_embeddings(
                    x=motions,
                    y={'lengths':m_lens}
                )
                mm_motion_embeddings.append(motion_embedings.unsqueeze(0))
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}', file=file, flush=True)
        eval_dict[model_name] = multimodality
    return eval_dict


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, replication_times, 
               diversity_times, mm_num_times, run_mm=False, eval_platform=None):
    
    with open(log_file, 'w') as f:    
        # Dctionary which maps captions into action labels
        all_metrics = OrderedDict({
            'FID': OrderedDict({}),
            'Diversity': OrderedDict({}),
            'Accuracy': OrderedDict({}),
            'MultiModality': OrderedDict({})
        })
        
        for replication in range(replication_times):
            motion_loaders, mm_motion_loaders = {}, {}
            motion_loaders['gt'] = gt_loader
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                motion_loader, mm_motion_loader = motion_loader_getter()
                motion_loaders[motion_loader_name] = motion_loader
                mm_motion_loaders[motion_loader_name] = mm_motion_loader

            print(f'\n==================== Replication {replication} ====================')
            print(f'==================== Replication {replication} ====================', file=f, flush=True)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            acti_dict = get_activations(eval_wrapper, motion_loaders)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            fid_score_dict = evaluate_fid(eval_wrapper, gt_loader, acti_dict, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            div_score_dict = evaluate_diversity(acti_dict, f, diversity_times)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            acc_score_dict = evaluate_accuracy(acti_dict, f)

            if run_mm:
                print(f'Time: {datetime.now()}')
                print(f'Time: {datetime.now()}', file=f, flush=True)
                mm_score_dict = evaluate_multimodality(eval_wrapper, mm_motion_loaders, f, mm_num_times)

            print(f'\n!!! DONE !!!\n')
            print(f'\n!!! DONE !!!\n', file=f, flush=True)
            
            # SCORES
            #for key, item in fid_score_dict.items():
            #    if key not in all_metrics['FID']:
            #        all_metrics['FID'][key] = [item]
            #    else:
            #        all_metrics['FID'][key] += [item]

            for key, item in div_score_dict.items():
                if key not in all_metrics['Diversity']:
                    all_metrics['Diversity'][key] = [item]
                else:
                    all_metrics['Diversity'][key] += [item]

            #for key, item in acc_score_dict.items():
            #    if key not in all_metrics['Accuracy']:
            #        all_metrics['Accuracy'][key] = [item]
            #    else:
            #        all_metrics['Accuracy'][key] += [item]

            if run_mm:
                for key, item in mm_score_dict.items():
                    if key not in all_metrics['MultiModality']:
                        all_metrics['MultiModality'][key] = [item]
                    else:
                        all_metrics['MultiModality'][key] += [item]

        mean_dict = {}
        for metric_name, metric_dict in all_metrics.items():
            print('========== %s Summary ==========' % metric_name)
            print('========== %s Summary ==========' % metric_name, file=f, flush=True)
            for model_name, values in metric_dict.items():
                
                mean, conf_interval = get_metric_statistics(np.array(values), replication_times)
                mean_dict[metric_name + ' ' + model_name] = mean
                
                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}')
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}', file=f, flush=True)
                elif isinstance(mean, np.ndarray):
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)):
                        line += '(top %d) Mean: %.4f CInt: %.4f;' % (i+1, mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)
                    
        # log results
        if eval_platform is not None:
            for k, v in mean_dict.items():
                if k.startswith('Accuracy'):
                    for i in range(len(v)):
                        eval_platform.report_scalar(name=f'top{i + 1}_' + k, value=v[i],
                                                            iteration=1, group_name='Eval')
                else:
                    eval_platform.report_scalar(name=k, value=v, iteration=1, group_name='Eval')
        
        return mean_dict


if __name__ == "__main__":

    from utils import dist_util
    from data_loaders.get_data import get_single_stream_dataloader
    from utils.parser_util import evaluation_parser
    from utils.fixseed import fixseed
    from utils.model_util import create_model_and_diffusion, load_saved_model
    from model.wrappers import wrap_w_classifier_free_sampling
    from eval import eval_ntu
    from eval.evaluators.evaluator_wrapper import EvaluatorWrapper

    args = evaluation_parser()
    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    
    s_stream = args.single_stream if args.model_type in ['MDM'] else 'target'
    active_streams = [s_stream] if args.model_type in ['MDM'] else args.stream_names

    data = {}
    for stream in active_streams :
        stream_args = getattr(args, stream)
        print(f"\n=== {stream.upper()} Data ===")
        print(f"creating [{stream}] data loader...")
        data[stream] = get_single_stream_dataloader(
            data_stream_args=stream_args,
            batch_size=args.batch_size,
            split='train', hml_mode='train',
            device=dist_util.dev(),
        )
        print(f"[{stream}] dataet len: {len(data[stream].dataset)}\n")

    print("\n===== Creating model and diffusion =====")
    model_for_eval, diffusion = create_model_and_diffusion(args, data, active_streams)
    data = data[s_stream] # deprecate un-used streams
    print(f"\nCheckpoints from [{args.model_path}]...")
    load_saved_model(model_for_eval, args.model_path, active_streams)
    if args.guidance_param != 1: # wrapping model with the classifier-free sampler
        model_for_eval = wrap_w_classifier_free_sampling(model_for_eval)
    model_for_eval.to(dist_util.dev())
    model_for_eval.eval()  # disable random masking
    
    def build_eval_components(args, stream):
        """
        Build evaluation components for the training loop.
        note that this is applied only to the 
        """
        stream_args = getattr(args, stream)       
    
        batch_size = 32
        eval_num_samples = 320
        mm_num_samples, mm_num_repeats = 100, 30  # reminder, mm = MultiModality
        SPLIT = 'val'

        # GT Dataloader
        print(f'\n=== building GT dataloader [{stream}] ===')
        eval_gt_data = get_single_stream_dataloader(
            data_stream_args=stream_args, batch_size=batch_size,
            split=SPLIT, hml_mode='gt',
            evaluator='stgcn', data_rep='xyz',
            device=dist_util.dev(),
        )

        # Dataloader to generate samples
        print(f'\n=== building GEN dataloader [{stream}] ===')
        gen_loader = get_single_stream_dataloader(
            data_stream_args=stream_args, batch_size=batch_size,
            split=SPLIT, hml_mode='eval',
            evaluator=None, data_rep=stream_args.data_rep,
            device=dist_util.dev(),
        )
        
        eval_data = {
            'test': lambda: eval_ntu.get_mdm_loader(
                args, model_for_eval, diffusion, batch_size,
                gen_loader, mm_num_samples, mm_num_repeats, gen_loader.dataset.opt.max_motion_length,
                eval_num_samples, scale=args.guidance_param, cond_mode=stream_args.cond_mode
                
            )
        }
        eval_wrapper = EvaluatorWrapper(stream_args.dataset, dist_util.dev(), task_split=stream_args.task_split, fewshot_id=stream_args.fewshot_id)
        
        return eval_wrapper, eval_data, eval_gt_data

    def evaluate(stream):
        """
        Perform evaluation for the given Dataset
        """
        assert stream == 'target', "Only target stream evaluation is supported."
        start_eval = time.time()
        
        print('\n>>> Running evaluation loop <<<\n')
        log_file = os.path.join(os.path.dirname(args.model_path), f'eval_{os.path.basename(args.model_path).replace(".pt", ".log")}')
        diversity_times = 300
        mm_num_times = 10  # mm is super slow hence we won't run it during training
        eval_rep_times = 5

        _ = eval_ntu.evaluation(
            eval_wrapper, eval_gt_data, eval_data, log_file,
            replication_times=eval_rep_times,
            diversity_times=diversity_times, mm_num_times=mm_num_times, run_mm=True
        )
        
        end_eval = time.time()
        print(f'Evaluation time: {(round(end_eval-start_eval)/60):.2f} min.')

    # RUN
    eval_wrapper, eval_data, eval_gt_data =  build_eval_components(args, s_stream)
    evaluate(s_stream)