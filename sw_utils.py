import numpy as np
import os
import re
from itertools import product
import numpy as np
import scipy.io as sio
import collections


def print_command(config : dict,
                   filename : str,
                   process_per_gpu_max=2, GPU_start=0, GPU_end=3,
                  teacher_path=None,
                  teacher_type=None,
                  attach=False,
                  noti_message=None,
                  main='main.py'
                 ):
    list_keys = []
    list_items = []
    total_process = 1
    
    # Firstly, we calculate the total number of processes that we run
    for key, item in config.items():
        # if item is list, we should consider all cases that each element in the list is applied
        if isinstance(item, list):
            total_process *= len(config[key])
            list_keys.append(key)
            list_items.append([str(i) for i in item]) # convert all elements to string types
            
    ############################## make a combination of parameters #################################
    command_template = ''
    for key in list_keys:
        command_template += '--'+key+' {} '
        
    command_combinations = []
    params_combination = list(product(*list_items))
    
    for param in params_combination:
        tmp = command_template.format(*param)
        command_combinations.append(tmp)
    
    ############################## make a directory and file pointer ################################
    if not os.path.isdir('./run'):
        os.mkdir('./run')
        
    file_path = os.path.join('run', filename)
    if not attach:
        file_pointer = open(file_path, 'w')
    else:
        file_pointer = open(file_path, 'a')

    ############################### start a loop ################################
    pre_command = 'CUDA_VISIBLE_DEVICES={} python ' + main + ' ' # --device {} '
    #pre_str = 'python main.py --device {} '
    # firstly, make a pre-command by using parameters that don't need to a combination
    pre_command += ' '.join(['--'+key+' '+str(item) for key, item in config.items() if key not in list_keys]) 

    # allocate each command with a combinated param list to each GPU
    process_per_cur_gpu =0
    GPU_cur = GPU_start
    cur_process = 0
    
    for variant_command in command_combinations:
        command = (pre_command + ' ' + variant_command).format(GPU_cur)
        
        ############################### add teacher ################################
        if teacher_path :
            seed_ = find_parameter(command, ['--seed'], spliter=' ')
            command += ' --teacher-path ' + teacher_path.format(seed_[0])
            command += ' --teacher-type {}'.format(teacher_type)
        ############################################################################
        
        process_per_cur_gpu += 1
        cur_process += 1

        if total_process > cur_process:
            # if the current GPU is full,
            if process_per_cur_gpu == process_per_gpu_max:
                process_per_cur_gpu = 0
                
                if GPU_cur<GPU_end:
                    command+=' &'
                    GPU_cur+=1
                    
                # if the current GPU is the last GPU, 
                else:
                    GPU_cur = GPU_start
                    command += ' &\nwait'
                    command += '\n'
            else:
                command += ' &'
        else:
            command += ' &\nwait'            
            
        print(command)
        file_pointer.write(command +'\n')

#     slack_message = 'date : {} / method : {} / dataset : {}'.format(config['date'], config['method'], config['dataset'])
    slack_run = 'python ../../slack_sender.py {}'.format(noti_message)
    print (slack_run)
    file_pointer.write(slack_run +'\n')
    file_pointer.close()    


def filter_file(filename, musthave, havenot):
    for term in musthave:
        if term not in filename:
            return False

    for term in havenot:
        if term in filename:
            return False
        
    return True

def find_parameter(string, param_list, spliter='_', fairness=False):
    results = [False for _ in param_list]
    terms = string.split(spliter)
    
    if fairness:
        for term in terms:
            for i, param in enumerate(param_list):
                param_name = re.findall('[a-zA-Z]+', term)[-1]
                if param==param_name:
                    number = re.findall('[\d\.\d]+', term)[-1]
#                     results[i] = float(number) if '.' in number else int(number)
                    results[i] = number

    else:
        for idx, term in enumerate(terms):
            for i, param in enumerate(param_list):
                if param == term:
                    results[i] = terms[idx+1]
                    
    return results

def cal_acc_eo(filename):
    M = sio.loadmat(filename, squeeze_me= True)
    # difference between each group
    groupkey = [i for i in list(M.keys()) if '__' not in i]
    num_groups = len(groupkey)
    num_classes = M[groupkey[0]].shape[0]
    #check total acc
    total = 0.0
    right_ans = 0.0
    group_acc = []
    for key in groupkey:
        s = M[key]
        total += s.sum() 
        right_ans += np.diag(s).sum()
        group_acc.append(np.diag(s).sum() / s.sum())
    acc = right_ans / total

    
    #check dp
    total_dp = []
    group_label_acc = np.zeros((num_groups, num_classes))
    for i in range(num_classes):
        tmp = []
        for j in range(num_groups):
            confu_mat = M[groupkey[j]]
            tmp.append(confu_mat[:, i].sum()/confu_mat.sum())

        total_dp.append(max(tmp) - min(tmp))
        
    #check eo
    total_eo = []
    for i in range(num_classes):
        tmp = []
        for j in range(num_groups):
            confu_mat = M[groupkey[j]]
            tmp.append(confu_mat[i, i]/confu_mat[i, :].sum() )
            group_label_acc[j,i] = confu_mat[i, i]/confu_mat[i, :].sum()
#         tmp = np.array(tmp)
#         total_eo.append(max(tmp) - np.average(np.array(tmp)))
#         total_eo.append(np.max(np.abs(tmp - np.average(tmp))))
        total_eo.append(max(tmp) - min(tmp))
    
    return acc, np.array(total_dp), np.array(total_eo), group_acc, group_label_acc

def cal_labelacc(filename):
    M = sio.loadmat(filename, squeeze_me= True)
    # difference between each group
    groupkey = [i for i in list(M.keys()) if '__' not in i]
    num_groups = len(groupkey)
    num_classes = M[groupkey[0]].shape[0]
    #check total acc
    total = 0.0
    right_ans = 0.0
    label_acc = []
    total_confu = np.zeros_like(M[groupkey[0]], dtype=np.float)
    for key in groupkey:
        total_confu += M[key]
    for l in range(num_classes):
        l_acc = total_confu[l,l] / total_confu[l,:].sum()
        label_acc.append(l_acc)
    return label_acc

def print_result(result_dict, cnt_dict, num_seeds):
    new_dict = {}
    for key in cnt_dict.keys():
        if cnt_dict[key] == num_seeds:
            result_dict[key] /= num_seeds
            acc, worstacc, avgeo, maxeo = list(result_dict[key])
            print('<{}>, acc : {:.4f} / avgeo : {:.4f} / maxeo : {:.4f} / worstacc : {:.4f}\
             '.format(key, acc, avgeo, maxeo, worstacc))
            new_dict[key] = result_dict[key]
    return new_dict


# def make_teacher_path(config, tconfig):
#     teacher_model = tconfig['model']
#     img_size = config['img-size']
#     date = tconfig['date']
#     dataset = config['dataset']
#     t_epoch = tconfig['epochs']
#     lr = config['lr']
#     bs = config['batch-size']

#     if 'wrn' not in teacher_model:
#         if 'pretrained' not in tconfig.keys():
#             teacher_model_name = '_'.join([teacher_model, 
#                                    'seed{}',
#                                    'epochs'+t_epoch, 
#                                    'bs'+bs, 'lr'+lr])
#         else:
#             teacher_model_name = '_'.join([teacher_model, 
#                                    'pretrained',
#                                    'seed{}',
#                                    'epochs'+t_epoch, 
#                                    'bs'+bs, 'lr'+lr])
#         if 'jointT' in config.keys():
#             teacher_model_name += '_joint'

#     else:
#         teacher_model_name = '_'.join([teacher_model,
#                                img_size+'img',
#                                'seed{}', 
#                                'epochs'+t_epoch, 
#                                'bs'+bs, 
#                                'lr'+lr])
    
#     teacher_model_name += '.pt'
#     teacher_path = os.path.join('trained_models', date, dataset, 'scratch', teacher_model_name)
#     return teacher_path

# def print_command_fairness(config, message='', process_per_gpu_max=2, GPU_start=0, GPU_end=3, teacher=False, file='', attach=False, parallel=False, tconfig=None):
#     list_keys = []
#     list_items = []
#     total_process = 1
#     for key, item in config.items():
#         if isinstance(item, list):
#             total_process *= len(config[key])
#             list_keys.append(key)
#             list_items.append([str(i) for i in item])
            
#     if teacher:
#         teacher_path = make_teacher_path(config, tconfig)
        
#     ############################## make a combination of parameters #################################

#     fixed_strs = ''
#     for key in list_keys:
#         fixed_strs += '--'+key+' {} '
        
#     variant_strs = []        
#     variant_params = list(product(*list_items))     
    
#     for param in variant_params:
#         variant_strs.append(fixed_strs.format(*param))
    
#     ############################### start a loop ################################
#     if file!='':
#         if not attach:
#             file_pointer = open(file, 'w')
#         else:
#             file_pointer = open(file, 'a')
#     pre_str = 'CUDA_VISIBLE_DEVICES={} python main.py '# --device {} '
#     #pre_str = 'python main.py --device {} '
#     pre_str += ' '.join(['--'+key+' '+str(item) for key, item in config.items() if key not in list_keys])

#     process_per_gpu_cur =0
#     GPU_cur = GPU_start
#     cur_process = 0
    
#     for variant_str in variant_strs:
#         string = (pre_str + ' ' + variant_str).format(GPU_cur)
#         ############ add teacher #############
#         if teacher:
#             seed_, gpu_ = find_parameter(string, ['--seed', '--device'], spliter=' ')
#             string += ' --teacher-path ' + teacher_path.format(seed_)
# #             string += ' --t-device {}'.format(gpu_)
#             string += ' --teacher-type {}'.format(tconfig['model'])
        
#         process_per_gpu_cur += 1
#         cur_process += 1

#         if total_process>cur_process:
#             if process_per_gpu_cur == process_per_gpu_max:
#                 process_per_gpu_cur = 0
#                 if GPU_cur<GPU_end:
#                     string+=' &'
#                     GPU_cur+=1
#                 else:
#                     GPU_cur = GPU_start
#                     string += ' &\nwait'
#                     string+= '\n'
#             else:
#                 string+=' &'
#         else:
#             string += ' &\nwait'            
#         print(string)
#         if file!='':
#             file_pointer.write(string +'\n')
#     string = 'python ../../slack_sender.py {}'.format(message)
#     print (string)
#     if file!='':
#         file_pointer.write(string +'\n')
#         file_pointer.close()    
    

