import torch
from .pgd import pgd
from .s_pgd import s_pgd
from .ibp import ibp
from .s_ibp import s_ibp
from .s_ibp_lp import s_ibp_lp
import pandas as pd
import numpy as np
import os
import warnings
import json
from torch import round as round_tensor

# from auto_LiRPA import BoundedModule, BoundedTensor, register_custom_op
# from auto_LiRPA.perturbations import PerturbationLpNorm

# from utils.auto_LiRPA_custom_operators.convolution_3d import BoundConv3D 



RESULTS_COLUMNS = [
    "ucf_101_id",
    "classification_prediction_idx",  # Note: ouputs of NN are defined as 'predictions'
    "classification_target_idx",  # Note: labels are defined as 'targets'
    "classification_correct",
    "classification_safe",
]

PRINT = False

def calculate_bounds_pgd(
    model: torch.nn.Module,
    dataloader,
    epsilon: float,
    k:int,
    batch_size: int,
    results_path: str | os.PathLike,
    device: torch.device,
    round_floats=False,
    begin_frame: int = 1,
    end_frame: int = 29,
):
    """Perturb the inputs to a neural network `model` with size `epsilon` using PGD
    to calculate their respective output bounds.
    Save these to file, in the `./verification_results` directory.

    Args:
        model (torch.nn.Module): NN Module to be verified
        dataloader (_type_): Dataloader for the test dataset
        epsilon (float): perturbation measure
        round_floats (bool, optional): If the floats need to be rounded. Defaults to False.

    Raises:
        Exception: NotImplementedError, in case the method is not implemented by autoLirpa

    """
    print(f"Performing PGD with an epsilon of {epsilon}")

    pgd_step_size = 0.1
    pgd_steps = 100

    df_results = pd.DataFrame(columns=RESULTS_COLUMNS)

    num_samples_verified = 0
    num_samples_correctly_classified = 0
    num_samples_safe = 0

    for dl_idx, (inputs, targets) in enumerate(dataloader):
        print(f"Starting new batch")

        if device == torch.device("cuda"):
            inputs = inputs.cuda()
            targets = targets.cuda()
            model = model.cuda()
                
        if dl_idx == 0:
            print("Running on", device)


        preds = model(inputs)
        pred_targets = torch.max(preds, dim=1).indices
        

        x_ = inputs.clone()

        '''
            Shared Constraints - reducing Perturbations to middle k frames
        '''
        # Shared Constraints -  k (perturb only k middle frames)
        # inputs has a shape [BATCH_SIZE, 1, 28, 64, 64], inp_subset should be [BATCH_SIZE, 1, k, 64, 64]
        # middle_frame = begin_frame + (end_frame - begin_frame)//2
        middle_frame = (end_frame - begin_frame)//2 
        inp_subset = x_[:, :, middle_frame - (k//2) : middle_frame + (k//2)]
        
        
        attack_input_ub = torch.cat((x_[:, :, :middle_frame - (k//2)], inp_subset + epsilon, x_[:, :, middle_frame + (k//2):]), dim=2)
        attack_input_lb = torch.cat((x_[:, :, :middle_frame - (k//2)], inp_subset - epsilon, x_[:, :, middle_frame + (k//2):]), dim=2)
        
        attack_results = pgd(
            model,
            attack_input_lb.clone(),
            attack_input_ub.clone(),
            targets.squeeze(dim=1),
            step_size=pgd_step_size,
            num_steps=pgd_steps,
        )

        # Iterate over each element in batch
        for i in range(len(targets)):
            # Each batch has len(inputs) images, and dl_idx indicates the batch number.
            # Therefore, id would be `i + dl_idx * len(inputs)`
            id = dl_idx * batch_size + i
            new_row = {"ucf_101_id": id}

            # num of samples verified
            num_samples_verified += 1

            (
                print(
                    f"Input {i} top-1 prediction is: {pred_targets[i]}, the "
                    f"ground-truth is: {targets[i]}"
                )
                if PRINT
                else None
            )

            classification_correct = (pred_targets[i] == targets[i]).item()

            num_samples_correctly_classified += 1 if classification_correct else 0

            
            truth_idx = int(targets[i].item())

            # Sanity check that the lower bound of the truth class is greater than the upper bound of all other classes
            if ( (pred_targets[i] == attack_results[i]).item() ):
                classification_safe = True
                print(
                    f"Classification is safe for this perturbation for video {id}"
                )
            else:
                classification_safe = False
                print(
                    f"Classification is NOT SAFE for this perturbation for video {id}"
                )

            if (attack_results[i] != targets[i] and classification_safe):
                
                a, adv_output, adv_input = pgd(
                    model,
                    attack_input_lb.clone(),
                    attack_input_ub.clone(),
                    targets.squeeze(dim=1),
                    step_size=pgd_step_size,
                    num_steps=pgd_steps,
                    return_model_output=True,
                )
                if PRINT:
                    print(f"For the input {np.array(inputs[i,0,...])}")
                    print(f"We have the classification {targets[i,...]}")
                    print(f"We have the adv input {np.array(adv_input[i,0, ...])}")
                    print(f"We have the adv output: {adv_output[i,...]}")
                warnings.warn(
                    "Attack was successful but the bounds were not safe! Inputs. Model output: "
                    + str(torch.argmax(adv_output[i]).item())
                    + " Expected output: "
                    + str(targets[i].item())
                )
                # raise Exception("The bounds should be sound")
                warnings.warn(f"The bounds should be sound for video {id}")

            new_row["classification_prediction_idx"] = pred_targets[i].item()
            new_row["classification_target_idx"] = targets[i].item()
            new_row["classification_correct"] = classification_correct
            safe = True if classification_safe else False
            new_row["classification_safe"] = safe
            if safe:
                num_samples_safe += 1

            new_row_df = pd.json_normalize(new_row)
            df_results = pd.concat([df_results, new_row_df], ignore_index=True)

    print(f"----\nSUMMARY\n----")
    print(f"For the method: pgd")
    print(f"Num samples verified: {num_samples_verified}")
    print(f"Num samples correctly classified: {num_samples_correctly_classified}")
    print(f"Num samples safe: {num_samples_safe}")
    print()

    results_summary = {
        "method": "pgd",
        "num_samples_verified": f"{num_samples_verified}",
        "num_samples_correctly_classified": f"{num_samples_correctly_classified}",
        "num_samples_safe": f"{num_samples_safe}",
        "epsilon": f"{epsilon}",
    }

    filename = f"results_pgd_{epsilon}_{k}"
    filename += f"_rounded" if round_floats else ""
    save_results_to_csv(df_results, results_summary, results_path, filename)


def calculate_bounds_s_pgd(
    model: torch.nn.Module,
    dataloader,
    epsilon: float,
    m: torch.Tensor,
    batch_size: int,
    results_path: str | os.PathLike,
    device: torch.device,
    round_floats=False,
    begin_frame: int = 1,
    end_frame: int = 29,
):
    """ Perturb the inputs to a neural network `model` with size `epsilon` using shared PGD 
    constraints defined by the Tensor `m` to calculate their respective output bounds.
    Save these to file, in the `./verification_results` directory.

    Args:
        model (torch.nn.Module): NN Module 
        dataloader (_type_): Dataloader for the test dataset
        epsilon (float): perturbation measure
        m (torch.Tensor): indicates the frames to be perturbed and the amount of perturbation needed [0,1]
        batch_size (int): batch size to keep track of input indices.
        results_path (str | os.PathLike): directory to store the results
        device (torch.device): Torch device (cuda / cpu)
        round_floats (bool, optional): If the floats need to be rounded. Defaults to False.
        begin_frame (int, optional): Index of the beginning frame in the dataset. Defaults to 1.
        end_frame (int, optional): Index of the end frame in the dataset. Defaults to 29.
    """
    print(f"Performing S-PGD with an epsilon of {epsilon}")

    pgd_step_size = 0.1
    pgd_steps = 100

    df_results = pd.DataFrame(columns=RESULTS_COLUMNS)

    num_samples_verified = 0
    num_samples_correctly_classified = 0
    num_samples_safe = 0

    for dl_idx, (inputs, targets) in enumerate(dataloader):
        print(f"Starting new batch")

        if device == torch.device("cuda"):
            inputs = inputs.cuda()
            targets = targets.cuda()
            model = model.cuda()
                
        if dl_idx == 0:
            print("Running on", device)


        preds = model(inputs)
        pred_targets = torch.max(preds, dim=1).indices
        


        '''
            Shared PGD attack with m - reducing Perturbations to frames defined by m
        '''
        attack_input_ub = inputs + epsilon
        attack_input_lb = inputs - epsilon

        
        attack_results = s_pgd(
            model,
            attack_input_lb,
            attack_input_ub,
            targets.squeeze(dim=1),
            m,
            step_size=pgd_step_size,
            num_steps=pgd_steps,
        )

        # Iterate over each element in batch
        for i in range(len(targets)):
            # Each batch has len(inputs) images, and dl_idx indicates the batch number.
            # Therefore, id would be `i + dl_idx * len(inputs)`
            id = dl_idx * batch_size + i
            new_row = {"ucf_101_id": id}

            # num of samples verified
            num_samples_verified += 1

            (
                print(
                    f"Input {i} top-1 prediction is: {pred_targets[i]}, the "
                    f"ground-truth is: {targets[i]}"
                )
                if PRINT
                else None
            )

            classification_correct = (pred_targets[i] == targets[i]).item()

            num_samples_correctly_classified += 1 if classification_correct else 0

            
            truth_idx = int(targets[i].item())

            # Sanity check that the lower bound of the truth class is greater than the upper bound of all other classes
            if ( (pred_targets[i] == attack_results[i]).item() ):
                classification_safe = True
                print(
                    f"Classification is safe for this perturbation for video {id}"
                )
            else:
                classification_safe = False
                print(
                    f"Classification is NOT SAFE for this perturbation for video {id}"
                )

            if (attack_results[i] != targets[i] and classification_safe):
                
                a, adv_output, adv_input = s_pgd(
                    model,
                    attack_input_lb,
                    attack_input_ub,
                    targets.squeeze(dim=1),
                    m,
                    step_size=pgd_step_size,
                    num_steps=pgd_steps,
                    return_model_output=True,
                )
                if PRINT:
                    print(f"For the input {np.array(inputs[i,0,...])}")
                    print(f"We have the classification {targets[i,...]}")
                    print(f"We have the adv input {np.array(adv_input[i,0, ...])}")
                    print(f"We have the adv output: {adv_output[i,...]}")
                warnings.warn(
                    "Attack was successful but the bounds were not safe! Inputs. Model output: "
                    + str(torch.argmax(adv_output[i]).item())
                    + " Expected output: "
                    + str(targets[i].item())
                )
                # raise Exception("The bounds should be sound")
                warnings.warn(f"The bounds should be sound for video {id}")

            new_row["classification_prediction_idx"] = pred_targets[i].item()
            new_row["classification_target_idx"] = targets[i].item()
            new_row["classification_correct"] = classification_correct
            safe = True if classification_safe else False
            new_row["classification_safe"] = safe
            if safe:
                num_samples_safe += 1

            new_row_df = pd.json_normalize(new_row)
            df_results = pd.concat([df_results, new_row_df], ignore_index=True)

    print(f"----\nSUMMARY\n----")
    print(f"For the method: spgd")
    print(f"Num samples verified: {num_samples_verified}")
    print(f"Num samples correctly classified: {num_samples_correctly_classified}")
    print(f"Num samples safe: {num_samples_safe}")
    print()

    results_summary = {
        "method": "spgd",
        "num_samples_verified": f"{num_samples_verified}",
        "num_samples_correctly_classified": f"{num_samples_correctly_classified}",
        "num_samples_safe": f"{num_samples_safe}",
        "epsilon": f"{epsilon}",
    }

    k = (m == 1).sum(dim=0).item() #number of 1s
    filename = f"results_s_pgd_{epsilon}_{k}"
    filename += f"_rounded" if round_floats else ""
    save_results_to_csv(df_results, results_summary, results_path, filename)



def calculate_bounds_ibp(
    model: torch.nn.Module,
    dataloader,
    epsilon: float,
    k:int,
    batch_size: int,
    results_path: str | os.PathLike,
    device: torch.device,
    round_floats=False,
    begin_frame: int = 1,
    end_frame: int = 29,
):
    """Verify the inputs to a neural network `model` with size `epsilon` using IBP
    to calculate their respective output bounds.
    Save these to file, in the `./verification_results` directory.

    Args:
        model (torch.nn.Module): NN Module to be verified
        dataloader (_type_): Dataloader for the test dataset
        epsilon (float): perturbation measure
        round_floats (bool, optional): If the floats need to be rounded. Defaults to False.

    Raises:
        Exception: NotImplementedError, in case the method is not implemented

    """
    print(f"Performing IBP with an epsilon of {epsilon}")


    df_results = pd.DataFrame(columns=RESULTS_COLUMNS)

    num_samples_verified = 0
    num_samples_correctly_classified = 0
    num_samples_safe = 0

    for dl_idx, (inputs, targets) in enumerate(dataloader):
        print(f"Starting new batch")

        if device == torch.device("cuda"):
            inputs = inputs.cuda()
            targets = targets.cuda()
            model = model.cuda()
                
        if dl_idx == 0:
            print("Running on", device)


        preds = model(inputs)
        pred_targets = torch.max(preds, dim=1).indices
        

        x_ = inputs.clone()

        '''
            Shared Constraints - reducing Perturbations to middle k frames
        '''
        # Shared Constraints -  k (perturb only k middle frames)
        # inputs has a shape [BATCH_SIZE, 1, 28, 64, 64], inp_subset should be [BATCH_SIZE, 1, k, 64, 64]
        # middle_frame = begin_frame + (end_frame - begin_frame)//2
        middle_frame = (end_frame - begin_frame)//2
        inp_subset = x_[:, :, middle_frame - (k//2) : middle_frame + (k//2)]
        
        
        attack_input_ub = torch.cat((x_[:, :, :middle_frame - (k//2)], inp_subset + epsilon, x_[:, :, middle_frame + (k//2):]), dim=2)
        attack_input_lb = torch.cat((x_[:, :, :middle_frame - (k//2)], inp_subset - epsilon, x_[:, :, middle_frame + (k//2):]), dim=2)
        
        ibp_results, final_lower, final_upper = ibp(
            model,
            attack_input_lb.clone(),
            attack_input_ub.clone(),
            targets,
            return_ub_lb = True
        )

        # Iterate over each element in batch
        for i in range(len(targets)):
            # Each batch has len(inputs) images, and dl_idx indicates the batch number.
            # Therefore, id would be `i + dl_idx * len(inputs)`
            id = dl_idx * batch_size + i
            new_row = {"ucf_101_id": id}

            # num of samples verified
            num_samples_verified += 1

            (
                print(
                    f"Input {i} top-1 prediction is: {pred_targets[i]}, the "
                    f"ground-truth is: {targets[i]}"
                )
                if PRINT
                else None
            )

            classification_correct = (pred_targets[i] == targets[i]).item()

            num_samples_correctly_classified += 1 if classification_correct else 0

            
            truth_idx = int(targets[i].item())
            # Sanity check that the lower bound of the truth class is greater than the upper bound of all other classes
            if ( ( final_lower[i][truth_idx] > torch.cat((final_upper[i][:truth_idx], final_upper[i][truth_idx + 1:], ))).all().item()):
                classification_safe = True
                print(
                    f"Classification is safe for this perturbation for video {id}"
                )
            else:
                classification_safe = False
                print(
                    f"Classification is NOT SAFE for this perturbation for video {id}"
                )

            if (ibp_results[i] != targets[i] and classification_safe):
                
                a, lb, ub = ibp(
                    model,
                    attack_input_lb.clone(),
                    attack_input_ub.clone(),
                    targets,
                    return_ub_lb=True,
                )
                if PRINT:
                    print(f"For the input {np.array(inputs[i,0,...])}")
                    print(f"We have the classification {targets[i,...]}")
                    print(f"We have the lb {np.array(lb[i,0, ...])}")
                    print(f"We have the adv output: {ub[i,...]}")
                warnings.warn(
                    "Attack was successful but the bounds were not safe! Inputs. Model output: "
                    + str(torch.argmax(lb[i]).item())
                    + " Expected output: "
                    + str(targets[i].item())
                )
                # raise Exception("The bounds should be sound")
                warnings.warn(f"The bounds should be sound for video {id}")

            new_row["classification_prediction_idx"] = pred_targets[i].item()
            new_row["classification_target_idx"] = targets[i].item()
            new_row["classification_correct"] = classification_correct
            safe = True if classification_safe else False
            new_row["classification_safe"] = safe
            if safe:
                num_samples_safe += 1

            new_row_df = pd.json_normalize(new_row)
            df_results = pd.concat([df_results, new_row_df], ignore_index=True)

    print(f"----\nSUMMARY\n----")
    print(f"For the method: ibp")
    print(f"Num samples verified: {num_samples_verified}")
    print(f"Num samples correctly classified: {num_samples_correctly_classified}")
    print(f"Num samples safe: {num_samples_safe}")
    print()

    results_summary = {
        "method": "ibp",
        "num_samples_verified": f"{num_samples_verified}",
        "num_samples_correctly_classified": f"{num_samples_correctly_classified}",
        "num_samples_safe": f"{num_samples_safe}",
        "epsilon": f"{epsilon}",
    }

    filename = f"results_ibp_{epsilon}_{k}"
    filename += f"_rounded" if round_floats else ""
    save_results_to_csv(df_results, results_summary, results_path, filename)


def calculate_bounds_s_ibp(
    model: torch.nn.Module,
    dataloader,
    epsilon: float,
    m: torch.Tensor,
    batch_size: int,
    results_path: str | os.PathLike,
    device: torch.device,
    round_floats=False,
    begin_frame: int = 1,
    end_frame: int = 29,
):
    """Verify the inputs to a neural network `model` with size `epsilon` using S-IBP
    to calculate their respective output bounds.
    Save these to file, in the `./verification_results` directory.

    Args:
        model (torch.nn.Module): NN Module to be verified
        dataloader (_type_): Dataloader for the test dataset
        epsilon (float): perturbation measure
        m (torch.Tensor): indicates the frames to be perturbed and the amount of perturbation needed [0,1]
        round_floats (bool, optional): If the floats need to be rounded. Defaults to False.

    Raises:
        Exception: NotImplementedError, in case the method is not implemented

    """
    print(f"Performing S-IBP with an epsilon of {epsilon}")


    df_results = pd.DataFrame(columns=RESULTS_COLUMNS)

    num_samples_verified = 0
    num_samples_correctly_classified = 0
    num_samples_safe = 0

    for dl_idx, (inputs, targets) in enumerate(dataloader):
        print(f"Starting new batch")

        if device == torch.device("cuda"):
            inputs = inputs.cuda()
            targets = targets.cuda()
            model = model.cuda()
                
        if dl_idx == 0:
            print("Running on", device)


        preds = model(inputs)
        pred_targets = torch.max(preds, dim=1).indices
        

        '''
            TODO: Shared IBP with m - reducing Perturbations to frames defined by m
        '''
        # attack_input_ub = inputs + epsilon
        # attack_input_lb = inputs - epsilon
        k = (m == 1).sum(dim=0).item() #number of 1s
        
        x_ = inputs.clone()
        '''
            Shared Constraints - reducing Perturbations to middle k frames
        '''
        # Shared Constraints -  k (perturb only k middle frames)
        # inputs has a shape [BATCH_SIZE, 1, 28, 64, 64], inp_subset should be [BATCH_SIZE, 1, k, 64, 64]
        # middle_frame = begin_frame + (end_frame - begin_frame)//2
        middle_frame = (end_frame - begin_frame)//2
        inp_subset = x_[:, :, middle_frame - (k//2) : middle_frame + (k//2)]
        
        attack_input_ub = torch.cat((x_[:, :, :middle_frame - (k//2)], inp_subset + epsilon, x_[:, :, middle_frame + (k//2):]), dim=2)
        attack_input_lb = torch.cat((x_[:, :, :middle_frame - (k//2)], inp_subset - epsilon, x_[:, :, middle_frame + (k//2):]), dim=2)
        
        
        (ibp_results, final_lower, final_upper) = s_ibp(
            model,
            attack_input_lb.clone(),
            attack_input_ub.clone(),
            targets,
            m,
            return_ub_lb = True
        )
        
        

        # Iterate over each element in batch
        for i in range(len(targets)):
            # Each batch has len(inputs) images, and dl_idx indicates the batch number.
            # Therefore, id would be `i + dl_idx * len(inputs)`
            id = dl_idx * batch_size + i
            new_row = {"ucf_101_id": id}

            # num of samples verified
            num_samples_verified += 1

            (
                print(
                    f"Input {i} top-1 prediction is: {pred_targets[i]}, the "
                    f"ground-truth is: {targets[i]}"
                )
                if PRINT
                else None
            )

            classification_correct = (pred_targets[i] == targets[i]).item()

            num_samples_correctly_classified += 1 if classification_correct else 0

            
            truth_idx = int(targets[i].item())

            # Sanity check that the lower bound of the truth class is greater than the upper bound of all other classes
            if ( ( final_lower[i][truth_idx] > torch.cat((final_upper[i][:truth_idx], final_upper[i][truth_idx + 1:], ))).all().item() ):
                classification_safe = True
                print(
                    f"Classification is safe for this perturbation for video {id}"
                )
            else:
                classification_safe = False
                print(
                    f"Classification is NOT SAFE for this perturbation for video {id}"
                )

            if (ibp_results[i] != targets[i] and classification_safe):
                
                a, lb, ub = s_ibp(
                    model,
                    attack_input_lb.clone(),
                    attack_input_ub.clone(),
                    targets,
                    m,
                    return_ub_lb=True,
                )
                if PRINT:
                    print(f"For the input {np.array(inputs[i,0,...])}")
                    print(f"We have the classification {targets[i,...]}")
                    print(f"We have the lb {np.array(lb[i,0, ...])}")
                    print(f"We have the adv output: {ub[i,...]}")
                warnings.warn(
                    "Attack was successful but the bounds were not safe! Inputs. Model output: "
                    + str(torch.argmax(lb[i]).item())
                    + " Expected output: "
                    + str(targets[i].item())
                )
                # raise Exception("The bounds should be sound")
                warnings.warn(f"The bounds should be sound for video {id}")

            new_row["classification_prediction_idx"] = pred_targets[i].item()
            new_row["classification_target_idx"] = targets[i].item()
            new_row["classification_correct"] = classification_correct
            safe = True if classification_safe else False
            new_row["classification_safe"] = safe
            if safe:
                num_samples_safe += 1

            new_row_df = pd.json_normalize(new_row)
            df_results = pd.concat([df_results, new_row_df], ignore_index=True)

    print(f"----\nSUMMARY\n----")
    print(f"For the method: s-ibp")
    print(f"Num samples verified: {num_samples_verified}")
    print(f"Num samples correctly classified: {num_samples_correctly_classified}")
    print(f"Num samples safe: {num_samples_safe}")
    print()

    results_summary = {
        "method": "s-ibp",
        "num_samples_verified": f"{num_samples_verified}",
        "num_samples_correctly_classified": f"{num_samples_correctly_classified}",
        "num_samples_safe": f"{num_samples_safe}",
        "epsilon": f"{epsilon}",
    }

    filename = f"results_s_ibp_{epsilon}_{k}"
    filename += f"_rounded" if round_floats else ""
    save_results_to_csv(df_results, results_summary, results_path, filename)


def calculate_bounds_s_ibp_lp(
    model: torch.nn.Module,
    dataloader,
    epsilon: float,
    k: int,
    batch_size: int,
    results_path: str | os.PathLike,
    device: torch.device,
    round_floats=False,
    begin_frame: int = 1,
    end_frame: int = 29,
):
    """Verify the inputs to a neural network `model` with size `epsilon` using S-IBP with LP (Gurobi)
    to calculate their respective output bounds.
    Save these to file, in the `./verification_results` directory.

    Args:
        model (torch.nn.Module): NN Module to be verified
        dataloader (_type_): Dataloader for the test dataset
        epsilon (float): perturbation measure
        m (torch.Tensor): indicates the frames to be perturbed and the amount of perturbation needed [0,1]
        round_floats (bool, optional): If the floats need to be rounded. Defaults to False.

    Raises:
        Exception: NotImplementedError, in case the method is not implemented

    """
    print(f"Performing S-IBP (LP) with an epsilon of {epsilon}")


    df_results = pd.DataFrame(columns=RESULTS_COLUMNS)

    num_samples_verified = 0
    num_samples_correctly_classified = 0
    num_samples_safe = 0

    for dl_idx, (inputs, targets) in enumerate(dataloader):
        print(f"Starting new batch")

        if device == torch.device("cuda"):
            inputs = inputs.cuda()
            targets = targets.cuda()
            model = model.cuda()
                
        if dl_idx == 0:
            print("Running on", device)


        preds = model(inputs)
        pred_targets = torch.max(preds, dim=1).indices
        

        '''
            TODO: Shared IBP with m - reducing Perturbations to frames defined by m
        '''
        # attack_input_ub = inputs + epsilon
        # attack_input_lb = inputs - epsilon
        
        x_ = inputs.clone()
        '''
            Shared Constraints - reducing Perturbations to middle k frames
        '''
        # Shared Constraints -  k (perturb only k middle frames)
        # inputs has a shape [BATCH_SIZE, 1, 28, 64, 64], inp_subset should be [BATCH_SIZE, 1, k, 64, 64]
        # middle_frame = begin_frame + (end_frame - begin_frame)//2
        
        
        middle_frame = (end_frame - begin_frame)//2
        # inp_subset = x_[:, :, middle_frame - (k//2) : middle_frame + (k//2)]
        
        # attack_input_ub = torch.cat((x_[:, :, :middle_frame - (k//2)], inp_subset + epsilon, x_[:, :, middle_frame + (k//2):]), dim=2)
        # attack_input_lb = torch.cat((x_[:, :, :middle_frame - (k//2)], inp_subset - epsilon, x_[:, :, middle_frame + (k//2):]), dim=2)
        
        # For S-IBP (lp), we pass the set of frames as a list. TODO: make implementations consistent with s-ibp
        sets_of_frames = [list(range(middle_frame - (k//2)+1, middle_frame + (k//2)+1))] # add 1 to account for offset
        
        (ibp_results, final_lower, final_upper) = s_ibp_lp(
            model,
            x_,
            [epsilon],
            sets_of_frames
        )
        
        

        # Iterate over each element in batch
        for i in range(len(targets)):
            # Each batch has len(inputs) images, and dl_idx indicates the batch number.
            # Therefore, id would be `i + dl_idx * len(inputs)`
            id = dl_idx * batch_size + i
            new_row = {"ucf_101_id": id}

            # num of samples verified
            num_samples_verified += 1

            (
                print(
                    f"Input {i} top-1 prediction is: {pred_targets[i]}, the "
                    f"ground-truth is: {targets[i]}"
                )
                if PRINT
                else None
            )

            classification_correct = (pred_targets[i] == targets[i]).item()

            num_samples_correctly_classified += 1 if classification_correct else 0

            
            truth_idx = int(targets[i].item())

            # Sanity check that the lower bound of the truth class is greater than the upper bound of all other classes
            if ( ( final_lower[i][truth_idx] > torch.cat((final_upper[i][:truth_idx], final_upper[i][truth_idx + 1:], ))).all().item() ):
                classification_safe = True
                print(
                    f"Classification is safe for this perturbation for video {id}"
                )
            else:
                classification_safe = False
                print(
                    f"Classification is NOT SAFE for this perturbation for video {id}"
                )

            if (ibp_results[i] != targets[i] and classification_safe):
                # Classification is incorrect, but verification was found safe!?
                
                lb, ub = final_lower, final_upper
                if PRINT:
                    print(f"For the input {np.array(inputs[i,0,...])}")
                    print(f"We have the classification {targets[i,...]}")
                    print(f"We have the lb {np.array(lb[i,0, ...])}")
                    print(f"We have the adv output: {ub[i,...]}")
                warnings.warn(
                    "Attack was successful but the bounds were not safe! Inputs. Model output: "
                    + str(torch.argmax(lb[i]).item())
                    + " Expected output: "
                    + str(targets[i].item())
                )
                # raise Exception("The bounds should be sound")
                warnings.warn(f"The bounds should be sound for video {id}")

            new_row["classification_prediction_idx"] = pred_targets[i].item()
            new_row["classification_target_idx"] = targets[i].item()
            new_row["classification_correct"] = classification_correct
            safe = True if classification_safe else False
            new_row["classification_safe"] = safe
            if safe:
                num_samples_safe += 1

            new_row_df = pd.json_normalize(new_row)
            df_results = pd.concat([df_results, new_row_df], ignore_index=True)

    print(f"----\nSUMMARY\n----")
    print(f"For the method: s-ibp")
    print(f"Num samples verified: {num_samples_verified}")
    print(f"Num samples correctly classified: {num_samples_correctly_classified}")
    print(f"Num samples safe: {num_samples_safe}")
    print()

    results_summary = {
        "method": "s-ibp-lp",
        "num_samples_verified": f"{num_samples_verified}",
        "num_samples_correctly_classified": f"{num_samples_correctly_classified}",
        "num_samples_safe": f"{num_samples_safe}",
        "epsilon": f"{epsilon}",
    }

    filename = f"results_s_ibp_lp_{epsilon}_{k}"
    filename += f"_rounded" if round_floats else ""
    save_results_to_csv(df_results, results_summary, results_path, filename)



def save_results_to_csv(results: pd.DataFrame, summary: dict, results_path: str | os.PathLike, filename: str):
    """Save a Pandas Dataframe as CSV

    Args:
        results (pd.DataFrame): Pandas Dataframe to be saved as CSV
        summary (dict): Summary of the results to be saved as JSON
        filename (str): Name of the file - based on the method, epsilon value
    """
    results.to_csv(
        results_path / f"{filename}.csv",
        index=False,
    )

    with open(
        results_path / f"{filename}.json",
        "w",
    ) as file:
        json.dump(summary, file, indent=4)

    print(f"Saved to file { results_path / f'{filename}.csv'}")