import torch
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score
from helper_functions import meanGDD, numAccurate,vague_belief_mass
from loss import lossFunc
from collections import Counter
import wandb


def test_result_log(js_result, prec_recall_f, acc, epoch, bestModel=False):
    if bestModel:
        wandb.log({
            f"TestB OverallJS": js_result[0], 
            f"TestB CompJS": js_result[1], 
            f"TestB SnglJS": js_result[2],
            f"TestB CompPreci": prec_recall_f[0], 
            f"TestB CompRecal": prec_recall_f[1], 
            f"TestB CompFscor": prec_recall_f[2], 
            f"TestB acc": acc}, step=epoch)
        print(f"TestBest acc: {acc:.4f}, \n \
            JS(O_V_N): {js_result}, P_R_F_compGTcnt_cmpPREDcnt: {prec_recall_f}\n")
        return 
    else:
        if epoch is None:
            wandb.log({
                f"TestF OverallJS": js_result[0], 
                f"TestF CompJS": js_result[1], 
                f"TestF SnglJS": js_result[2],
                f"TestF CompPreci": prec_recall_f[0], 
                f"TestF CompRecal": prec_recall_f[1], 
                f"TestF CompFscor": prec_recall_f[2], 
                f"TestF acc": acc}, step=epoch)
            print(f"TestF acc: {acc:.4f}, \n \
                JS(O_V_N): {js_result}, P_R_F_compGTcnt_cmpPREDcnt: {prec_recall_f}\n")
        else:
            wandb.log({
                f"Test OverallJS": js_result[0], 
                f"Test CompJS": js_result[1], 
                f"Test SnglJS": js_result[2],
                f"Test CompPreci": prec_recall_f[0], 
                f"Test CompRecal": prec_recall_f[1], 
                f"Test CompFscor": prec_recall_f[2], 
                f"Test acc": acc}, step=epoch)
            print(f"Test acc: {acc:.4f}, \n \
                JS(O_V_N): {js_result}, P_R_F_compGTcnt_cmpPREDcnt: {prec_recall_f}\n")


@torch.no_grad()
def evaluate(
    model, val_loader, 
    num_single_classes, kappa, a_copy, 
    annealing_coefficient, 
    device
    ):
    model.eval()
    results = {
        'accuracy': 0.0,
        'mean_val_loss': 0.0
        }
    total_correct = 0.0
    total_samples = 0
    val_losses = []
    for batch in val_loader:
        images, _, labels = batch
        images, labels = images.to(device), labels.to(device)
        one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=kappa)
        output = model(images)
        loss = lossFunc(output, one_hot_labels, a_copy, num_single_classes, annealing_coefficient)
        # batch_loss, _, _ = lossFunc(r, one_hot_labels, a_copy, annealing_coefficient)
        # loss = torch.mean(batch_loss)

        total_correct += numAccurate(output, labels)
        total_samples += len(labels)
        val_loss = loss.detach()
        val_losses.append(val_loss)
    results['mean_val_loss'] = torch.stack(val_losses).mean().item()
    results['accuracy'] = total_correct / total_samples
    return results


@torch.no_grad()
def evaluate_vague_nonvague_ENN(model, val_loader, R, num_singles, epoch, device, bestModel=False):
    model.eval()
    outputs_all = []
    labels_all = []
    preds_all = []
    correct = 0
    for batch in val_loader:
        images, _, labels = batch
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        preds = output.argmax(dim=1)
        correct += torch.sum(preds == labels.data)
        outputs_all.append(output)
        labels_all.append(labels)
        preds_all.append(preds)

    outputs_all = torch.cat(outputs_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    preds_all = torch.cat(preds_all, dim=0)
    acc = correct / len(labels_all)

    stat_result, GT_Pred_res = calculate_metrics_ENN(outputs_all, labels_all, R)

    avg_js_nonvague = stat_result[0] / (stat_result[2]+1e-10)
    # print(f"Average Jaccard Similarity Nonvague Classes:{avg_js_nonvague}")
    avg_js_vague = stat_result[1] / (stat_result[3]+1e-10)
    # print(f"Average Jaccard Similarity Vague Classes:{avg_js_vague}")
    overall_js = (stat_result[0] + stat_result[1])/(stat_result[2] + stat_result[3]+1e-10)
    # print(f"Overall JS: {overall_js}")
    js_result = [overall_js, avg_js_vague, avg_js_nonvague]
    
    # check precision, recall, f-score for composite classes
    prec_recall_f = precision_recall_f_v1(labels_all, preds_all, num_singles) 

    test_result_log(js_result, prec_recall_f, acc, epoch, bestModel)
    
    return acc 


@torch.no_grad()
def evaluate_nonvague_HENN_final(
    model,
    test_loader,
    K,
    device
    ):
    model.eval()
    output_all = []
    true_labels_all = []
    # losses = []
    for batch in test_loader:
        images, single_labels_GT, _ = batch
        images = images.to(device)
        labels = single_labels_GT.to(device)
        output = model(images)
        output_all.append(output)
        true_labels_all.append(labels)
    output_all = torch.cat(output_all, dim=0)
    true_labels = torch.cat(true_labels_all, dim=0)

    alpha = torch.add(output_all[:,:K], 1)
    # Get the predicted prob and labels
    p_exp = meanGDD(alpha, output_all)
    predicted_labels = torch.argmax(p_exp, dim=1) # 
    corr_num = torch.sum(true_labels.cpu() == predicted_labels.cpu())

    return corr_num / len(true_labels)


def precision_recall_f_v1(y_test, y_pred, num_singles):
    # make singleton labels 0, and composite labels 1
    y_test = y_test.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    y_test = y_test >= num_singles
    y_pred = y_pred >= num_singles
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    label_value_cnt = Counter(y_test)
    pred_value_cnt = Counter(y_pred)

    comp_GT_cnt = label_value_cnt[True]
    cmp_pred_cnt = pred_value_cnt[True]
    
    return precision, recall, f1, comp_GT_cnt, cmp_pred_cnt

# def precision_recall_f(GT_Pred_res): #todo: need to complete this
#     GTs = GT_Pred_res[0]
#     Predicteds = GT_Pred_res[1]
#     Predicteds_new = GT_Pred_res[2]
    
#     cnt_single = 0
#     cnt_comp = 0
#     cnt_corr = 0
#     cnt_wrong = 0

#     for idx, (gt, pred_1, pred_2) in enumerate(zip(GTs, Predicteds, Predicteds_new)):
#         if len(gt) == 1:
#     #         print(idx, gt, pred_1, pred_2)
#             if len(pred_2) == 1:
#                 cnt_single += 1
#                 if pred_2 == gt:
#                     cnt_corr += 1
#                 else:
#                     cnt_wrong += 1
#             else:
#                 cnt_comp += 1
#     print(cnt_single, cnt_comp)
#     print(cnt_corr, cnt_wrong)


def calculate_metrics(output, labels, R, K, W, a):
    GTs = []
    Predicteds = []
    Predicteds_new = []
    
    correct_vague = 0.0
    correct_nonvague = 0.0
    vague_total = 0
    nonvague_total = 0
    
    alpha = torch.add(output[:,:K], torch.mul(W, a))

    # Get the predicted labels
    p_exp = meanGDD(alpha, output)
    predicted_labels = torch.argmax(p_exp, dim=1) # 

    # Calculate vaguenesses
    b = output / (torch.sum(output, dim=1) + W)[:, None]
    total_vaguenesses = torch.sum(b[:, K:], dim=1)
    b_v = vague_belief_mass(b)

    for i in range(len(labels)):
        k = labels[i].item()
        predicted_set = set(R[torch.argmax(output[i])])
        Predicteds.append(predicted_set)
        
        if len(predicted_set) == 1:
            predicted_set = set(R[predicted_labels[i].item()])
        
        Predicteds_new.append(predicted_set)

        ground_truth_set = set(R[k])
        GTs.append(ground_truth_set)
        
        intersect = predicted_set.intersection(ground_truth_set)
        union = predicted_set.union(ground_truth_set)
        if len(predicted_set) == 1:
            correct_nonvague += float(len(intersect)) / len(union)
            nonvague_total += 1
        else:
            correct_vague += float(len(intersect)) / len(union)
            vague_total += 1
    stat_result = [correct_nonvague, correct_vague, nonvague_total, vague_total] #todo check this with calculate_metric
    GT_Pred_res = [GTs, Predicteds, Predicteds_new]
    return stat_result, GT_Pred_res


def calculate_metrics_ENN(output, labels, R):
    GTs = []
    Predicteds = []

    correct_vague = 0.0
    correct_nonvague = 0.0
    vague_total = 0
    nonvague_total = 0

    for i in range(len(labels)):
        k = labels[i].item()
        predicted_set = set(R[torch.argmax(output[i])])
        Predicteds.append(predicted_set)

        ground_truth_set = set(R[k])
        GTs.append(ground_truth_set)
        
        intersect = predicted_set.intersection(ground_truth_set)
        union = predicted_set.union(ground_truth_set)
        if len(predicted_set) == 1:
            correct_nonvague += float(len(intersect)) / len(union)
            nonvague_total += 1
        else:
            correct_vague += float(len(intersect)) / len(union)
            vague_total += 1
    stat_result = [correct_nonvague, correct_vague, nonvague_total, vague_total] #todo check this with calculate_metric
    GT_Pred_res = [GTs, Predicteds]
    return stat_result, GT_Pred_res


# def evaluate_set(model, data_loader, W, K, device):
#     vaguenesses = []
#     is_vague = []
#     for batch in data_loader:
#         images, labels = batch
#         images, labels = images.to(device), labels.to(device)
#         output = model(images)
#         b = output / (torch.sum(output, dim=1) + W)[:, None]
#         total_vaguenesses = torch.sum(b[:, K:], dim=1)
#         is_vague += [y >= K for y in labels.detach().cpu().numpy().tolist()]
#         vaguenesses += total_vaguenesses.detach().cpu().numpy().tolist()
#     return is_vague, vaguenesses         


# def draw_roc(model, data_loader):
#     is_vague, vaguenesses = evaluate_set(model, data_loader)
#     fpr, tpr, thresholds = metrics.roc_curve(is_vague, vaguenesses)
#     plt.plot(fpr, tpr)
#     plt.ylabel('True Positive Rate')
#     plt.xlabel('false Positive Rate')
#     plt.show()