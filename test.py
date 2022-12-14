import torch
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score
from helper_functions import projection_prob, meanGDD, js_subset, vague_belief_mass
from loss import lossFunc
from collections import Counter
import wandb


def test_vague_result_log(
    js_result,
    prec_recall_f,
    acc,
    js_comp, js_singl,
    epoch, bestModel=False):
    if bestModel:
        tag = "TestB"
    else:
        if epoch is None:
            tag = "TestF"
        else:
            tag = "Test"
    wandb.log({
        f"{tag} JSoverall": js_result[0], 
        f"{tag} JScomp": js_result[1], 
        f"{tag} JSsngl": js_result[2],
        f"{tag} CmpPreci": prec_recall_f[0], 
        f"{tag} CmpRecal": prec_recall_f[1], 
        f"{tag} CmpFscor": prec_recall_f[2], 
        f"{tag} acc": acc,
        f"{tag} js_comp": js_comp,
        f"{tag} js_singl": js_singl}, step=epoch)
    print(f"{tag} acc: {acc:.4f},\n\
        JS(O_V_N): {js_result[0]:.4f}, {js_result[1]:.4f}, {js_result[2]:.4f},\n\
        P_R_F_compGTcnt_cmpPREDcnt: {prec_recall_f}\n")


def test_nonvague_result_log(
    nonvague_acc1, #from meanGDD
    nonvague_acc,  #from projection_prob
    nonvague_acc_singl, #from projection_prob based on singleton
    epoch, bestModel=False):
    if bestModel:
        tag = "TestB"
    else:
        if epoch is None:
            tag = "TestF"
        else:
            tag = "Test"
    wandb.log({
        f"{tag} nonVagueAcc1": nonvague_acc1, 
        f"{tag} nonVagueAcc": nonvague_acc, 
        f"{tag} nonVagueAccSingl": nonvague_acc_singl}, step=epoch)
    print(f"{tag} nonVagueAcc1: {nonvague_acc1:.4f},\n\
        nonVagueAcc: {nonvague_acc:.4f},\n\
        nonVagueAccSingl: {nonvague_acc_singl:.4f}\n")


def acc_subset(idx, labels_true, labels_pred):
    labels_true_subs = labels_true[idx]
    labels_pred_subs = labels_pred[idx]
    corr_subs = torch.sum(labels_true_subs == labels_pred_subs).item()
    acc_subs = corr_subs / len(labels_true_subs)
    return acc_subs


@torch.no_grad()
def evaluate_vague_nonvague_ENN(
    model, 
    val_loader, 
    R, 
    num_singles,
    num_comp,
    vague_classes_ids,
    epoch, 
    device, 
    bestModel=False):
    model.eval()
    outputs_all = []
    labels_all = [] # including composite labels
    true_labels_all = [] # singleton ground truth
    preds_all = []
    correct = 0
    for batch in val_loader:
        images, single_labels_GT, labels = batch
        images, labels = images.to(device), labels.to(device)
        single_labels_GT = single_labels_GT.to(device)
        output = model(images)
        preds = output.argmax(dim=1)
        correct += torch.sum(preds == labels.data)
        outputs_all.append(output)
        labels_all.append(labels)
        true_labels_all.append(single_labels_GT)
        preds_all.append(preds)

    outputs_all = torch.cat(outputs_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    true_labels = torch.cat(true_labels_all, dim=0)
    preds_all = torch.cat(preds_all, dim=0)
    acc = correct / len(labels_all)

    # calculate the accuracy among singleton examples
    # acc of composite examples
    comp_idx = labels_all > num_singles-1
    # acc_comp = acc_subset(comp_idx, labels_all, preds_all)
    js_comp = js_subset(comp_idx, labels_all, preds_all, R)
    # acc of singleton examples
    singl_idx = labels_all < num_singles
    # acc_singl = acc_subset(singl_idx, labels_all, preds_all)
    js_singl = js_subset(singl_idx, labels_all, preds_all, R)
    
    stat_result, GT_Pred_res = calculate_metrics_ENN(outputs_all, labels_all, R)
    avg_js_nonvague = stat_result[0] / (stat_result[2]+1e-10)
    avg_js_vague = stat_result[1] / (stat_result[3]+1e-10)
    overall_js = (stat_result[0] + stat_result[1])/(stat_result[2] + stat_result[3]+1e-10)
    js_result = [overall_js, avg_js_vague, avg_js_nonvague]

    # check precision, recall, f-score for composite classes
    prec_recall_f = precision_recall_f_v1(labels_all, preds_all, num_singles) 
    test_vague_result_log(js_result, prec_recall_f, acc, js_comp, js_singl, epoch, bestModel)

    ##### nonVagueAcc for all examples
    alpha = torch.add(outputs_all[:,:num_singles], 1)
    # Get the predicted prob and labels
    p_exp1 = meanGDD(vague_classes_ids, alpha, outputs_all, num_singles, num_comp, device)
    predicted_labels1 = torch.argmax(p_exp1, dim=1) # 
    corr_num1 = torch.sum(true_labels.cpu() == predicted_labels1.cpu())
    nonvague_acc_meanGDD = corr_num1 / len(true_labels)

    p_exp = projection_prob(num_singles, num_comp, R, outputs_all.cpu())
    predicted_labels = torch.argmax(p_exp, dim=1) # 
    pred_corr_or_not = true_labels.cpu() == predicted_labels.cpu()
    corr_num = torch.sum(pred_corr_or_not)
    nonvague_acc = corr_num / len(true_labels)

    ##### nonVagueAcc for singleton examples
    pred_corr_or_not_singl = pred_corr_or_not[singl_idx]
    corr_num_singl = torch.sum(pred_corr_or_not_singl)
    nonvague_acc_singl = corr_num_singl / len(pred_corr_or_not_singl)
    test_nonvague_result_log(nonvague_acc_meanGDD, nonvague_acc, nonvague_acc_singl, epoch, bestModel)

    return acc 


@torch.no_grad()
def evaluate_nonvague_HENN_final(
    model,
    test_loader,
    K,
    device,
    num_comp,
    vague_classes_ids,
    R
    ):
    model.eval()
    output_all = []
    true_labels_all = []
    for batch in test_loader:
        images, single_labels_GT, _ = batch
        images = images.to(device)
        labels = single_labels_GT.to(device)
        output = model(images)
        output_all.append(output)
        true_labels_all.append(labels)
    output_all = torch.cat(output_all, dim=0)
    true_labels = torch.cat(true_labels_all, dim=0)

    # nonVagueAcc for all examples
    alpha = torch.add(output_all[:,:K], 1)
    # Get the predicted prob and labels
    p_exp1 = meanGDD(vague_classes_ids, alpha, output_all, K, num_comp, device)
    predicted_labels1 = torch.argmax(p_exp1, dim=1) # 
    corr_num1 = torch.sum(true_labels.cpu() == predicted_labels1.cpu())
    acc1 = corr_num1 / len(true_labels)
    
    p_exp = projection_prob(K, num_comp, R, output_all.cpu())
    predicted_labels = torch.argmax(p_exp, dim=1) # 
    corr_num = torch.sum(true_labels.cpu() == predicted_labels.cpu())
    acc = corr_num / len(true_labels)

    # nonVagueAcc for singleton examples
    p_exp = projection_prob(K, num_comp, R, output_all.cpu())
    predicted_labels = torch.argmax(p_exp, dim=1) # 
    corr_num = torch.sum(true_labels.cpu() == predicted_labels.cpu())
    nonVagueAcc_singl = corr_num / len(true_labels)
    
    return acc1, acc, nonVagueAcc_singl


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


# def calculate_metrics(output, labels, R, K, W, a):
#     GTs = []
#     Predicteds = []
#     Predicteds_new = []
    
#     correct_vague = 0.0
#     correct_nonvague = 0.0
#     vague_total = 0
#     nonvague_total = 0
    
#     alpha = torch.add(output[:,:K], torch.mul(W, a))

#     # Get the predicted labels
#     p_exp = meanGDD(alpha, output)
#     predicted_labels = torch.argmax(p_exp, dim=1) # 

#     # Calculate vaguenesses
#     b = output / (torch.sum(output, dim=1) + W)[:, None]
#     total_vaguenesses = torch.sum(b[:, K:], dim=1)
#     b_v = vague_belief_mass(b)

#     for i in range(len(labels)):
#         k = labels[i].item()
#         predicted_set = set(R[torch.argmax(output[i])])
#         Predicteds.append(predicted_set)
        
#         if len(predicted_set) == 1:
#             predicted_set = set(R[predicted_labels[i].item()])
        
#         Predicteds_new.append(predicted_set)

#         ground_truth_set = set(R[k])
#         GTs.append(ground_truth_set)
        
#         intersect = predicted_set.intersection(ground_truth_set)
#         union = predicted_set.union(ground_truth_set)
#         if len(predicted_set) == 1:
#             correct_nonvague += float(len(intersect)) / len(union)
#             nonvague_total += 1
#         else:
#             correct_vague += float(len(intersect)) / len(union)
#             vague_total += 1
#     stat_result = [correct_nonvague, correct_vague, nonvague_total, vague_total] #todo check this with calculate_metric
#     GT_Pred_res = [GTs, Predicteds, Predicteds_new]
#     return stat_result, GT_Pred_res


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