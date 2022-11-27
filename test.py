import torch


@torch.no_grad()
def evaluate(model, val_loader, 
             num_single_classes, kappa, a_copy, 
             annealing_coefficient, 
             device):
    model.eval()
    results = {
                'accuracy': 0.0,
                'mean_val_loss': 0.0
              }
    total_correct = 0.0
    total_samples = 0
    val_losses = []
    for batch in val_loader:
        images, labels = batch
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


def vague_belief_mass(b):
    b_v = torch.zeros(len(b), K).to(device)
    sum_beliefs = torch.zeros(len(b), K).to(device)

    for k in range(K):
        for l in range(len(C)):
            relative_base_rate = torch.zeros(1).to(device)
            intersection_set = set(R[k]).intersection(set(C[l]))
            if len(intersection_set) > 0:
                relative_base_rate = a_copy[R.index(list(intersection_set))] / a_copy[K + l]
            sum_beliefs[:, k] = sum_beliefs[:, k] + relative_base_rate * b[:, K + l]
        b_v[:, k] = sum_beliefs[:,k]
    
    return b_v


def calculate_metrics(r, labels):
    correct_vague = 0.0
    correct_nonvague = 0.0
    vague_total = 0
    nonvague_total = 0
    
    alpha = torch.add(r[:,:K], torch.mul(W, a))

    # Get the predicted labels
    p_exp = meanGDD(alpha, r)
    predicted_labels = torch.argmax(p_exp, dim=1)

    # Calculate vaguenesses
    b = r / (torch.sum(r, dim=1) + W)[:, None]
    total_vaguenesses = torch.sum(b[:, K:], dim=1)
    b_v = vague_belief_mass(b)

    for i in range(len(labels)):
        k = labels[i].item()
        predicted_set = set(R[torch.argmax(r[i])])

        if len(predicted_set) == 1:
            predicted_set = set(R[predicted_labels[i].item()])

        ground_truth_set = set(R[k])

        if len(predicted_set) == 1:
            correct_nonvague += float(len(predicted_set.intersection(ground_truth_set))) / len(predicted_set.union(ground_truth_set))
            nonvague_total += 1
        else:
            correct_vague += float(len(predicted_set.intersection(ground_truth_set))) / len(predicted_set.union(ground_truth_set))
            vague_total += 1
      
    return [correct_nonvague, correct_vague, nonvague_total, vague_total]
