import torch

def train_model(epochs, model, lr, train_loader, val_loader, 
          num_single_classes, kappa, a_copy, device, 
          weight_decay=0, model_save_dir=""):
#     ignored_params = list(map(id, model.fc.parameters()))
#     base_params = filter(lambda p: id(p) not in ignored_params,
#                          model.parameters())

#     optimizer = torch.optim.Adam([
#                 {'params': base_params},
#                 {'params': model.fc.parameters(), 'lr': lr}
#             ], lr=lr*0.1, weight_decay=weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs_stage_1], gamma=0.1)
    global_step = 0
    annealing_step = 1000.0 * args.n_batches
    for epoch in range(epochs):
        # Training Phase 
        print(f" get last lr:{scheduler.get_last_lr()}")
        model.train()
        train_losses = []
        squared_losses = []
        kls = []
        for batch in train_loader:
            optimizer.zero_grad()
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=kappa)
            output = model(images)
            annealing_coefficient = min(1.0, global_step/annealing_step)
            loss = lossFunc(output, one_hot_labels, a_copy, num_single_classes, annealing_coefficient)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            # batch_loss, squared_loss, kl = lossFunc(r, one_hot_labels, a_copy, annealing_coefficient)
            # loss = torch.mean(batch_loss)
            # train_losses.append(loss)
            # squared_losses.append(torch.mean(squared_loss))
            # kls.append(torch.mean(kl))
            # loss.backward()
            # optimizer.step()
            global_step += 1
        mean_train_loss = torch.stack(train_losses).mean().item()
        # mean_squared_loss = torch.stack(squared_losses).mean().item()
        # mean_kl_loss = torch.stack(kls).mean().item()
        
        # Validation phase
        results = evaluate(model, val_loader, annealing_coefficient)
        print(f"Epoch [{epoch}], Mean Training Loss: {mean_train_loss:.4f}, Mean Validation Loss: {results['mean_val_loss']:.4f}, Validation Accuracy: {results['accuracy']:.4f}")
    
        if (epoch + 1) % 5 == 0:
            saved_path = os.path.join(model_save_dir, f'model_traditional_{epoch}.pt')
            torch.save(model.state_dict(), saved_path)
            # torch.save(model.state_dict(), PATH_HENN)
    return model 
