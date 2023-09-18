## Takeaways

> classical image classification task

### Modify on existing model
- finetuning
- as a feature extractor


### `data_transform` with `Dict`
```python
# data augmentation, in dict
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
```

### train model with validation & update best check points 
```python
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
  ...
  # create temporary to save training chekpoints
  with TemporaryDirectory() as tempdir:
    ...
    # begin training
    for epoch in range(num_epochs):
      ...
      for phase in ['train', 'val']:    # train & validation
        if phase == 'train':
          model.train()
        else:
          model.eval()

       ...

        for inputs, labels in dataloaders[phase]:
          ...
          # Context-manager that sets gradient calculation on or off
          with torch.set_grad_enabled(phase == 'train'):
            ...
            # backward + optimize only if in training phase
            if phase == 'train':
                loss.backward()
                optimizer.step()

          ...

        if phase == 'train':
          scheduler.step()
        ...
        # deep copy the model (update best model)
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), best_model_params_path)
      print()
    ...

    # load best model weights
    model.load_state_dict(torch.load(best_model_params_path))

  return model
```
