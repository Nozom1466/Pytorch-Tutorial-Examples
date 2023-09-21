## Takeaways

Fast Gradient Sign Attack (FGSA) is a strategy designed for adversarial sample generating. Gradients during backpropagation
are used to generate noise, which will be weighted by $\varepsilon$ and added to original image. ([paper](https://arxiv.org/abs/1412.6572))
The work on defense from attacks like FGSA also leas into the idea of making models more robust in general, 
to both naturally perturbed and adversarially crafted inputs.


![FGSA_panda](https://pytorch.org/tutorials/_images/fgsm_panda_image.png)
> (QUSTION) Does this strategy has something to do with data augmentation? 🤔:


### Load `.pth` checkpoint
Layer names and properties will be recorded in the `.pth` file, which means a new model with incorrect name (and a different parameters) 
will fail to load `.pth` file.


Newly restyled module `Net` failed to load checkpoint file `lenet_mnist_model.pth`
```python
# LeNet (restyled, CANNOT LOAD pretrained model .pth in tutorial)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32,
                                             kernel_size=3, stride=1),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=32, out_channels=364,
                                             kernel_size=3, stride=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2))

        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.5)

        self.MLP = nn.Sequential(nn.Flatten(),
                                 nn.Linear(in_features=9216, out_features=128),
                                 nn.ReLU())

        self.classifier = nn.Linear(in_features=128, out_features=10)


    def forward(self, x):
        x = self.convs(x)
        x = self.dropout1(x)
        x = self.MLP(x)
        x = self.dropout2(x)
        x = self.classifier(x)
        output = nn.LogSoftmax(x, dim=1)
        return output

```



### denormalization
```python
def denorm(batch, mean=[0.1307], std=[0.3081]):
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

```



### `imshow` image in `matplotlib`
The `adv_ex` is the misclassified image after FGSA attack
```python
adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
```


### `plt.tight_layout()`
Before:
```python
import matplotlib.pyplot as plt

#-- In your case, you'd do something more like:
# from matplotlib.figure import Figure
# fig = Figure()
#-- ...but we want to use it interactive for a quick example, so 
#--    we'll do it this way
fig, axes = plt.subplots(nrows=4, ncols=4)

for i, ax in enumerate(axes.flat, start=1):
    ax.set_title('Test Axes {}'.format(i))
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')

plt.show()
```
![Before tight layout](https://i.stack.imgur.com/tNHym.png)



After:
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=4, ncols=4)

for i, ax in enumerate(axes.flat, start=1):
    ax.set_title('Test Axes {}'.format(i))
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')

fig.tight_layout()

plt.show()

```
![After tight layout](https://i.stack.imgur.com/qGNyL.png)


