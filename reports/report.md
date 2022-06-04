# Report about training model **TorchSensorNN**
## Architecture summary
```

TorchSensorNN(
  (sequential): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=256, out_features=600, bias=True)
    (2): ReLU()
    (3): Linear(in_features=600, out_features=600, bias=True)
    (4): ReLU()
    (5): Linear(in_features=600, out_features=600, bias=True)
    (6): ReLU()
    (7): Linear(in_features=600, out_features=4096, bias=True)
    (8): ReLU()
    (9): Unflatten(dim=1, unflattened_size=torch.Size([64, 64]))
  )
)

```
![l_curve](l_curve.png)