## Build and Run
```bash
docker build -t cuda .
docker run -it --rm --gpus all -v $(pwd):/workspace cuda bash run.sh  
```

## Results (aws ec2 g4dn.xlarge)
```
2D Convolution
Forward: 1.7635s (1763.5ms)
```