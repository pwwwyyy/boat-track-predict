<div align="center">
<h2>1D-UNet for boat track predicting</h2>
</div>

## 📝 TODO
- [x] test data and scripts
- [x] normalizer modified
- [x] ......

## Requirements
> - Python 3.9, PyTorch >= 1.9.1
> - Platforms: Ubuntu 22.04, cuda 11.8
> - (similar env is ok)

## generate stimulation data
```bash
python generate_data.py
```

## Training
```bash
python main.py --train --config_path boat_track_stimu.cfg
```

## Testing
```bash

```
