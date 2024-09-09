# NER-Net
Official repository for the CVPR 2024 paper, "[Seeing Motion at Nighttime with an Event Camera](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Seeing_Motion_at_Nighttime_with_an_Event_Camera_CVPR_2024_paper.pdf)”

[![CVPR2024 Seeing Motion at Nighttime with an Event Camera](https://i.ytimg.com/vi/zpfTLCF1Kw4/maxresdefault.jpg)](https://youtu.be/zpfTLCF1Kw4 "CVPR2024 Seeing Motion at Nighttime with an Event Camera")

## Real Low-light Event Dataset (RLED)

### Download

The publicly available RLED dataset includes 100 training sequences (about 171GB) and 12 test sequences (about 21GB), including 5 scenes such as city, suburbs, town, valley, and village. To reduce storage and transmission costs, all data is stored in .h5 format, and the images have been converted to grayscale.
We offer both Baidu Cloud Drive and Microsoft OneDrive for data download:

[Baidu Disk](https://pan.baidu.com/share/init?surl=h-iI5H_5DUIc2dQKvGXUkA) (code: `cknc`)

[OneDrive](https://1drv.ms/f/c/33d779a79b43175a/EvwW6jKyq_dOj7aP0IjXqJgBKNdz6yRvE7m0GRA-0ZPpyw?e=yUbQed)

RLED provides spatially and temporally aligned image GTs and low-light events:
![RLED_sample0](samples/RLED_sample_0.gif)
![RLED_sample1](samples/RLED_sample_1.gif)

### Data Format
The H5 file is based on [h5_tools](https://github.com/TimoStoff/events_contrast_maximization/tree/d6241dc90ec4dc2b4cffbb331a2389ff179bf7ab/tools).

The structure of the h5 file is the following:

```
├── city_0000_25fps.h5
│   └── events
│       ├── xs
│       ├── ys
│       ├── ts
│       ├── ps
│   └── images
│       ├── image000000000
        │   ├── event_idx
        │   ├── size
        │   └── timestamp
│       ├── image000000001
│       ├── image000000002
│       │   ...
│   └── event_indices
│
│   ─── attribute info
│       ├── duration
│       ├── num_events
│       ├── num_neg
│       ├── num_pos
│       ├── num_imgs
│       ├── sensor_resolution
│       ├── t0
        └── tk
```

- events: xs(column idx), ys(row idx), ts(timestamp in microseconds), ps(polarity).
- images: the collection of all images.
  - event_idx: the index of the event in the entire event stream sequence corresponding to the end of exposure time for the current image.
  - size: image size (height * width).
  - timestamp: the timestamp of the end of image exposure in microseconds.
- event_indices(Optional): An [n, 2] array, where n is the number of images in the current h5 file. To ensure a strict correspondence between events and images, we extend 10 ms before and after the midpoint of each image's exposure time and calculate the indices of the events corresponding to the start and end times.
- attribute info:
  - duration: the duration of the current h5 file.
  - num_events: the total number of events.
  - num_neg: the total number of negtive events.
  - num_pos: the total number of positive events.
  - num_imgs: the total number of images.
  - sensor_resolution: image size (width, height).
  - t0: the timestamp of the first event.
  - tk: the timestamp of the last event.

## Citation

Please cite us if our work is useful for your research, thank you!

```
@inproceedings{liu2024seeing,
  title={Seeing Motion at Nighttime with an Event Camera},
  author={Liu, Haoyue and Peng, Shihan and Zhu, Lin and Chang, Yi and Zhou, Hanyu and Yan, Luxin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={25648--25658},
  year={2024}
}
```